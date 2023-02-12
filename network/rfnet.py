from copy import deepcopy
import torch.nn as nn
import torch
import numpy as np
from itertools import chain # 串联多个迭代对象

from .util import _BNReluConv, _BNReluConv_adapter, upsample
from network.resnet.resnet_single_scale_single_attention import *
from network.resnet.resnet_single_scale_single_attention import resnet18_adapter

import torch.nn.functional as F

import PIL
import torchvision.transforms as transforms
import network.my_transforms as my_transforms

# 返回N张图片按channel为单位计算的std和mean
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    #计算（N,C,W*H)中第三维的var和mean
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

# 计算最小熵loss
def entropy_calc(x):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    
    # interp = nn.Upsample(size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
    # x = interp(x)
    
    x = F.softmax(x)
    
    assert x.dim() == 4
    n, c, h, w = x.size()

    # entropy_loss = -torch.sum(torch.mul(x, torch.log2(x + 1e-30)))
    # entropy_loss1 = entropy_loss / (n * h * w * np.log2(c))

    entropy_map = torch.cuda.FloatTensor(x.shape[2], x.shape[3]).fill_(0)
    for batch_idx in range(x.shape[0]):
        for c in range(x.shape[1]):
            entropy_map = entropy_map - (x[batch_idx, c, :, :] * torch.log2(x[batch_idx, c, :, :] + 1e-30))
    entropy_loss2 = entropy_map.mean()
    
    return entropy_loss2

# 计算最小熵loss
def entropy_calc_two(x1, x2): 
    x1 = F.softmax(x1)
    x2 = F.softmax(x2)
    
    assert x1.dim() == 4
    n, c, h, w = x1.size()
    entropy_loss = -torch.sum(torch.mul(x2, torch.log2(x1 + 1e-30))) / (n * h * w * np.log2(c))
    
    return entropy_loss

class RFNet(nn.Module):
    def __init__(self, num_classes, backbone, criterion=None, criterion_aux=None, args=None):
        super(RFNet, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.criterion = criterion
        self.criterion_aux = criterion_aux
        self.args = args
        self.logits = _BNReluConv(self.backbone.num_features, self.num_classes, batch_norm=True)

    def forward(self, x, gts=None, aux_gts=None, img_gt=None, visualize=False, cal_covstat=False, apply_wtloss=True, dsbn=False, mode=None, domain_label=None):
        x_size = x.size()
        x, _ = self.backbone(x, None)
        logits = self.logits.forward(x)
        logits_upsample = upsample(logits, x_size[2:])

        # 当前是训练/推理阶段
        if self.training:
            loss1 = self.criterion(logits_upsample, gts)
            loss2 = loss1 # 没有设置aux_loss
            return_loss = [loss1, loss2]

            return return_loss
        else:
            return logits_upsample

    def random_init_params(self):
        return chain(*([self.logits.parameters(), self.backbone.random_init_params()]))

    def fine_tune_params(self):
        return self.backbone.fine_tune_params()

    def calc_style_std_mean(self, target):
        input_mean, input_std = calc_mean_std(target)
        input_std=input_std.cpu()
        input_mean=input_mean.cpu()
        mean = input_mean.detach().numpy()
        std = input_std.detach().numpy()

        return np.append(mean, std)


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor

def get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False):
    # img_shape = (32, 32, 3)
    # n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose([
        my_transforms.Clip(0.0, 1.0), 
        my_transforms.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        # transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),  
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            resample=PIL.Image.BILINEAR,
            fillcolor=None
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        # transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        my_transforms.GaussianNoise(0, gaussian_std),
        my_transforms.Clip(clip_min, clip_max)
    ])
    return tta_transforms

def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

class RFNet_adapter(nn.Module):
    def __init__(self, num_classes, backbone, criterion=None, criterion_aux=None, args=None):
        super(RFNet_adapter, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.criterion = criterion
        self.criterion_aux = criterion_aux
        self.args = args
        self.logits = _BNReluConv_adapter(self.backbone.num_features, self.num_classes, batch_norm=True)
        self.transform = get_tta_transforms()

    def forward(self, x, gts=None, aux_gts=None, img_gt=None, visualize=False, cal_covstat=False, apply_wtloss=True, dsbn=False, mode=None, domain_label=None, task=None, run_mode=None, method=None):
        x_size = x.size()
        output, _, style_feats = self.backbone(x, task=task)
        logits = self.logits.forward(output, task=task)
        logits_upsample = upsample(logits, x_size[2:])

        # 计算style_code
        style_code = self.calc_style_std_mean(torch.unsqueeze(style_feats[0][0, :, :, :], 0))
        style_code = np.append(style_code, self.calc_style_std_mean(torch.unsqueeze(style_feats[1][0, :, :, :], 0)))

        # 当前是推理时适应/训练/推理阶段
        if self.training and run_mode == "test-adaptation": # 推理时适应
            # assuming that ground truth is accessable
            if method == "PL_real":
                loss1 = self.criterion(logits_upsample, gts)
                loss2 = loss1 # 没有设置aux_loss
                return_loss = [loss1, loss2]
            # hard label self-training
            elif method == "PL_hard":
                gts = logits_upsample.data.max(1)[1]
                loss1 = self.criterion(logits_upsample, gts)
                loss2 = loss1 # 没有设置aux_loss
                return_loss = [loss1, loss2]
            # only for CoTTA
            elif method == "CoTTA":
                # Teacher prediction
                # output_anchor, _, _ = self.backbone_anchor(x, task=task)
                # logits_anchor = self.logits_anchor.forward(output_anchor, task=task)
                # logts_upsample_anchor = upsample(logits_anchor, x_size[2:])
                output_ema_standard, _, _ = self.ema_backbone(x, task=task)
                logits_ema_standard = self.ema_logits(output_ema_standard, task=task)
                logits_upsample_ema_standard = upsample(logits_ema_standard, x_size[2:])
                if self.args.CoTTA_Aug:
                    # Augmentation-averaged Prediction
                    N = 32 
                    outputs_emas = []
                    for i in range(N):
                        output_ema, _, _ = self.ema_backbone(self.transform(x), task=task)
                        output_ema = output_ema.detach()
                        logits_ema = self.ema_logits(output_ema, task=task).detach()
                        logits_upsample_ema = upsample(logits_ema, x_size[2:]).detach()
                        outputs_emas.append(logits_upsample_ema)
                    outputs_ema = torch.stack(outputs_emas).mean(0)
                else:
                    outputs_ema = logits_upsample_ema_standard
                # student loss
                loss1 = entropy_calc_two(logits_upsample, outputs_ema)
                loss2 = loss1 # 没有设置aux_loss
                return_loss = [loss1, loss2]
            # entropy minimization
            else:
                loss1 = entropy_calc(logits_upsample)
                loss2 = loss1 # 没有设置aux_loss
                return_loss = [loss1, loss2]

            return return_loss, logits_upsample, style_code

        elif self.training and run_mode == "train": # 训练
            loss1 = self.criterion(logits_upsample, gts)
            loss2 = loss1 # 没有设置aux_loss
            return_loss = [loss1, loss2]

            return return_loss

        else: # 推理
            return 0, logits_upsample, style_code

    def random_init_params(self):
        return chain(*([self.logits.parameters(), self.backbone.random_init_params()]))

    def fine_tune_params(self):
        return self.backbone.fine_tune_params()

    def calc_style_std_mean(self, target):
        input_mean, input_std = calc_mean_std(target)
        input_std=input_std.cpu()
        input_mean=input_mean.cpu()
        mean = input_mean.detach().numpy()
        std = input_std.detach().numpy()

        return np.append(mean, std)
    
    def copy_state(self):
        if self.args.method == "CoTTA":
            self.backbone_state = deepcopy(self.backbone.state_dict())
            self.logits_state = deepcopy(self.logits.state_dict())
            # self.backbone_anchor = deepcopy(self.backbone)
            # self.logits_anchor = deepcopy(self.logits)
            self.ema_backbone = deepcopy(self.backbone)
            self.ema_logits = deepcopy(self.logits)
            for param in self.ema_backbone.parameters():
                param.detach_()
            for param in self.ema_logits.parameters():
                param.detach_()
    
    def teacher_update(self):
        self.ema_backbone = update_ema_variables(ema_model = self.ema_backbone, model = self.backbone, alpha_teacher=0.99)
        self.ema_logits = update_ema_variables(ema_model = self.ema_logits, model = self.logits, alpha_teacher=0.99)

    def restore(self):
        if True:
            for nm, m  in self.backbone.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape)<0.1).float().cuda() 
                        with torch.no_grad():
                            p.data = self.backbone_state[f"{nm}.{npp}"] * mask + p * (1.-mask)
            for nm, m  in self.logits.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape)<0.1).float().cuda() 
                        with torch.no_grad():
                            p.data = self.logits_state[f"{nm}.{npp}"] * mask + p * (1.-mask)


def rfnet(args, num_classes, criterion, criterion_aux):
    """
    rfnet Network
    """
    resnet = resnet18(pretrained=True, efficient=False, use_bn= True) # 骨干网络选取的是ResNet-18

    print("Model : RFNet, Backbone : ResNet-18")
    return RFNet(num_classes, resnet, criterion=criterion, criterion_aux=criterion_aux, args=args)

def rfnet_adapter(args, num_classes, criterion, criterion_aux):
    """
    rfnet Network
    """
    resnet = resnet18_adapter(pretrained=False, efficient=False, use_bn= True) # 骨干网络选取的是ResNet-18

    print("Model : RFNet, Backbone : ResNet-18_adapter")
    return RFNet_adapter(num_classes, resnet, criterion=criterion, criterion_aux=criterion_aux, args=args)