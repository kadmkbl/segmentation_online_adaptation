# -*- coding: utf-8 -*-
from torch.utils import model_zoo

import numpy as np
from network import Resnet, VGG
from network.oprations import *
from network.oprations import ASPP_module_adapter
from network.cov_settings import CovMatrix_ISW, CovMatrix_IRW
from network.instance_whitening import instance_whitening_loss, get_covariance_matrix
from network.mynn import initialize_weights, Norm2d, Upsample, freeze_weights, unfreeze_weights

import torchvision.models as models

#返回N张图片按channel为单位计算的std和mean
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

class DeepV2(nn.Module):
    def __init__(self, num_classes, trunk='resnet-101', pyramids=[6, 12, 18, 24], criterion=None, criterion_aux=None,
                variant='D', skip='m1', skip_num=48, args=None):
        super(DeepV2, self).__init__()
        self.criterion = criterion
        self.criterion_aux = criterion_aux
        self.variant = variant
        self.args = args
        self.trunk = trunk
        self.dsbn = args.dsbn
        self.num_domains = args.num_domains


        if trunk == 'resnet-18':
            resnet = Resnet.resnet18(wt_layer=self.args.wt_layer)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnet-50': # Deeplabv3+ 之 ResNet-50 from ResNet子类
            if self.dsbn:
                resnet = Resnet.dsbnresnet50(wt_layer=self.args.wt_layer, num_domains=self.num_domains) # DSBN版
                resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool) # 把ResNet残差模块之前的部分单独弄了一个 —— conv1 → bn1 → relu → maxpool
            else:
                resnet = Resnet.resnet50(wt_layer=self.args.wt_layer)
                resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool) # 把ResNet残差模块之前的部分单独弄了一个 —— conv1 → bn1 → relu → maxpool
        elif trunk == 'resnet-101':
            """ three 3 X 3
            resnet = Resnet.resnet101(pretrained=True, wt_layer=self.args.wt_layer)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1,
                                        resnet.conv2, resnet.bn2, resnet.relu2,
                                        resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            """
            # 7 X 7
            resnet = Resnet.resnet101(pretrained=True, wt_layer=self.args.wt_layer)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool) # 把ResNet残差模块之前的部分单独弄了一个 —— conv1 → bn1 → relu → maxpool
        elif trunk == 'resnet-152':
            resnet = Resnet.resnet152()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnext-50':
            resnet = models.resnext50_32x4d(pretrained=True)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnext-101':
            resnet = models.resnext101_32x8d(pretrained=True)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'wide_resnet-50':
            resnet = models.wide_resnet50_2(pretrained=True)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'wide_resnet-101':
            resnet = models.wide_resnet101_2(pretrained=True)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            raise ValueError("Not a valid network arch")

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4 # ResNet主干

        self.aspp1 = ASPP_module(2048, num_classes, pyramids[0])
        self.aspp2 = ASPP_module(2048, num_classes, pyramids[1])
        self.aspp3 = ASPP_module(2048, num_classes, pyramids[2])
        self.aspp4 = ASPP_module(2048, num_classes, pyramids[3])

        initialize_weights(self.aspp1)
        initialize_weights(self.aspp2)
        initialize_weights(self.aspp3)
        initialize_weights(self.aspp4)

    def forward(self, x, gts=None, aux_gts=None, img_gt=None, visualize=False, cal_covstat=False, apply_wtloss=True, dsbn=False, mode=None, domain_label=None, task=None):
        w_arr = []
        
        if cal_covstat:
            x = torch.cat(x, dim=0)
        
        x_size = x.size()  # 800
        x = self.layer0[0](x)
        if self.args.wt_layer[2] == 1 or self.args.wt_layer[2] == 2:
            x, w = self.layer0[1](x)
            w_arr.append(w)
        else:
            if dsbn:
                x, _ = self.layer0[1](x, domain_label=domain_label)
            else:
                x = self.layer0[1](x)
        x = self.layer0[2](x)
        x = self.layer0[3](x)

        x_tuple = self.layer1([x, w_arr]) 
        x_tuple = self.layer2(x_tuple)
        x_tuple = self.layer3(x_tuple)
        x_tuple = self.layer4(x_tuple)
        x = x_tuple[0]
        w_arr = x_tuple[1]

        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        x = x1 + x2 + x3 + x4

        main_out = F.upsample(x, size=x_size[2:], mode='bilinear', align_corners=True)
        
        # 当前是训练/推理阶段
        if self.training:
            loss1 = self.criterion(main_out, gts)
            loss2 = loss1 # 没有设置aux_loss
            return_loss = [loss1, loss2]

            return return_loss
        else:
            return main_out

    def get_1x_lr_params(self):
        b = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        for i in range(len(b)):
            for k in b[i].parameters():
                if k.requires_grad:
                    yield k

    def get_10x_lr_params(self):
        b = [self.aspp1, self.aspp2, self.aspp3, self.aspp4]
        for j in range(len(b)):
            for k in b[j].parameters():
                if k.requires_grad:
                    yield k

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze_backbone_bn(self):
        self.bn1.eval()

        for m in self.layer1:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        for m in self.layer2:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        for m in self.layer3:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        for m in self.layer4:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

class DeepV2VGG(nn.Module):
    def __init__(self, num_classes, trunk='vgg-16', pyramids=[6, 12, 18, 24], criterion=None, criterion_aux=None, args=None):
        super(DeepV2VGG, self).__init__()
        self.criterion = criterion
        self.criterion_aux = criterion_aux
        self.args = args
        self.trunk = trunk

        if trunk == 'vgg-16':
            resnet = VGG.vgg16_bn()
        else:
            raise ValueError("Not a valid network arch")

        self.layer1, self.layer2, self.layer3, self.layer4, self.layer5 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4, resnet.layer5 # ResNet主干

        self.aspp1 = ASPP_module(512, num_classes, pyramids[0])
        self.aspp2 = ASPP_module(512, num_classes, pyramids[1])
        self.aspp3 = ASPP_module(512, num_classes, pyramids[2])
        self.aspp4 = ASPP_module(512, num_classes, pyramids[3])

        initialize_weights(self.aspp1)
        initialize_weights(self.aspp2)
        initialize_weights(self.aspp3)
        initialize_weights(self.aspp4)

    def forward(self, x, gts=None, aux_gts=None, img_gt=None, visualize=False, cal_covstat=False, apply_wtloss=True, dsbn=False, mode=None, domain_label=None, task=None):
        if cal_covstat:
            x = torch.cat(x, dim=0)
        
        x_size = x.size()  # 800
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        x1 = self.aspp1(out)
        x2 = self.aspp2(out)
        x3 = self.aspp3(out)
        x4 = self.aspp4(out)

        x = x1 + x2 + x3 + x4

        main_out = F.upsample(x, size=x_size[2:], mode='bilinear', align_corners=True)
        
        # 当前是训练/推理阶段
        if self.training:
            loss1 = self.criterion(main_out, gts)
            loss2 = loss1 # 没有设置aux_loss
            return_loss = [loss1, loss2]

            return return_loss
        else:
            return main_out

    def get_1x_lr_params(self):
        b = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        for i in range(len(b)):
            for k in b[i].parameters():
                if k.requires_grad:
                    yield k

    def get_10x_lr_params(self):
        b = [self.aspp1, self.aspp2, self.aspp3, self.aspp4]
        for j in range(len(b)):
            for k in b[j].parameters():
                if k.requires_grad:
                    yield k

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze_backbone_bn(self):
        self.bn1.eval()

        for m in self.layer1:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        for m in self.layer2:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        for m in self.layer3:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        for m in self.layer4:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

class DeepV2VGG_adapter(nn.Module):
    def __init__(self, num_classes, trunk='vgg-16', pyramids=[6, 12, 18, 24], criterion=None, criterion_aux=None, args=None):
        super(DeepV2VGG_adapter, self).__init__()
        self.criterion = criterion
        self.criterion_aux = criterion_aux
        self.args = args
        self.trunk = trunk

        if trunk == 'vgg-16':
            resnet = VGG.vgg16_bn_adapter()
        else:
            raise ValueError("Not a valid network arch")

        self.layer1, self.layer2, self.layer3, self.layer4, self.layer5 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4, resnet.layer5 # ResNet主干

        self.aspp1 = ASPP_module_adapter(512, num_classes, pyramids[0])
        self.aspp2 = ASPP_module_adapter(512, num_classes, pyramids[1])
        self.aspp3 = ASPP_module_adapter(512, num_classes, pyramids[2])
        self.aspp4 = ASPP_module_adapter(512, num_classes, pyramids[3])

        initialize_weights(self.aspp1)
        initialize_weights(self.aspp2)
        initialize_weights(self.aspp3)
        initialize_weights(self.aspp4)

    def forward(self, x, gts=None, aux_gts=None, img_gt=None, visualize=False, cal_covstat=False, apply_wtloss=True, dsbn=False, mode=None, domain_label=None, task=None):
        if cal_covstat:
            x = torch.cat(x, dim=0)

        style_feats = []

        x_size = x.size()  # 800
        out = self.layer1(x)
        style_feats.append(out)
        out = self.layer2(out)
        style_feats.append(out)
        out = self.layer3(out, task=task)
        out = self.layer4(out, task=task)
        out = self.layer5(out, task=task)

        # 计算style_code
        style_code = self.calc_style_std_mean(torch.unsqueeze(style_feats[0][0, :, :, :], 0))
        style_code = np.append(style_code, self.calc_style_std_mean(torch.unsqueeze(style_feats[1][0, :, :, :], 0)))

        x1 = self.aspp1(out, task=task)
        x2 = self.aspp2(out, task=task)
        x3 = self.aspp3(out, task=task)
        x4 = self.aspp4(out, task=task)

        x = x1 + x2 + x3 + x4

        main_out = F.upsample(x, size=x_size[2:], mode='bilinear', align_corners=True)
        
        # 当前是训练/推理阶段
        if self.training:
            loss1 = self.criterion(main_out, gts)
            loss2 = loss1 # 没有设置aux_loss
            return_loss = [loss1, loss2]

            return return_loss
        else:
            return main_out, style_code

    def get_1x_lr_params(self):
        b = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        for i in range(len(b)):
            for k in b[i].parameters():
                if k.requires_grad:
                    yield k

    def get_10x_lr_params(self):
        b = [self.aspp1, self.aspp2, self.aspp3, self.aspp4]
        for j in range(len(b)):
            for k in b[j].parameters():
                if k.requires_grad:
                    yield k

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze_backbone_bn(self):
        self.bn1.eval()

        for m in self.layer1:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        for m in self.layer2:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        for m in self.layer3:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        for m in self.layer4:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def calc_style_std_mean(self, target):
        input_mean, input_std = calc_mean_std(target)
        input_std=input_std.cpu()
        input_mean=input_mean.cpu()
        mean = input_mean.detach().numpy()
        std = input_std.detach().numpy()

        return np.append(mean, std)


def DeepR101V2(args, num_classes, criterion, criterion_aux):
    """
    Resnet 101 Based Network
    """
    print("Model : DeepLabv2, Backbone : ResNet-101")
    return DeepV2(num_classes, trunk='resnet-101',  pyramids=[6, 12, 18, 24], criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)

def DeepV16V2(args, num_classes, criterion, criterion_aux):
    """
    VGG 16 Based Network
    """
    print("Model : DeepLabv2, Backbone : VGG-16")
    return DeepV2VGG(num_classes, trunk='vgg-16', pyramids=[6, 12, 18, 24], criterion=criterion, criterion_aux=criterion_aux, args=args)

def DeepV16V2_adapter(args, num_classes, criterion, criterion_aux):
    """
    VGG 16 Based Network
    """
    print("Model : DeepLabv2_adapter, Backbone : VGG-16")
    return DeepV2VGG_adapter(num_classes, trunk='vgg-16', pyramids=[6, 12, 18, 24], criterion=criterion, criterion_aux=criterion_aux, args=args)