# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import network.mynn as mynn
import numpy as np

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


def conv1x1_fonc(in_planes, out_planes=None, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)

class conv1x1(nn.Module):
    def __init__(self, planes, out_planes=None, stride=1):
        super(conv1x1, self).__init__()
        self.conv = conv1x1_fonc(planes, out_planes, stride) 

    def forward(self, x):
        y = self.conv(x)
        return y

def conv1x1_transpose_fonc(in_planes, out_planes=None, stride=1, bias=False):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, output_padding=1, bias=bias)

class conv1x1_transpose(nn.Module):
    def __init__(self, planes, out_planes=None, stride=1):
        super(conv1x1_transpose, self).__init__()
        self.conv = conv1x1_transpose_fonc(planes, out_planes, stride) 

    def forward(self, x):
        y = self.conv(x)
        return y


class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        self.a = self.conv(input)
        self.b = self.pool(input)
        output = torch.cat([self.a, self.b], 1)
        output = self.bn(output)
        return F.relu(output)

class DownsamplerBlock_adapter(nn.Module):
    tasks_preset = 100
    
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.parallel_convs = nn.ModuleList([conv1x1(ninput, noutput-ninput, stride=2) for i in range(self.tasks_preset)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(noutput, eps=1e-3) for i in range(self.tasks_preset)])
        self.pool = nn.MaxPool2d(2, stride=2)
        # self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input, task=None):
        self.a = self.conv(input)
        # self.a = self.a + self.parallel_convs[task](input)
        self.b = self.pool(input)
        output = torch.cat([self.a, self.b], 1)
        # output = self.bn(output)
        output = self.bns[task](output)
        return F.relu(output)
    

class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)
        

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        return F.relu(output+input)    #+input = identity (residual connection)
    
class non_bottleneck_1d_adapter(nn.Module):
    tasks_preset = 100
    
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)
        
        self.parallel_convs1 = nn.ModuleList([conv1x1(chann, chann, stride=1) for i in range(self.tasks_preset)])

        self.bns1 = nn.ModuleList([nn.BatchNorm2d(chann, eps=1e-03) for i in range(self.tasks_preset)])

        # self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1, dilated))

        self.parallel_convs2 = nn.ModuleList([conv1x1(chann, chann, stride=1) for i in range(self.tasks_preset)])

        self.bns2 = nn.ModuleList([nn.BatchNorm2d(chann, eps=1e-03) for i in range(self.tasks_preset)])

        # self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)
        

    def forward(self, input, task=None):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        # output = output + self.parallel_convs1[task](input)
        output = self.bns1[task](output)
        # output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        # output = output + self.parallel_convs2[task](output)
        output = self.bns2[task](output)
        # output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        return F.relu(output+input)    #+input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3,16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16,64))

        for x in range(0, 5):    #5 times
           self.layers.append(non_bottleneck_1d(64, 0.03, 1)) 

        self.layers.append(DownsamplerBlock(64,128))

        for x in range(0, 2):    #2 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

        #Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output

class Encoder_adapter(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3,16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16,64))

        for x in range(0, 5):    #5 times
           self.layers.append(non_bottleneck_1d(64, 0.03, 1)) 

        # specific adapters
        self.layers.append(DownsamplerBlock_adapter(64,128))

        for x in range(0, 2):    #2 times
            self.layers.append(non_bottleneck_1d_adapter(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d_adapter(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d_adapter(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d_adapter(128, 0.3, 16))

        #Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False, task=None):
        style_feats = []
        output = self.initial_block(input)        
        style_feats.append(output)

        cnt = 0
        for layer in self.layers:
            if cnt < 6:
                output = layer(output)
                style_feats.append(output)
                cnt = cnt + 1
            else:
                output = layer(output, task=task)

        if predict:
            output = self.output_conv(output)

        return output, style_feats


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)

class UpsamplerBlock_adapter(nn.Module):
    tasks_preset = 100
    
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.parallel_convs = nn.ModuleList([conv1x1_transpose(ninput, noutput, stride=2) for i in range(self.tasks_preset)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(noutput, eps=1e-3) for i in range(self.tasks_preset)])
        # self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input, task=None):
        output = self.conv(input)
        # output = output + self.parallel_convs[task](input)
        output = self.bns[task](output)
        # output = self.bn(output)
        return F.relu(output)


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128,64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64,16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d( 16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output

class Decoder_adapter(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock_adapter(128,64))
        self.layers.append(non_bottleneck_1d_adapter(64, 0, 1))
        self.layers.append(non_bottleneck_1d_adapter(64, 0, 1))

        self.layers.append(UpsamplerBlock_adapter(64,16))
        self.layers.append(non_bottleneck_1d_adapter(16, 0, 1))
        self.layers.append(non_bottleneck_1d_adapter(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input, task=None):
        output = input

        for layer in self.layers:
            output = layer(output, task=task)

        output = self.output_conv(output)

        return output

#ERFNet
class ERFNet(nn.Module):
    def __init__(self, num_classes, encoder=None, pyramids=[6, 12, 18, 24], criterion=None, criterion_aux=None,
                variant='D', skip='m1', skip_num=48, args=None):
        super(ERFNet, self).__init__()
        self.criterion = criterion
        self.criterion_aux = criterion_aux
        self.variant = variant
        self.args = args
        self.encoder = encoder
        self.dsbn = args.dsbn
        self.num_domains = args.num_domains

        if (self.encoder == None):
            self.encoder = Encoder(num_classes)
        else:
            self.encoder = self.encoder
        self.decoder = Decoder(num_classes)

    def forward(self, x, gts=None, aux_gts=None, img_gt=None, visualize=False, cal_covstat=False, apply_wtloss=True, dsbn=False, mode=None, domain_label=None, only_encode=False):     
        if cal_covstat:
            x = torch.cat(x, dim=0)

        if only_encode:
            return self.encoder.forward(x, predict=True)
        else:
            output = self.encoder(x)    #predict=False by default
            main_out = self.decoder.forward(output)

            # 当前是训练/推理阶段
            if self.training:
                loss1 = self.criterion(main_out, gts)
                loss2 = loss1 # 没有设置aux_loss
                return_loss = [loss1, loss2]

                return return_loss
            else:
                return main_out

#ERFNet_adapter
class ERFNet_adapter(nn.Module):
    def __init__(self, num_classes, encoder=None, pyramids=[6, 12, 18, 24], criterion=None, criterion_aux=None,
                variant='D', skip='m1', skip_num=48, args=None):
        super(ERFNet_adapter, self).__init__()
        self.criterion = criterion
        self.criterion_aux = criterion_aux
        self.variant = variant
        self.args = args
        self.encoder = encoder
        self.dsbn = args.dsbn
        self.num_domains = args.num_domains

        if (self.encoder == None):
            self.encoder = Encoder_adapter(num_classes)
        else:
            self.encoder = self.encoder
        self.decoder = Decoder_adapter(num_classes)

    def forward(self, x, gts=None, aux_gts=None, img_gt=None, visualize=False, cal_covstat=False, apply_wtloss=True, dsbn=False, mode=None, domain_label=None, only_encode=False, task=None):     
        if cal_covstat:
            x = torch.cat(x, dim=0)

        if only_encode:
            return self.encoder.forward(x, predict=True)
        else:
            output, style_feats = self.encoder(x, task=task)    #predict=False by default
            main_out = self.decoder.forward(output, task=task)

            # 计算style_code
            style_code = self.calc_style_std_mean(torch.unsqueeze(style_feats[0][0, :, :, :], 0))
            style_code = np.append(style_code, self.calc_style_std_mean(torch.unsqueeze(style_feats[1][0, :, :, :], 0)))
            style_code = np.append(style_code, self.calc_style_std_mean(torch.unsqueeze(style_feats[2][0, :, :, :], 0)))
            style_code = np.append(style_code, self.calc_style_std_mean(torch.unsqueeze(style_feats[3][0, :, :, :], 0)))
            style_code = np.append(style_code, self.calc_style_std_mean(torch.unsqueeze(style_feats[4][0, :, :, :], 0)))
            style_code = np.append(style_code, self.calc_style_std_mean(torch.unsqueeze(style_feats[5][0, :, :, :], 0)))
            style_code = np.append(style_code, self.calc_style_std_mean(torch.unsqueeze(style_feats[6][0, :, :, :], 0)))

            # 当前是训练/推理阶段
            if self.training:
                loss1 = self.criterion(main_out, gts)
                loss2 = loss1 # 没有设置aux_loss
                return_loss = [loss1, loss2]

                return return_loss
            else:
                return main_out, style_code
            
    def calc_style_std_mean(self, target):
        input_mean, input_std = calc_mean_std(target)
        input_std=input_std.cpu()
        input_mean=input_mean.cpu()
        mean = input_mean.detach().numpy()
        std = input_std.detach().numpy()

        return np.append(mean, std)


def erfnet(args, num_classes, criterion, criterion_aux, pretrained=True):
    """
    ERFNet
    """
    print("Model : ERFNet")
    model = ERFNet(num_classes, encoder=None,  pyramids=[6, 12, 18, 24], criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
        print("########### pretrained ##############")
        # model.load_state_dict(torch.load('./pretrained/resnet101-imagenet.pth', map_location="cpu"))
        mynn.forgiving_state_restore(model, torch.load('/data/user21100736/Work/Pretrained_Models/erfnet_encoder_pretrained.pth.tar', map_location="cpu"))

    return model

def erfnet_adapter(args, num_classes, criterion, criterion_aux, pretrained=True):
    """
    ERFNet
    """
    print("Model : ERFNet_adapter")
    model = ERFNet_adapter(num_classes, encoder=None,  pyramids=[6, 12, 18, 24], criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
        print("########### pretrained ##############")
        # model.load_state_dict(torch.load('./pretrained/resnet101-imagenet.pth', map_location="cpu"))
        mynn.forgiving_state_restore(model, torch.load('/data/user21100736/Work/Pretrained_Models/erfnet_encoder_pretrained.pth.tar', map_location="cpu"))

    return model