import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import network.mynn as mynn


__all__ = ['VGG16_bn']

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
}


def conv1x1_fonc(in_planes, out_planes=None, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)

class conv1x1(nn.Module):
    def __init__(self, planes, out_planes=None, stride=1):
        super(conv1x1, self).__init__()
        self.conv = conv1x1_fonc(planes, out_planes, stride) 

    def forward(self, x):
        y = self.conv(x)
        return y


# VGG模块
# def conv_layer(chann_in, chann_out, k_size, p_size):
#     layer = nn.Sequential(
#         nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
#         nn.BatchNorm2d(chann_out),
#         nn.ReLU()
#     )
#     return layer

class conv_layer(nn.Module):
    def __init__(self, chann_in, chann_out, k_size, p_size):
        super(conv_layer, self).__init__()
        self.conv = nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size)
        self.bn = nn.BatchNorm2d(chann_out)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x

class conv_layer_adapter(nn.Module):
    tasks_preset = 100

    def __init__(self, chann_in, chann_out, k_size, p_size):
        super(conv_layer_adapter, self).__init__()
        self.conv = nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size)
        self.parallel_conv = nn.ModuleList([conv1x1(chann_in, chann_out, stride=1) for i in range(self.tasks_preset)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(chann_out) for i in range(self.tasks_preset)])
        # self.bn = nn.BatchNorm2d(chann_out)
        self.relu = nn.ReLU()
    
    def forward(self, x, task=None):
        out = self.conv(x)
        # out = out + self.parallel_conv[task](x)
        out = self.bns[task](out)
        # out = self.bn(out)
        out = self.relu(out)
        
        return out


# def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s, num_domains=1):
#     layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
#     layers += [nn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
#     return nn.Sequential(*layers)

class vgg_conv_block(nn.Module):
    def __init__(self, in_list, out_list, k_list, p_list, pooling_k, pooling_s, num_domains=1):
        super(vgg_conv_block, self).__init__()
        if len(in_list) == 2:
            self.layercnt = 2
            self.layer0 = conv_layer(in_list[0], out_list[0], k_list[0], p_list[0])
            self.layer1 = conv_layer(in_list[1], out_list[1], k_list[1], p_list[1])
            self.maxpool = nn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)
        elif len(in_list) == 3:
            self.layercnt = 3
            self.layer0 = conv_layer(in_list[0], out_list[0], k_list[0], p_list[0])
            self.layer1 = conv_layer(in_list[1], out_list[1], k_list[1], p_list[1])
            self.layer2 = conv_layer(in_list[2], out_list[2], k_list[2], p_list[2])
            self.maxpool = nn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)

    def forward(self, x):
        if self.layercnt == 2:
            out = self.layer0(x)
            out = self.layer1(out)
        elif self.layercnt == 3:
            out = self.layer0(x)
            out = self.layer1(out)
            out = self.layer2(out)
        
        out = self.maxpool(out)
        
        return out

class vgg_conv_block_adapter(nn.Module):
    def __init__(self, in_list, out_list, k_list, p_list, pooling_k, pooling_s, num_domains=1):
        super(vgg_conv_block_adapter, self).__init__()
        if len(in_list) == 2:
            self.layercnt = 2
            self.layer0 = conv_layer_adapter(in_list[0], out_list[0], k_list[0], p_list[0])
            self.layer1 = conv_layer_adapter(in_list[1], out_list[1], k_list[1], p_list[1])
            self.maxpool = nn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)
        elif len(in_list) == 3:
            self.layercnt = 3
            self.layer0 = conv_layer_adapter(in_list[0], out_list[0], k_list[0], p_list[0])
            self.layer1 = conv_layer_adapter(in_list[1], out_list[1], k_list[1], p_list[1])
            self.layer2 = conv_layer_adapter(in_list[2], out_list[2], k_list[2], p_list[2])
            self.maxpool = nn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)

    def forward(self, x, task=None):
        if self.layercnt == 2:
            out = self.layer0(x, task=task)
            out = self.layer1(out, task=task)
        elif self.layercnt == 3:
            out = self.layer0(x, task=task)
            out = self.layer1(out, task=task)
            out = self.layer2(out, task=task)

        out = self.maxpool(out)
        
        return out


def vgg_fc_layer(size_in, size_out, dropout):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU(),
        nn.Dropout(p=dropout),
    )
    return layer


# VGG搭建 —— https://github.com/msyim/VGG16/blob/master/VGG16.py
class VGG16_bn(nn.Module):
    def __init__(self, num_classes=1000, dropout=0.5):
        super(VGG16_bn, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        # FC layers
        self.layer6 = vgg_fc_layer(7*7*512, 4096, dropout)
        self.layer7 = vgg_fc_layer(4096, 4096, dropout)

        # Final layer
        self.layer8 = nn.Linear(4096, num_classes)

        # Initial
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = self.layer5(out)
        out = vgg16_features.view(out.size(0), -1)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return vgg16_features, out

class VGG16_bn_adapter(nn.Module):
    def __init__(self, num_classes=1000, dropout=0.5):
        super(VGG16_bn_adapter, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        # Specific Adapters
        self.layer3 = vgg_conv_block_adapter([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.layer4 = vgg_conv_block_adapter([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.layer5 = vgg_conv_block_adapter([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        # FC layers
        self.layer6 = vgg_fc_layer(7*7*512, 4096, dropout)
        self.layer7 = vgg_fc_layer(4096, 4096, dropout)

        # Final layer
        self.layer8 = nn.Linear(4096, num_classes)

        # Initial
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, task=None):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out, task=task)
        out = self.layer4(out, task=task)
        vgg16_features = self.layer5(out, task=task)
        out = vgg16_features.view(out.size(0), -1)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return vgg16_features, out


def vgg16_bn(pretrained=True, **kwargs):
    """Constructs a vgg_16_bn model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = VGG16_bn(**kwargs)
    if pretrained:
        # model.load_state_dict(torch.load("/data/user21100736/Work/Pretrained_Models/vgg16_bn-6c64b313.pth"), False)
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
        print("########### pretrained ##############")
        mynn.forgiving_state_restore(model, torch.load("/data/user21100736/Work/Pretrained_Models/vgg16_bn-6c64b313.pth"))
    return model

def vgg16_bn_adapter(pretrained=True, **kwargs):
    """Constructs a vgg_16_bn_adapter model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = VGG16_bn_adapter(**kwargs)
    if pretrained:
        # model.load_state_dict(torch.load("/data/user21100736/Work/Pretrained_Models/vgg16_bn-6c64b313.pth"), False)
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
        print("########### pretrained ##############")
        mynn.forgiving_state_restore(model, torch.load("/data/user21100736/Work/Pretrained_Models/vgg16_bn-6c64b313.pth"))
    return model