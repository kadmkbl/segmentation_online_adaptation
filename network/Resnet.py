"""
# Code Adapted from:
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
#
# BSD 3-Clause License
#
# Copyright (c) 2017,
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import network.mynn as mynn
from network.sync_switchwhiten import SyncSwitchWhiten2d
from network.instance_whitening import InstanceWhitening
from network.dsbn import DomainSpecificBatchNorm2d

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _ntuple
from collections import OrderedDict
import operator
from itertools import islice

_pair = _ntuple(2)

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnet_adapt101']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# 专门用于downsample的卷积，返回两个值
class DownsampleConv(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(DownsampleConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode='zeros')

    def forward(self, input, domain_label):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups), domain_label

class BasicBlock(nn.Module):
    """
    Basic Block for Resnet
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, iw=0, dsbn=False, num_domains=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = mynn.Norm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = mynn.Norm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.iw = iw
        if self.iw == 1:
            self.instance_norm_layer = InstanceWhitening(planes * self.expansion)
            self.relu = nn.ReLU(inplace=False)
        elif self.iw == 2:
            self.instance_norm_layer = InstanceWhitening(planes * self.expansion)
            self.relu = nn.ReLU(inplace=False)
        elif self.iw == 3:
            self.instance_norm_layer = nn.InstanceNorm2d(planes * self.expansion, affine=False)
            self.relu = nn.ReLU(inplace=True)
        elif self.iw == 4:
            self.instance_norm_layer = nn.InstanceNorm2d(planes * self.expansion, affine=True)
            self.relu = nn.ReLU(inplace=True)
        elif self.iw == 5:
            self.instance_norm_layer = SyncSwitchWhiten2d(planes * self.expansion,
                                                          num_pergroup=16,
                                                          sw_type=2,
                                                          T=5,
                                                          tie_weight=False,
                                                          eps=1e-5,
                                                          momentum=0.99,
                                                          affine=True)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x_tuple):
        if len(x_tuple) == 2:
            w_arr = x_tuple[1]
            x = x_tuple[0]
        else:
            print("error!!!")
            return

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.iw >= 1:
            if self.iw == 1 or self.iw == 2:
                out, w = self.instance_norm_layer(out)
                w_arr.append(w)
            else:
                out = self.instance_norm_layer(out)

        out = self.relu(out)

        return [out, w_arr]

class Bottleneck_bfDSBN(nn.Module):
    """
    Bottleneck Layer for Resnet
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, iw=0):
        super(Bottleneck_bfDSBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = mynn.Norm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = mynn.Norm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = mynn.Norm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

        self.iw = iw
        if self.iw == 1:
            self.instance_norm_layer = InstanceWhitening(planes * self.expansion)
            self.relu = nn.ReLU(inplace=False)
        elif self.iw == 2:
            self.instance_norm_layer = InstanceWhitening(planes * self.expansion)
            self.relu = nn.ReLU(inplace=False)
        elif self.iw == 3:
            self.instance_norm_layer = nn.InstanceNorm2d(planes * self.expansion, affine=False)
            self.relu = nn.ReLU(inplace=True)
        elif self.iw == 4:
            self.instance_norm_layer = nn.InstanceNorm2d(planes * self.expansion, affine=True)
            self.relu = nn.ReLU(inplace=True)
        elif self.iw == 5:
            self.instance_norm_layer = SyncSwitchWhiten2d(planes * self.expansion,
                                                          num_pergroup=16,
                                                          sw_type=2,
                                                          T=5,
                                                          tie_weight=False,
                                                          eps=1e-5,
                                                          momentum=0.99,
                                                          affine=True)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x_tuple):
        if len(x_tuple) == 2:
            w_arr = x_tuple[1]
            x = x_tuple[0]
        else:
            print("error!!!")
            return

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.iw >= 1:
            if self.iw == 1 or self.iw == 2:
                out, w = self.instance_norm_layer(out)
                w_arr.append(w)
            else:
                out = self.instance_norm_layer(out)

        out = self.relu(out)

        return [out, w_arr]

class Bottleneck(nn.Module):
    """
    Bottleneck Layer for Resnet
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, iw=0, dsbn=False, num_domains=1):
        super(Bottleneck, self).__init__()
        self.dsbn = dsbn
        self.num_domains = num_domains
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if self.dsbn:
            self.bn1 = DomainSpecificBatchNorm2d(planes, self.num_domains)
        else:
            self.bn1 = mynn.Norm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        if self.dsbn:
            self.bn2 = DomainSpecificBatchNorm2d(planes, self.num_domains)
        else:
            self.bn2 = mynn.Norm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        if self.dsbn:
            self.bn3 = DomainSpecificBatchNorm2d(planes * self.expansion, self.num_domains)
        else:
            self.bn3 = mynn.Norm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

        self.iw = iw
        if self.iw == 1:
            self.instance_norm_layer = InstanceWhitening(planes * self.expansion)
            self.relu = nn.ReLU(inplace=False)
        elif self.iw == 2:
            self.instance_norm_layer = InstanceWhitening(planes * self.expansion)
            self.relu = nn.ReLU(inplace=False)
        elif self.iw == 3:
            self.instance_norm_layer = nn.InstanceNorm2d(planes * self.expansion, affine=False)
            self.relu = nn.ReLU(inplace=True)
        elif self.iw == 4:
            self.instance_norm_layer = nn.InstanceNorm2d(planes * self.expansion, affine=True)
            self.relu = nn.ReLU(inplace=True)
        elif self.iw == 5:
            self.instance_norm_layer = SyncSwitchWhiten2d(planes * self.expansion,
                                                          num_pergroup=16,
                                                          sw_type=2,
                                                          T=5,
                                                          tie_weight=False,
                                                          eps=1e-5,
                                                          momentum=0.99,
                                                          affine=True)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x_tuple, domain_label):
        if len(x_tuple) == 2:
            w_arr = x_tuple[1]
            x = x_tuple[0]
        else:
            print("error!!!")
            return

        residual = x

        out = self.conv1(x)
        if self.dsbn:
            out, _ = self.bn1(out, domain_label)
        else:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.dsbn:
            out, _ = self.bn2(out, domain_label)
        else:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.dsbn:
            out, _ = self.bn3(out, domain_label)
        else:
            out = self.bn3(out)

        if self.downsample is not None:
            residual, _ = self.downsample(x, domain_label)

        out += residual

        if self.iw >= 1:
            if self.iw == 1 or self.iw == 2:
                out, w = self.instance_norm_layer(out)
                w_arr.append(w)
            else:
                out = self.instance_norm_layer(out)

        out = self.relu(out)

        return [out, w_arr], domain_label

class ResNet_bfDSBN(nn.Module):
    """
    Resnet Global Module for Initialization
    """

    def __init__(self, block, layers, wt_layer=None, num_classes=1000):
        self.inplanes = 64
        super(ResNet_bfDSBN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if wt_layer[2] == 1: # wt_layer[2]会选择normalization
            self.bn1 = InstanceWhitening(64)
            self.relu = nn.ReLU(inplace=False)
        elif wt_layer[2] == 2:
            self.bn1 = InstanceWhitening(64)
            self.relu = nn.ReLU(inplace=False)
        elif wt_layer[2] == 3:
            self.bn1 = nn.InstanceNorm2d(64, affine=False)
            self.relu = nn.ReLU(inplace=True)
        elif wt_layer[2] == 4:
            self.bn1 = nn.InstanceNorm2d(64, affine=True)
            self.relu = nn.ReLU(inplace=True)
        elif wt_layer[2] == 5:
            self.bn1 = SyncSwitchWhiten2d(self.inplanes,
                                          num_pergroup=16,
                                          sw_type=2,
                                          T=5,
                                          tie_weight=False,
                                          eps=1e-5,
                                          momentum=0.99,
                                          affine=True)
            self.relu = nn.ReLU(inplace=True)
        elif wt_layer[2] == 0:
            self.bn1 = mynn.Norm2d(64)
            self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], wt_layer=wt_layer[3])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, wt_layer=wt_layer[4])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, wt_layer=wt_layer[5])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, wt_layer=wt_layer[6])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.wt_layer = wt_layer

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, wt_layer=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                mynn.Norm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, iw=0))
        self.inplanes = planes * block.expansion
        for index in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                iw=0 if (wt_layer > 0 and index < blocks - 1) else wt_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        w_arr = []
        x_size = x.size()  # 800

        x = self.conv1(x)
        if self.wt_layer[2] == 1 or self.wt_layer[2] == 2:
            x, w = self.bn1(x)
            w_arr.append(w)
        else:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # ResNet的四个残差模块之前：conv1 → bn1 → relu → maxpool

        x_tuple = self.layer1([x, w_arr])  # 400
        low_level = x_tuple[0]

        x_tuple = self.layer2(x_tuple)  # 100
        x_tuple = self.layer3(x_tuple)  # 100
        aux_out = x_tuple[0]
        x_tuple = self.layer4(x_tuple)  # 100

        x = x_tuple[0]
        w_arr = x_tuple[1]
        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return x

"""
class ResNet3X3(nn.Module):
    
    # Resnet Global Module for Initialization
    

    def __init__(self, block, layers, wt_layer=None, num_classes=1000):
        self.inplanes = 128
        super(ResNet3X3, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.bn1 = mynn.Norm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                                bias=False)

        if wt_layer[0] == 1:
            self.bn1 = InstanceWhitening(64)
            self.relu1 = nn.ReLU(inplace=False)
        elif wt_layer[0] == 2:
            self.bn1 = InstanceWhitening(64)
            self.relu1 = nn.ReLU(inplace=False)
        elif wt_layer[0] == 3:
            self.bn1 = nn.InstanceNorm2d(64, affine=False)
            self.relu1 = nn.ReLU(inplace=True)
        elif wt_layer[0] == 4:
            self.bn1 = nn.InstanceNorm2d(64, affine=True)
            self.relu1 = nn.ReLU(inplace=True)
        elif wt_layer[0] == 5:
            self.bn1 = SyncSwitchWhiten2d(64,
                                          num_pergroup=16,
                                          sw_type=2,
                                          T=5,
                                          tie_weight=False,
                                          eps=1e-5,
                                          momentum=0.99,
                                          affine=True)
            self.relu1 = nn.ReLU(inplace=True)
        else:
            self.bn1 = mynn.Norm2d(64)
            self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        if wt_layer[1] == 1:
            self.bn2 = InstanceWhitening(64)
            self.relu2 = nn.ReLU(inplace=False)
        elif wt_layer[1] == 2:
            self.bn2 = InstanceWhitening(64)
            self.relu2 = nn.ReLU(inplace=False)
        elif wt_layer[1] == 3:
            self.bn2 = nn.InstanceNorm2d(64, affine=False)
            self.relu2 = nn.ReLU(inplace=True)
        elif wt_layer[1] == 4:
            self.bn2 = nn.InstanceNorm2d(64, affine=True)
            self.relu2 = nn.ReLU(inplace=True)
        elif wt_layer[1] == 5:
            self.bn2 = SyncSwitchWhiten2d(64,
                                          num_pergroup=16,
                                          sw_type=2,
                                          T=5,
                                          tie_weight=False,
                                          eps=1e-5,
                                          momentum=0.99,
                                          affine=True)
            self.relu2 = nn.ReLU(inplace=True)
        else:
            self.bn2 = mynn.Norm2d(64)
            self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1,
                               bias=False)
        if wt_layer[2] == 1:
            self.bn3 = InstanceWhitening(self.inplanes)
            self.relu3 = nn.ReLU(inplace=False)
        elif wt_layer[2] == 2:
            self.bn3 = InstanceWhitening(self.inplanes)
            self.relu3 = nn.ReLU(inplace=False)
        elif wt_layer[2] == 3:
            self.bn3 = nn.InstanceNorm2d(self.inplanes, affine=False)
            self.relu3 = nn.ReLU(inplace=True)
        elif wt_layer[2] == 4:
            self.bn3 = nn.InstanceNorm2d(self.inplanes, affine=True)
            self.relu3 = nn.ReLU(inplace=True)
        elif wt_layer[2] == 5:
            self.bn3 = SyncSwitchWhiten2d(self.inplanes,
                                          num_pergroup=16,
                                          sw_type=2,
                                          T=5,
                                          tie_weight=False,
                                          eps=1e-5,
                                          momentum=0.99,
                                          affine=True)
            self.relu3 = nn.ReLU(inplace=True)
        else:
            self.bn3 = mynn.Norm2d(self.inplanes)
            self.relu3 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], wt_layer=wt_layer[3])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, wt_layer=wt_layer[4])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, wt_layer=wt_layer[5])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, wt_layer=wt_layer[6])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.wt_layer = wt_layer

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, wt_layer=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                mynn.Norm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, iw=0))
        self.inplanes = planes * block.expansion
        for index in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                iw=0 if (wt_layer > 0 and index < blocks - 1) else wt_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        w_arr = []
        x_size = x.size()  # 800

        x = self.conv1(x)
        if self.wt_layer[0] == 1 or self.wt_layer[0] == 2:
            x, w = self.bn1(x)
            w_arr.append(w)
        else:
            x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        if self.wt_layer[1] == 1 or self.wt_layer[1] == 2:
            x, w = self.bn2(x)
            w_arr.append(w)
        else:
            x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        if self.wt_layer[2] == 1 or self.wt_layer[2] == 2:
            x, w = self.bn3(x)
            w_arr.append(w)
        else:
            x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool(x)

        x_tuple = self.layer1([x, w_arr])  # 400
        low_level = x_tuple[0]

        x_tuple = self.layer2(x_tuple)  # 100
        x_tuple = self.layer3(x_tuple)  # 100
        aux_out = x_tuple[0]
        x_tuple = self.layer4(x_tuple)  # 100

        x = x_tuple[0]
        w_arr = x_tuple[1]
        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return x
"""

class ResNet(nn.Module):
    """
    Resnet Global Module for Initialization
    """

    def __init__(self, block, layers, wt_layer=None, dsbn=False, num_domains=1, num_classes=1000):
        self.inplanes = 64
        self.dsbn = dsbn
        self.num_domains = num_domains
        # self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        if wt_layer[2] == 1: # wt_layer[2]会选择normalization
            self.bn1 = InstanceWhitening(64)
            self.relu = nn.ReLU(inplace=False)
        elif wt_layer[2] == 2:
            self.bn1 = InstanceWhitening(64)
            self.relu = nn.ReLU(inplace=False)
        elif wt_layer[2] == 3:
            self.bn1 = nn.InstanceNorm2d(64, affine=False)
            self.relu = nn.ReLU(inplace=True)
        elif wt_layer[2] == 4:
            self.bn1 = nn.InstanceNorm2d(64, affine=True)
            self.relu = nn.ReLU(inplace=True)
        elif wt_layer[2] == 5:
            self.bn1 = SyncSwitchWhiten2d(self.inplanes,
                                          num_pergroup=16,
                                          sw_type=2,
                                          T=5,
                                          tie_weight=False,
                                          eps=1e-5,
                                          momentum=0.99,
                                          affine=True)
            self.relu = nn.ReLU(inplace=True)
        elif wt_layer[2] == 0 and self.dsbn:
            self.bn1 = DomainSpecificBatchNorm2d(64, self.num_domains)
            self.relu = nn.ReLU(inplace=True)
        else: # wt_layer[2] == 0且不使用DSBN
            self.bn1 = mynn.Norm2d(64)
            self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], wt_layer=wt_layer[3]) # wt_layer[3]会选择normalization
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, wt_layer=wt_layer[4]) # wt_layer[4]会选择normalization
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, wt_layer=wt_layer[5])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, wt_layer=wt_layer[6])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.wt_layer = wt_layer

        # 初始化modules
        for m in self.modules():
            # print(m._get_name)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, wt_layer=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = TwoInputSequential(
                DownsampleConv(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                DomainSpecificBatchNorm2d(planes * block.expansion, self.num_domains),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, iw=0, dsbn=self.dsbn, num_domains=self.num_domains)) # 一个bottleneck
        self.inplanes = planes * block.expansion
        for index in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                iw=0 if (wt_layer > 0 and index < blocks - 1) else wt_layer, dsbn=self.dsbn, num_domains=self.num_domains)) # [3-1, 4-1, 6-1, 3-1]个bottleneck
        
        return TwoInputSequential(*layers)

    def forward(self, x): # 实际上deepv3用的是这个类（ResNet）里面的模块，并没有直接用这个类，所以这一部分forward()其实没起作用
        w_arr = []
        x_size = x.size()  # 800

        x = self.conv1(x)
        if self.wt_layer[2] == 1 or self.wt_layer[2] == 2:
            x, w = self.bn1(x)
            w_arr.append(w)
        else:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # ResNet的四个残差模块之前：conv1 → bn1 → relu → maxpool

        x_tuple = self.layer1([x, w_arr])  # 400
        low_level = x_tuple[0]

        x_tuple = self.layer2(x_tuple)  # 100
        x_tuple = self.layer3(x_tuple)  # 100
        aux_out = x_tuple[0]
        x_tuple = self.layer4(x_tuple)  # 100

        x = x_tuple[0] # 没有用ResNet后面的AvgPool和FC层
        w_arr = x_tuple[1]

        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return x


class TwoInputSequential(nn.Module):
    r"""A sequential container forward with two inputs.
    """

    def __init__(self, *args):
        super(TwoInputSequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return TwoInputSequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(TwoInputSequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, input1, input2):
        for module in self._modules.values():
            input1, input2 = module(input1, input2)
        return input1, input2


def resnet18(pretrained=True, wt_layer=None, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if wt_layer is None:
        wt_layer = [0, 0, 0, 0, 0, 0, 0]
    model = ResNet(BasicBlock, [2, 2, 2, 2], wt_layer=wt_layer, dsbn=False, num_domains=1, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), False)
        print("########### pretrained ##############")
        mynn.forgiving_state_restore(model, model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=True, wt_layer=None, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if wt_layer is None:
        wt_layer = [0, 0, 0, 0, 0, 0, 0]
    model = ResNet_bfDSBN(Bottleneck_bfDSBN, [3, 4, 6, 3], wt_layer=wt_layer, **kwargs) # ResNet50及以上使用Bottleneck模块；位置参数与关键参数（形参）
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), False)
        print("########### pretrained ##############")
        mynn.forgiving_state_restore(model, model_zoo.load_url(model_urls['resnet50']))
    return model


def dsbnresnet50(pretrained=True, wt_layer=None, num_domains=None, **kwargs):
    """Constructs a DBSNResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if wt_layer is None:
        wt_layer = [0, 0, 0, 0, 0, 0, 0]
    model = ResNet(Bottleneck, [3, 4, 6, 3], wt_layer=wt_layer, dsbn=True, num_domains=num_domains, **kwargs) # ResNet50及以上使用Bottleneck模块；位置参数与关键参数（形参）
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), False)
        print("########### pretrained ##############")
        mynn.forgiving_state_restore(model, model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=True, wt_layer=None, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        :param pretrained:
    """
    if wt_layer is None:
        wt_layer = [0, 0, 0, 0, 0, 0, 0]
    # model = ResNet3X3(Bottleneck_bfDSBN, [3, 4, 23, 3], wt_layer=wt_layer, **kwargs)
    model = ResNet_bfDSBN(Bottleneck_bfDSBN, [3, 4, 23, 3], wt_layer=wt_layer, **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
        print("########### pretrained ##############")
        # model.load_state_dict(torch.load('./pretrained/resnet101-imagenet.pth', map_location="cpu"))
        mynn.forgiving_state_restore(model, torch.load('/data/user21100736/Work/Pretrained_Models/resnet101-imagenet.pth', map_location="cpu"))
    return model


def resnet_adapt101(args, pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        args: arguments that contain adapt_layer information
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        :param pretrained:
    """
    # model = ResNet3X3(args, **kwargs)
    model = ResNet_bfDSBN(args, **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
        print("########### pretrained ##############")
        model.load_state_dict(torch.load('./pretrained/resnet_adapt101-imagenet.pth', map_location="cpu"))
        # mynn.forgiving_state_restore(model, torch.load('./pretrained/resnet101-imagenet.pth', map_location="cpu"))
    return model


def resnet152(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
