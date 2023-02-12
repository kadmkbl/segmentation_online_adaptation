import logging
import torch
from torch import nn
from network import VGG
from network.cov_settings import CovMatrix_ISW, CovMatrix_IRW
from network.instance_whitening import instance_whitening_loss, get_covariance_matrix
from network.mynn import initialize_weights, Norm2d, Upsample, freeze_weights, unfreeze_weights

import torchvision.models as models

# This is implemented in full accordance with the original one (https://github.com/shelhamer/fcn.berkeleyvision.org)
class FCN8s(nn.Module):
    def __init__(self, num_classes, criterion=None, criterion_aux=None, args=None):
        super(FCN8s, self).__init__()
        self.criterion = criterion
        self.criterion_aux = criterion_aux
        self.args = args
        self.dsbn = args.dsbn
        self.num_domains = args.num_domains

        vgg = VGG.vgg16_bn(dsbn=self.dsbn, num_domains=self.num_domains)
        
        # FCN
        # 前三层（融合层）
        self.layer1 = vgg.layer1
        self.layer2 = vgg.layer2
        self.layer3 = vgg.layer3 # dim：256
        # 第四层（融合层）
        self.layer4 = vgg.layer4 # dim：512
        # 第五层
        self.layer5 = vgg.layer5 # dim：512

        # FC → 全卷积，最终输出
        fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.score_fr = nn.Sequential(
            fc6, nn.ReLU(inplace=True), nn.Dropout(), fc7, nn.ReLU(inplace=True), nn.Dropout(), score_fr
        )
        
        # 融合操作前，最终输出
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)

        # 逆卷积（上采样）
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False) # pool5输出上采样
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False) # pool4输出，融合，继续上采样
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False) # pool3输出，融合，继续上采样

        # Initial
        initialize_weights(self.score_pool3)
        initialize_weights(self.score_pool4)
        initialize_weights(self.score_fr)
        initialize_weights(self.upscore2)
        initialize_weights(self.upscore_pool4)
        initialize_weights(self.upscore8)

        # Setting the flags
        self.eps = 1e-5
        self.whitening = False

    def forward(self, x, gts=None, aux_gts=None, img_gt=None, visualize=False, cal_covstat=False, apply_wtloss=True, dsbn=False, mode=None, domain_label=None):
        x_size = x.size()
        # print(x_size) # torch.Size([4, 3, 768, 768])
        pool1, _ = self.layer1(x, domain_label)
        pool2, _ = self.layer2(pool1, domain_label)
        pool3, _ = self.layer3(pool2, domain_label)
        pool4, _ = self.layer4(pool3, domain_label)
        pool5, _ = self.layer5(pool4, domain_label)
        # print("pool3: ", pool5.size()) # pool3:  torch.Size([4, 512, 24, 24])
        # print("pool4: ", pool5.size()) # pool4:  torch.Size([4, 512, 24, 24])
        # print("pool5: ", pool5.size()) # pool5:  torch.Size([4, 512, 24, 24])

        # pool5输出，上采样
        score_fr = self.score_fr(pool5)
        upscore2 = self.upscore2(score_fr)
        # print("score_fr: ", score_fr.size()) # score_fr:  torch.Size([4, 19, 18, 18])
        # print("upscore2: ", upscore2.size()) # upscore2:  torch.Size([4, 19, 38, 38])

        # pool4输出，融合，继续上采样
        score_pool4 = self.score_pool4(0.01 * pool4)
        # print("score_pool4: ", score_pool4.size()) # score_pool4:  torch.Size([4, 19, 48, 48])
        # print("bf_score_pool4: ", score_pool4[:, :, 5: (5 + upscore2.size()[2]), 5: (5 + upscore2.size()[3])].size()) # bf_score_pool4:  torch.Size([4, 19, 38, 38])
        upscore_pool4 = self.upscore_pool4(score_pool4[:, :, 5: (5 + upscore2.size()[2]), 5: (5 + upscore2.size()[3])]
                                           + upscore2)
        # print("upscore_pool4: ", upscore_pool4.size()) # upscore_pool4:  torch.Size([4, 19, 78, 78])

        # pool3输出，融合，继续上采样
        score_pool3 = self.score_pool3(0.0001 * pool3)
        # print("score_pool3: ", score_pool3.size()) # score_pool3:  torch.Size([4, 19, 96, 96])
        # print("bf_score_pool3: ", score_pool3[:, :, 9: (9 + upscore_pool4.size()[2]), 9: (9 + upscore_pool4.size()[3])].size()) # bf_score_pool3:  torch.Size([4, 19, 78, 78])
        upscore8 = self.upscore8(score_pool3[:, :, 9: (9 + upscore_pool4.size()[2]), 9: (9 + upscore_pool4.size()[3])]
                                + upscore_pool4)
        # print("upscore8: ", upscore8.size()) # upscore8:  torch.Size([4, 19, 632, 632])

        main_out = Upsample(upscore8, x_size[2:]) # 为什么语义分割网络不同尺寸的输入图像都可以用：https://blog.csdn.net/justsolow/article/details/110140818
        # print("main_out: ", main_out.size()) # main_out:  torch.Size([4, 19, 768, 768])
        # 原：main_out = upscore8[:, :, 31: (31 + x_size[2]), 31: (31 + x_size[3])].contiguous() # main_out torch.Size([4, 19, 601, 601]) —— 溢出了

        # print(main_out)
        # print(gts)
        # 当前是训练/推理阶段
        if self.training:
            loss1 = self.criterion(main_out, gts)
            loss2 = loss1 # 没有设置aux_loss
            return_loss = [loss1, loss2]

            return return_loss
        else:
            return main_out


def FCN(args, num_classes, criterion, criterion_aux):
    """
    FCN Network
    """
    print("Model : FCN, Backbone : VGG-16") # FCN暂时只提供VGG-16的默认主干，且是VGG-16 with BN的变体
    return FCN8s(num_classes, criterion=criterion, criterion_aux=criterion_aux, args=args)