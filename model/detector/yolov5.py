# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import torch
import torch.nn as nn
from model.backbone import build_backbone
from model.neck import build_neck
from model.head import build_head
import sys

class Yolov5(nn.Module):
    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 loss=None
                 ):

        super(Yolov5, self).__init__()
        self.with_neck=True if neck else False
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)

        self.head = build_head(bbox_head)
        self.initialize_weights()

    def initialize_weights(self,):
        for m in self.modules():
            t = type(m)
            #######卷积使用默认的初始化
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True

    def load_weight(self,weight,device):
        pretrained = torch.load(weight)
        state_dict = self.state_dict()
        state_dict.update(pretrained["state_dict"])
        self.load_state_dict(state_dict)

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self,x):
        feat=self.extract_feat(x)
        return self.head.detect(feat)





