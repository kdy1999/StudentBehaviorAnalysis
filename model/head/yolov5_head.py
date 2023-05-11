import numpy as np
import torch
import torch.nn as nn
from model.ops import ConvModule,multi_apply,bias_init_with_prob,DepthwiseSeparableConvModule
import math
import sys
########四种输出头形式:(1)只有预测层;(2)具有预测层和堆叠卷积层，但是不解耦;(3)解耦输出头;(4)三个解耦头，输出类别，属性，回归向量
##########将cls分支替换为窗口注意力模式，以及全局注意力，对比两者的优劣
class YOLOV5Head(nn.Module):
    stride = torch.tensor([8,16,32])  # strides computed during build
    def __init__(self,
                 num_classes=80,
                 anchors=(),
                 in_channels=(),
                 feat_channels=256,
                 stacked_convs=3,
                 decouple=False,
                 width=1,
                 inplace=False,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='SiLU'),
                 conv_bias="auto",
                 use_depthwise=False,
                 stride=[8,16,32],
                 #############将属性，类别，回归化为三个分支，分别学习不同的特征
                 trident=False,
                 big_conv=False
                 ):
        super().__init__()

        self.norm_cfg=norm_cfg
        self.act_cfg=act_cfg
        self.decouple=decouple
        self.trident=trident
        self.conv_bias=conv_bias
        self.width=width
        self.in_channels=in_channels
        self.feat_channel=feat_channels
        self.nc = num_classes # number of classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.stacked_convs=stacked_convs
        self.stride=torch.tensor(stride)
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.inplace=inplace
        self.use_depthwise=use_depthwise
        self.build_predictor(in_channels=in_channels)

    def build_predictor(self,
                        in_channels):
        conv=DepthwiseSeparableConvModule if self.use_depthwise else ConvModule
        if not self.decouple:
            self.predict_layers = nn.ModuleList()
        else:
            self.reg_pred_layers=nn.ModuleList()
            self.cls_pred_layers=nn.ModuleList()
            if self.trident:
                self.attr_pred_layers=nn.ModuleList()

        if self.stacked_convs>0:
            if not self.decouple:
                self.stackd_convs_all_levels=nn.ModuleList()
                for i in range(self.nl):
                    stack_convs_level=[]
                    for j in range(self.stacked_convs):
                        in_channel=self.in_channels[i] if j==0 else self.feat_channel

                        stack_convs_level.append(conv(
                            in_channels=in_channel,
                            out_channels=self.feat_channel,
                            kernel_size=3,
                            padding=1,
                            inplace=self.inplace,
                            conv_cfg=None,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg,
                            bias=self.conv_bias
                        ))
                    self.stackd_convs_all_levels.append(nn.Sequential(*stack_convs_level))
                    self.predict_layers.append(nn.Conv2d(in_channels=self.feat_channel,
                                                         out_channels=self.no*self.na,kernel_size=1))

            else:
                self.reg_stackd_convs_all_levels = nn.ModuleList()
                self.cls_stackd_convs_all_levels = nn.ModuleList()
                if self.trident:
                    self.attr_stackd_convs_all_levels=nn.ModuleList()
                for i in range(self.nl):
                    reg_stack_convs_level=[]
                    cls_stack_convs_level = []
                    if self.trident:
                        attr_stack_convs_level=[]
                    for j in range(self.stacked_convs):
                        in_channel=self.in_channels[i] if j==0 else self.feat_channel
                        reg_stack_convs_level.append(
                            conv(
                                in_channels=in_channel,
                                out_channels=self.feat_channel,
                                kernel_size=3,
                                padding=1,
                                inplace=self.inplace,
                                conv_cfg=None,
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg,
                                bias=self.conv_bias

                            )
                        )
                        cls_stack_convs_level.append(
                            conv(
                                in_channels=in_channel,
                                out_channels=self.feat_channel,
                                kernel_size=3,
                                padding=1,
                                inplace=self.inplace,
                                conv_cfg=None,
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg,
                                bias=self.conv_bias

                            )
                        )
                        if self.trident:
                            ############可在属性分支增大kernel_size，或者添加多尺度卷积
                            attr_stack_convs_level.append(
                                conv(in_channels=in_channel,
                                     out_channels=self.feat_channel,
                                     kernel_size=3,
                                     padding=1,
                                     inplace=self.inplace,
                                     conv_cfg=None,
                                     norm_cfg=self.norm_cfg,
                                     act_cfg=self.act_cfg,
                                     bias=self.conv_bias
                                     )
                            )

                    self.reg_stackd_convs_all_levels.append(nn.Sequential(*reg_stack_convs_level))
                    self.cls_stackd_convs_all_levels.append(nn.Sequential(*cls_stack_convs_level))
                    if self.trident:
                        self.attr_stackd_convs_all_levels.append(nn.Sequential(*attr_stack_convs_level))

                    if self.trident:
                        self.attr_pred_layers.append(nn.Conv2d(in_channels=self.feat_channel,
                                                               out_channels=(self.nc-2)*self.na,
                                                               kernel_size=1))
                        self.reg_pred_layers.append(nn.Conv2d(in_channels=self.feat_channel,
                                                              out_channels=5 * self.na,
                                                              kernel_size=1))
                        self.cls_pred_layers.append(nn.Conv2d(in_channels=self.feat_channel,
                                                              out_channels=2 * self.na,
                                                              kernel_size=1))
                    else:
                        self.reg_pred_layers.append(nn.Conv2d(in_channels=self.feat_channel,
                                                              out_channels=5*self.na,
                                                              kernel_size=1))
                        self.cls_pred_layers.append(nn.Conv2d(in_channels=self.feat_channel,
                                                              out_channels=self.nc * self.na,
                                                              kernel_size=1))

        else:
            for channel in in_channels:
                self.predict_layers.append(nn.Conv2d(in_channels=channel*self.width,
                                                     out_channels=self.no*self.na,
                                                     kernel_size=1))

    def detect(self, x):
        z = []  # inference output
        x=self.forward(x)
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                if self.inplace:
                    ###########查询标签的定义
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                    ##########[cx,cy,w,h]
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)



    def init_weights(self):
        # 提升训练稳定性，防止训练初期梯度被负样本主导
        bias_init = bias_init_with_prob(0.01)
        if self.decouple:
            if self.trident:
                for m_reg,m_cls,m_attr,s in zip(self.reg_pred_layers,self.cls_pred_layers,self.attr_pred_layers,list(self.stride)):
                    bias_reg=m_reg.bias.view(self.na,5)
                    bias_cls=m_cls.bias.view(self.na,2)
                    bias_attr=m_attr.bias.view(self.na,self.nc-2)
                    bias_reg.data[:,-1]+=math.log(32 / (1024 / s) ** 2)  # obj (32 objects per 1024 image)
                    bias_cls.data.fill_(bias_init)
                    bias_attr.data.fill_(bias_init)
                    m_reg.bias = torch.nn.Parameter(bias_reg.view(-1), requires_grad=True)
                    m_cls.bias = torch.nn.Parameter(bias_cls.view(-1), requires_grad=True)
                    m_attr.bias = torch.nn.Parameter(bias_attr.view(-1), requires_grad=True)

            else:
                for m_reg,m_cls,s in zip(self.reg_pred_layers,self.cls_pred_layers,list(self.stride)):
                    bias_reg = m_reg.bias.view(self.na, 5)
                    bias_cls = m_cls.bias.view(self.na, self.nc)
                    bias_reg.data[:, -1] += math.log(32 / (1024 / s) ** 2)  # obj (32 objects per 1024 image)
                    bias_cls.data.fill_(bias_init)
                    m_reg.bias = torch.nn.Parameter(bias_reg.view(-1), requires_grad=True)
                    m_cls.bias = torch.nn.Parameter(bias_cls.view(-1), requires_grad=True)
        else:
            for mi, s in zip(self.predict_layers, list(self.stride)):  # from
                b = mi.bias.view(self.na, -1)  # conv.bias(255) to (3,85)
                ########focal loss的obj bias应该尽可能设得小一点，因为如果太小导致正负样本输出概率一致，负样本主导了梯度
                b.data[:, 4] += math.log(32 / (1024 / s) ** 2)  # obj (8 objects per 640 image)
                ########将类别分类器的bias也设小一些，使得sigmoid的输出尽可能小，平衡正负样本的梯度；
                # 同时将样本数量多的属性的分类器的bias设得大一些，平衡梯度
                b.data[:, 5:] += math.log(0.6 / (self.nc - 0.99))
                mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        ######对anchor使用stride进行编码
        self.anchors /= self.stride.view(-1, 1, 1).cuda()


    def forward_single(self, x, stack_convs, reg_stack_convs,cls_stack_convs,attr_stack_convs,
                       predict_layer,reg_pred_layer,cls_pred_layer,attr_pre_layer):
        if self.stacked_convs:
            if not self.decouple:
                stack_feature=stack_convs(x)
                out=predict_layer(stack_feature)
                return out
            else:
                if self.trident:
                    reg_stack_feature = reg_stack_convs(x)
                    cls_stack_feature = cls_stack_convs(x)
                    attr_stack_feature= attr_stack_convs(x)
                    reg_pred = reg_pred_layer(reg_stack_feature)
                    cls_pred = cls_pred_layer(cls_stack_feature)
                    attr_pred= attr_pre_layer(attr_stack_feature)
                    b,_,h,w= reg_pred.shape
                    reg_pred = reg_pred.view(b, self.na, 5, h, w)
                    cls_pred = cls_pred.view(b, self.na, 2, h, w)
                    attr_pred=attr_pred.view(b,self.na,self.nc-2,h,w)
                    out = torch.cat([reg_pred, cls_pred,attr_pred], dim=2)
                    out = out.view(b, self.na*self.no, h, w).contiguous()
                else:
                    reg_stack_feature=reg_stack_convs(x)
                    cls_stack_feature=cls_stack_convs(x)
                    reg_pred=reg_pred_layer(reg_stack_feature)
                    cls_pred=cls_pred_layer(cls_stack_feature)
                    b,_,h,w=reg_pred.shape
                    reg_pred=reg_pred.view(b,self.na,5,h,w)
                    cls_pred=cls_pred.view(b,self.na,self.nc,h,w)
                    out=torch.cat([reg_pred,cls_pred],dim=2)
                    out=out.view(b,self.no*self.na,h,w).contiguous()
                return out
        else:
            out=predict_layer(x)
            return out

    def forward(self,feats):
        if self.stacked_convs:
            if not self.decouple:
                return multi_apply(self.forward_single,
                                   feats,
                                   self.stackd_convs_all_levels,
                                   [[],[],[]],
                                   [[],[],[]],
                                   [[], [], []],
                                   self.predict_layers,
                                   [[],[],[]],
                                   [[],[],[]],
                                   [[],[],[]],)
            else:
                if self.trident:
                    return multi_apply(self.forward_single,
                                       feats,
                                       [[], [], []],
                                       self.reg_stackd_convs_all_levels,
                                       self.cls_stackd_convs_all_levels,
                                       self.attr_stackd_convs_all_levels,
                                       [[], [], []],
                                       self.reg_pred_layers,
                                       self.cls_pred_layers,
                                       self.attr_pred_layers, )
                else:
                    return multi_apply(self.forward_single,
                                       feats,
                                       [[],[],[]],
                                       self.reg_stackd_convs_all_levels,
                                       self.cls_stackd_convs_all_levels,
                                       [[], [], []],
                                       [[],[],[]],
                                       self.reg_pred_layers,
                                       self.cls_pred_layers,
                                       [[],[],[]],)
        else:
            return multi_apply(self.forward_single,
                               feats,
                               [[],[],[]],
                               [[],[],[]],
                               [[],[],[]],
                               [[], [], []],
                               self.predict_layers,
                               [[],[],[]],
                               [[],[],[]],
                               [[],[],[]],)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()



