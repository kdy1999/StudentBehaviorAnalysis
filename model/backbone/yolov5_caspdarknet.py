import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from model.ops import DepthwiseSeparableConvModule,ConvModule,CSPLayer_YOLOV5

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=None,
                 inplace=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        #####使用1x1卷积调整维度
        self.conv = ConvModule(
            in_channels * 4,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size-1)//2,
            inplace=inplace,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
            )

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)

        return self.conv(torch.cat([x[..., ::2, ::2],
                                    x[..., 1::2, ::2],
                                    x[..., ::2, 1::2],
                                    x[..., 1::2, 1::2]],
                                   1))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=(5, 9, 13),
                 inplace=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super().__init__()
        mid_channels = in_channels // 2  # hidden channels
        self.conv1 = ConvModule(
            in_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            inplace=inplace,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            mid_channels * (len(kernel_sizes) + 1),
            out_channels,
            kernel_size=1,
            stride=1,
            inplace=inplace,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.poolings = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
            for k in kernel_sizes])

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [pool(x) for pool in self.poolings], 1))


class CSPDarknet_YOLOV5(nn.Module):

    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    #######最后一个阶段使用spp，不使用短连接
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 9, True, False],
               [256, 512, 9, True, False], [512, 1024, 3, False, True]],
        'P6': [[64, 128, 3, True, False], [128, 256, 9, True, False],
               [256, 512, 9, True, False], [512, 768, 3, True, False],
               [768, 1024, 3, False, True]]
    }

    def __init__(self,
                 arch='P5',
                 deepen_factor=1.0,
                 widen_factor=1.0,
                 out_indices=(2, 3, 4),
                 frozen_stages=-1,
                 use_depthwise=False,
                 arch_ovewrite=None,
                 spp_kernal_sizes=(5, 9, 13),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='SiLU'),
                 inplace=False,
                 ##########是否停止更新bn统计全局std,mean
                 norm_eval=False,
                 ):
        super().__init__()

        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        assert set(out_indices).issubset(
            i for i in range(len(arch_setting) + 1))
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError('frozen_stages must be in range(-1, '
                             'len(arch_setting) + 1). But received '
                             f'{frozen_stages}')

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        #####是否使用组卷积
        self.use_depthwise = use_depthwise
        #####是否将BN的统计均值和标准差固定，可适用于使用预训练模型的微调
        self.norm_eval = norm_eval
        self.inplace=inplace
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        ######使用Focus层降采样，可以将FOCUS层替换为一个6x6的卷积
        self.stem = Focus(
            3,
            int(arch_setting[0][0] * widen_factor),
            kernel_size=3,
            inplace=self.inplace,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.layers = ['stem']

        for i, (in_channels, out_channels, num_blocks, add_identity,
                use_spp) in enumerate(arch_setting):
            in_channels = int(in_channels * widen_factor)
            out_channels = int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            stage = []
            ######每个stage第一个卷积进行下采样
            conv_layer = conv(
                in_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                inplace=self.inplace,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            #######使用spp提高感受野
            stage.append(conv_layer)
            if use_spp:
                spp = SPP(
                    out_channels,
                    out_channels,
                    kernel_sizes=spp_kernal_sizes,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                stage.append(spp)
            ####csp_layer放在最后阶段的开头，能够极大提高感受野
            csp_layer = CSPLayer_YOLOV5(
                out_channels,
                out_channels,
                num_blocks=num_blocks,
                add_identity=add_identity,
                use_depthwise=use_depthwise,
                inplace=inplace,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            stage.append(csp_layer)
            self.add_module(f'stage{i + 1}', nn.Sequential(*stage))
            self.layers.append(f'stage{i + 1}')

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                ########不对std,mean进行更新
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(CSPDarknet_YOLOV5, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

