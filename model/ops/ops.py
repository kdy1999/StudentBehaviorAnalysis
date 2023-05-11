import copy
import logging
from logging.handlers import TimedRotatingFileHandler
import os
import warnings
from datetime import datetime
from functools import partial

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from easydict import EasyDict as edict
from model.datasets.augmentations import letterbox
from tools.config import Config
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm

from .basic import (build_activation_layer, build_conv_layer, build_norm_layer,
                    build_padding_layer, constant_init, kaiming_init)


class YamlParser(edict):
    """
    This is yaml parser based on EasyDict.
    """

    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if config_file is not None:
            assert (os.path.isfile(config_file))
            with open(config_file, 'r') as fo:
                yaml_ = yaml.load(fo.read(), Loader=yaml.FullLoader)
                cfg_dict.update(yaml_)

        super(YamlParser, self).__init__(cfg_dict)

    def merge_from_file(self, config_file):
        with open(config_file, 'r') as fo:
            yaml_ = yaml.load(fo.read(), Loader=yaml.FullLoader)
            self.update(yaml_)

    def merge_from_dict(self, config_dict):
        self.update(config_dict)


def get_config(config_file=None):
    if config_file is not None:
        if config_file.endswith(".py"):
            return Config.fromfile(config_file)
        else:
            return YamlParser(config_file=config_file)
    else:
        return YamlParser(config_file=config_file)


def set_logger(name, log_rotator, backup_count):
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                           '..', 'log')
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(name)
    logger.root.setLevel(level=logging.NOTSET)
    log_file_path = os.path.abspath(os.path.join(log_dir, name))

    # print(log_file_path)

    # file_handler = logging.FileHandler(log_file_path)
    file_handler = TimedRotatingFileHandler(log_file_path,
                                            when=log_rotator,
                                            backupCount=backup_count)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.WARNING)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def iou_overlap(bboxes1, bboxes2, eps=1e-6):

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] -
                                                   bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] -
                                                   bboxes2[..., 1])

    lt = torch.max(bboxes1[:, None, :2],
                   bboxes2[None, :, :2])  # [n ,1, 2]----[1,n,2]---->[n,n,2]

    rb = torch.min(bboxes1[:, None, 2:],
                   bboxes2[None, :, 2:])  # [n,1, 2],[1,n,2]------>[n,n,2]

    wh = (rb - lt).clamp(min=0)  # [n,n, 2]
    overlap = wh[..., 0] * wh[..., 1]

    union = area1[..., None] + area2[..., None, :] - overlap

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union

    return ious


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return list(map_results)
    # return tuple(map(list, zip(*map_results)))


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


#########基础的CBA模块
class ConvModule(nn.Module):
    _abbr_ = 'conv_block'

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 inplace=True,
                 with_spectral_norm=False,
                 padding_mode='zeros',
                 order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        official_padding_mode = ['zeros', 'circular']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        #####使用显示的非常规padding
        if self.with_explicit_padding:
            pad_cfg = dict(type=padding_mode)
            self.padding_layer = build_padding_layer(pad_cfg, padding)

        # reset padding to 0 for conv module
        conv_padding = 0 if self.with_explicit_padding else padding
        # build convolution layer
        self.conv = build_conv_layer(conv_cfg,
                                     in_channels,
                                     out_channels,
                                     kernel_size,
                                     stride=stride,
                                     padding=conv_padding,
                                     dilation=dilation,
                                     groups=groups,
                                     bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
            if self.with_bias:
                if isinstance(norm, (_BatchNorm, _InstanceNorm)):
                    warnings.warn(
                        'Unnecessary conv bias before batch/instance norm')
        else:
            self.norm_name = None

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            # nn.Tanh has no 'inplace' argument
            if act_cfg_['type'] not in [
                    'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish'
            ]:
                act_cfg_.setdefault('inplace', inplace)
            self.activate = build_activation_layer(act_cfg_)

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners by calling their own ``init_weights()``,
        #    and we do not want ConvModule to override the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners (that is, they don't have their own ``init_weights()``)
        #    and PyTorch's conv layers, they will be initialized by
        #    this method with default ``kaiming_init``.
        # Note: For PyTorch's conv layers, they will be overwritten by our
        #    initialization implementation using default ``kaiming_init``.
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
        return x

    def forward_fuse(self, x):
        if self.with_activation:
            return self.activate(self.conv(x))
        else:
            return self.conv(x)

    def fuse_conv_and_bn(self, ):
        if self.norm:
            fusedconv = nn.Conv2d(self.conv.in_channels,
                                  self.conv.out_channels,
                                  kernel_size=self.conv.kernel_size,
                                  stride=self.conv.stride,
                                  padding=self.conv.padding,
                                  groups=self.conv.groups,
                                  bias=True).requires_grad_(False).to(
                                      self.conv.weight.device)

            # prepare filters
            w_conv = self.conv.weight.clone().view(self.conv.out_channels, -1)
            w_bn = torch.diag(
                self.norm.weight.div(
                    torch.sqrt(self.norm.eps + self.norm.running_var)))
            fusedconv.weight.copy_(
                torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

            # prepare spatial bias
            b_conv = torch.zeros(
                self.conv.weight.size(0), device=self.conv.weight.device
            ) if self.conv.bias is None else self.conv.bias
            b_bn = self.norm.bias - self.norm.weight.mul(
                self.norm.running_mean).div(
                    torch.sqrt(self.norm.running_var + self.norm.eps))
            fusedconv.bias.copy_(
                torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
            self.conv = fusedconv
            delattr(self, self.norm_name)
            self.forward = self.forward_fuse


class DepthwiseSeparableConvModule(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 inplace=False,
                 dilation=1,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 dw_norm_cfg='default',
                 dw_act_cfg='default',
                 pw_norm_cfg='default',
                 pw_act_cfg='default',
                 **kwargs):
        super(DepthwiseSeparableConvModule, self).__init__()
        assert 'groups' not in kwargs, 'groups should not be specified'

        # if norm/activation config of depthwise/pointwise ConvModule is not
        # specified, use default config.
        dw_norm_cfg = dw_norm_cfg if dw_norm_cfg != 'default' else norm_cfg
        dw_act_cfg = dw_act_cfg if dw_act_cfg != 'default' else act_cfg
        pw_norm_cfg = pw_norm_cfg if pw_norm_cfg != 'default' else norm_cfg
        pw_act_cfg = pw_act_cfg if pw_act_cfg != 'default' else act_cfg

        # depthwise convolution
        self.depthwise_conv = ConvModule(in_channels,
                                         in_channels,
                                         kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         inplace=inplace,
                                         dilation=dilation,
                                         groups=in_channels,
                                         norm_cfg=dw_norm_cfg,
                                         act_cfg=dw_act_cfg,
                                         **kwargs)

        self.pointwise_conv = ConvModule(in_channels,
                                         out_channels,
                                         1,
                                         inplace=inplace,
                                         norm_cfg=pw_norm_cfg,
                                         act_cfg=pw_act_cfg,
                                         **kwargs)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class DarknetBottleneck(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=0.5,
                 add_identity=True,
                 use_depthwise=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=None):
        super().__init__(init_cfg)
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = ConvModule(in_channels,
                                hidden_channels,
                                1,
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg,
                                act_cfg=act_cfg)
        self.conv2 = conv(hidden_channels,
                          out_channels,
                          3,
                          stride=1,
                          padding=1,
                          conv_cfg=conv_cfg,
                          norm_cfg=norm_cfg,
                          act_cfg=act_cfg)
        self.add_identity = \
            add_identity and in_channels == out_channels

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out


class CSPLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 expand_ratio=0.5,
                 num_blocks=1,
                 add_identity=True,
                 use_depthwise=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=None):
        super().__init__(init_cfg)
        mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvModule(in_channels,
                                    mid_channels,
                                    1,
                                    conv_cfg=conv_cfg,
                                    norm_cfg=norm_cfg,
                                    act_cfg=act_cfg)
        self.short_conv = ConvModule(in_channels,
                                     mid_channels,
                                     1,
                                     conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg,
                                     act_cfg=act_cfg)
        self.final_conv = ConvModule(2 * mid_channels,
                                     out_channels,
                                     1,
                                     conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg,
                                     act_cfg=act_cfg)

        self.blocks = nn.Sequential(*[
            DarknetBottleneck(mid_channels,
                              mid_channels,
                              1.0,
                              add_identity,
                              use_depthwise,
                              conv_cfg=conv_cfg,
                              norm_cfg=norm_cfg,
                              act_cfg=act_cfg) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = torch.cat((x_main, x_short), dim=1)
        return self.final_conv(x_final)


class Bottleneck_YOLOV5(nn.Module):
    # Standard bottleneck
    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=0.5,
                 add_identity=True,
                 use_depthwise=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 inplace=False):
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = ConvModule(in_channels,
                                hidden_channels,
                                1,
                                inplace=inplace,
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg,
                                act_cfg=act_cfg)
        self.conv2 = conv(hidden_channels,
                          out_channels,
                          3,
                          padding=1,
                          inplace=inplace,
                          conv_cfg=conv_cfg,
                          norm_cfg=norm_cfg,
                          act_cfg=act_cfg)
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x):
        return x + self.conv2(
            self.conv1(x)) if self.add_identity else self.conv2(self.conv1(x))


class CSPLayer_YOLOV5(nn.Module):
    # CSP Bottleneck with 3 convolutions    csp结构
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=1,
                 expand_ratio=0.5,
                 add_identity=True,
                 use_depthwise=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='SiLU'),
                 inplace=False
                 ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        mid_channels = int(out_channels * expand_ratio)  # hidden channels
        self.main_conv = ConvModule(in_channels,
                                    mid_channels,
                                    kernel_size=1,
                                    inplace=inplace,
                                    conv_cfg=conv_cfg,
                                    norm_cfg=norm_cfg,
                                    act_cfg=act_cfg)
        ######shortcut分支
        self.short_conv = ConvModule(in_channels,
                                     mid_channels,
                                     kernel_size=1,
                                     conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg,
                                     act_cfg=act_cfg)
        #####对concat特征卷积
        self.final_conv = ConvModule(2 * mid_channels,
                                     out_channels,
                                     kernel_size=1,
                                     inplace=inplace,
                                     conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg,
                                     act_cfg=act_cfg)  # act=FReLU(c2)
        self.blocks = nn.Sequential(*[
            Bottleneck_YOLOV5(mid_channels,
                              mid_channels,
                              expansion=1.0,
                              add_identity=add_identity,
                              use_depthwise=use_depthwise,
                              conv_cfg=conv_cfg,
                              norm_cfg=norm_cfg,
                              act_cfg=act_cfg,
                              inplace=inplace) for _ in range(num_blocks)
        ])

    def forward(self, x):

        return self.final_conv(
            torch.cat((self.blocks(self.main_conv(x)), self.short_conv(x)),
                      dim=1))


def fuse_model(model):
    for m in model.modules():
        if type(m) is ConvModule and hasattr(m, 'norm'):
            m.fuse_conv_and_bn()


class block_dataset:

    def __init__(self, block_num=2, factor=0.25, stride=32, img_size=1024):

        self.block_num = block_num
        self.factor = 0.25
        self.stride = stride
        self.img_size = img_size

    def clip_cordinate(self, cordinates, shape):
        for sub_cor in cordinates:
            sub_cor[0] = max(0, min(sub_cor[0], shape))
            sub_cor[1] = max(0, min(sub_cor[1], shape))

    def get_blocks(self, image):
        # 获取分块的图像
        block_images_rect = []
        block_images = []
        block_cordinates = []
        shape = image.shape
        row_block = shape[0] // self.block_num
        col_block = shape[1] // self.block_num
        row_cordinates = [[
            row * row_block,
            (row + 1) * row_block + int(row_block * self.factor)
        ] for row in range(self.block_num)]
        col_cordinates = [[
            col * col_block,
            (col + 1) * col_block + int(col_block * self.factor)
        ] for col in range(self.block_num)]
        self.clip_cordinate(row_cordinates, shape[0])
        self.clip_cordinate(col_cordinates, shape[1])

        for row in range(self.block_num):
            for col in range(self.block_num):
                # index = row * self.block_num + col
                row_cordinate = row_cordinates[row]
                col_cordinate = col_cordinates[col]
                block_image = image[row_cordinate[0]:row_cordinate[1],
                                    col_cordinate[0]:col_cordinate[1], :]
                block_image_rect = letterbox(block_image,
                                             self.img_size,
                                             stride=self.stride)[0]
                block_image_rect = block_image_rect.transpose(
                    (2, 0, 1))[::-1][None]
                # print(block_image_rect.shape)
                block_images_rect.append(
                    np.ascontiguousarray(block_image_rect))
                block_images.append(block_image)
                block_cordinates.append([row_cordinate, col_cordinate])
        image_rect = letterbox(image, self.img_size, stride=self.stride)[0]
        image_rect = np.ascontiguousarray(
            image_rect.transpose((2, 0, 1))[::-1][None])
        return block_images_rect, block_images, block_cordinates, image_rect, image


def clip_border(box, shape):
    box[0] = max(0, min(box[0], shape[1]))
    box[2] = max(0, min(box[2], shape[1]))
    box[1] = max(0, min(box[1], shape[0]))
    box[3] = max(0, min(box[3], shape[0]))


def _reduce(loss, reduction, **kwargs):
    if reduction == 'none':
        ret = loss
    elif reduction == 'mean':
        normalizer = loss.numel()
        if kwargs.get('normalizer', None):
            normalizer = kwargs['normalizer']
        ret = loss.sum() / normalizer
    elif reduction == 'sum':
        ret = loss.sum()
    else:
        raise ValueError(reduction + ' is not valid')
    return ret


def dynamic_normalizer(input, target, alpha, gamma):

    def reduce_(tensor, gamma):
        return tensor.pow(gamma).sum()

    target = target.reshape(-1).long()
    input_p = input.detach().sigmoid()
    ########提取正样本序号
    pos_mask = torch.nonzero(target >= 1).squeeze()
    ########提取合法标签序号
    valid_mask = torch.nonzero(target >= 0).squeeze()
    ########1减去正样本输出后求平方
    pos_normalizer = reduce_((1 - input_p[pos_mask, target[pos_mask] - 1]),
                             gamma)
    ########所有负样本输出的平方
    neg_normalizer = reduce_(input_p[valid_mask], gamma) - reduce_(
        input_p[pos_mask, target[pos_mask] - 1], gamma)
    pos_normalizer *= alpha
    neg_normalizer *= 1 - alpha
    normalizer = torch.clamp(pos_normalizer + neg_normalizer, min=1)
    return normalizer


class BasicBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=1,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert self.expansion == 1
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.norm1_name, norm1 = build_norm_layer(norm_cfg,
                                                  self.mid_channels,
                                                  postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg,
                                                  out_channels,
                                                  postfix=2)

        self.conv1 = build_conv_layer(conv_cfg,
                                      in_channels,
                                      self.mid_channels,
                                      3,
                                      stride=stride,
                                      padding=dilation,
                                      dilation=dilation,
                                      bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(conv_cfg,
                                      self.mid_channels,
                                      out_channels,
                                      3,
                                      padding=1,
                                      bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        out = _inner_forward(x)
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=4,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        assert style in ['pytorch', 'caffe']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg,
                                                  self.mid_channels,
                                                  postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg,
                                                  self.mid_channels,
                                                  postfix=2)
        self.norm3_name, norm3 = build_norm_layer(norm_cfg,
                                                  out_channels,
                                                  postfix=3)

        self.conv1 = build_conv_layer(conv_cfg,
                                      in_channels,
                                      self.mid_channels,
                                      kernel_size=1,
                                      stride=self.conv1_stride,
                                      bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(conv_cfg,
                                      self.mid_channels,
                                      self.mid_channels,
                                      kernel_size=3,
                                      stride=self.conv2_stride,
                                      padding=dilation,
                                      dilation=dilation,
                                      bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(conv_cfg,
                                      self.mid_channels,
                                      out_channels,
                                      kernel_size=1,
                                      bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: the normalization layer named "norm3" """
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        out = _inner_forward(x)
        out = self.relu(out)

        return out


def get_expansion(block, expansion=None):
    if isinstance(expansion, int):
        assert expansion > 0
    elif expansion is None:
        if hasattr(block, 'expansion'):
            expansion = block.expansion
        elif issubclass(block, BasicBlock):
            expansion = 1
        elif issubclass(block, Bottleneck):
            expansion = 4
        else:
            raise TypeError(f'expansion is not specified for {block.__name__}')
    else:
        raise TypeError('expansion must be an integer or None')

    return expansion


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def flip_back(output_flipped, flip_pairs, target_type='GaussianHeatmap'):
    assert output_flipped.ndim == 4, \
        'output_flipped should be [batch_size, num_keypoints, height, width]'
    shape_ori = output_flipped.shape
    channels = 1

    if target_type.lower() == 'CombinedTarget'.lower():
        channels = 3
        output_flipped[:, 1::3, ...] = -output_flipped[:, 1::3, ...]
    output_flipped = output_flipped.reshape(shape_ori[0], -1, channels,
                                            shape_ori[2], shape_ori[3])

    output_flipped_back = output_flipped.copy()

    # Swap left-right parts
    for left, right in flip_pairs:
        output_flipped_back[:, left, ...] = output_flipped[:, right, ...]
        output_flipped_back[:, right, ...] = output_flipped[:, left, ...]
    output_flipped_back = output_flipped_back.reshape(shape_ori)
    # Flip horizontally  调换配对的顺序，然后水平翻转
    output_flipped_back = output_flipped_back[..., ::-1]
    return output_flipped_back


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()
    assert (num_channels % groups == 0), ('num_channels should be '
                                          'divisible by groups')
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x


import thop


def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def profile(x, ops, n=100, device=None):
    # profile a pytorch module or list of modules. Example usage:
    #     x = torch.randn(16, 3, 640, 640)  # input
    #     m1 = lambda x: x * torch.sigmoid(x)
    #     m2 = nn.SiLU()
    #     profile(x, [m1, m2], n=100)  # profile speed over 100 iterations

    device = device  #or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    x.requires_grad = True
    print(torch.__version__, device.type,
          torch.cuda.get_device_properties(0) if device.type == 'cuda' else '')
    print(
        f"\n{'Params':>12s}{'GFLOPs':>12s}{'forward (ms)':>16s}{'backward (ms)':>16s}{'input':>24s}{'output':>24s}"
    )
    for m in ops if isinstance(ops, list) else [ops]:
        m = m.to(device) if hasattr(m, 'to') else m  # device
        # m = m.half() if hasattr(m, 'half') and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m  # type
        dtf, dtb, t = 0., 0., [0., 0., 0.]  # dt forward, backward
        try:
            flops = thop.profile(m, inputs=(x, ),
                                 verbose=False)[0] / 1E9 * 2  # GFLOPs
        except:
            flops = 0

        for _ in range(n):
            t[0] = time_sync()
            y = m(x)
            t[1] = time_sync()
            try:
                _ = y.sum().backward()
                t[2] = time_sync()
            except:  # no backward method
                t[2] = float('nan')
            dtf += (t[1] - t[0]) * 1000 / n  # ms per op forward
            dtb += (t[2] - t[1]) * 1000 / n  # ms per op backward

        s_in = tuple(x.shape) if isinstance(x, torch.Tensor) else 'list'
        s_out = tuple(y.shape) if isinstance(y, torch.Tensor) else 'list'
        p = sum(list(x.numel() for x in m.parameters())) if isinstance(
            m, nn.Module) else 0  # parameters
        print(
            f'{p:12}{flops:12.4g}{dtf:16.4g}{dtb:16.4g}{str(s_in):>24s}{str(s_out):>24s}'
        )
