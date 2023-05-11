from .ops import multi_apply, DepthwiseSeparableConvModule,ConvModule,CSPLayer,\
    CSPLayer_YOLOV5,initialize_weights,block_dataset,clip_border,_reduce,dynamic_normalizer,\
    fuse_model,Bottleneck,BasicBlock,get_expansion,resize,flip_back,channel_shuffle,\
    set_logger,iou_overlap,get_config
# from .ops import profile
from .basic import build_padding_layer,build_norm_layer,build_activation_layer,\
    build_conv_layer,bias_init_with_prob

buider = {"DWConv": DepthwiseSeparableConvModule, "ConvModule": ConvModule}


def builder_ops(cfg):
    return buider[cfg.pop("type")]