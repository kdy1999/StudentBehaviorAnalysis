from .yolov5_caspdarknet import CSPDarknet_YOLOV5
from .hrnet import HRNet
from .litehrnet import LiteHRNet
BACKBONES={
    "yolov5": CSPDarknet_YOLOV5,
    'hrnet': HRNet,
    'LiteHRNet': LiteHRNet
}

def build_backbone(cfg):
    type=cfg.pop("type")
    return BACKBONES[type](**cfg)





