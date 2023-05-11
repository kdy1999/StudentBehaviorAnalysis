from .yolov5pafpn import YOLOV5PAFPN

NECKS=dict(
    yolov5=YOLOV5PAFPN
)

def build_neck(cfg):
    type=cfg.pop("type")
    return NECKS[type](**cfg)