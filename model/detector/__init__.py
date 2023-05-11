from .yolov5 import Yolov5
from .top_down import TopDown
DETECTORS=dict(
    yolov5=Yolov5,
    top_down=TopDown
)

def build_detector(cfg):
    type=cfg.pop("type")
    return DETECTORS[type](**cfg)