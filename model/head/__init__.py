from .yolov5_head import YOLOV5Head
from .top_down import TopdownHeatmapSimpleHead
HEADERS={
    "yolov5": YOLOV5Head,
    "top_down_head": TopdownHeatmapSimpleHead
}

def build_head(cfg):
    type=cfg.pop("type")
    return HEADERS[type](**cfg)