import torch.nn as nn
from model.backbone import build_backbone
from model.neck import build_neck
from model.head import build_head
class TopDown(nn.Module):
    def __init__(self,
                 backbone,
                 neck=None,
                 keypoint_head=None,
                 test_cfg=None,
                 pretrained=None):

        super().__init__()
        self.backbone = build_backbone(backbone)
        self.test_cfg = test_cfg
        if neck is not None:
            self.neck = build_neck(neck)

        if keypoint_head is not None:
            keypoint_head['test_cfg'] = test_cfg
            self.keypoint_head = build_head(keypoint_head)

    @property
    def with_neck(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'neck')

    @property
    def with_keypoint(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'keypoint_head')

    def forward(self, img):
        output = self.backbone(img)
        if self.with_neck:
            output = self.neck(output)
        if self.with_keypoint:
            output = self.keypoint_head(output)
        return output
