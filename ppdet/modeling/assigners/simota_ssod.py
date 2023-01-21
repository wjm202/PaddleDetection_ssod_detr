import paddle
import numpy as np
import paddle.nn.functional as F

from ppdet.modeling.losses.varifocal_loss import varifocal_loss
from ppdet.modeling.bbox_utils import batch_bbox_overlaps
from ppdet.core.workspace import register
@register
class SimOTA_SSOD(object):
    def __init__(self,
                 center_radius=2.5,
                 candidate_topk=10,
                 iou_weight=3.0,
                 cls_weight=1.0,
                 num_classes=80,
                 use_vfl=True):
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight
        self.num_classes = num_classes
        self.use_vfl = use_vfl
    def __call__():
        print(1)