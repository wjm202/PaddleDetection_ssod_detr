# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register

from ..bbox_utils import decode_yolo, xywh2xyxy, iou_similarity

__all__ = ['YOLOv3Loss', 'YOLOXLoss']


def bbox_transform(pbox, anchor, downsample):
    pbox = decode_yolo(pbox, anchor, downsample)
    pbox = xywh2xyxy(pbox)
    return pbox


@register
class YOLOv3Loss(nn.Layer):

    __inject__ = ['iou_loss', 'iou_aware_loss']
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 ignore_thresh=0.7,
                 label_smooth=False,
                 downsample=[32, 16, 8],
                 scale_x_y=1.,
                 iou_loss=None,
                 iou_aware_loss=None):
        """
        YOLOv3Loss layer

        Args:
            num_calsses (int): number of foreground classes
            ignore_thresh (float): threshold to ignore confidence loss
            label_smooth (bool): whether to use label smoothing
            downsample (list): downsample ratio for each detection block
            scale_x_y (float): scale_x_y factor
            iou_loss (object): IoULoss instance
            iou_aware_loss (object): IouAwareLoss instance  
        """
        super(YOLOv3Loss, self).__init__()
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh
        self.label_smooth = label_smooth
        self.downsample = downsample
        self.scale_x_y = scale_x_y
        self.iou_loss = iou_loss
        self.iou_aware_loss = iou_aware_loss
        self.distill_pairs = []

    def obj_loss(self, pbox, gbox, pobj, tobj, anchor, downsample):
        # pbox
        pbox = decode_yolo(pbox, anchor, downsample)
        pbox = xywh2xyxy(pbox)
        pbox = paddle.concat(pbox, axis=-1)
        b = pbox.shape[0]
        pbox = pbox.reshape((b, -1, 4))
        # gbox
        gxy = gbox[:, :, 0:2] - gbox[:, :, 2:4] * 0.5
        gwh = gbox[:, :, 0:2] + gbox[:, :, 2:4] * 0.5
        gbox = paddle.concat([gxy, gwh], axis=-1)

        iou = iou_similarity(pbox, gbox)
        iou.stop_gradient = True
        iou_max = iou.max(2)  # [N, M1]
        iou_mask = paddle.cast(iou_max <= self.ignore_thresh, dtype=pbox.dtype)
        iou_mask.stop_gradient = True

        pobj = pobj.reshape((b, -1))
        tobj = tobj.reshape((b, -1))
        obj_mask = paddle.cast(tobj > 0, dtype=pbox.dtype)
        obj_mask.stop_gradient = True

        loss_obj = F.binary_cross_entropy_with_logits(
            pobj, obj_mask, reduction='none')
        loss_obj_pos = (loss_obj * tobj)
        loss_obj_neg = (loss_obj * (1 - obj_mask) * iou_mask)
        return loss_obj_pos + loss_obj_neg

    def cls_loss(self, pcls, tcls):
        if self.label_smooth:
            delta = min(1. / self.num_classes, 1. / 40)
            pos, neg = 1 - delta, delta
            # 1 for positive, 0 for negative
            tcls = pos * paddle.cast(
                tcls > 0., dtype=tcls.dtype) + neg * paddle.cast(
                    tcls <= 0., dtype=tcls.dtype)

        loss_cls = F.binary_cross_entropy_with_logits(
            pcls, tcls, reduction='none')
        return loss_cls

    def yolov3_loss(self, p, t, gt_box, anchor, downsample, scale=1.,
                    eps=1e-10):
        na = len(anchor)
        b, c, h, w = p.shape
        if self.iou_aware_loss:
            ioup, p = p[:, 0:na, :, :], p[:, na:, :, :]
            ioup = ioup.unsqueeze(-1)
        p = p.reshape((b, na, -1, h, w)).transpose((0, 1, 3, 4, 2))
        x, y = p[:, :, :, :, 0:1], p[:, :, :, :, 1:2]
        w, h = p[:, :, :, :, 2:3], p[:, :, :, :, 3:4]
        obj, pcls = p[:, :, :, :, 4:5], p[:, :, :, :, 5:]
        self.distill_pairs.append([x, y, w, h, obj, pcls])

        t = t.transpose((0, 1, 3, 4, 2))
        tx, ty = t[:, :, :, :, 0:1], t[:, :, :, :, 1:2]
        tw, th = t[:, :, :, :, 2:3], t[:, :, :, :, 3:4]
        tscale = t[:, :, :, :, 4:5]
        tobj, tcls = t[:, :, :, :, 5:6], t[:, :, :, :, 6:]

        tscale_obj = tscale * tobj
        loss = dict()

        x = scale * F.sigmoid(x) - 0.5 * (scale - 1.)
        y = scale * F.sigmoid(y) - 0.5 * (scale - 1.)

        if abs(scale - 1.) < eps:
            loss_x = F.binary_cross_entropy(x, tx, reduction='none')
            loss_y = F.binary_cross_entropy(y, ty, reduction='none')
            loss_xy = tscale_obj * (loss_x + loss_y)
        else:
            loss_x = paddle.abs(x - tx)
            loss_y = paddle.abs(y - ty)
            loss_xy = tscale_obj * (loss_x + loss_y)

        loss_xy = loss_xy.sum([1, 2, 3, 4]).mean()

        loss_w = paddle.abs(w - tw)
        loss_h = paddle.abs(h - th)
        loss_wh = tscale_obj * (loss_w + loss_h)
        loss_wh = loss_wh.sum([1, 2, 3, 4]).mean()

        loss['loss_xy'] = loss_xy
        loss['loss_wh'] = loss_wh

        if self.iou_loss is not None:
            # warn: do not modify x, y, w, h in place
            box, tbox = [x, y, w, h], [tx, ty, tw, th]
            pbox = bbox_transform(box, anchor, downsample)
            gbox = bbox_transform(tbox, anchor, downsample)
            loss_iou = self.iou_loss(pbox, gbox)
            loss_iou = loss_iou * tscale_obj
            loss_iou = loss_iou.sum([1, 2, 3, 4]).mean()
            loss['loss_iou'] = loss_iou

        if self.iou_aware_loss is not None:
            box, tbox = [x, y, w, h], [tx, ty, tw, th]
            pbox = bbox_transform(box, anchor, downsample)
            gbox = bbox_transform(tbox, anchor, downsample)
            loss_iou_aware = self.iou_aware_loss(ioup, pbox, gbox)
            loss_iou_aware = loss_iou_aware * tobj
            loss_iou_aware = loss_iou_aware.sum([1, 2, 3, 4]).mean()
            loss['loss_iou_aware'] = loss_iou_aware

        box = [x, y, w, h]
        loss_obj = self.obj_loss(box, gt_box, obj, tobj, anchor, downsample)
        loss_obj = loss_obj.sum(-1).mean()
        loss['loss_obj'] = loss_obj
        loss_cls = self.cls_loss(pcls, tcls) * tobj
        loss_cls = loss_cls.sum([1, 2, 3, 4]).mean()
        loss['loss_cls'] = loss_cls
        return loss

    def forward(self, inputs, targets, anchors):
        np = len(inputs)
        gt_targets = [targets['target{}'.format(i)] for i in range(np)]
        gt_box = targets['gt_bbox']
        yolo_losses = dict()
        self.distill_pairs.clear()
        for x, t, anchor, downsample in zip(inputs, gt_targets, anchors,
                                            self.downsample):
            yolo_loss = self.yolov3_loss(x, t, gt_box, anchor, downsample,
                                         self.scale_x_y)
            for k, v in yolo_loss.items():
                if k in yolo_losses:
                    yolo_losses[k] += v
                else:
                    yolo_losses[k] = v

        loss = 0
        for k, v in yolo_losses.items():
            loss += v

        yolo_losses['loss'] = loss
        return yolo_losses


@register
class YOLOXLoss(nn.Layer):
    """
    YOLOXLoss
    Args:
        loss_type (str): 
        reduction (str):
        reg_weight (float): weight for location loss
    """

    def __init__(self,
                 loss_type="iou",
                 reduction=None,
                 reg_weight=5.0):
        super(YOLOXLoss, self).__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        self.reg_weight = reg_weight
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")

    def iou_loss(self, pred, target):
        assert pred.shape[0] == target.shape[0]
        boxes1 = pred
        boxes2 = target

        # cxcywh to x1y1x2y2
        boxes1_x0y0x1y1 = paddle.concat([boxes1[:, :2] - boxes1[:, 2:] * 0.5,
                                         boxes1[:, :2] + boxes1[:, 2:] * 0.5], axis=-1)
        boxes2_x0y0x1y1 = paddle.concat([boxes2[:, :2] - boxes2[:, 2:] * 0.5,
                                         boxes2[:, :2] + boxes2[:, 2:] * 0.5], axis=-1)

        boxes1_area = (boxes1_x0y0x1y1[:, 2] - boxes1_x0y0x1y1[:, 0]) * (boxes1_x0y0x1y1[:, 3] - boxes1_x0y0x1y1[:, 1])
        boxes2_area = (boxes2_x0y0x1y1[:, 2] - boxes2_x0y0x1y1[:, 0]) * (boxes2_x0y0x1y1[:, 3] - boxes2_x0y0x1y1[:, 1])

        left_up = paddle.maximum(boxes1_x0y0x1y1[:, :2], boxes2_x0y0x1y1[:, :2])
        right_down = paddle.minimum(boxes1_x0y0x1y1[:, 2:], boxes2_x0y0x1y1[:, 2:])

        inter_section = F.relu(right_down - left_up)
        inter_area = inter_section[:, 0] * inter_section[:, 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / (union_area + 1e-16)


        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            enclose_left_up = paddle.minimum(boxes1_x0y0x1y1[:, :2], boxes2_x0y0x1y1[:, :2])
            enclose_right_down = paddle.maximum(boxes1_x0y0x1y1[:, 2:], boxes2_x0y0x1y1[:, 2:])

            enclose_wh = enclose_right_down - enclose_left_up
            enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]

            giou = iou - (enclose_area - union_area) / enclose_area
            giou = paddle.clip(giou, -1.0, 1.0)
            loss = 1 - giou
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

    def forward(self, num_fg, bbox_preds, obj_preds, cls_preds, origin_preds,
                fg_masks, reg_targets, obj_targets, cls_targets, l1_targets, num_classes, use_l1):
        """
        Calculate the loss for classification, location and centerness

        Return:
            loss (dict): loss composed by classification loss, bounding box
        """
        num_fg = max(num_fg, 1)
        bbox_preds = paddle.reshape(bbox_preds, [-1, 4])  # [N*A, 4]
        pos_index = paddle.nonzero(fg_masks > 0)[:, 0]    # [num_fg, ]
        pos_bbox_preds = paddle.gather(bbox_preds, pos_index)  # [num_fg, 4] xywh
        loss_iou = (
            self.iou_loss(pos_bbox_preds, reg_targets)
        ).sum() / num_fg
        loss_obj = (
            self.bcewithlog_loss(paddle.reshape(obj_preds, [-1, 1]), obj_targets)
        ).sum() / num_fg

        cls_preds = paddle.reshape(cls_preds, [-1, num_classes])  # [N*A, 80]
        pos_cls_preds = paddle.gather(cls_preds, pos_index)       # [num_fg, 80]
        loss_cls = (
            self.bcewithlog_loss(pos_cls_preds, cls_targets)
        ).sum() / num_fg
        if use_l1:
            origin_preds = paddle.reshape(origin_preds, [-1, 4])       # [N*A, 4]
            pos_origin_preds = paddle.gather(origin_preds, pos_index)  # [num_fg, 4]
            loss_l1 = (
                self.l1_loss(pos_origin_preds, l1_targets)
            ).sum() / num_fg

        losses = {
            "loss_iou": self.reg_weight * loss_iou,
            "loss_obj": loss_obj,
            "loss_cls": loss_cls,
        }
        if use_l1:
            losses["loss_l1"] = loss_l1
        return losses
