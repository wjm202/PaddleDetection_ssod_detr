# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from ppdet.modeling import ops

__all__ = ['YOLOXLoss']


@register
class YOLOXLoss(nn.Layer):
    """
    YOLOXLoss
    Args:
        loss_alpha (float): alpha in focal loss
        loss_gamma (float): gamma in focal loss
        iou_loss_type (str): location loss type, IoU/GIoU/LINEAR_IoU
        reg_weights (float): weight for location loss
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
        '''
        输入矩形的格式是cx cy w h
        '''
        assert pred.shape[0] == target.shape[0]

        boxes1 = pred
        boxes2 = target

        # 变成左上角坐标、右下角坐标
        boxes1_x0y0x1y1 = paddle.concat([boxes1[:, :2] - boxes1[:, 2:] * 0.5,
                                         boxes1[:, :2] + boxes1[:, 2:] * 0.5], axis=-1)
        boxes2_x0y0x1y1 = paddle.concat([boxes2[:, :2] - boxes2[:, 2:] * 0.5,
                                         boxes2[:, :2] + boxes2[:, 2:] * 0.5], axis=-1)

        # 两个矩形的面积
        boxes1_area = (boxes1_x0y0x1y1[:, 2] - boxes1_x0y0x1y1[:, 0]) * (boxes1_x0y0x1y1[:, 3] - boxes1_x0y0x1y1[:, 1])
        boxes2_area = (boxes2_x0y0x1y1[:, 2] - boxes2_x0y0x1y1[:, 0]) * (boxes2_x0y0x1y1[:, 3] - boxes2_x0y0x1y1[:, 1])

        # 相交矩形的左上角坐标、右下角坐标
        left_up = paddle.maximum(boxes1_x0y0x1y1[:, :2], boxes2_x0y0x1y1[:, :2])
        right_down = paddle.minimum(boxes1_x0y0x1y1[:, 2:], boxes2_x0y0x1y1[:, 2:])

        # 相交矩形的面积inter_area。iou
        inter_section = F.relu(right_down - left_up)
        inter_area = inter_section[:, 0] * inter_section[:, 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / (union_area + 1e-16)


        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            # 包围矩形的左上角坐标、右下角坐标
            enclose_left_up = paddle.minimum(boxes1_x0y0x1y1[:, :2], boxes2_x0y0x1y1[:, :2])
            enclose_right_down = paddle.maximum(boxes1_x0y0x1y1[:, 2:], boxes2_x0y0x1y1[:, 2:])

            # 包围矩形的面积
            enclose_wh = enclose_right_down - enclose_left_up
            enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]

            giou = iou - (enclose_area - union_area) / enclose_area
            # giou限制在区间[-1.0, 1.0]内
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
        Args:
            cls_logits (list): list of Tensor, which is predicted
                score for all anchor points with shape [N, M, C]
            bboxes_reg (list): list of Tensor, which is predicted
                offsets for all anchor points with shape [N, M, 4]
            centerness (list): list of Tensor, which is predicted
                centerness for all anchor points with shape [N, M, 1]
            tag_labels (list): list of Tensor, which is category
                targets for each anchor point
            tag_bboxes (list): list of Tensor, which is bounding
                boxes targets for positive samples
            tag_center (list): list of Tensor, which is centerness
                targets for positive samples
        Return:
            loss (dict): loss composed by classification loss, bounding box
        """
        num_fg = max(num_fg, 1)
        bbox_preds = paddle.reshape(bbox_preds, [-1, 4])  # [N*A, 4]
        pos_index = paddle.nonzero(fg_masks > 0)[:, 0]    # [num_fg, ]
        pos_bbox_preds = paddle.gather(bbox_preds, pos_index)  # [num_fg, 4]   每个最终正样本预测的xywh
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
