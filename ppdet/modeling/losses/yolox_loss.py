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
from IPython import embed
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register

__all__ = ['YOLOXLoss']


def bboxes_iou_batch(bboxes_a, bboxes_b, xyxy=True):
    """
    Args:
        bboxes_a: (tensor) bounding boxes, Shape: [N, A, 4].
        bboxes_b: (tensor) bounding boxes, Shape: [N, B, 4].
    Return:
      (tensor) iou, Shape: [N, A, B].
    """
    N = bboxes_a.shape[0]
    A = bboxes_a.shape[1]
    B = bboxes_b.shape[1]
    if xyxy:
        box_a = bboxes_a
        box_b = bboxes_b
    else:  # cxcywh
        box_a = paddle.concat([bboxes_a[:, :, :2] - bboxes_a[:, :, 2:] * 0.5,
                               bboxes_a[:, :, :2] + bboxes_a[:, :, 2:] * 0.5], axis=-1)
        box_b = paddle.concat([bboxes_b[:, :, :2] - bboxes_b[:, :, 2:] * 0.5,
                               bboxes_b[:, :, :2] + bboxes_b[:, :, 2:] * 0.5], axis=-1)

    box_a_rb = paddle.reshape(box_a[:, :, 2:], (N, A, 1, 2))
    box_a_rb = paddle.tile(box_a_rb, [1, 1, B, 1])
    box_b_rb = paddle.reshape(box_b[:, :, 2:], (N, 1, B, 2))
    box_b_rb = paddle.tile(box_b_rb, [1, A, 1, 1])
    max_xy = paddle.minimum(box_a_rb, box_b_rb)

    box_a_lu = paddle.reshape(box_a[:, :, :2], (N, A, 1, 2))
    box_a_lu = paddle.tile(box_a_lu, [1, 1, B, 1])
    box_b_lu = paddle.reshape(box_b[:, :, :2], (N, 1, B, 2))
    box_b_lu = paddle.tile(box_b_lu, [1, A, 1, 1])
    min_xy = paddle.maximum(box_a_lu, box_b_lu)

    inter = F.relu(max_xy - min_xy)
    inter = inter[:, :, :, 0] * inter[:, :, :, 1]

    box_a_w = box_a[:, :, 2]-box_a[:, :, 0]
    box_a_h = box_a[:, :, 3]-box_a[:, :, 1]
    area_a = box_a_h * box_a_w
    area_a = paddle.reshape(area_a, (N, A, 1))
    area_a = paddle.tile(area_a, [1, 1, B])  # [N, A, B]

    box_b_w = box_b[:, :, 2]-box_b[:, :, 0]
    box_b_h = box_b[:, :, 3]-box_b[:, :, 1]
    area_b = box_b_h * box_b_w
    area_b = paddle.reshape(area_b, (N, 1, B))
    area_b = paddle.tile(area_b, [1, A, 1])  # [N, A, B]

    union = area_a + area_b - inter + 1e-9
    return inter / union  # [N, A, B]


class IOUloss(nn.Layer):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.reshape([-1, 4])
        target = target.reshape([-1, 4])
        tl = paddle.maximum(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = paddle.minimum(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = paddle.prod(pred[:, 2:], 1)
        area_g = paddle.prod(target[:, 2:], 1)

        en = (tl < br).astype(tl.dtype).prod(axis=1)
        area_i = paddle.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = paddle.minimum(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = paddle.maximum(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = paddle.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clip(1e-16)
            loss = 1 - giou.clip(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

@register
class YOLOXLoss(nn.Layer):
    __shared__ = ['num_classes']
    """
    YOLOXLoss
    Args:
        loss_type (str): 
        reduction (str):
        reg_weight (float): weight for location loss
    """
    def __init__(self,
                 num_classes=80,
                 loss_type="iou",
                 reduction=None,
                 reg_weight=5.0):
        super(YOLOXLoss, self).__init__()
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.reduction = reduction
        self.reg_weight = reg_weight
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")

    def iou_loss_bug(self, pred, target):
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

    def get_yolox_loss(self, num_fg, bbox_preds, obj_preds, cls_preds, origin_preds,
                fg_masks, reg_targets, obj_targets, cls_targets, l1_targets, use_l1):
        """
        Calculate the loss for classification, location and centerness

        Return:
            loss (dict): loss composed by classification loss, bounding box
        """
        num_fg = max(num_fg, 1)
        bbox_preds = paddle.reshape(bbox_preds, [-1, 4])  # [N*A, 4]
        pos_index = paddle.nonzero(fg_masks > 0)[:, 0]    # [num_fg, ]
        pos_bbox_preds = paddle.gather(bbox_preds, pos_index)  # [num_fg, 4] xywh
        # 1 [15525.62792969] [15284.11523438] 
        # 2 [3264.59082031] [3550.83007812]
        #print('  reg_targets.sum()  ', pos_bbox_preds.sum(), reg_targets.sum())

        loss_iou = (
            self.iou_loss(pos_bbox_preds, reg_targets)
        ).sum() / num_fg

        loss_obj = (
            self.bcewithlog_loss(paddle.reshape(obj_preds, [-1, 1]), obj_targets)
        ).sum() / num_fg

        cls_preds = paddle.reshape(cls_preds, [-1, self.num_classes])  # [N*A, 80]
        pos_cls_preds = paddle.gather(cls_preds, pos_index)       # [num_fg, 80]
        loss_cls = (
            self.bcewithlog_loss(pos_cls_preds, cls_targets)
        ).sum() / num_fg

        if use_l1:
            # origin_preds is already concated
            origin_preds = paddle.reshape(origin_preds, [-1, 4])       # [N*A, 4]
            pos_origin_preds = paddle.gather(origin_preds, pos_index)  # [num_fg, 4]
            loss_l1 = (
                self.l1_loss(pos_origin_preds, l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = paddle.to_tensor([0.])
            loss_l1.stop_gradient = False

        losses = {
            "loss_iou": self.reg_weight * loss_iou,
            "loss_conf": loss_obj,
            "loss_cls": loss_cls,
            "loss_l1": loss_l1,
        }
        return losses




    def forward(self, outputs, x_shifts, y_shifts, expanded_strides, origin_preds, labels, dtype, use_l1):
        #N, A = outputs.shape[:2]

        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        nlabel = (labels.sum(axis=2) > 0).sum(axis=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = paddle.concat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = paddle.concat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = paddle.concat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = paddle.concat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0


        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = paddle.zeros((0, self.num_classes))
                reg_target = paddle.zeros((0, 4))
                l1_target = paddle.zeros((0, 4))
                obj_target = paddle.zeros((total_num_anchors, 1))
                fg_mask = paddle.zeros([total_num_anchors], dtype=bool)
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.astype(paddle.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        paddle.zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            if len(reg_target.shape) == 1:
                reg_target.unsqueeze_(0)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.astype(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = paddle.concat(cls_targets, 0)
        reg_targets = paddle.concat(reg_targets, 0)
        obj_targets = paddle.concat(obj_targets, 0)
        fg_masks = paddle.concat(fg_masks, 0)
        if self.use_l1:
            l1_targets = paddle.concat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
            self.iou_loss(bbox_preds.reshape([-1, 4])[fg_masks], reg_targets)
        ).sum() / num_fg

        loss_obj = (
            self.bcewithlog_loss(obj_preds.reshape([-1, 1]), obj_targets)
        ).sum() / num_fg

        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.reshape([-1, self.num_classes])[fg_masks], cls_targets
            )
        ).sum() / num_fg

        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.reshape([-1, 4])[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )


    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        if len(gt.shape) == 1:
            gt.unsqueeze_(0)
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = paddle.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = paddle.log(gt[:, 3] / stride + eps)
        l1_target.stop_gradient = True #
        return l1_target


    @paddle.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        bbox_preds,
        obj_preds,
        labels,
        imgs,
        mode="gpu",
    ):
        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().astype('float32')
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().astype('float32')
            gt_classes = gt_classes.cpu().astype('float32')
            expanded_strides = expanded_strides.cpu().astype('float32')
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )
        #print(fg_mask.sum(0).item(), fg_mask.shape)
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        # print(num_in_boxes_anchor)
        gt_cls_per_image = (
            F.one_hot(gt_classes.astype(paddle.int64), self.num_classes)
            .astype('float32')
            .unsqueeze(1)
            .tile([1, num_in_boxes_anchor, 1])
        )
        pair_wise_ious_loss = -paddle.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with paddle.amp.auto_cast(enable=False):
            cls_preds_ = (
                F.sigmoid(cls_preds_.astype('float32').unsqueeze(0).tile([num_gt, 1, 1]))
                * F.sigmoid(obj_preds_.astype('float32').unsqueeze(0).tile([num_gt, 1, 1]))
            )
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
            fg_mask,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .tile([num_gt, 1])
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .tile([num_gt, 1])
        )

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .tile([1, total_num_anchors])
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .tile([1, total_num_anchors])
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .tile([1, total_num_anchors])
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .tile([1, total_num_anchors])
        )

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = paddle.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(axis=-1) > 0.0
        is_in_boxes_all = is_in_boxes.astype(paddle.int64).sum(axis=0) > 0
        # in fixed center

        center_radius = 2.5

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).tile(
            [1, total_num_anchors]
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).tile(
            [1, total_num_anchors]
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).tile(
            [1, total_num_anchors]
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).tile(
            [1, total_num_anchors]
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = paddle.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(axis=-1) > 0.0
        is_in_centers_all = is_in_centers.astype(paddle.int64).sum(axis=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        '''
        is_in_boxes_and_center = (
            is_in_boxes.astype('float32').masked_select(is_in_boxes_anchor.tile([is_in_boxes.shape[0], 1])).reshape([is_in_boxes.shape[0], -1]).astype(bool) & \
            is_in_centers.astype('float32').masked_select(is_in_boxes_anchor.tile([is_in_centers.shape[0], 1])).reshape([is_in_centers.shape[0], -1]).astype(bool)
        )
        '''
        is_in_boxes_and_center = (
            paddle.to_tensor(is_in_boxes.numpy()[:, is_in_boxes_anchor.numpy()]) & paddle.to_tensor(is_in_centers.numpy()[:, is_in_boxes_anchor.numpy()])
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        # print(cost.shape, pair_wise_ious.shape, gt_classes.shape, fg_mask.shape)
        cost, pair_wise_ious, gt_classes, fg_mask = cost.cpu().numpy(), pair_wise_ious.cpu().numpy(), gt_classes.cpu().numpy(), fg_mask.cpu().numpy()
        matching_matrix = np.zeros_like(cost, dtype=np.uint8)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.shape[1])
        topk_ious, _ = paddle.topk(paddle.to_tensor(ious_in_boxes_matrix), n_candidate_k, axis=1)
        topk_ious = topk_ious.numpy()
        dynamic_ks = np.clip(topk_ious.sum(1).astype('int32'), a_min=1, a_max=None)
        dynamic_ks = dynamic_ks.tolist()
        for gt_idx in range(num_gt):
            _, pos_idx = paddle.topk(
                paddle.to_tensor(cost[gt_idx]), k=dynamic_ks[gt_idx], largest=False
            )
            pos_idx = pos_idx.numpy()
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            cost_argmin = np.argmin(cost[:, anchor_matching_gt > 1], axis=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_fg = fg_mask_inboxes.sum().item()
        fg_mask[fg_mask] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, paddle.to_tensor(gt_matched_classes), paddle.to_tensor(pred_ious_this_matching), paddle.to_tensor(matched_gt_inds), paddle.to_tensor(fg_mask)
