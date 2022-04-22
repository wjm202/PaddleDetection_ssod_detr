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
from ..bbox_utils import decode_yolo, xywh2xyxy, iou_similarity, bbox_iou

__all__ = ['YOLOv3Loss', 'YOLOv5Loss']


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
class YOLOv5Loss(nn.Layer):
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 downsample_ratios=[8, 16, 32],
                 balance=[4.0, 1.0, 0.4],
                 box_weight=0.05,
                 obj_weight=1.0,
                 cls_weght=0.5,
                 bias=0.5,
                 anchor_t=4.0,
                 label_smooth_eps=0.):
        super(YOLOv5Loss, self).__init__()
        self.num_classes = num_classes
        self.balance = balance
        self.na = 3  # not len(anchors) # anchors.shape [na, 3, 2]
        self.gr = 1.0

        self.BCEcls = nn.BCEWithLogitsLoss(
            pos_weight=paddle.to_tensor([1.0]), reduction='none')
        self.BCEobj = nn.BCEWithLogitsLoss(
            pos_weight=paddle.to_tensor([1.0]), reduction='none')

        self.loss_weights = {
            'box': box_weight,
            'obj': obj_weight,
            'cls': cls_weght,
        }

        eps = label_smooth_eps if label_smooth_eps > 0 else 0.
        self.cls_pos_label = 1.0 - 0.5 * eps
        self.cls_neg_label = 0.5 * eps

        self.downsample_ratios = downsample_ratios
        self.bias = bias
        self.off = paddle.to_tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
            ],
            dtype='float32') * self.bias
        self.anchor_t = anchor_t

    def build_targets(self, outputs, targets, anchors):
        gt_nums = [len(bbox) for bbox in targets['gt_bbox']]
        nt = int(sum(gt_nums))
        na = len(anchors) # anchors.shape[1]  # not len(anchors) # shape [na, 3, 2]
        tcls, tbox, indices, anch = [], [], [], []

        gain = paddle.ones([7], dtype='float32')  # normalized to gridspace gain
        if nt:
            ai = paddle.tile(
                paddle.cast(
                    paddle.arange(na), dtype='float32').reshape([na, 1]),
                [1, nt])  # [na, num_gts]

        batch_size = outputs[0].shape[0]
        gt_labels = []
        for idx in range(batch_size):
            gt_num = gt_nums[idx]
            if gt_num == 0:
                continue
            gt_bbox = paddle.cast(
                targets['gt_bbox'][idx][:gt_num], dtype='float32')
            gt_class = paddle.cast(
                targets['gt_class'][idx][:gt_num], dtype='float32') * 1.0
            # img_idx = np.repeat(np.array([[idx]]), gt_num, axis=0)
            img_idx = paddle.tile(
                paddle.to_tensor(
                    [[idx]], dtype='float32'), [gt_num, 1])
            gt_labels.append(
                paddle.concat(
                    (img_idx, gt_class, gt_bbox), axis=-1))
        if (len(gt_labels)):
            gt_labels = paddle.concat(gt_labels)  # [num_gts, 6]

        if nt:
            targets_labels = paddle.concat(
                (paddle.tile(gt_labels.unsqueeze(0), [na, 1, 1]),
                 ai.unsqueeze(-1)),
                axis=2)
        # targets_labels.shape [na, num_gts, 7]

        for i in range(len(anchors)):
            anchor = anchors[i] / self.downsample_ratios[i]
            # gain[2:6] = np.array(outputs[i].shape, dtype=np.float32)[[3, 2, 3, 2]] # xyxy gain
            gain[2:6] = paddle.to_tensor(
                outputs[i].shape, dtype='float32')[[3, 2, 3, 2]]

            # Match targets_labels to
            if nt:
                t = targets_labels * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchor[:, None]  # wh ratio
                j = paddle.maximum(r, 1 / r).max(2) < self.anchor_t
                t = t[j]  # filter

                # Offsets
                if len(t):
                    gxy = t[:, 2:4]  # grid xy
                    gxi = gain[[2, 3]] - gxy  # inverse
                    j, k = ((gxy % 1 < self.bias) & (gxy > 1)).T
                    l, m = ((gxi % 1 < self.bias) & (gxi > 1)).T
                    j = paddle.concat(
                        (paddle.ones_like(j).unsqueeze(0), j.unsqueeze(0),
                         k.unsqueeze(0), l.unsqueeze(0), m.unsqueeze(0)))
                    t = paddle.tile(t, [5, 1, 1])[j]
                    offsets = (
                        paddle.zeros_like(gxy)[None] + self.off[:, None])[j]
                else:
                    indices.append(
                        (paddle.to_tensor([]), paddle.to_tensor([]),
                         paddle.to_tensor([]),
                         paddle.to_tensor([])))  # image, anchor, grid indices
                    tbox.append(paddle.to_tensor([]))  # box
                    anch.append(paddle.to_tensor([]))  #
                    tcls.append(paddle.to_tensor([]))  # class
                    continue
            else:
                # t = targets_labels[0]
                # offsets = 0
                indices.append(
                    (paddle.to_tensor([]), paddle.to_tensor([]),
                     paddle.to_tensor([]),
                     paddle.to_tensor([])))  # image, anchor, grid indices
                tbox.append(paddle.to_tensor([]))  # box
                anch.append(paddle.to_tensor([]))  # anchor
                tcls.append(paddle.to_tensor([]))  # class
                continue

            if len(t):
                # Define
                b, c = paddle.cast(t[:, :2], 'int64').T  # image, class
                gxy = t[:, 2:4]  # grid xy
                gwh = t[:, 4:6]  # grid wh
                gij = paddle.cast((gxy - offsets), 'int64')
                gi, gj = gij.T  # grid xy indices

                # Append
                a = paddle.cast(t[:, 6], 'int64')  # anchor indices
                gj, gi = gj.clip(0, gain[3] - 1), gi.clip(0, gain[2] - 1)  # clamp_
                indices.append((b, a, gj, gi))  # image, anchor, grid indices
                tbox.append(paddle.concat((gxy - gij, gwh), 1))
                anch.append(anchor[a])
                tcls.append(c)
            else:
                indices.append(
                    (paddle.to_tensor([]), paddle.to_tensor([]),
                     paddle.to_tensor([]),
                     paddle.to_tensor([])))  # image, anchor, grid indices
                tbox.append(paddle.to_tensor([]))  # box
                anch.append(paddle.to_tensor([]))  #
                tcls.append(paddle.to_tensor([]))  # class

        return tcls, tbox, indices, anch

    def yolov5_loss(self, pi, t_cls, t_box, t_indices, t_anchor, balance):
        loss = dict()
        b, a, gj, gi = t_indices  # image, anchor, gridy, gridx
        n = b.shape[0]  # number of targets
        tobj = paddle.zeros_like(pi[:, :, :, :, 0])
        tobj.stop_gradient = True

        loss_box = paddle.to_tensor([0.])
        loss_cls = paddle.to_tensor([0.])
        if n:
            ps = pi[b, a, gj, gi]  # [4, 3, 80, 80, 85] -> [21, 85]

            if len(ps.shape) == 1:
                ps = ps.unsqueeze(0)
            # loss_box
            pxy = F.sigmoid(ps[:, :2]) * 2 - 0.5
            pwh = (F.sigmoid(ps[:, 2:4]) * 2)**2 * t_anchor
            pbox = paddle.concat((pxy, pwh), 1)
            iou = bbox_iou(pbox.T, t_box.T, x1y1x2y2=False, ciou=True)
            # iou.stop_gradient = True
            loss_box = (1.0 - iou).mean()

            # loss_obj
            score_iou = paddle.cast(iou.detach().clip(0), tobj.dtype)
            tobj[b, a, gj, gi] = (1.0 - self.gr
                                  ) + self.gr * score_iou  # iou ratio

            # loss_cls
            t = paddle.full_like(ps[:, 5:], self.cls_neg_label)
            t[range(n), t_cls] = self.cls_pos_label
            t.stop_gradient = True
            loss_cls = self.BCEcls(ps[:, 5:], t).mean()

        # loss_obj
        obji = self.BCEobj(pi[:, :, :, :, 4], tobj).mean()  # [4, 3, 80, 80]
        loss_obj = obji * balance

        loss['loss_box'] = loss_box * self.loss_weights['box']
        loss['loss_obj'] = loss_obj * self.loss_weights['obj']
        loss['loss_cls'] = loss_cls * self.loss_weights['cls']
        return loss

    def forward(self, inputs, targets, anchors):
        assert len(inputs) == len(anchors)
        assert len(inputs) == len(self.downsample_ratios)
        yolo_losses = dict()
        tcls, tbox, indices, anch = self.build_targets(inputs, targets, anchors)

        for i, (p_det, balance) in enumerate(zip(inputs, self.balance)):
            t_cls = tcls[i]
            t_box = tbox[i]
            t_anchor = anch[i]
            t_indices = indices[i]

            bs, ch, h, w = p_det.shape
            pi = p_det.reshape((bs, self.na, -1, h, w)).transpose(
                (0, 1, 3, 4, 2))

            yolo_loss = self.yolov5_loss(pi, t_cls, t_box, t_indices, t_anchor,
                                         balance)

            for k, v in yolo_loss.items():
                if k in yolo_losses:
                    yolo_losses[k] += v
                else:
                    yolo_losses[k] = v

        loss = 0
        for k, v in yolo_losses.items():
            loss += v

        batch_size = inputs[0].shape[0]
        batch_size.stop_gradient = True
        num_gpus = targets.get('num_gpus', 8)
        num_gpus.stop_gradient = True
        yolo_losses['loss'] = loss * batch_size * num_gpus
        return yolo_losses
