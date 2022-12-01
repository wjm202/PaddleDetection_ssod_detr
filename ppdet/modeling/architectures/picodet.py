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
from pkgutil import get_data

import paddle
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch
from ..ssod_utils import QFLv2
from ..losses import GIoULoss
__all__ = ['PicoDet']


@register
class PicoDet(BaseArch):
    """
    Generalized Focal Loss network, see https://arxiv.org/abs/2006.04388

    Args:
        backbone (object): backbone instance
        neck (object): 'FPN' instance
        head (object): 'PicoHead' instance
    """

    __category__ = 'architecture'

    def __init__(self, backbone, neck, head='PicoHead'):
        super(PicoDet, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.export_post_process = True
        self.export_nms = True

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])

        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        kwargs = {'input_shape': neck.out_shape}
        head = create(cfg['head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "head": head,
        }

    def _forward(self):
        is_teacher = self.inputs.get('is_teacher', False)
        get_data = self.inputs.get('get_data', False)
        body_feats = self.backbone(self.inputs)
        fpn_feats = self.neck(body_feats)
        head_outs = self.head(fpn_feats, self.export_post_process,is_teacher=is_teacher)

        if self.training or (not self.export_post_process) or is_teacher :
            if is_teacher or get_data:
               return head_outs[0:3]
            else:
                loss = {}
                loss_gfl = self.head.get_loss(head_outs, self.inputs)
                loss.update(loss_gfl)
                total_loss = paddle.add_n(list(loss.values()))
                loss.update({'loss': total_loss})
                return loss
        else:
            if not self.export_post_process:
               return {'picodet': head_outs}
            elif self.export_nms:
                scale_factor = self.inputs['scale_factor']
                bboxes, bbox_num = self.head.post_process(
                    head_outs, scale_factor, export_nms=self.export_nms)
                return bboxes, bbox_num
            else:
                bboxes, mlvl_scores = self.head.post_process(
                    head_outs, scale_factor, export_nms=self.export_nms)
                output = {'bbox': bboxes, 'scores': mlvl_scores}
                return output

    def get_loss(self,):
        return self._forward()

    def get_pred(self):
        return self._forward()

    def get_loss_keys(self):
        return ['loss_cls', 'loss_iou', 'loss_dfl']
    def get_distill_loss(self, head_outs, teacher_head_outs, ratio=0.1):
        # student_probs: already sigmoid
        student_probs, student_dfl, student_deltas = head_outs
        teacher_probs, teacher_dfl, teacher_deltas = teacher_head_outs
        nc = student_probs.shape[-1]
        student_probs = student_probs.reshape([-1, nc])
        student_deltas = student_deltas.reshape([-1, 4])
        teacher_probs = teacher_probs.reshape([-1, nc])
        teacher_deltas = teacher_deltas.reshape([-1, 4])
        student_dfl = student_dfl.reshape([-1, 4,8])
        teacher_dfl = teacher_dfl.reshape([-1, 4,8])

        with paddle.no_grad():
            # Region Selection
            count_num = int(teacher_probs.shape[0] * ratio)
            #teacher_probs = F.sigmoid(teacher_probs) # already sigmoid
            max_vals = paddle.max(teacher_probs, 1)
            sorted_vals, sorted_inds = paddle.topk(max_vals,
                                                   teacher_probs.shape[0])
            mask = paddle.zeros_like(max_vals)
            mask[sorted_inds[:count_num]] = 1.
            fg_num = sorted_vals[:count_num].sum()
            b_mask = mask > 0.

        loss_logits = QFLv2(
            student_probs, teacher_probs, weight=mask, reduction="sum") / fg_num
        # [88872, 80] [88872, 80]

        inputs = paddle.concat(
            (-student_deltas[b_mask][..., :2], student_deltas[b_mask][..., 2:]),
            axis=-1)
        targets = paddle.concat(
            (-teacher_deltas[b_mask][..., :2], teacher_deltas[b_mask][..., 2:]),
            axis=-1)
        iou_loss = GIoULoss(reduction='mean')
        loss_deltas = iou_loss(inputs, targets)

        #loss_dfl = paddle.to_tensor([0])  # todo
        # student_dfl_pred = student_dfl[b_mask].reshape([-1, 17])
        # teacher_dfl_tar = teacher_dfl[b_mask].reshape([-1, 17])

        # loss_dfl = self.distribution_focal_loss(student_dfl_pred,
        #                                         teacher_dfl_tar)
        # todo: weight_targets

        return {
            "distill_loss_cls": loss_logits,
            "distill_loss_iou": loss_deltas,
            # "distill_loss_dfl": loss_dfl,
            "fg_sum": fg_num,
        }

    def _df_loss(self, pred_dist, target):  # [810, 4, 17]  [810, 4]
        target_left = paddle.cast(target, 'int64')
        target_right = target_left + 1
        weight_left = target_right.astype('float32') - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_dist, target_left, reduction='none') * weight_left
        loss_right = F.cross_entropy(
            pred_dist, target_right, reduction='none') * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

    def distribution_focal_loss(self,
                                pred_corners,
                                target_corners,
                                weight_targets=None):
        target_corners_label = F.softmax(target_corners, axis=-1)
        loss_dfl = F.cross_entropy(
            pred_corners,
            target_corners_label,
            soft_label=True,
            reduction='none')
        loss_dfl = loss_dfl.sum(1)
        if weight_targets is not None:
            loss_dfl = loss_dfl * (weight_targets.expand([-1, 4]).reshape([-1]))
            loss_dfl = loss_dfl.sum(-1) / weight_targets.sum()
        else:
            loss_dfl = loss_dfl.mean(-1)
        loss_dfl = loss_dfl / 4.  # 4 direction
        return loss_dfl

