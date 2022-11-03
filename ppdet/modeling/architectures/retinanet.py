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

from ppdet.core.workspace import register, create
from .meta_arch import BaseArch
import paddle
import paddle.nn.functional as F
from ..ssod_utils import permute_to_N_HWA_K, QFLv2, giou_loss
from IPython import embed

__all__ = ['RetinaNet']


@register
class RetinaNet(BaseArch):
    __category__ = 'architecture'

    def __init__(self, backbone, neck, head):
        super(RetinaNet, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

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
            'head': head,
        }

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        neck_feats = self.neck(body_feats)

        is_teacher = self.inputs.get('is_teacher', False)
        if self.training or is_teacher:
            return self.head(neck_feats, self.inputs)
        else:
            head_outs = self.head(neck_feats)
            bbox, bbox_num = self.head.post_process(
                head_outs, self.inputs['im_shape'], self.inputs['scale_factor'])
            return {'bbox': bbox, 'bbox_num': bbox_num}

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()

    def decode_head_outs(self, head_outs):
        cls_logits_list, bboxes_reg_list = head_outs
        anchors = self.head.anchor_generator(cls_logits_list)
        cls_logits = [_.transpose([0, 2, 3, 1]) for _ in cls_logits_list]
        bboxes_reg = [_.transpose([0, 2, 3, 1]) for _ in bboxes_reg_list]
        bboxes, scores = self.head.decode(
            anchors, cls_logits, bboxes_reg, self.inputs['im_shape'], self.inputs['scale_factor'])
        return scores, bboxes

    def get_distill_loss(self, head_outs, teacher_head_outs, ratio=0.1):
        student_logits, student_deltas = self.decode_head_outs(head_outs) # [2, 80, 4729] [2, 4729, 4]
        teacher_logits, teacher_deltas = self.decode_head_outs(teacher_head_outs)

        nc = student_logits.shape[1]
        student_logits = student_logits.transpose([0, 2, 1]).reshape([-1, nc])
        student_deltas = student_deltas.reshape([-1, 4])
        teacher_logits = teacher_logits.transpose([0, 2, 1]).reshape([-1, nc])
        teacher_deltas = teacher_deltas.reshape([-1, 4])

        with paddle.no_grad():
            # Region Selection
            count_num = int(teacher_logits.shape[0] * ratio)
            teacher_probs = teacher_logits #F.sigmoid(teacher_logits) # [4729*2, 80]
            max_vals = paddle.max(teacher_probs, 1)
            sorted_vals, sorted_inds = paddle.topk(max_vals, teacher_logits.shape[0])
            mask = paddle.zeros_like(max_vals)
            mask[sorted_inds[:count_num]] = 1.
            fg_num = sorted_vals[:count_num].sum()
            b_mask = mask > 0.

        loss_logits = QFLv2(
            #F.sigmoid(student_logits),
            student_logits,
            teacher_probs,
            weight=mask,
            reduction="sum") / fg_num

        inputs = paddle.concat(
            (-student_deltas[b_mask][..., :2], student_deltas[b_mask][..., 2:]),
            axis=-1)
        targets = paddle.concat(
            (-teacher_deltas[b_mask][..., :2], teacher_deltas[b_mask][..., 2:]),
            axis=-1)
        loss_deltas = giou_loss(inputs, targets).mean()

        return {
            "distill_loss_cls": loss_logits,
            "distill_loss_reg": loss_deltas,
            "fg_sum": fg_num,
        }
