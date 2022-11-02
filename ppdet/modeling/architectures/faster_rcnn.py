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
import paddle.nn.functional as F
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch
from ..ssod_utils import permute_to_N_HWA_K, QFLv2, giou_loss

__all__ = ['FasterRCNN']


@register
class FasterRCNN(BaseArch):
    """
    Faster R-CNN network, see https://arxiv.org/abs/1506.01497

    Args:
        backbone (object): backbone instance
        rpn_head (object): `RPNHead` instance
        bbox_head (object): `BBoxHead` instance
        bbox_post_process (object): `BBoxPostProcess` instance
        neck (object): 'FPN' instance
    """
    __category__ = 'architecture'
    __inject__ = ['bbox_post_process']

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_head,
                 bbox_post_process,
                 neck=None):
        super(FasterRCNN, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.rpn_head = rpn_head
        self.bbox_head = bbox_head
        self.bbox_post_process = bbox_post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])
        kwargs = {'input_shape': backbone.out_shape}
        neck = cfg['neck'] and create(cfg['neck'], **kwargs)

        out_shape = neck and neck.out_shape or backbone.out_shape
        kwargs = {'input_shape': out_shape}
        rpn_head = create(cfg['rpn_head'], **kwargs)
        bbox_head = create(cfg['bbox_head'], **kwargs)
        return {
            'backbone': backbone,
            'neck': neck,
            "rpn_head": rpn_head,
            "bbox_head": bbox_head,
        }

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        if self.neck is not None:
            body_feats = self.neck(body_feats)

        is_teacher = self.inputs.get('is_teacher', False)
        if is_teacher:
            rois, rois_num, _ = self.rpn_head(body_feats, self.inputs)
            head_outs, _ = self.bbox_head(body_feats, rois, rois_num, self.inputs)
            return head_outs

        if self.training:
            rois, rois_num, rpn_loss = self.rpn_head(body_feats, self.inputs)
            bbox_loss, _ = self.bbox_head(body_feats, rois, rois_num,
                                          self.inputs)
            loss = {}
            loss.update(rpn_loss)
            loss.update(bbox_loss)
            total_loss = paddle.add_n(list(loss.values()))
            loss.update({'loss': total_loss})
            return loss
        else:
            rois, rois_num, _ = self.rpn_head(body_feats, self.inputs)
            preds, _ = self.bbox_head(body_feats, rois, rois_num, None)

            im_shape = self.inputs['im_shape']
            scale_factor = self.inputs['scale_factor']
            bbox, bbox_num = self.bbox_post_process(preds, (rois, rois_num),
                                                    im_shape, scale_factor)

            # rescale the prediction back to origin image
            bboxes, bbox_pred, bbox_num = self.bbox_post_process.get_pred(
                bbox, bbox_num, im_shape, scale_factor)
            return {'bbox': bbox_pred, 'bbox_num': bbox_num}

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()

    def get_distill_loss(self, head_outs, teacher_head_outs, ratio=0.1):
        student_logits, student_deltas = head_outs
        teacher_logits, teacher_deltas = teacher_head_outs
        nc = student_logits[0].shape[1]

        student_logits = paddle.concat(
            [permute_to_N_HWA_K(x, nc)
             for x in student_logits], axis=1).reshape([-1, nc])
        teacher_logits = paddle.concat(
            [permute_to_N_HWA_K(x, nc)
             for x in teacher_logits], axis=1).reshape([-1, nc])

        student_deltas = paddle.concat(
            [permute_to_N_HWA_K(x, 4)
             for x in student_deltas], axis=1).reshape([-1, 4])
        teacher_deltas = paddle.concat(
            [permute_to_N_HWA_K(x, 4)
             for x in teacher_deltas], 1).reshape([-1, 4])

        with paddle.no_grad():
            # Region Selection
            count_num = int(teacher_logits.shape[0] * ratio)
            teacher_probs = F.sigmoid(teacher_logits) # [8184, 80]
            max_vals = paddle.max(teacher_probs, 1)
            sorted_vals, sorted_inds = paddle.topk(max_vals, teacher_logits.shape[0])
            mask = paddle.zeros_like(max_vals)
            mask[sorted_inds[:count_num]] = 1.
            fg_num = sorted_vals[:count_num].sum()
            b_mask = mask > 0.

        loss_logits = QFLv2(
            F.sigmoid(student_logits),
            teacher_probs,
            weight=mask,
            reduction="sum") / fg_num

        inputs = paddle.concat(
            (-student_deltas[b_mask][..., :2], student_deltas[b_mask][..., 2:]),
            axis=-1)  #wjm add
        targets = paddle.concat(
            (-teacher_deltas[b_mask][..., :2], teacher_deltas[b_mask][..., 2:]),
            axis=-1)  #wjm add
        loss_deltas = giou_loss(inputs, targets).mean()

        return {
            "distill_loss_cls": loss_logits,
            "distill_loss_box": loss_deltas,
            "fg_sum": fg_num,
        }
