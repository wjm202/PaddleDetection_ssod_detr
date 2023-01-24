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

import numpy as np
import paddle
import paddle.nn.functional as F
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch
from ..ssod_utils import permute_to_N_HWA_K, QFLv2
from ..losses import GIoULoss
from ppdet.modeling.bbox_utils import bbox_overlaps
from ppdet.modeling.losses import IouLoss
from ..losses import GIoULoss

__all__ = ['FCOS']


@register
class FCOS(BaseArch):
    """
    FCOS network, see https://arxiv.org/abs/1904.01355

    Args:
        backbone (object): backbone instance
        neck (object): 'FPN' instance
        fcos_head (object): 'FCOSHead' instance
    """

    __category__ = 'architecture'

    def __init__(self, backbone, neck='FPN', fcos_head='FCOSHead'):
        super(FCOS, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.fcos_head = fcos_head
        self.is_teacher = False

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])

        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        kwargs = {'input_shape': neck.out_shape}
        fcos_head = create(cfg['fcos_head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "fcos_head": fcos_head,
        }

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        fpn_feats = self.neck(body_feats)

        self.is_teacher = self.inputs.get('is_teacher', False)
        if self.training or self.is_teacher:
            losses = self.fcos_head(fpn_feats, self.inputs)
            return losses
        else:
            fcos_head_outs = self.fcos_head(fpn_feats)
            bbox_pred, bbox_num = self.fcos_head.post_process(
                fcos_head_outs, self.inputs['scale_factor'])
            return {'bbox': bbox_pred, 'bbox_num': bbox_num}

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()

    def get_loss_keys(self):
        return ['loss_cls', 'loss_box', 'loss_quality','loss_asa_cls','loss_asa_iou']

    def get_distill_loss(self,
                         fcos_head_outs,
                         teacher_fcos_head_outs,
                         ratio=0.01):
        student_logits, student_deltas, student_quality = fcos_head_outs
        teacher_logits, teacher_deltas, teacher_quality = teacher_fcos_head_outs
        nc=student_logits.shape[2]
        student_logits=paddle.reshape(student_logits, [-1,nc])
        student_deltas=paddle.reshape(student_deltas, [-1,4])
        student_quality=paddle.reshape( student_quality, [-1,1])
        teacher_logits=paddle.reshape(teacher_logits, [-1,nc])
        teacher_deltas=paddle.reshape(teacher_deltas, [-1,4])
        teacher_quality=paddle.reshape(teacher_quality, [-1,1])
        with paddle.no_grad():
            # Region Selection
            count_num = int(teacher_logits.shape[0] * ratio)
            teacher_probs = teacher_logits
            max_vals = paddle.max(teacher_probs, 1)
            sorted_vals, sorted_inds = paddle.topk(max_vals,
                                                   teacher_logits.shape[0])
            mask = paddle.zeros_like(max_vals)
            mask[sorted_inds[:count_num]] = 1.
            fg_num = sorted_vals[:count_num].sum()
            b_mask = mask > 0

        # distill_loss_cls
        loss_logits = QFLv2(
            student_logits,
            teacher_probs,
            weight=mask,
            reduction="sum") / fg_num

        # distill_loss_box
        inputs = paddle.concat(
            (-student_deltas[b_mask][..., :2], student_deltas[b_mask][..., 2:]),
            axis=-1)
        targets = paddle.concat(
            (-teacher_deltas[b_mask][..., :2], teacher_deltas[b_mask][..., 2:]),
            axis=-1)
        iou_loss = GIoULoss(reduction='mean')
        loss_deltas = iou_loss(inputs, targets)

        # distill_loss_quality
        loss_quality = F.binary_cross_entropy(
            F.sigmoid(student_quality[b_mask]),
            F.sigmoid(teacher_quality[b_mask]),
            reduction='mean')

        return {
            "distill_loss_cls": loss_logits,
            "distill_loss_box": loss_deltas,
            "distill_loss_quality": loss_quality,
            "fg_sum": fg_num,
        }
        
    def semi_loss(self,pred_cls, pred_bboxes,label_list,bbox_target_list,pos_num_list):
                if type(bbox_target_list)==int:
                    return paddle.to_tensor(0.0),paddle.to_tensor(0.0)
                labels = paddle.to_tensor(np.stack(label_list, axis=0)).unsqueeze(0)
                bbox_targets = paddle.to_tensor(np.stack(bbox_target_list, axis=0)).unsqueeze(0)
                pred_bboxes=pred_bboxes.unsqueeze(0)
                pred_cls   =pred_cls.unsqueeze(0)
                # bbox_targets /= stride_tensor  # rescale bbox
                iou_loss=GIoULoss(reduction='mean')    
                # 1. obj score loss
                mask_positive = (labels != 80)
                num_pos = pos_num_list
                num_classes=80
                if num_pos > 0:
                    num_pos = paddle.to_tensor(num_pos, dtype='float32').clip(min=1)
                    # loss_obj /= num_pos

                    # 2. iou loss
                    bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
                    pred_bboxes_pos = paddle.masked_select(pred_bboxes,
                                                        bbox_mask).reshape([-1, 4])
                    assigned_bboxes_pos = paddle.masked_select(
                        bbox_targets, bbox_mask).reshape([-1, 4])
                    bbox_iou = bbox_overlaps(pred_bboxes_pos, assigned_bboxes_pos)
                    bbox_iou = paddle.diag(bbox_iou)

                    loss_iou = iou_loss(
                        pred_bboxes_pos,
                        assigned_bboxes_pos)
                    # loss_iou = loss_iou.sum() / num_pos

                    # 3. cls loss
                    cls_mask = mask_positive.unsqueeze(-1).tile(
                        [1, 1, num_classes])
                    pred_cls_pos = paddle.masked_select(
                        pred_cls, cls_mask).reshape([-1, num_classes])
                    assigned_cls_pos = paddle.masked_select(labels, mask_positive)
                    assigned_cls_pos = F.one_hot(assigned_cls_pos,
                                                num_classes + 1)[..., :-1]
                    assigned_cls_pos *= bbox_iou.unsqueeze(-1)
                    loss_cls = F.binary_cross_entropy(
                        pred_cls_pos, assigned_cls_pos, reduction='sum')
                    loss_cls /= num_pos

                    return loss_cls,loss_iou