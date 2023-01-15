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
from ..ssod_utils import permute_to_N_HWA_K, QFLv2
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

        self.queue_ptr=0
        self.queue_size = 100*672
        # self.queue_feats = paddle.zeros([self.queue_size, 80]).cuda()
        # self.queue_probs = paddle.zeros([self.queue_size, 80]).cuda()
        self.queue_probs=[]
        self.queue_feats=[]
        self.it=0
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
        return ['loss_cls', 'loss_box', 'loss_quality']

    def get_distill_loss(self,
                         fcos_head_outs,
                         teacher_fcos_head_outs,
                         ratio=0.01,
                         preds_feat=None):
        student_logits, student_deltas, student_quality = fcos_head_outs
        teacher_logits, teacher_deltas, teacher_quality = teacher_fcos_head_outs
        nc = student_logits[0].shape[1]

        student_logits = paddle.concat(
            [
                _.transpose([0, 2, 3, 1]).reshape([-1, nc])
                for _ in student_logits
            ],
            axis=0)
        teacher_logits = paddle.concat(
            [
                _.transpose([0, 2, 3, 1]).reshape([-1, nc])
                for _ in teacher_logits
            ],
            axis=0)

        student_deltas = paddle.concat(
            [
                _.transpose([0, 2, 3, 1]).reshape([-1, 4])
                for _ in student_deltas
            ],
            axis=0)
        teacher_deltas = paddle.concat(
            [
                _.transpose([0, 2, 3, 1]).reshape([-1, 4])
                for _ in teacher_deltas
            ],
            axis=0)

        student_quality = paddle.concat(
            [
                _.transpose([0, 2, 3, 1]).reshape([-1, 1])
                for _ in student_quality
            ],
            axis=0)
        teacher_quality = paddle.concat(
            [
                _.transpose([0, 2, 3, 1]).reshape([-1, 1])
                for _ in teacher_quality
            ],
            axis=0)

        student_probs=F.sigmoid(student_logits)

        with paddle.no_grad():
            # Region Selection
            count_num = int(teacher_logits.shape[0] * ratio)
            teacher_probs = F.sigmoid(teacher_logits) # already sigmoid
            max_vals = paddle.max(teacher_probs, 1)
            sorted_vals, sorted_inds = paddle.topk(max_vals,
                                                   teacher_probs.shape[0])
            mask = paddle.zeros_like(max_vals)
            mask[sorted_inds[:count_num]] = 1.
            fg_num = sorted_vals[:count_num].sum()
            b_mask = mask > 0.
        #comatch 
            probs=teacher_probs[b_mask].detach()
            temperature=0.2
            alpha=0.9
            if self.it>2: # memory-smoothing 
                
                    A = paddle.exp(paddle.mm(preds_feat[b_mask], self.queue_feats_tensor.t())/temperature)       
                    A = A/A.sum(1,keepdim=True)                    
                    probs = alpha*probs + (1-alpha)*paddle.mm(A, self.queue_probs_tensor) 
                # queue_ptr= student_probs.shape[0]
            n = teacher_logits[b_mask].shape[0]   
        sim = paddle.exp(paddle.mm(student_probs[b_mask], teacher_probs[b_mask].t())/0.2) #feats_u_s0.shape 448,64 sim.shape 448,448
        sim_probs = sim / sim.sum(1, keepdim=True)
        Q = paddle.mm(probs, probs.t())
        # Q2 = paddle.mm(probs,  self.queue_probs.t())    
        # paddle.cat([Q,Q2],dim=1)
        Q.fill_diagonal_(1)    
        pos_mask = (Q>=0.5).astype("float")
            
        Q = Q * pos_mask
        Q = Q / Q.sum(1, keepdim=True)
        # paddle.zeros([672,672]).fill_diagonal_(1)
        # contrastive loss
        #如果是多尺度的话建议直接建立一个list每间隔100个迭代pop(0),并且self.queue_=paddle.concat list
        self.queue_feats.append(preds_feat[b_mask].detach())
        self.queue_probs.append(teacher_probs[b_mask].detach())
        if len( self.queue_feats)>2:
            self.queue_probs.pop(0)
            self.queue_feats.pop(0)
        self.queue_probs_tensor=paddle.concat([_ for _ in self.queue_probs])
        self.queue_feats_tensor=paddle.concat([_ for _ in self.queue_feats])
        self.it+=1
        loss_contrast = - (paddle.log(sim_probs + 1e-7) * Q).sum(1)
        loss_contrast = loss_contrast.mean()  

        # distill_loss_cls
        loss_logits = QFLv2(
            student_probs,
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
            "distill_loss_contrast": loss_contrast,
            "fg_sum": fg_num,
        }
