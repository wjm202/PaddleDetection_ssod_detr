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
from typing_extensions import Self

from ppdet.core.workspace import register, create
from .meta_arch import BaseArch
from ..post_process import JDEBBoxPostProcess
import paddle
import paddle.nn.functional as F
from ..ssod_utils import QFLv2
from ..losses import GIoULoss
from IPython import embed

__all__ = ['YOLOv3']


@register
class YOLOv3(BaseArch):
    __category__ = 'architecture'
    __shared__ = ['data_format']
    __inject__ = ['post_process']

    def __init__(self,
                 backbone='DarkNet',
                 neck='YOLOv3FPN',
                 yolo_head='YOLOv3Head',
                 post_process='BBoxPostProcess',
                 data_format='NCHW',
                 for_mot=False):
        """
        YOLOv3 network, see https://arxiv.org/abs/1804.02767
        Args:
            backbone (nn.Layer): backbone instance
            neck (nn.Layer): neck instance
            yolo_head (nn.Layer): anchor_head instance
            bbox_post_process (object): `BBoxPostProcess` instance
            data_format (str): data format, NCHW or NHWC
            for_mot (bool): whether return other features for multi-object tracking
                models, default False in pure object detection models.
        """
        super(YOLOv3, self).__init__(data_format=data_format)
        self.backbone = backbone
        self.neck = neck
        self.yolo_head = yolo_head
        self.post_process = post_process
        self.for_mot = for_mot
        self.return_idx = isinstance(post_process, JDEBBoxPostProcess)
        self.is_teacher = False
        self.queue_ptr=0
        self.queue_size = 100*672
        self.queue_feats = paddle.zeros([self.queue_size, 80]).cuda()
        self.queue_probs = paddle.zeros([self.queue_size, 80]).cuda()
        self.it=0
    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])

        # fpn
        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        # head
        kwargs = {'input_shape': neck.out_shape}
        yolo_head = create(cfg['yolo_head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "yolo_head": yolo_head,
        }

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        if self.for_mot:
            neck_feats = self.neck(body_feats, self.for_mot)
        else:
            neck_feats = self.neck(body_feats)

        if isinstance(neck_feats, dict):
            assert self.for_mot == True
            emb_feats = neck_feats['emb_feats']
            neck_feats = neck_feats['yolo_feats']

        self.is_teacher = self.inputs.get('is_teacher', False)
        if self.training or self.is_teacher:
            yolo_losses = self.yolo_head(neck_feats, self.inputs)

            if self.for_mot:
                return {'det_losses': yolo_losses, 'emb_feats': emb_feats}
            else:
                return yolo_losses

        else:
            yolo_head_outs = self.yolo_head(neck_feats)

            if self.for_mot:
                boxes_idx, bbox, bbox_num, nms_keep_idx = self.post_process(
                    yolo_head_outs, self.yolo_head.mask_anchors)
                output = {
                    'bbox': bbox,
                    'bbox_num': bbox_num,
                    'boxes_idx': boxes_idx,
                    'nms_keep_idx': nms_keep_idx,
                    'emb_feats': emb_feats,
                }
            else:
                if self.return_idx:
                    _, bbox, bbox_num, _ = self.post_process(
                        yolo_head_outs, self.yolo_head.mask_anchors)
                elif self.post_process is not None:
                    bbox, bbox_num = self.post_process(
                        yolo_head_outs, self.yolo_head.mask_anchors,
                        self.inputs['im_shape'], self.inputs['scale_factor'])
                else:
                    bbox, bbox_num = self.yolo_head.post_process(
                        yolo_head_outs, self.inputs['scale_factor'])
                output = {'bbox': bbox, 'bbox_num': bbox_num}

            return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()

    def get_loss_keys(self):
        return ['loss_cls', 'loss_iou', 'loss_dfl','loss_contrast']

    def get_distill_loss(self, head_outs, teacher_head_outs, ratio=0.01):
        # student_probs: already sigmoid
        student_probs, student_deltas = head_outs
        teacher_probs, teacher_deltas= teacher_head_outs
        nc = student_probs.shape[-1]
        student_probs = student_probs.reshape([-1, nc])
        teacher_probs = teacher_probs.reshape([-1, nc])
        student_deltas = student_deltas.reshape([-1, 4])
        teacher_deltas = teacher_deltas.reshape([-1, 4])


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
        #comatch 
            probs=teacher_probs[b_mask].detach()
            temperature=0.2
            alpha=0.9
            if self.it>100: # memory-smoothing 
                
                    A = paddle.exp(paddle.mm(teacher_probs[b_mask], self.queue_probs.t())/temperature)       
                    A = A/A.sum(1,keepdim=True)                    
                    probs = alpha*probs + (1-alpha)*paddle.mm(A, self.queue_probs) 
                # queue_ptr= student_probs.shape[0]
            n = student_probs[b_mask].shape[0]   
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
        self.queue_feats[self.queue_ptr:self.queue_ptr + n,:] = teacher_probs[b_mask].detach()
        self.queue_probs[self.queue_ptr:self.queue_ptr + n,:] = teacher_probs[b_mask].detach()
        self.queue_ptr = (self.queue_ptr+n)%self.queue_size
        self.it+=1
        loss_contrast = - (paddle.log(sim_probs + 1e-7) * Q).sum(1)
        loss_contrast = loss_contrast.mean()   
        
        
        loss_logits = QFLv2(
            student_probs, teacher_probs, weight=mask, reduction="sum") / fg_num
        # [88872, 80] [88872, 80]
        
        iou_loss = GIoULoss(reduction='mean')
        loss_deltas = iou_loss(student_deltas[b_mask],teacher_deltas[b_mask])



        return {
            "distill_loss_cls": loss_logits,
            "distill_loss_iou": loss_deltas,
            "distill_loss_dfl": paddle.to_tensor(0).astype('float32'),
            "distill_loss_contrast": loss_contrast,
            "fg_sum": fg_num,
        }
