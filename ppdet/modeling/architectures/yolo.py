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
        return ['loss_cls', 'loss_iou', 'loss_dfl']

    def get_distill_loss(self, head_outs, teacher_head_outs, ratio=0.01,thr=None):
        # student_probs: already sigmoid
        student_probs, student_deltas, student_dfl = head_outs
        teacher_probs, teacher_deltas, teacher_dfl = teacher_head_outs
        nc = student_probs.shape[-1]
        student_probs = student_probs.reshape([-1, nc])
        teacher_probs = teacher_probs.reshape([-1, nc])
        student_deltas = student_deltas.reshape([-1, 4])
        teacher_deltas = teacher_deltas.reshape([-1, 4])
        student_dfl = student_dfl.reshape([-1, 4, 17])
        teacher_dfl = teacher_dfl.reshape([-1, 4, 17])
        with paddle.no_grad():#ratio=0.03
            # Region Selection
            # with paddle.no_grad():
            # # Region Selection
            
            # teacher_probs =student_probs
            # max_vals = paddle.max(teacher_probs, 1)
            # # print(max_vals.max())
            # max_class = paddle.argmax(teacher_probs, 1).astype('float32')
            # max_values=paddle.concat( [max_class.unsqueeze(1), max_vals.unsqueeze(1)],axis=-1)
            # mask=[]
            # for i in range(nc):
            #     mask.append(paddle.multiply(max_values[:,0]==i , max_values[:,1]>0.2).unsqueeze(1))
            # mask=paddle.concat([_ for _ in mask],axis=1)
            max_vals=paddle.max(teacher_probs, 1).unsqueeze(1)
            mask=max_vals>0.2#试试改成numpy！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            mask=mask.sum(1)>0
            b_mask=mask.astype('bool')
            mask=mask.astype('int32')
        if  mask.sum()==0:
            return {
            "distill_loss_cls": paddle.to_tensor(0.0),
            "distill_loss_iou": paddle.to_tensor(0.0),
            "distill_loss_dfl": paddle.to_tensor(0.0),
            "fg_sum": paddle.to_tensor(0.0),
        }
        else:
            fg_num = paddle.max(teacher_probs[b_mask], 1)
            fg_num =fg_num.sum()
    # distill_loss_cls 
            # loss_logits = QFLv2(
            #     F.sigmoid(student_logits),
            #     teacher_probs,
            #     weight=mask,
            #     fg_num=fg_num) 
            loss_logits = QFLv2(
                student_probs,
                teacher_probs,
                weight=mask,
                reduction = "sum"
                ) /fg_num
            if loss_logits>50:
                print('nan')

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
        loss_dfl = F.cross_entropy(
            F.softmax(student_dfl[b_mask].reshape([-1, 17])),
            F.softmax(teacher_dfl[b_mask].reshape([-1, 17])),
            soft_label=True,
            reduction='mean')
        # student_dfl_pred = student_dfl[b_mask].reshape([-1, 17])
        # teacher_dfl_tar = teacher_dfl[b_mask].reshape([-1, 17])
        # loss_dfl = self.distribution_focal_loss(student_dfl_pred,
        #                                         teacher_dfl_tar)
        # todo: weight_targets

        return {
            "distill_loss_cls": loss_logits,
            "distill_loss_iou": loss_deltas,
            "distill_loss_dfl": loss_dfl,
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
