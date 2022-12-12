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

from numpy import int32
from sqlalchemy import true
import numpy as np
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch
from ..post_process import JDEBBoxPostProcess
import paddle
import paddle.nn.functional as F
from ..ssod_utils import QFLv2
from ..losses import GIoULoss
from IPython import embed
import copy
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
        self.cls_thr = [0.9] * 80
        self.cls_thr_ig = [0.2] * 80
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

        is_teacher = self.inputs.get('is_teacher', False)
        if self.training or is_teacher:
            yolo_losses = self.yolo_head(neck_feats, self.inputs)
            if is_teacher:
                  bbox, bbox_num = self.yolo_head.post_process(
                        yolo_losses, paddle.ones_like(self.inputs['scale_factor']),is_teacher=True,)
                  return bbox, bbox_num  
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
                        yolo_head_outs, self.inputs['scale_factor'],cls_thr=self.cls_thr)
                output = {'bbox': bbox, 'bbox_num': bbox_num}

            return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()

    def get_loss_keys(self):
        return ['loss_cls', 'loss_iou', 'loss_dfl']

    def get_distill_loss(self, head_outs, teacher_head_outs):
        bbox,bbox_num=teacher_head_outs
        cls_thr=copy.deepcopy(self.cls_thr)
        bbox_num_list=[bbox_num[i].numpy() for i in range(len(bbox_num))]
        bbox_list=[bbox[i].numpy() for i in range(len(bbox))]
        gt_meta={}
        pad_gt_mask=[]
        gt_bbox=[]
        for i in range(len(bbox_list)):
                a=np.ones([40,1])
                b=np.zeros([40,6])
                b_mask= [False for _ in range(len(bbox_list[i]))]
                for j in range(len(bbox_list[i])):
                        ids=int(bbox_list[i][:,0][j])
                        if bbox_list[i][:,1][j] >cls_thr[ids]:
                            b_mask[j] = True
                if sum(b_mask)==0:
                    gt_bbox.append(b)
                else:
                    b[:len(bbox[i][b_mask])]=bbox[i][b_mask]
                    gt_bbox.append(b)
                a[int(sum(b_mask)):]=0
                pad_gt_mask.append(a)
        gt_bbox=np.stack( [_ for _ in gt_bbox],axis=0)       
        pad_gt_mask=np.stack([_ for _ in pad_gt_mask] ,axis=0)
        gt_meta['gt_class'] = np.expand_dims(gt_bbox[:,:,0],axis=-1)
        gt_meta['gt_bbox'] = gt_bbox[:,:,2:6]
        gt_meta['pad_gt_mask']= pad_gt_mask
        gt_meta['epoch_id'] =100
        # gt_meta['gt_class'] = paddle.to_tensor(np.expand_dims(gt_bbox[:,:,0],axis=-1),dtype="int32")
        # gt_meta['gt_bbox'] = paddle.to_tensor(gt_bbox[:,:,2:6],dtype="float32")
        # gt_meta['pad_gt_mask']=paddle.to_tensor(pad_gt_mask,dtype="int32")
        # gt_meta['epoch_id'] =100

        # loss=self.yolo_head.get_loss(head_outs,gt_meta)
        return gt_meta
