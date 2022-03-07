# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved. 
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
import paddle.nn as nn

from ppdet.core.workspace import register, create
from .meta_arch import BaseArch
from IPython import embed

__all__ = ['YOLOX']


def yolox_resize(inputs, targets, inputs_dim, tsize):
    # targets.shape [n, 120, 4]
    scale_y = tsize[0] / inputs_dim[0]
    scale_x = tsize[1] / inputs_dim[1]
    if scale_x != 1 or scale_y != 1:
        inputs = F.interpolate(
            inputs, size=tsize, mode="bilinear", align_corners=False
        )
        targets[:, :, 0::2] = targets[:, :, 0::2] * scale_x
        targets[:, :, 1::2] = targets[:, :, 1::2] * scale_y
    return inputs, targets


@register
class YOLOX(BaseArch):
    __category__ = 'architecture'
    __shared__ = ['data_format']
    __inject__ = ['post_process']

    def __init__(self,
                 backbone='CSPDarkNet',
                 neck='YOLOPAFPN',
                 yolox_head='YOLOXHead',
                 post_process='YOLOXPostProcess',
                 data_format='NCHW',
                 for_mot=False):
        """
        YOLOX network, see https://arxiv.org/abs/2107.08430

        Args:
            backbone (nn.Layer): backbone instance
            neck (nn.Layer): neck instance
            yolox_head (nn.Layer): anchor_head instance
            post_process (object): `YOLOXPostProcess` instance
        """
        super(YOLOX, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.yolox_head = yolox_head
        self.post_process = post_process
        self.for_mot = for_mot
        self.target_dim = [416, 416] # initial scale, will be reset by self.inputs['target_size'].shape

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])

        # fpn
        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        # head
        kwargs = {'input_shape': neck.out_shape}
        yolox_head = create(cfg['yolox_head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "yolox_head": yolox_head,
        }

    def _forward(self):
        # YOLOX random resizing every 10 iters
        if self.training:
            interval = 10
            inputs_dim = self.inputs['image'].shape[2:]
            if self.inputs['step_id'] % interval == 0:
                self.target_dim = list(map(int, self.inputs['target_size'][0].numpy()))
            #print('inputs_dim   target_dim', tuple(inputs_dim), tuple(self.target_dim))
            self.inputs['image'], self.inputs['gt_bbox'] = yolox_resize(
                self.inputs['image'], self.inputs['gt_bbox'],
                inputs_dim, self.target_dim)
            target_size_tensor = paddle.to_tensor(self.target_dim[0])
            target_size_tensor.stop_gradient = True

        '''
        print('self.inputs ', self.inputs['image'].shape, self.inputs['image'].sum())
        print('self.inputs  sum 0 ',  self.inputs['image'][:,0,:,:].sum())
        print('self.inputs  sum 1 ',  self.inputs['image'][:,1,:,:].sum())
        print('self.inputs  sum 2 ',  self.inputs['image'][:,2,:,:].sum())
        '''
        '''
        body_feats = self.backbone(self.inputs)
        print('body_feats ', [x.shape for x in body_feats])
        print('body_feats sum ', [x.sum() for x in body_feats])
        neck_feats = self.neck(body_feats, self.for_mot)
        print('neck_feats ', [x.shape for x in neck_feats])
        print('neck_feats sum ', [x.sum() for x in neck_feats])
        '''
        body_feats = self.backbone(self.inputs)
        neck_feats = self.neck(body_feats, self.for_mot)

        if self.training:
            yolo_losses = self.yolox_head(neck_feats, self.inputs)
            yolo_losses.update({'size': target_size_tensor})
            return yolo_losses
        else:
            yolo_head_outs = self.yolox_head(neck_feats)
            bbox, bbox_num = self.post_process(
                yolo_head_outs, 
                self.inputs['im_shape'], self.inputs['scale_factor'])
            output = {'bbox': bbox, 'bbox_num': bbox_num}
            return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()
