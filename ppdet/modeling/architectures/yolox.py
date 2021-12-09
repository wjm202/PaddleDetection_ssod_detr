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

import paddle
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch

__all__ = ['YOLOX']


@register
class YOLOX(BaseArch):
    __category__ = 'architecture'
    #__shared__ = ['data_format']
    __inject__ = ['post_process']

    def __init__(self,
                 backbone='CSPDarkNet',
                 neck='YOLOPAFPN',
                 yolox_head='YOLOXHead',
                 post_process='YOLOXPostProcess'):
        """
        YOLOX network, see https://arxiv.org/abs/...

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
        body_feats = self.backbone(self.inputs)
        neck_feats = self.neck(body_feats)

        if self.training:
            gt_bbox = self.inputs['gt_bbox']     # [N, 120, 4]
            gt_class = self.inputs['gt_class']   # [N, 120]
            gt_class = gt_class.unsqueeze(2)     # [N, 120, 1]
            gt_class = gt_class.astype(gt_bbox.dtype)
            gt_class_bbox = paddle.concat([gt_class, gt_bbox], 2)
            epoch_id = self.inputs['epoch_id']
            yolo_losses = self.yolox_head.get_loss(neck_feats, gt_class_bbox, epoch_id)
            loss = {}
            loss.update(yolo_losses)
            total_loss = paddle.add_n(list(loss.values()))
            loss.update({'loss': total_loss})
            return loss
        else:
            scale_factor = self.inputs['scale_factor']
            yolox_head_outs = self.yolox_head.get_prediction(neck_feats, scale_factor)
            bbox_pred, bbox_num = self.post_process(yolox_head_outs, scale_factor)
            output = {'bbox': bbox_pred, 'bbox_num': bbox_num}
            # print(bbox_pred)
            return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()
