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

__all__ = ["RetinaNet"]


@register
class RetinaNet(BaseArch):
    """
    RetinaNet, see https://arxiv.org/abs/1708.02002

    Args:
        backbone (object): backbone instance
        neck (object): 'FPN' instance
        head (object): 'RetinaNetHead' instance
        post_process (object): 'RetinaNetPostProcess' instance
    """
    __category__ = 'architecture'
    __inject__ = ["postprocess", "anchor_generator"]

    def __init__(self,
                 backbone,
                 neck,
                 anchor_generator="AnchorGenerator",
                 head="RetinaNetHead",
                 postprocess="RetinaNetPostProcess"):
        super(RetinaNet, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.anchor_generator = anchor_generator
        self.head = head
        self.postprocess = postprocess

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])

        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        anchor_generator = create(cfg["anchor_generator"])
        num_anchors = anchor_generator.num_anchors

        kwargs = {'input_shape': neck.out_shape, 'num_anchors': num_anchors}
        head = create(cfg['head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "anchor_generator": anchor_generator,
            "head": head,
        }

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        fpn_feats = self.neck(body_feats)
        anchors_list = self.anchor_generator(fpn_feats)
        pred_scores_list, pred_boxes_list = self.head(fpn_feats)

        if not self.training:
            im_shape = self.inputs['im_shape']
            scale_factor = self.inputs['scale_factor']

            bboxes, bbox_num = self.postprocess(pred_scores_list,
                                                pred_boxes_list, anchors_list,
                                                scale_factor, im_shape)
            return bboxes, bbox_num
        else:
            return anchors_list, [pred_scores_list, pred_boxes_list]

    def get_loss(self):
        anchors_list, preds = self._forward()

        loss_dict = self.head.losses(anchors_list, preds, self.inputs)
        total_loss = sum(loss_dict.values())

        loss_dict.update({"loss": total_loss})

        return loss_dict

    def get_pred(self):
        bboxes, bbox_num = self._forward()
        return {'bbox': bboxes, 'bbox_num': bbox_num}
