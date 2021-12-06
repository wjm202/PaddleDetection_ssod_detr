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
from IPython import embed
import math
import paddle
import paddle.nn as nn
from ppdet.core.workspace import register
from .. import initializer as init


@register
class RetinaNetHead(nn.Layer):
    """
    RetinaNetHead for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.

    Args:
        in_channel (int): the channel number of input.
        out_channel (int): the channel number of output.
        num_classes (int): the number of classes, 80 (COCO dataset) by default.
        num_convs (int): the number of convs in the head.
        num_anchors (int): number of generated anchors.
        norm (str or callable):
            Normalization for conv layers except for the two output layers.
            See :func:`detectron2.layers.get_norm` for supported types.
        loss_func (object): the class is used to compute loss.
        prior_prob (float): Prior weight for computing bias.


    """
    __shared__ = ['num_classes']
    __inject__ = ['loss_func']

    def __init__(self,
                 in_channel=256,
                 out_channel=256,
                 num_classes=80,
                 num_convs=4,
                 num_anchors=9,
                 norm="",
                 loss_func="RetinaNetLoss",
                 prior_prob=0.01):
        super(RetinaNetHead, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.get_loss = loss_func
        self.prior_prob = prior_prob

        cls_net = []
        reg_net = []
        for i in range(num_convs):
            in_ch = in_channel if i == 0 else out_channel

            cls_net.append(
                nn.Conv2D(
                    in_ch, out_channel, kernel_size=3, stride=1, padding=1))
            if norm == "bn":
                cls_net.append(nn.BatchNorm2D(out_channel))
            cls_net.append(nn.ReLU())

            reg_net.append(
                nn.Conv2D(
                    in_ch, out_channel, kernel_size=3, stride=1, padding=1))
            if norm == "bn":
                reg_net.append(nn.BatchNorm2D(out_channel))
            reg_net.append(nn.ReLU())

        self.cls_net = nn.Sequential(*cls_net)
        self.reg_net = nn.Sequential(*reg_net)

        self.cls_score = nn.Conv2D(
            out_channel,
            num_anchors * num_classes,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bbox_pred = nn.Conv2D(
            out_channel, num_anchors * 4, kernel_size=3, stride=1, padding=1)

        init.reset_initialized_parameter(self)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                init.normal_(m.weight, mean=0., std=0.01)
                init.constant_(m.bias, 0)

        init.constant_(self.cls_score.bias, bias_value)

    def forward(self, feats):
        pred_scores_list = []
        pred_boxes_list = []

        for feat in feats:
            pred_score = self.cls_score(self.cls_net(feat))
            pred_box = self.bbox_pred(self.reg_net(feat))

            n, c, h, w = pred_score.shape
            pred_score = paddle.transpose(
                pred_score, perm=[0, 2, 3, 1])  # n, h, w, c
            pred_score = paddle.reshape(
                pred_score,
                shape=[n, h * w * self.num_anchors, self.num_classes])

            pred_box = paddle.transpose(pred_box, perm=[0, 2, 3, 1])
            pred_box = paddle.reshape(
                pred_box, shape=[n, h * w * self.num_anchors, 4])

            pred_scores_list.append(pred_score)
            pred_boxes_list.append(pred_box)

        return pred_scores_list, pred_boxes_list

    def losses(self, anchors, preds, inputs):
        anchors = paddle.concat(anchors)

        return self.get_loss(anchors, preds, inputs)
