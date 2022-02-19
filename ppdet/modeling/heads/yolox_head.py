#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from IPython import embed
import math
import copy
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from paddle.nn.initializer import Constant
from ppdet.core.workspace import register
from ..backbones.csp_darknet import BaseConv, DWConv

__all__ = ['YOLOXHead']


@register
class YOLOXHead(nn.Layer):
    __shared__ = ['num_classes', 'width_factor', 'act']
    __inject__ = ['yolox_loss']

    def __init__(self,
                 num_classes=80,
                 width_factor=1.0,
                 strides=[8, 16, 32],
                 in_channels=[128, 256, 512],
                 channels=[256, 512, 1024],
                 mosaic_epoch=285,
                 act="silu",
                 yolox_loss='YOLOXLoss',
                 depthwise=False,
                 prior_prob=0.01):
        """
        Head for YOLOX network

        Args:
            num_classes (int):
        """
        super(YOLOXHead, self).__init__()
        assert len(channels) > 0, "channels length should > 0"
        self.n_anchors = 1
        self.num_classes = num_classes
        self.mosaic_epoch = mosaic_epoch
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.LayerList()
        self.reg_convs = nn.LayerList()
        self.cls_preds = nn.LayerList()
        self.reg_preds = nn.LayerList()
        self.obj_preds = nn.LayerList()
        self.stems = nn.LayerList()
        Conv = DWConv if depthwise else BaseConv

        self.prior_prob = prior_prob
        bias_init_value = -math.log((1 - self.prior_prob) / self.prior_prob)

        for i in range(len(channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(channels[i] * width_factor),
                    out_channels=int(256 * width_factor),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width_factor),
                            out_channels=int(256 * width_factor),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width_factor),
                            out_channels=int(256 * width_factor),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width_factor),
                            out_channels=int(256 * width_factor),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width_factor),
                            out_channels=int(256 * width_factor),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )

            battr = ParamAttr(
                initializer=Constant(bias_init_value),
                regularizer=L2Decay(0.))
            self.cls_preds.append(
                nn.Conv2D(
                    in_channels=int(256 * width_factor),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2D(
                    in_channels=int(256 * width_factor),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2D(
                    in_channels=int(256 * width_factor),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.strides = strides
        self.yolox_loss = yolox_loss
        self.grids = [paddle.zeros((1, ))] * len(channels)
        self.initialize_biases(1e-2)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.reshape([self.n_anchors, -1])
            #b = paddle.reshape(conv.bias, [self.n_anchors, -1])
            conv.bias = paddle.create_parameter(shape=b.reshape([-1]).shape, dtype='float32',
                        default_initializer=paddle.nn.initializer.Constant(-math.log((1 - prior_prob) / prior_prob)))

        for conv in self.obj_preds:
            b = conv.bias.reshape([self.n_anchors, -1])
            #b = paddle.reshape(conv.bias, [self.n_anchors, -1])
            conv.bias = paddle.create_parameter(shape=b.reshape([-1]).shape, dtype='float32',
                        default_initializer=paddle.nn.initializer.Constant(-math.log((1 - prior_prob) / prior_prob)))

    def forward(self, feats, targets=None):
        if self.training:
            gt_bbox = targets['gt_bbox']     # [N, 120, 4]
            gt_class = targets['gt_class'].unsqueeze(-1).astype(gt_bbox.dtype)   # [N, 120, 1]
            gt_class_bbox = paddle.concat([gt_class, gt_bbox], axis=2)
            gt_class_bbox.stop_gradient = False

            self.use_l1 = True if targets['epoch_id'] >= self.mosaic_epoch else False

            outputs, x_shifts, y_shifts, expanded_strides, origin_preds = self.get_outputs(feats)

            outputs = paddle.concat(outputs, axis=1)
            yolox_losses = self.yolox_loss(
                outputs,
                x_shifts,
                y_shifts,
                expanded_strides,
                origin_preds,
                gt_class_bbox,
                dtype=feats[0].dtype,
                use_l1=self.use_l1,
            )
            return yolox_losses
        else:
            outputs, x_shifts, y_shifts, expanded_strides, origin_preds = self.get_outputs(feats)

            self.hw = [x.shape[-2:] for x in outputs]
            outputs = paddle.concat(
                [paddle.reshape(x, (x.shape[0], x.shape[1], -1)) for x in outputs], axis=2
            )
            outputs = paddle.transpose(outputs, [0, 2, 1]) # [bs, 85, A] -> [bs, A, 85]

            if self.decode_in_inference:
                outputs = self.decode_outputs(outputs)
            return outputs

    def get_outputs(self, feats):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        #print('  conv cls_preds [0,:,:,:]  ///////////.............', self.cls_preds[0].weight[0,:,:,:].sum())
        #print('  conv cls_preds   ///////////.............', self.cls_preds[0].weight.sum())
        #print('  conv reg_preds [0,:,:,:]   ///////////.............', self.reg_preds[0].weight[0,:,:,:].sum())
        #print('  conv reg_preds     ///////////.............', self.reg_preds[0].weight.sum())

        for i, (cls_conv, reg_conv, stride, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, feats)
        ):
            #print('..... fpn feat //', i, x.sum())
            x = self.stems[i](x)
            #print('..... stems feat //', i, x.sum())

            cls_x = x
            reg_x = x
            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[i](cls_feat)
            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[i](reg_feat)
            obj_output = self.obj_preds[i](reg_feat)

            #print('..... reg cls feat //', i, reg_feat.sum(), cls_feat.sum())
            #print('// reg obj cls_output //', i, reg_output.sum(), obj_output.sum(), cls_output.sum())
            if self.training:
                output = paddle.concat([reg_output, obj_output, cls_output], 1)
                # output.shape=[N, 1 * 80 * 80, 85]  # xywh
                # grid.shape=  [1, 1 * 80 * 80, 2]
                output, grid = self.get_output_and_grid(
                    output, i, stride
                )
                x_shifts.append(grid[:, :, 0])   # [1, 1 * 80 * 80]
                y_shifts.append(grid[:, :, 1])   # [1, 1 * 80 * 80]

                expanded_stride = paddle.ones((1, grid.shape[1]), dtype=feats[0].dtype) * stride
                expanded_strides.append(expanded_stride)

                if self.use_l1:  # not use mosaic
                    bs, _, hsize, wsize = reg_output.shape
                    
                    reg_output = reg_output.reshape((bs, self.n_anchors, 4, hsize, wsize)).transpose((0, 1, 3, 4, 2))

                    reg_output = reg_output.reshape((bs, -1, 4))
                    origin_preds.append(reg_output.clone())
            else:
                objs = F.sigmoid(obj_output)
                clses = F.sigmoid(cls_output)
                output = paddle.concat([reg_output, objs, clses], axis=1)
            
            outputs.append(output)

        return outputs, x_shifts, y_shifts, expanded_strides, origin_preds

    def get_output_and_grid(self, output, k, stride):
        grid = self.grids[k]
        bs = output.shape[0]
        n_ch = self.num_classes + 5
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = paddle.meshgrid([paddle.arange(hsize), paddle.arange(wsize)])
            grid = paddle.stack((xv, yv), 2)
            grid = paddle.reshape(grid, (1, 1, hsize, wsize, 2))
            grid = paddle.cast(grid, dtype=output.dtype)
            self.grids[k] = grid

        output = paddle.reshape(output, (bs, self.n_anchors, n_ch, hsize, wsize))
        output = paddle.transpose(output, [0, 1, 3, 4, 2])
        output = paddle.reshape(output, (bs, self.n_anchors * hsize * wsize, -1))   # [N, 1 * 80 * 80, 85]
        grid = paddle.reshape(grid, (1, -1, 2))   # [1, 1 * 80 * 80, 2]

        xy = (output[:, :, :2] + grid) * stride       # [N, 1 * 80 * 80, 2]
        wh = paddle.exp(output[:, :, 2:4]) * stride   # [N, 1 * 80 * 80, 2]
        output = paddle.concat([xy, wh, output[:, :, 4:]], 2)   # [N, 1 * 80 * 80, 85]
        return output, grid

    def decode_outputs(self, outputs): # [1, 8400, 85]
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides): # [[80, 80], [40, 40], [20, 20]]  [8, 16, 32]
            yv, xv = paddle.meshgrid([paddle.arange(hsize), paddle.arange(wsize)]) # [80, 80], [80, 80]
            grid = paddle.reshape(paddle.stack((xv, yv), axis=2), (1, hsize * wsize, 2)) # [80, 80, 2] -> [1, 6400, 2]
            grids.append(grid)
            strides.append(paddle.full((1, hsize * wsize, 1), stride))
        grids = paddle.concat(grids, axis=1)
        strides = paddle.concat(strides, axis=1)

        outputs[:, :, 0:2] = (outputs[:, :, 0:2] + grids) * strides
        outputs[:, :, 2:4] = paddle.exp(outputs[:, :, 2:4]) * strides
        return outputs

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }
