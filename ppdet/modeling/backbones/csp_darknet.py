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

import paddle
import paddle.nn as nn
from paddle.regularizer import L2Decay
from paddle import ParamAttr
from IPython import embed
from ppdet.core.workspace import register, serializable
from ..shape_spec import ShapeSpec

__all__ = ['CSPDarkNet', 'BaseConv', 'DWConv']


def get_activation(name="silu"):
    if name == "silu":
        module = nn.Silu()
    elif name == "relu":
        module = nn.ReLU()
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Layer):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super(BaseConv, self).__init__()
        # same padding
        pad = (ksize - 1) // 2

        conv_battr = False
        bias_init = None
        if bias:
            blr = 1.0
            conv_battr = ParamAttr(learning_rate=blr,
                                   initializer=bias_init,
                                   regularizer=L2Decay(0.))
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias_attr=conv_battr,
        )

        norm_lr = 1.0
        norm_decay = 0.0
        pattr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay))
        battr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay))

        self.bn = nn.BatchNorm2D(out_channels, weight_attr=pattr, bias_attr=battr, momentum=0.97, epsilon=1e-3)
        self.act = get_activation(act)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

    def freeze(self):
        if self.conv.weight is not None:
            self.conv.weight.stop_gradient = True
        if self.conv.bias is not None:
            self.conv.bias.stop_gradient = True
        self.bn.weight.stop_gradient = True
        self.bn.bias.stop_gradient = True

    def fix_bn(self):
        self.bn.eval()


class DWConv(nn.Layer):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super(DWConv, self).__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)

    def freeze(self):
        self.dconv.freeze()
        self.pconv.freeze()

    def fix_bn(self):
        self.dconv.fix_bn()
        self.pconv.fix_bn()


class Bottleneck(nn.Layer):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super(Bottleneck, self).__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        temp1 = self.conv1(x)
        y = self.conv2(temp1)
        if self.use_add:
            y = y + x
        return y

    def freeze(self):
        self.conv1.freeze()
        self.conv2.freeze()

    def fix_bn(self):
        self.conv1.fix_bn()
        self.conv2.fix_bn()


class SPPBottleneck(nn.Layer):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super(SPPBottleneck, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.LayerList(
            [
                nn.MaxPool2D(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = paddle.concat([x] + [m(x) for m in self.m], axis=1)
        x = self.conv2(x)
        return x

    def freeze(self):
        self.conv1.freeze()
        self.conv2.freeze()

    def fix_bn(self):
        self.conv1.fix_bn()
        self.conv2.fix_bn()


class CSPLayer(paddle.nn.Layer):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = paddle.concat([x_1, x_2], axis=1)
        return self.conv3(x)

    def freeze(self):
        self.conv1.freeze()
        self.conv2.freeze()
        self.conv3.freeze()
        for layer in self.m:
            layer.freeze()

    def fix_bn(self):
        self.conv1.fix_bn()
        self.conv2.fix_bn()
        self.conv3.fix_bn()
        for layer in self.m:
            layer.fix_bn()


class Focus(paddle.nn.Layer):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super(Focus, self).__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[:, :, 0::2, 0::2]
        patch_top_right = x[:, :, 0::2, 1::2]
        patch_bot_left = x[:, :, 1::2, 0::2]
        patch_bot_right = x[:, :, 1::2, 1::2]

        y = paddle.concat(
            [
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ],
            axis=1,
        )
        #x.stop_gradient = True
        return self.conv(y)

    def freeze(self):
        self.conv.freeze()

    def fix_bn(self):
        self.conv.fix_bn()


@register
@serializable
class CSPDarkNet(nn.Layer):
    __shared__ = ['dep_mul', 'wid_mul', 'act']

    def __init__(self,
                 dep_mul=1.0,
                 wid_mul=1.0,
                 return_idx=[2, 3, 4],
                 depthwise=False,
                 act='silu',
                 freeze_at=-1,
                 fix_bn_mean_var_at=-1):
        """
        CSPDarknet, see https://github.com/Megvii-BaseDetection/YOLOX

        Args:
            dep_mul (float):
            wid_mul (float):
            return_idx (list): index of stages whose feature maps are returned
            depthwise (bool): use depth-wise conv layer
            act (str): activation function type, default 'silu'
            freeze_at (int): freeze the backbone at which stage
            fix_bn_mean_var_at (int): fix the batch norm layer at which stage
        """
        super(CSPDarkNet, self).__init__()
        self.return_idx = return_idx
        Conv = DWConv if depthwise else BaseConv

        self.freeze_at = freeze_at
        self.fix_bn_mean_var_at = fix_bn_mean_var_at

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3
        self._out_channels = []

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)
        if 0 in self.return_idx:
            self._out_channels.append(base_channels)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )
        if 1 in self.return_idx:
            self._out_channels.append(base_channels * 2)

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )
        if 2 in self.return_idx:
            self._out_channels.append(base_channels * 4)

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )
        if 3 in self.return_idx:
            self._out_channels.append(base_channels * 8)

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )
        if 4 in self.return_idx:
            self._out_channels.append(base_channels * 16)
        self.freeze()
        self.fix_bn()

    def forward(self, inputs):
        x = inputs['image']
        blocks = []
        x = self.stem(x)
        if 0 in self.return_idx:
            blocks.append(x)
        x = self.dark2(x)
        if 1 in self.return_idx:
            blocks.append(x)
        x = self.dark3(x)
        if 2 in self.return_idx:
            blocks.append(x)
        x = self.dark4(x)
        if 3 in self.return_idx:
            blocks.append(x)
        x = self.dark5(x)
        if 4 in self.return_idx:
            blocks.append(x)
        return blocks

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]

    def freeze(self):
        freeze_at = self.freeze_at
        if freeze_at >= 0:
            self.stem.freeze()
        if freeze_at >= 1:
            for layer in self.dark2:
                layer.freeze()
        if freeze_at >= 2:
            for layer in self.dark3:
                layer.freeze()
        if freeze_at >= 3:
            for layer in self.dark4:
                layer.freeze()
        if freeze_at >= 4:
            for layer in self.dark5:
                layer.freeze()

    def fix_bn(self):
        fix_bn_mean_var_at = self.fix_bn_mean_var_at
        if fix_bn_mean_var_at >= 0:
            self.stem.fix_bn()
        if fix_bn_mean_var_at >= 1:
            for layer in self.dark2:
                layer.fix_bn()
        if fix_bn_mean_var_at >= 2:
            for layer in self.dark3:
                layer.fix_bn()
        if fix_bn_mean_var_at >= 3:
            for layer in self.dark4:
                layer.fix_bn()
        if fix_bn_mean_var_at >= 4:
            for layer in self.dark5:
                layer.fix_bn()
