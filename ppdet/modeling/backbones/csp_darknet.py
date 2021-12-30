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
from IPython import embed
import math
import paddle
import paddle.nn as nn
from paddle.regularizer import L2Decay
from paddle import ParamAttr

from ppdet.core.workspace import register, serializable
from ..shape_spec import ShapeSpec

__all__ = ['CSPDarkNet', 'BaseConv', 'DWConv', 'Bottleneck', 'SPPLayer', 'SPPFLayer']


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


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = (
        nn.Conv2D(
            conv._in_channels,
            conv._out_channels,
            kernel_size=conv._kernel_size,
            stride=conv._stride,
            padding=conv._padding,
            groups=conv._groups,
            bias_attr=True,
        )
    )
    fusedconv.stop_gradient = True

    # prepare filters
    w_conv = paddle.reshape(conv.weight, [conv._out_channels, -1])
    w_bn = paddle.diag(paddle.divide(bn.weight, paddle.sqrt(bn._epsilon + bn._variance)))
    fusedconv.weight.set_value(paddle.reshape(paddle.mm(w_bn, w_conv), fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = (
        paddle.zeros([conv.weight.shape[0]])
        if conv.bias is None
        else conv.bias
    )
    b_bn = bn.bias - paddle.divide(paddle.multiply(bn.weight, bn._mean), paddle.sqrt(bn._epsilon + bn._variance))

    fusedconv.bias.set_value(paddle.mm(w_bn, b_conv.unsqueeze(-1)).squeeze(-1) + b_bn)
    return fusedconv


class BaseConv(nn.Layer):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super(BaseConv, self).__init__()
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
        # bn not eps=1e-5, momentum=0.1
        self.act = get_activation(act)

    def forward(self, x):
        if self.training:
            return self.act(self.bn(self.conv(x)))
        else:
            #self.conv = fuse_conv_and_bn(self.conv, self.bn)
            #return self.act(self.conv(x))
            return self.act(self.bn(self.conv(x)))

    def fuse_forward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Layer):
    """Depthwise Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, bias=False, act="silu"):
        super(DWConv, self).__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            bias=bias,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, bias=bias, act=act
        )

    def forward(self, x):
        return self.pconv(self.dconv(x))


class Focus(nn.Layer):
    """Focus width and height information into channel space, used in YOLOX."""

    def __init__(self, in_channels, out_channels, ksize=3, stride=1, bias=False, act="silu"):
        super(Focus, self).__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize=ksize, stride=stride, bias=bias, act=act)

    def forward(self, x):
        # x: [bs, c, w, h] -> y: [b, 4c, w/2, h/2]
        patch_top_left = x[:, :, 0::2, 0::2]
        patch_top_right = x[:, :, 0::2, 1::2]
        patch_bot_left = x[:, :, 1::2, 0::2]
        patch_bot_right = x[:, :, 1::2, 1::2]

        y = paddle.concat([
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ],
            axis=1,
        )
        out = self.conv(y)
        return out


class Bottleneck(nn.Layer):
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5,
                 depthwise=False, bias=False, act="silu"):
        super(Bottleneck, self).__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = Conv(in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, ksize=3, stride=1, bias=bias, act=act)
        self.add_shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.add_shortcut:
            y = y + x
        return y


class SPPLayer(nn.Layer):
    """Spatial Pyramid Pooling (SPP) layer used in YOLOv3-SPP and YOLOX"""

    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), bias=False, act="silu"):
        super(SPPLayer, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.maxpoolings = nn.LayerList([
            nn.MaxPool2D(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, ksize=1, stride=1, bias=bias, act=act)

    def forward(self, x):
        x = self.conv1(x)
        concats = paddle.concat([x] + [mp(x) for mp in self.maxpoolings], axis=1)
        out = self.conv2(concats)
        return out


class SPPFLayer(nn.Layer):
    """ Spatial Pyramid Pooling - Fast (SPPF) layer used in YOLOv5 by Glenn Jocher,
        equivalent to SPP(k=(5, 9, 13))
    """
    def __init__(self, in_channels, out_channels, ksize=5, bias=False, act='silu'):
        super(SPPFLayer, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.maxpooling = nn.MaxPool2D(kernel_size=ksize, stride=1, padding=ksize // 2)
        conv2_channels = hidden_channels * 4
        self.conv2 = BaseConv(conv2_channels, out_channels, ksize=1, stride=1, bias=bias, act=act)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.maxpooling(x)
        y2 = self.maxpooling(y1)
        y3 = self.maxpooling(y2)
        concats = paddle.concat([x, y1, y2, y3], axis=1)
        out = self.conv2(concats)
        return out


class CSPLayer(nn.Layer):
    """CSP (Cross Stage Partial) layer with 3 convs, named C3 in YOLOv5"""

    def __init__(self, in_channels, out_channels, num_blocks=1, shortcut=True,
                 expansion=0.5, depthwise=False, bias=False, act="silu"):
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        module_list = [
            Bottleneck(
                hidden_channels,
                hidden_channels,
                shortcut=shortcut,
                expansion=1.0,
                depthwise=depthwise,
                bias=bias,
                act=act)
            for _ in range(num_blocks)
        ]
        self.bottlenecks = nn.Sequential(*module_list)
        self.conv3 = BaseConv(hidden_channels * 2, out_channels, ksize=1, stride=1, bias=bias, act=act)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.bottlenecks(x_1)
        x = paddle.concat([x_1, x_2], axis=1)
        return self.conv3(x)


@register
@serializable
class CSPDarkNet(nn.Layer):
    __shared__ = ['depth_factor', 'width_factor', 'act']

    # in_channels, out_channels, num_blocks, add_shortcut, use_spp(use_sppf)
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 9, True, False], [512, 1024, 3, True, True]],
        'P6': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 9, True, False], [512, 768, 3, True, False],
               [768, 1024, 3, True, True]],
        'X': [[64, 128, 3, True, False], [128, 256, 9, True, False],
             [256, 512, 9, True, False], [512, 1024, 3, False, True]],
    }
    """
    CSPDarkNet backbone.

    Args:
        arch (str): Architecture of CSPDarkNet, from {P5, P6, X}, default as P5,
            and 'X' means used in YOLOX.
        depth_factor (float): Depth multiplier, multiply number of channels in
            each layer, default as 1.0.
        width_factor (float): Width multiplier, multiply number of blocks in
            CSPLayer, default as 1.0.
        depthwise (bool): Whether to use depth-wise conv layer.
        act (str): Activation function type, default as 'silu'.
        return_idx (list): Index of stages whose feature maps are returned.
        freeze_at (int): Freeze the backbone at which stage.
    """
    def __init__(self,
                 arch='P5',
                 depth_factor=1.0,
                 width_factor=1.0,
                 depthwise=False,
                 act='silu',
                 return_idx=[2, 3, 4],
                 freeze_at=-1):
        super(CSPDarkNet, self).__init__()
        self.arch = arch
        self.return_idx = return_idx
        Conv = DWConv if depthwise else BaseConv

        arch_setting = self.arch_settings[arch]
        base_channels = int(arch_setting[0][0] * width_factor)
        bias = False 
        # only for fuse_conv_bn debug. the bn'bias will be computed and convert to conv'bias

        # Note: different between the latest YOLOv5 and the original YOLOX
        # 1. self.stem
        # 2. use SPPF(in YOLOv5) or SPP(in YOLOX)
        # 3. put SPPF before(YOLOv5) or SPP after(YOLOX) the last cspdark block's CSPLayer
        # 4. whether SPPF(SPP)'CSPLayer add shortcut, True in YOLOv5, False in YOLOX

        if arch in ['P5', 'P6']:
            # in the latest YOLOv5, use Conv stem, and SPPF (fast, only single spp kernal size)
            self.stem = Conv(3, base_channels, ksize=6, stride=2, bias=bias, act=act)
            spp_kernal_sizes = 5
        else:
            # in the original YOLOX, use Focus stem, and SPP (three spp kernal sizes)
            self.stem = Focus(3, base_channels, ksize=3, stride=1, bias=bias, act=act)
            spp_kernal_sizes = (5, 9, 13)

        _out_channels = [base_channels]
        layers_num = 1
        self.csp_dark_blocks = []

        for i, (in_channels, out_channels, num_blocks, shortcut, use_spp) in enumerate(arch_setting):
            _out_channels.append(out_channels)
            in_channels = int(in_channels * width_factor)
            out_channels = int(out_channels * width_factor)
            num_blocks = max(round(num_blocks * depth_factor), 1)
            stage = []

            conv_layer = self.add_sublayer(
                'layers{}.stage{}.conv_layer'.format(layers_num, i + 1),
                Conv(in_channels, out_channels, 3, 2, bias=bias, act=act)
            )
            stage.append(conv_layer)
            layers_num += 1
                
            if use_spp and arch in ['X']:
                # in YOLOX use SPPLayer
                spp_layer = self.add_sublayer(
                    'layers{}.stage{}.spp_layer'.format(layers_num, i + 1),
                    SPPLayer(out_channels, out_channels, kernel_sizes=spp_kernal_sizes, bias=bias, act=act)
                )
                stage.append(spp_layer)
                layers_num += 1

            csp_layer = self.add_sublayer(
                'layers{}.stage{}.csp_layer'.format(layers_num, i + 1), 
                CSPLayer(out_channels, 
                         out_channels,
                         num_blocks=num_blocks,
                         shortcut=shortcut,
                         depthwise=depthwise,
                         bias=bias,
                         act=act)
            )
            stage.append(csp_layer)
            layers_num += 1

            if use_spp and arch in ['P5', 'P6']:
                # in latest YOLOv5 use SPPFLayer instead of SPPLayer
                sppf_layer = self.add_sublayer(
                    'layers{}.stage{}.sppf_layer'.format(layers_num, i + 1),
                    SPPFLayer(out_channels, out_channels, ksize=5, bias=bias, act=act)
                )
                stage.append(sppf_layer)
                layers_num += 1
            
            self.csp_dark_blocks.append(nn.Sequential(*stage))

        self._out_channels = [_out_channels[i] for i in self.return_idx]
        self.strides = [[2, 4, 8, 16, 32][i] for i in self.return_idx]

    def forward(self, inputs):
        x = inputs['image']
        outputs = []
        x = self.stem(x)
        for i, layer in enumerate(self.csp_dark_blocks):
            x = layer(x)
            if i + 1 in self.return_idx:
                outputs.append(x)
        return outputs

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]
