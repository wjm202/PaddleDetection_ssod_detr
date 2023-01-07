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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ..initializer import bias_init_with_prob
from ppdet.core.workspace import register

__all__ = ['SimpleConvHead']

norm_cfg = {
    "BN": ("bn", nn.BatchNorm2D),
    "SyncBN": ("syncbn", nn.SyncBatchNorm),
    "GN": ("gn", nn.GroupNorm),
}


def build_norm_layer(cfg, num_features, postfix=""):
    """Build normalization layer
    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            create named layer.
    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and "type" in cfg
    cfg_ = cfg.copy()  # dict

    layer_type = cfg_.pop("type")
    if layer_type not in norm_cfg:
        raise KeyError("Unrecognized norm type {}".format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop("requires_grad", True)
    #cfg_.setdefault("eps", 1e-5)
    if layer_type == "SyncBN":
        print("layer_type", layer_type)
    if layer_type != "GN":
        layer = norm_layer(num_features, **cfg_)
        if layer_type == "SyncBN" and hasattr(layer, "_specify_ddp_gpu_num"):
            layer._specify_ddp_gpu_num(1)
    else:
        assert "num_groups" in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer


activations = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "ReLU6": nn.ReLU6,
    "SELU": nn.SELU,
    "ELU": nn.ELU,
    "GELU": nn.GELU,
    "PReLU": nn.PReLU,
    "silu": nn.Silu,
    "HardSwish": nn.Hardswish,
    "Hardswish": nn.Hardswish,
    None: nn.Identity,
}


def act_layers(name):
    assert name in activations.keys()
    if name == "LeakyReLU":
        return nn.LeakyReLU(negative_slope=0.1)
    elif name == "GELU":
        return nn.GELU()
    elif name == "PReLU":
        return nn.PReLU()
    else:
        return activations[name]()


def kaiming_init(module,
                 a=0,
                 mode="fan_out",
                 nonlinearity="relu",
                 bias=0,
                 distribution="normal"):
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.initializer.kaiming_uniform(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.initializer.KaimingNormal()
    if hasattr(module, "bias") and module.bias is not None:
        nn.initializer.Constant(val)


def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.initializer.Constant(val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.initializer.Constant(val)


class ConvModule(nn.Layer):
    """A conv block that contains conv/norm/activation layers.
    Args:
        in_channels (int): Same as nn.Conv2D.
        out_channels (int): Same as nn.Conv2D.
        kernel_size (int or tuple[int]): Same as nn.Conv2D.
        stride (int or tuple[int]): Same as nn.Conv2D.
        padding (int or tuple[int]): Same as nn.Conv2D.
        dilation (int or tuple[int]): Same as nn.Conv2D.
        groups (int): Same as nn.Conv2D.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
        activation (str): activation layer, "ReLU" by default.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias="auto",
            conv_cfg=None,
            norm_cfg=None,
            activation="ReLU",
            order=("conv", "norm", "act"), ):
        super(ConvModule, self).__init__()
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert activation is None or isinstance(activation, str)
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.order = order

        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == {"conv", "norm", "act"}

        self.with_norm = norm_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == "auto":
            bias = False if self.with_norm else True
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn("ConvModule has norm and bias at the same time")

        # build convolution layer
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias_attr=bias, )
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # build normalization layers
        if self.with_norm:  # true
            # norm layer is after conv layer
            if order.index("norm") > order.index("conv"):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_sublayer(self.norm_name, norm)
        else:
            self.norm_name = None

        # build activation layer
        if self.activation:
            self.act = act_layers(self.activation)

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        if self.activation == "LeakyReLU":
            nonlinearity = "leaky_relu"
        else:
            nonlinearity = "relu"
        kaiming_init(self.conv, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, norm=True):
        for layer in self.order:  #("conv", "norm", "act")
            if layer == "conv":  # x: [8, 576, 10, 10]
                x = self.conv(x)
            elif layer == "norm" and norm and self.with_norm:
                x = self.norm(x)
            elif layer == "act" and self.activation:
                x = self.act(x)
        return x


class Scale(nn.Layer):
    """
    A learnable scale parameter
    """

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        scale = paddle.to_tensor(scale)
        self.scale = paddle.create_parameter(
            shape=scale.shape,
            dtype=paddle.float32,
            default_initializer=paddle.nn.initializer.Assign(scale))

    def forward(self, x):
        return x * self.scale


def normal_init(module, mean=0, std=1, bias=0):
    nn.initializer.Normal(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.initializer.Constant(bias)


@register
class SimpleConvHead(nn.Layer):
    __shared__ = ['num_classes']

    def __init__(
            self,
            in_channels=[1024, 512, 256],
            num_classes=80,
            norm_cfg=dict(type="BN", requires_grad=True),  #  GN num_groups=32,  SyncBN
            act='LeakyReLU', #"LeakyReLU",
            reg_max=16):
        super(SimpleConvHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels  # 576
        self.reg_max = reg_max

        self.norm_cfg = norm_cfg
        print("norm_cfg:", norm_cfg)
        self.activation = act

        self._init_layers()
        self.init_weights()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_layers(self):
        self.cls_convs = nn.LayerList()
        self.reg_convs = nn.LayerList()
        for in_c in self.in_channels:
            self.cls_convs.append(
                ConvModule(
                    in_c * 2,
                    in_c,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=None,
                    norm_cfg=self.norm_cfg,
                    activation=self.activation))
            self.reg_convs.append(
                ConvModule(
                    in_c * 2,
                    in_c,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=None,
                    norm_cfg=self.norm_cfg,
                    activation=self.activation))
        # pred head
        self.gfl_cls = nn.LayerList()
        self.gfl_reg = nn.LayerList()
        self.scales = nn.LayerList()
        for in_c in self.in_channels:
            self.gfl_cls.append(
                nn.Conv2D(in_c, self.num_classes, 3, padding=1))
            self.gfl_reg.append(
                nn.Conv2D(in_c, 4 * (self.reg_max + 1), 3, padding=1))
            self.scales.append(Scale(1.0))

    def init_weights(self):
        for cls_, reg_ in zip(self.cls_convs, self.reg_convs):
            normal_init(cls_.conv, std=0.01)
            normal_init(reg_.conv, std=0.01)

        bias_cls = -4.595
        #bias_cls = bias_init_with_prob(0.01)
        for cls_, reg_ in zip(self.gfl_cls, self.gfl_reg):
            normal_init(cls_, std=0.01, bias=bias_cls)
            normal_init(reg_, std=0.01)

    def forward(self, feats):
        cls_scores = []
        bbox_preds = []
        for i, feat in enumerate(feats):
            cls_score = self.gfl_cls[i](self.cls_convs[i](feat))
            bbox_pred = self.gfl_reg[i](self.reg_convs[i](feat))
            cls_score = F.sigmoid(cls_score)
            cls_score = cls_score.flatten(2).transpose([0, 2, 1])

            bbox_pred = self.scales[i](bbox_pred)
            bbox_pred = bbox_pred.flatten(2).transpose([0, 2, 1])

            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)

        cls_scores = paddle.concat(cls_scores, axis=1)
        bbox_preds = paddle.concat(bbox_preds, axis=1)
        return cls_scores, bbox_preds
