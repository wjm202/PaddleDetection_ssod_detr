#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn.functional as F


def permute_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.reshape([N, -1, K, H, W]).transpose([0, 3, 4, 1, 2])
    tensor = tensor.reshape([N, -1, K])
    return tensor


def QFLv2(pred_sigmoid, teacher_sigmoid, weight=None, beta=2.0, reduction='mean'):
    # all goes to 0
    pt = pred_sigmoid
    zerolabel = paddle.zeros_like(pt)
    loss = F.binary_cross_entropy(
        pred_sigmoid, zerolabel, reduction='none') * pt.pow(beta)
    pos = weight > 0

    # positive goes to bbox quality
    pt = teacher_sigmoid[pos] - pred_sigmoid[pos]
    loss[pos] = F.binary_cross_entropy(
        pred_sigmoid[pos], teacher_sigmoid[pos],
        reduction='none') * pt.pow(beta)

    valid = weight >= 0
    if reduction == "mean":
        loss = loss[valid].mean()
    elif reduction == "sum":
        loss = loss[valid].sum()
    return loss


def giou_loss(inputs, targets, eps=1e-7):
    inputs_area = (inputs[..., 2] - inputs[..., 0]).clip_(min=0) \
        * (inputs[..., 3] - inputs[..., 1]).clip_(min=0)
    targets_area = (targets[..., 2] - targets[..., 0]).clip_(min=0) \
        * (targets[..., 3] - targets[..., 1]).clip_(min=0)

    w_intersect = (paddle.minimum(inputs[..., 2], targets[..., 2]) -
                   paddle.maximum(inputs[..., 0], targets[..., 0])).clip_(min=0)
    h_intersect = (paddle.minimum(inputs[..., 3], targets[..., 3]) -
                   paddle.maximum(inputs[..., 1], targets[..., 1])).clip_(min=0)

    area_intersect = w_intersect * h_intersect
    area_union = targets_area + inputs_area - area_intersect

    ious = area_intersect / area_union.clip(min=eps)

    g_w_intersect = paddle.maximum(inputs[..., 2], targets[..., 2]) \
        - paddle.minimum(inputs[..., 0], targets[..., 0])
    g_h_intersect = paddle.maximum(inputs[..., 3], targets[..., 3]) \
        - paddle.minimum(inputs[..., 1], targets[..., 1])
    ac_uion = g_w_intersect * g_h_intersect
    gious = ious - (ac_uion - area_union) / ac_uion.clip(min=eps)
    loss = 1 - gious

    return loss