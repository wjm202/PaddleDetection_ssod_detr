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
import paddle.nn as nn
import paddle.nn.functional as F
from .distill import SSIM

from ppdet.core.workspace import register, create, load_config
from ppdet.utils.checkpoint import load_pretrain_weight
from .distill import parameter_init
from ppdet.modeling.losses.iou_loss import GIoULoss
from ppdet.utils.logger import setup_logger

logger = setup_logger(__name__)


class PPYOLOEDistillModel(nn.Layer):
    def __init__(self, cfg, slim_cfg):
        super(PPYOLOEDistillModel, self).__init__()
        self.student_model = create(cfg.architecture)
        logger.debug('Load student model pretrain_weights:{}'.format(
            cfg.pretrain_weights))
        load_pretrain_weight(self.student_model, cfg.pretrain_weights)

        slim_cfg = load_config(slim_cfg)
        self.teacher_model = create(slim_cfg.architecture)
        self.distill_loss = create(slim_cfg.distill_loss)
        logger.debug('Load teacher model pretrain_weights:{}'.format(
            slim_cfg.pretrain_weights))
        load_pretrain_weight(self.teacher_model, slim_cfg.pretrain_weights)

        for param in self.teacher_model.parameters():
            param.trainable = False

    def parameters(self):
        return self.student_model.parameters()

    def forward(self, inputs, alpha=0.125, beta=1.5):
        if self.training:
            with paddle.no_grad():
                teacher_out = self.teacher_model(inputs)
            if hasattr(self.teacher_model.yolo_head, "assigned_labels"):
                self.student_model.yolo_head.assigned_labels, self.student_model.yolo_head.assigned_bboxes, self.student_model.yolo_head.assigned_scores, self.student_model.yolo_head.mask_positive = \
                    self.teacher_model.yolo_head.assigned_labels, self.teacher_model.yolo_head.assigned_bboxes, self.teacher_model.yolo_head.assigned_scores, self.teacher_model.yolo_head.mask_positive
                delattr(self.teacher_model.yolo_head, "assigned_labels")
                delattr(self.teacher_model.yolo_head, "assigned_bboxes")
                delattr(self.teacher_model.yolo_head, "assigned_scores")
                delattr(self.teacher_model.yolo_head, "mask_positive")

            student_out = self.student_model(inputs)

            # head loss concerned
            soft_loss, feat_loss, distill_loss_dict = self.distill_loss(
                self.teacher_model, self.student_model)
            # print(distill_loss_dict)
            # base loss
            stu_loss = student_out['det_losses']
            oriloss = stu_loss['loss']

            # conbined distill
            stu_loss['loss'] = soft_loss + alpha * feat_loss + alpha * oriloss
            stu_loss['feat_loss'] = feat_loss
            stu_loss['soft_loss'] = soft_loss
            return stu_loss
        else:
            return self.student_model(inputs)


@register
class DistillPPYOLOELoss(nn.Layer):
    def __init__(self,
                 teacher_width_mult=1.0,
                 student_width_mult=0.75,
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5,
                 },
                 kd_neck=True,
                 kd_type='fgd'):
        super(DistillPPYOLOELoss, self).__init__()
        self.loss_bbox = GIoULoss()
        self.bbox_loss_weight = loss_weight['iou']
        self.dfl_loss_weight = loss_weight['dfl']
        self.qfl_loss_weight = loss_weight['class']
        self.loss_num = 3
        neck_out_channels = [768, 384, 192]  # default as L model
        self.kd_neck = kd_neck
        self.kd_type = kd_type

        if self.kd_neck:
            # Knowledge Distillation for Detectors in necks
            distill_loss_module_list = []
            self.t_channel_list = [
                int(c * teacher_width_mult) for c in neck_out_channels
            ]
            self.s_channel_list = [
                int(c * student_width_mult) for c in neck_out_channels
            ]
            for i in range(self.loss_num):
                if self.kd_type == 'fgd':
                    distill_loss_module = FGDFeatureLoss_norm(
                        student_channels=self.s_channel_list[i],
                        teacher_channels=self.t_channel_list[i])
                elif self.kd_type == 'pkd':
                    distill_loss_module = PKDLoss(
                        student_channels=self.s_channel_list[i],
                        teacher_channels=self.t_channel_list[i],
                        resize_stu=False)
                elif self.kd_type == 'mgd':
                    distill_loss_module = MGDSSIMLoss(
                        student_channels=self.s_channel_list[i],
                        teacher_channels=self.t_channel_list[i]) 
                else:
                    raise ValueError
                distill_loss_module_list.append(distill_loss_module)
            self.distill_loss_module_list = nn.LayerList(
                distill_loss_module_list)

    def bbox_loss(self, s_bbox, t_bbox, weight_targets=None):
        # sx, sy, sw, sh 
        # tx, ty, tw, th
        if weight_targets is not None:
            loss_bbox = paddle.sum(
                self.loss_bbox(s_bbox, t_bbox) * weight_targets)
            avg_factor = weight_targets.sum()
            loss_bbox = loss_bbox / avg_factor
        else:
            loss_bbox = paddle.mean(self.loss_bbox(s_bbox, t_bbox))
        return loss_bbox

    def quality_focal_loss(self, pred_logits, soft_target_logits, beta=2.0, \
            use_sigmoid=True, label_weights=None, num_total_pos=None, pos_mask=None):

        if use_sigmoid:
            func = F.binary_cross_entropy_with_logits
            soft_target = F.sigmoid(soft_target_logits)
            pred_sigmoid = F.sigmoid(pred_logits)
            preds = pred_logits
        else:
            func = F.binary_cross_entropy
            soft_target = soft_target_logits
            pred_sigmoid = pred_logits
            preds = pred_sigmoid

        scale_factor = pred_sigmoid - soft_target
        loss = func(
            preds, soft_target, reduction='none') * scale_factor.abs().pow(beta)
        loss = loss
        if pos_mask is not None:
            loss *= pos_mask
        # (func(pred_logits, soft_target, reduction='none') * scale_factor.abs().pow(beta)).sum(1).sum(0) [12.66135025])

        loss = loss.sum(1)  # (N, )
        if label_weights is not None:
            loss = loss * label_weights
        if num_total_pos is not None:
            loss = loss.sum() / num_total_pos
        else:
            loss = loss.mean()
        return loss

    def distribution_focal_loss(self, pred_corners, target_corners,
                                weight_targets):

        target_corners_label = paddle.nn.functional.softmax(
            target_corners, axis=-1)
        loss_dfl = paddle.nn.functional.cross_entropy(
            pred_corners,
            target_corners_label,
            soft_label=True,
            reduction='none')
        loss_dfl = loss_dfl.sum(1)
        if weight_targets is not None:
            loss_dfl = loss_dfl * (weight_targets.expand([-1, 4]).reshape([-1]))
            loss_dfl = loss_dfl.sum(-1) / weight_targets.sum()
        else:
            loss_dfl = loss_dfl.mean(-1)
        loss_dfl = loss_dfl / 4.  # 4个方向
        return loss_dfl

    def forward(self, teacher_model, student_model):
        teacher_distill_pairs = teacher_model.yolo_head.distill_pairs
        student_distill_pairs = student_model.yolo_head.distill_pairs
        distill_bbox_loss, distill_dfl_loss, distill_cls_loss = [], [], []
        distill_bbox_loss.append(
            self.bbox_loss(student_distill_pairs['pred_bboxes_pos'],
                            teacher_distill_pairs['pred_bboxes_pos'].detach(),
                            weight_targets=student_distill_pairs['bbox_weight']
                ) if 'pred_bboxes_pos' in student_distill_pairs and \
                    'pred_bboxes_pos' in teacher_distill_pairs and \
                        'bbox_weight' in student_distill_pairs
                else student_distill_pairs['null_loss']
            )
        distill_dfl_loss.append(self.distribution_focal_loss(
                    student_distill_pairs['pred_dist_pos'].reshape((-1, student_distill_pairs['pred_dist_pos'].shape[-1])),
                    teacher_distill_pairs['pred_dist_pos'].detach().reshape((-1, teacher_distill_pairs['pred_dist_pos'].shape[-1])), \
                    weight_targets=student_distill_pairs['bbox_weight']
                ) if 'pred_dist_pos' in student_distill_pairs and \
                    'pred_dist_pos' in teacher_distill_pairs and \
                        'bbox_weight' in student_distill_pairs
                else student_distill_pairs['null_loss']
            )
        distill_cls_loss.append(
            self.quality_focal_loss(
                student_distill_pairs['pred_cls_scores'].reshape((
                    -1, student_distill_pairs['pred_cls_scores'].shape[-1])),
                teacher_distill_pairs['pred_cls_scores'].detach().reshape((
                    -1, teacher_distill_pairs['pred_cls_scores'].shape[-1])),
                num_total_pos=student_distill_pairs['pos_num'],  ###
                use_sigmoid=False))
        distill_bbox_loss = paddle.add_n(distill_bbox_loss)
        distill_cls_loss = paddle.add_n(distill_cls_loss)
        distill_dfl_loss = paddle.add_n(distill_dfl_loss)

        if self.kd_neck:
            # Knowledge Distillation for Detectors in necks
            distill_neck_global_loss = []
            inputs = student_model.inputs
            teacher_fpn_feats = teacher_distill_pairs['emb_feats']
            student_fpn_feats = student_distill_pairs['emb_feats']
            assert 'gt_bbox' in inputs
            for i, distill_loss_module in enumerate(
                    self.distill_loss_module_list):
                distill_neck_global_loss.append(
                    distill_loss_module(student_fpn_feats[i], teacher_fpn_feats[
                        i], inputs))
            distill_neck_global_loss = paddle.add_n(distill_neck_global_loss)
        else:
            distill_neck_global_loss = paddle.to_tensor([0])

        soft_loss = (distill_bbox_loss * self.bbox_loss_weight + distill_cls_loss *
                self.qfl_loss_weight + distill_dfl_loss * self.dfl_loss_weight)
        student_model.yolo_head.distill_pairs.clear()
        teacher_model.yolo_head.distill_pairs.clear()
        return soft_loss, \
            distill_neck_global_loss, \
            {'dfl_loss': distill_dfl_loss, 'qfl_loss': distill_cls_loss, 'bbox_loss': distill_bbox_loss}


@register
class FGDFeatureLoss_norm(nn.Layer):
    """
    The code is reference from https://github.com/yzd-v/FGD/blob/master/mmdet/distillation/losses/fgd.py
    Paddle version of `Focal and Global Knowledge Distillation for Detectors`
   
    Args:
        student_channels(int): The number of channels in the student's FPN feature map. Default to 256.
        teacher_channels(int): The number of channels in the teacher's FPN feature map. Default to 256.
        temp (float, optional): The temperature coefficient. Defaults to 0.5.
        alpha_fgd (float, optional): The weight of fg_loss. Defaults to 0.001
        beta_fgd (float, optional): The weight of bg_loss. Defaults to 0.0005
        gamma_fgd (float, optional): The weight of mask_loss. Defaults to 0.001
        lambda_fgd (float, optional): The weight of relation_loss. Defaults to 0.000005
    """

    def __init__(
            self,
            student_channels=256,
            teacher_channels=256,
            temp=0.5,
            alpha_fgd=0.00001,  # 0.001
            beta_fgd=0.000005,  # 0.0005
            gamma_fgd=0.00001,  # 0.001
            lambda_fgd=0.00000005):  # 0.000005
        super(FGDFeatureLoss_norm, self).__init__()
        self.temp = temp
        self.alpha_fgd = alpha_fgd
        self.beta_fgd = beta_fgd
        self.gamma_fgd = gamma_fgd
        self.lambda_fgd = lambda_fgd
        self.normalize = True

        kaiming_init = parameter_init("kaiming")
        zeros_init = parameter_init("constant", 0.0)

        if student_channels != teacher_channels:
            self.align = nn.Conv2D(
                student_channels,
                teacher_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                weight_attr=kaiming_init)
            # self.align = ESEAttn(student_channels, teacher_channels)
            student_channels = teacher_channels
        else:
            self.align = None

        self.conv_mask_s = nn.Conv2D(
            student_channels, 1, kernel_size=1, weight_attr=kaiming_init)
        self.conv_mask_t = nn.Conv2D(
            teacher_channels, 1, kernel_size=1, weight_attr=kaiming_init)

        self.stu_conv_block = nn.Sequential(
            nn.Conv2D(
                student_channels,
                student_channels // 2,
                kernel_size=1,
                weight_attr=zeros_init),
            nn.LayerNorm([student_channels // 2, 1, 1]),
            nn.ReLU(),
            nn.Conv2D(
                student_channels // 2,
                student_channels,
                kernel_size=1,
                weight_attr=zeros_init))
        self.tea_conv_block = nn.Sequential(
            nn.Conv2D(
                teacher_channels,
                teacher_channels // 2,
                kernel_size=1,
                weight_attr=zeros_init),
            nn.LayerNorm([teacher_channels // 2, 1, 1]),
            nn.ReLU(),
            nn.Conv2D(
                teacher_channels // 2,
                teacher_channels,
                kernel_size=1,
                weight_attr=zeros_init))

    def norm(self, feat):
        """Normalize the feature maps to have zero mean and unit variances.
        Args:
            feat (torch.Tensor): The original feature map with shape
                (N, C, H, W).
        """
        assert len(feat.shape) == 4
        N, C, H, W = feat.shape
        feat = feat.transpose([1, 0, 2, 3]).reshape([C, -1])
        mean = feat.mean(axis=-1, keepdim=True)
        std = feat.std(axis=-1, keepdim=True)
        feat = (feat - mean) / (std + 1e-6)
        return feat.reshape([C, N, H, W]).transpose([1, 0, 2, 3])

    def spatial_channel_attention(self, x, t=0.5):
        shape = paddle.shape(x)
        N, C, H, W = shape

        _f = paddle.abs(x)
        spatial_map = paddle.reshape(
            paddle.mean(
                _f, axis=1, keepdim=True) / t, [N, -1])
        spatial_map = F.softmax(spatial_map, axis=1, dtype="float32") * H * W
        spatial_att = paddle.reshape(spatial_map, [N, H, W])

        channel_map = paddle.mean(
            paddle.mean(
                _f, axis=2, keepdim=False), axis=2, keepdim=False)
        channel_att = F.softmax(channel_map / t, axis=1, dtype="float32") * C
        return [spatial_att, channel_att]

    def spatial_pool(self, x, mode="teacher"):
        batch, channel, width, height = x.shape
        x_copy = x
        x_copy = paddle.reshape(x_copy, [batch, channel, height * width])
        x_copy = x_copy.unsqueeze(1)
        if mode.lower() == "student":
            context_mask = self.conv_mask_s(x)
        else:
            context_mask = self.conv_mask_t(x)

        context_mask = paddle.reshape(context_mask, [batch, 1, height * width])
        context_mask = F.softmax(context_mask, axis=2)
        context_mask = context_mask.unsqueeze(-1)
        context = paddle.matmul(x_copy, context_mask)
        context = paddle.reshape(context, [batch, channel, 1, 1])

        return context

    def mask_loss(self, stu_channel_att, tea_channel_att, stu_spatial_att,
                  tea_spatial_att):
        def _func(a, b):
            return paddle.sum(paddle.abs(a - b)) / len(a)

        mask_loss = _func(stu_channel_att, tea_channel_att) + _func(
            stu_spatial_att, tea_spatial_att)

        return mask_loss

    def feature_loss(self, stu_feature, tea_feature, Mask_fg, Mask_bg,
                     tea_channel_att, tea_spatial_att):
        Mask_fg = Mask_fg.unsqueeze(axis=1)
        Mask_bg = Mask_bg.unsqueeze(axis=1)

        tea_channel_att = tea_channel_att.unsqueeze(axis=-1)
        tea_channel_att = tea_channel_att.unsqueeze(axis=-1)

        tea_spatial_att = tea_spatial_att.unsqueeze(axis=1)

        fea_t = paddle.multiply(tea_feature, paddle.sqrt(tea_spatial_att))
        fea_t = paddle.multiply(fea_t, paddle.sqrt(tea_channel_att))
        fg_fea_t = paddle.multiply(fea_t, paddle.sqrt(Mask_fg))
        bg_fea_t = paddle.multiply(fea_t, paddle.sqrt(Mask_bg))

        fea_s = paddle.multiply(stu_feature, paddle.sqrt(tea_spatial_att))
        fea_s = paddle.multiply(fea_s, paddle.sqrt(tea_channel_att))
        fg_fea_s = paddle.multiply(fea_s, paddle.sqrt(Mask_fg))
        bg_fea_s = paddle.multiply(fea_s, paddle.sqrt(Mask_bg))

        fg_loss = F.mse_loss(fg_fea_s, fg_fea_t, reduction="sum") / len(Mask_fg)
        bg_loss = F.mse_loss(bg_fea_s, bg_fea_t, reduction="sum") / len(Mask_bg)

        return fg_loss, bg_loss

    def relation_loss(self, stu_feature, tea_feature):
        context_s = self.spatial_pool(stu_feature, "student")
        context_t = self.spatial_pool(tea_feature, "teacher")

        out_s = stu_feature + self.stu_conv_block(context_s)
        out_t = tea_feature + self.tea_conv_block(context_t)

        rela_loss = F.mse_loss(out_s, out_t, reduction="sum") / len(out_s)

        return rela_loss

    def mask_value(self, mask, xl, xr, yl, yr, value):
        mask[xl:xr, yl:yr] = paddle.maximum(mask[xl:xr, yl:yr], value)
        return mask

    def forward(self, stu_feature, tea_feature, inputs):
        """Forward function.
        Args:
            stu_feature(Tensor): Bs*C*H*W, student's feature map
            tea_feature(Tensor): Bs*C*H*W, teacher's feature map
            inputs: The inputs with gt bbox and input shape info.
        """
        assert stu_feature.shape[-2:] == stu_feature.shape[-2:], \
            f'The shape of Student feature {stu_feature.shape} and Teacher feature {tea_feature.shape} should be the same.'
        assert "gt_bbox" in inputs.keys() and "im_shape" in inputs.keys(
        ), "ERROR! FGDFeatureLoss need gt_bbox and im_shape as inputs."
        gt_bboxes = inputs['gt_bbox']
        ins_shape = [
            inputs['im_shape'][i] for i in range(inputs['im_shape'].shape[0])
        ]

        if self.align is not None:
            stu_feature = self.align(stu_feature)

        if self.normalize:
            stu_feature, tea_feature = self.norm(stu_feature), self.norm(
                tea_feature)

        N, C, H, W = stu_feature.shape

        tea_spatial_att, tea_channel_att = self.spatial_channel_attention(
            tea_feature, self.temp)
        stu_spatial_att, stu_channel_att = self.spatial_channel_attention(
            stu_feature, self.temp)

        Mask_fg = paddle.zeros(tea_spatial_att.shape)
        Mask_bg = paddle.ones_like(tea_spatial_att)
        one_tmp = paddle.ones([*tea_spatial_att.shape[1:]])
        zero_tmp = paddle.zeros([*tea_spatial_att.shape[1:]])
        wmin, wmax, hmin, hmax, area = [], [], [], [], []

        if gt_bboxes.shape[1] != 0:
            for i in range(N):
                tmp_box = paddle.ones_like(gt_bboxes[i])
                tmp_box[:, 0] = gt_bboxes[i][:, 0] / ins_shape[i][1] * W
                tmp_box[:, 2] = gt_bboxes[i][:, 2] / ins_shape[i][1] * W
                tmp_box[:, 1] = gt_bboxes[i][:, 1] / ins_shape[i][0] * H
                tmp_box[:, 3] = gt_bboxes[i][:, 3] / ins_shape[i][0] * H

                zero = paddle.zeros_like(tmp_box[:, 0], dtype="int32")
                ones = paddle.ones_like(tmp_box[:, 2], dtype="int32")
                wmin.append(
                    paddle.cast(paddle.floor(tmp_box[:, 0]), "int32").maximum(
                        zero))
                wmax.append(paddle.cast(paddle.ceil(tmp_box[:, 2]), "int32"))
                hmin.append(
                    paddle.cast(paddle.floor(tmp_box[:, 1]), "int32").maximum(
                        zero))
                hmax.append(paddle.cast(paddle.ceil(tmp_box[:, 3]), "int32"))

                area_recip = 1.0 / (
                    hmax[i].reshape([1, -1]) + 1 - hmin[i].reshape([1, -1])) / (
                        wmax[i].reshape([1, -1]) + 1 - wmin[i].reshape([1, -1]))

                for j in range(len(gt_bboxes[i])):
                    if gt_bboxes[i][j].sum() > 0:
                        Mask_fg[i] = self.mask_value(
                            Mask_fg[i], hmin[i][j], hmax[i][j] + 1, wmin[i][j],
                            wmax[i][j] + 1, area_recip[0][j])

                Mask_bg[i] = paddle.where(Mask_fg[i] > zero_tmp, zero_tmp,
                                          one_tmp)

                if paddle.sum(Mask_bg[i]):
                    Mask_bg[i] /= paddle.sum(Mask_bg[i])

            fg_loss, bg_loss = self.feature_loss(
                stu_feature, tea_feature, Mask_fg, Mask_bg, tea_channel_att,
                tea_spatial_att)
            mask_loss = self.mask_loss(stu_channel_att, tea_channel_att,
                                       stu_spatial_att, tea_spatial_att)
            rela_loss = self.relation_loss(stu_feature, tea_feature)
            loss = self.alpha_fgd * fg_loss + self.beta_fgd * bg_loss \
                        + self.gamma_fgd * mask_loss + self.lambda_fgd * rela_loss
        else:
            rela_loss = self.relation_loss(stu_feature, tea_feature)
            loss = self.lambda_fgd * rela_loss
        return loss


@register
class PKDLoss(nn.Layer):
    """PyTorch version of `PKD: General Distillation Framework for Object
    Detectors via Pearson Correlation Coefficient.
    <https://arxiv.org/abs/2207.02039>`_.
    Args:
        loss_weight (float): Weight of loss. Defaults to 1.0.
        resize_stu (bool): If True, we'll down/up sample the features of the
            student model to the spatial size of those of the teacher model if
            their spatial sizes are different. And vice versa. Defaults to
            True.
    """

    def __init__(self,
                 student_channels=256,
                 teacher_channels=256,
                 loss_weight=1.0,
                 resize_stu=True):
        super(PKDLoss, self).__init__()
        self.loss_weight = loss_weight
        self.resize_stu = resize_stu

        kaiming_init = parameter_init("kaiming")
        if student_channels != teacher_channels:
            self.align = nn.Conv2D(
                student_channels,
                teacher_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                weight_attr=kaiming_init)
        else:
            self.align = None

    def norm(self, feat):
        """Normalize the feature maps to have zero mean and unit variances.
        Args:
            feat (torch.Tensor): The original feature map with shape
                (N, C, H, W).
        """
        assert len(feat.shape) == 4
        N, C, H, W = feat.shape
        feat = feat.transpose([1, 0, 2, 3]).reshape([C, -1])
        mean = feat.mean(axis=-1, keepdim=True)
        std = feat.std(axis=-1, keepdim=True)
        feat = (feat - mean) / (std + 1e-6)
        return feat.reshape([C, N, H, W]).transpose([1, 0, 2, 3])

    def forward(self, pred_S, pred_T, input):
        if self.align is not None:
            pred_S = self.align(pred_S)

        loss = 0.
        size_S, size_T = pred_S.shape[2:], pred_T.shape[2:]
        if size_S[0] != size_T[0]:
            if self.resize_stu:
                pred_S = F.interpolate(pred_S, size_T, mode='bilinear')
            else:
                pred_T = F.interpolate(pred_T, size_S, mode='bilinear')
        assert pred_S.shape == pred_T.shape

        norm_S, norm_T = self.norm(pred_S), self.norm(pred_T)

        # First conduct feature normalization and then calculate the
        # MSE loss. Methematically, it is equivalent to firstly calculate
        # the Pearson Correlation Coefficient (r) between two feature
        # vectors, and then use 1-r as the new feature imitation loss.
        loss += F.mse_loss(norm_S, norm_T) / 2

        return loss * self.loss_weight


@register
class MGDSSIMLoss(nn.Layer):

    def __init__(self,
                 student_channels=256,
                 teacher_channels=256,
                 loss_weight=1.0,
                 max_alpha=1.0,
                 min_alpha=0.2):
        super(MGDSSIMLoss, self).__init__()
        self.loss_weight = loss_weight
        self.max_alpha = max_alpha
        self.min_alpha = min_alpha
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.ssim_loss = SSIM(11)

        kaiming_init = parameter_init("kaiming")
        if student_channels != teacher_channels:
            self.align_layer = nn.Conv2D(
                student_channels,
                teacher_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                weight_attr=kaiming_init,
                bias_attr=False)
        else:
            self.align_layer = None

        self.generations= nn.Sequential(
                nn.Conv2D(
                    teacher_channels,
                    teacher_channels,
                    kernel_size=3,
                    padding=1),
                nn.ReLU(),
                nn.Conv2D(
                    teacher_channels,
                    teacher_channels,
                    kernel_size=3,
                    padding=1))
        self.use_norm = False

    def norm(self, feat):
        """Normalize the feature maps to have zero mean and unit variances.
        Args:
            feat (torch.Tensor): The original feature map with shape
                (N, C, H, W).
        """
        assert len(feat.shape) == 4
        N, C, H, W = feat.shape
        feat = feat.transpose([1, 0, 2, 3]).reshape([C, -1])
        mean = feat.mean(axis=-1, keepdim=True)
        std = feat.std(axis=-1, keepdim=True)
        feat = (feat - mean) / (std + 1e-6)
        return feat.reshape([C, N, H, W]).transpose([1, 0, 2, 3])

    def get_ppyoloe_mgd_loss(self, stu_feature, tea_feature, ssim=False):
        N, C, H, W = stu_feature.shape
        masked_fea = stu_feature
        masked_fea = self.align_layer(masked_fea)
        new_fea = self.generations(masked_fea)

        if self.use_norm:
            new_fea = self.norm(new_fea)
            tea_feature = self.norm(tea_feature)

        if ssim is False:
            dis_loss = self.mse_loss(new_fea, tea_feature) / N
        else:
            ssim = self.ssim_loss(new_fea, tea_feature)
            dis_loss = paddle.clip((1 - ssim) / 2, 0, 1)
        return dis_loss

    def forward(self, stu_fpn_feat, tea_fpn_feat, input):
        loss = self.get_ppyoloe_mgd_loss(
            stu_fpn_feat, tea_fpn_feat, ssim=True)
        return loss * self.loss_weight
