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
from ppdet.core.workspace import register

from ..bbox_utils import batch_distance2bbox
from ..losses import GIoULoss
from ..initializer import bias_init_with_prob, constant_, normal_
from ..assigners.utils import generate_anchors_for_grid_cell
from ppdet.modeling.backbones.cspresnet import ConvBNLayer
from ppdet.modeling.ops import get_static_shape, get_act_fn
from ppdet.modeling.layers import MultiClassNMS
from IPython import embed

__all__ = ['PPYOLOEHead']


class ESEAttn(nn.Layer):
    def __init__(self, feat_channels, act='swish'):
        super(ESEAttn, self).__init__()
        self.fc = nn.Conv2D(feat_channels, feat_channels, 1)
        self.conv = ConvBNLayer(feat_channels, feat_channels, 1, act=act)

        self._init_weights()

    def _init_weights(self):
        normal_(self.fc.weight, std=0.001)

    def forward(self, feat, avg_feat):
        weight = F.sigmoid(self.fc(avg_feat))
        return self.conv(feat * weight)


@register
class PPYOLOEHead(nn.Layer):
    __shared__ = [
        'num_classes', 'eval_size', 'trt', 'exclude_nms',
        'exclude_post_process', 'use_shared_conv'
    ]
    __inject__ = ['static_assigner', 'assigner', 'nms']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=80,
                 act='swish',
                 fpn_strides=(32, 16, 8),
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 reg_max=16,
                 reg_range=None,
                 static_assigner_epoch=4,
                 use_varifocal_loss=True,
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner',
                 nms='MultiClassNMS',
                 eval_size=None,
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5,
                 },
                 trt=False,
                 exclude_nms=False,
                 exclude_post_process=False,
                 use_shared_conv=True):
        super(PPYOLOEHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        if reg_range:
            self.sm_use = True
            self.reg_range = reg_range
        else:
            self.sm_use = False
            self.reg_range = (0, reg_max + 1)
        self.reg_channels = self.reg_range[1] - self.reg_range[0]
        self.iou_loss = GIoULoss()
        self.loss_weight = loss_weight
        self.use_varifocal_loss = use_varifocal_loss
        self.eval_size = eval_size

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.nms = nms
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.exclude_nms = exclude_nms
        self.exclude_post_process = exclude_post_process
        self.use_shared_conv = use_shared_conv

        # stem
        self.stem_cls = nn.LayerList()
        self.stem_reg = nn.LayerList()
        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act
        for in_c in self.in_channels:
            self.stem_cls.append(ESEAttn(in_c, act=act))
            self.stem_reg.append(ESEAttn(in_c, act=act))
        # pred head
        self.pred_cls = nn.LayerList()
        self.pred_reg = nn.LayerList()
        for in_c in self.in_channels:
            self.pred_cls.append(
                nn.Conv2D(
                    in_c, self.num_classes, 3, padding=1))
            self.pred_reg.append(
                nn.Conv2D(
                    in_c, 4 * self.reg_channels, 3, padding=1))
        # projection conv
        self.proj_conv = nn.Conv2D(self.reg_channels, 1, 1, bias_attr=False)
        self.proj_conv.skip_quant = True
        self._init_weights()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        for cls_, reg_ in zip(self.pred_cls, self.pred_reg):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.weight)
            constant_(reg_.bias, 1.0)

        proj = paddle.linspace(self.reg_range[0], self.reg_range[1] - 1,
                               self.reg_channels).reshape(
                                   [1, self.reg_channels, 1, 1])
        self.proj_conv.weight.set_value(proj)
        self.proj_conv.weight.stop_gradient = True
        if self.eval_size:
            anchor_points, stride_tensor = self._generate_anchors()
            self.anchor_points = anchor_points
            self.stride_tensor = stride_tensor

    def forward_train_fake(self, feats, targets):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_score_list, reg_distri_list = [], []
        for i, feat in enumerate(feats):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
            reg_distri_list.append(reg_distri.flatten(2).transpose([0, 2, 1]))
        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_distri_list = paddle.concat(reg_distri_list, axis=1)

        is_teacher = targets.get('is_teacher', False)
        assert is_teacher == True
        if 1:
            anchor_points_s = anchor_points / stride_tensor
            pred_bboxes, new_reg_distri_list = self._bbox_decode(
                anchor_points_s, reg_distri_list)
            return cls_score_list, pred_bboxes, new_reg_distri_list

    def forward_train(self, feats, targets):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_score_list, reg_distri_list = [], []
        for i, feat in enumerate(feats):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
            reg_distri_list.append(reg_distri.flatten(2).transpose([0, 2, 1]))
        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_distri_list = paddle.concat(reg_distri_list, axis=1)

        is_teacher = targets.get('is_teacher', False)
        if is_teacher:
            anchor_points_s = anchor_points / stride_tensor
            pred_bboxes, new_reg_distri_list = self._bbox_decode(
                anchor_points_s, reg_distri_list)
            return cls_score_list, pred_bboxes, new_reg_distri_list

        get_data = targets.get('get_data', False)
        if get_data:
            # anchor_points_s = anchor_points / stride_tensor
            # pred_bboxes, new_reg_distri_list = self._bbox_decode(
            #     anchor_points_s, reg_distri_list)
            return cls_score_list, reg_distri_list, anchors, anchor_points,num_anchors_list, stride_tensor

        return self.get_loss([
            cls_score_list, reg_distri_list, anchors, anchor_points,
            num_anchors_list, stride_tensor
        ], targets)

    def _generate_anchors(self, feats=None, dtype='float32'):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_strides):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = int(self.eval_size[0] / stride)
                w = int(self.eval_size[1] / stride)
            shift_x = paddle.arange(end=w) + self.grid_cell_offset
            shift_y = paddle.arange(end=h) + self.grid_cell_offset
            shift_y, shift_x = paddle.meshgrid(shift_y, shift_x)
            anchor_point = paddle.cast(
                paddle.stack(
                    [shift_x, shift_y], axis=-1), dtype=dtype)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(paddle.full([h * w, 1], stride, dtype=dtype))
        anchor_points = paddle.concat(anchor_points)
        stride_tensor = paddle.concat(stride_tensor)
        return anchor_points, stride_tensor

    #def forward_eval(self, feats, targets=None):
    def forward_eval(self, feats):
        if self.eval_size:
            anchor_points, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points, stride_tensor = self._generate_anchors(feats)
        cls_score_list, reg_dist_list = [], []
        for i, feat in enumerate(feats):
            _, _, h, w = feat.shape
            l = h * w
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            reg_dist = reg_dist.reshape(
                [-1, 4, self.reg_channels, l]).transpose([0, 2, 3, 1])
            if self.use_shared_conv:
                reg_dist = self.proj_conv(F.softmax(
                    reg_dist, axis=1)).squeeze(1)
            else:
                reg_dist = F.softmax(reg_dist, axis=1)
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([-1, self.num_classes, l]))
            reg_dist_list.append(reg_dist)

        cls_score_list = paddle.concat(cls_score_list, axis=-1)
        if self.use_shared_conv:
            reg_dist_list = paddle.concat(reg_dist_list, axis=1)
        else:
            reg_dist_list = paddle.concat(reg_dist_list, axis=2)
            reg_dist_list = self.proj_conv(reg_dist_list).squeeze(1)

        # if targets is not None:
        #     is_teacher = targets.get('is_teacher', False)
        #     if is_teacher:
        #         pred_bboxes = batch_distance2bbox(anchor_points, reg_dist_list)
        #         return cls_score_list, pred_bboxes, reg_dist_list

        return cls_score_list, reg_dist_list, anchor_points, stride_tensor

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"

        if self.training:
            return self.forward_train(feats, targets)
        else:
            if targets is not None:
                is_teacher = targets.get('is_teacher', False)
                if is_teacher:
                    return self.forward_eval(feats)

            return self.forward_eval(feats)

    @staticmethod
    def _focal_loss(score, label, alpha=0.25, gamma=2.0):
        weight = (score - label).pow(gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight *= alpha_t
        loss = F.binary_cross_entropy(
            score, label, weight=weight, reduction='sum')
        return loss

    @staticmethod
    def _varifocal_loss(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        loss = F.binary_cross_entropy(
            pred_score, gt_score, weight=weight, reduction='sum')
        return loss

    def _bbox_decode(self, anchor_points, pred_dist):
        _, l, _ = get_static_shape(pred_dist)
        tmp_pred_dist = pred_dist.reshape([-1, l, 4, self.reg_channels])
        pred_dist = F.softmax(tmp_pred_dist)  # [16, 6069, 4, 17]
        pred_dist = self.proj_conv(pred_dist.transpose([0, 3, 1, 2])).squeeze(1)
        return batch_distance2bbox(anchor_points, pred_dist), tmp_pred_dist

    def _bbox2distance(self, points, bbox):
        x1y1, x2y2 = paddle.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return paddle.concat([lt, rb], -1).clip(self.reg_range[0],
                                                self.reg_range[1] - 1 - 0.01)

    def _df_loss(self, pred_dist, target, lower_bound=0):
        target_left = paddle.cast(target.floor(), 'int64')
        target_right = target_left + 1
        weight_left = target_right.astype('float32') - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_dist, target_left - lower_bound,
            reduction='none') * weight_left
        loss_right = F.cross_entropy(
            pred_dist, target_right - lower_bound,
            reduction='none') * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

    def _bbox_loss(self, pred_dist, pred_bboxes, anchor_points, assigned_labels,
                   assigned_bboxes, assigned_scores, assigned_scores_sum):
        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.sum()
        # pos/neg loss
        if num_pos > 0:
            # l1 + iou
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
            pred_bboxes_pos = paddle.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = paddle.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = paddle.masked_select(
                assigned_scores.sum(-1), mask_positive).unsqueeze(-1)

            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)

            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum

            dist_mask = mask_positive.unsqueeze(-1).tile(
                [1, 1, self.reg_channels * 4])
            pred_dist_pos = paddle.masked_select(
                pred_dist, dist_mask).reshape([-1, 4, self.reg_channels])
            assigned_ltrb = self._bbox2distance(anchor_points, assigned_bboxes)
            assigned_ltrb_pos = paddle.masked_select(
                assigned_ltrb, bbox_mask).reshape([-1, 4])
            loss_dfl = self._df_loss(pred_dist_pos, assigned_ltrb_pos,
                                     self.reg_range[0]) * bbox_weight
            loss_dfl = loss_dfl.sum() / assigned_scores_sum
        else:
            loss_l1 = paddle.zeros([1])
            loss_iou = paddle.zeros([1])
            loss_dfl = pred_dist.sum() * 0.
        return loss_l1, loss_iou, loss_dfl

    def get_loss(self, head_outs, gt_meta):
        pred_scores, pred_distri, anchors,\
        anchor_points, num_anchors_list, stride_tensor = head_outs

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes, _ = self._bbox_decode(anchor_points_s, pred_distri)

        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        pad_gt_mask = gt_meta['pad_gt_mask']
        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.static_assigner(
                    anchors,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes,
                    pred_bboxes=pred_bboxes.detach() * stride_tensor)
            alpha_l = 0.25
        else:
            if self.sm_use:
                assigned_labels, assigned_bboxes, assigned_scores = \
                    self.assigner(
                    pred_scores.detach(),
                    pred_bboxes.detach() * stride_tensor,
                    anchor_points,
                    stride_tensor,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes)
            else:
                assigned_labels, assigned_bboxes, assigned_scores = \
                    self.assigner(
                    pred_scores.detach(),
                    pred_bboxes.detach() * stride_tensor,
                    anchor_points,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes)
            alpha_l = -1
        # rescale bbox
        assigned_bboxes /= stride_tensor
        # cls loss
        if self.use_varifocal_loss:
            one_hot_label = F.one_hot(assigned_labels,
                                      self.num_classes + 1)[..., :-1]
            loss_cls = self._varifocal_loss(pred_scores, assigned_scores,
                                            one_hot_label)
        else:
            loss_cls = self._focal_loss(pred_scores, assigned_scores, alpha_l)

        assigned_scores_sum = assigned_scores.sum()
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.all_reduce(assigned_scores_sum)
            assigned_scores_sum /= paddle.distributed.get_world_size()
        assigned_scores_sum = paddle.clip(assigned_scores_sum, min=1.)
        loss_cls /= assigned_scores_sum

        loss_l1, loss_iou, loss_dfl = \
            self._bbox_loss(pred_distri, pred_bboxes, anchor_points_s,
                            assigned_labels, assigned_bboxes, assigned_scores,
                            assigned_scores_sum)
        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl
        out_dict = {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_dfl': loss_dfl,
            'loss_l1': loss_l1,
        }
        return out_dict

    def post_process(self, head_outs, scale_factor,is_teacher=False,cls_thr=None):
        cls=[0.7971377968788147, 0.5335614681243896, 0.6894877552986145, 0.7627964615821838, 0.7426165342330933, 0.8953188061714172, 0.9076625108718872, 0.638511061668396, 0.5249035358428955, 0.5750279426574707, 0.9273083209991455, 0.9307509660720825, 0.6070082783699036, 0.5306237936019897, 0.5239234566688538, 0.8608815670013428, 0.828597366809845, 0.7883327603340149, 0.6916555762290955, 0.7202829718589783, 0.8636602163314819, 0.8950977325439453, 0.9282777905464172, 0.9328149557113647, 0.49826765060424805, 0.62161785364151, 0.4690295159816742, 0.5356476902961731, 0.5508789420127869, 0.8920686841011047, 0.44862228631973267, 0.44502493739128113, 0.6615172624588013, 0.6176568865776062, 0.7074908018112183, 0.6941214799880981, 0.6821920275688171, 0.6759849786758423, 0.846224844455719, 0.6022900342941284, 0.5386207699775696, 0.6477895379066467, 0.454955518245697, 0.44457951188087463, 0.45751553773880005, 0.6607925295829773, 0.4562230706214905, 0.4382580518722534, 0.643734872341156, 0.4389798939228058, 0.6201882362365723, 0.5130181908607483, 0.5291361212730408, 0.799691379070282, 0.6155484914779663, 0.6252705454826355, 0.6011744141578674, 0.7009831666946411, 0.5844095945358276, 0.7438904643058777, 0.6576117277145386, 0.8585709929466248, 0.8740172982215881, 0.8746117949485779, 0.868269145488739, 0.526948869228363, 0.865877091884613, 0.5768778324127197, 0.6691982746124268, 0.630863606929779, 0.44945234060287476, 0.6373603940010071, 0.8172448873519897, 0.41741517186164856, 0.8494108319282532, 0.5757522583007812, 0.452577143907547, 0.5866093635559082, 0.13069918751716614, 0.4245533049106598]
        cls_thr_ig=[0.3739641606807709, 0.3334673345088959, 0.40132248401641846, 0.3862171769142151, 0.2930634021759033, 0.4429801404476166, 0.4017512798309326, 0.4111578166484833, 0.32809436321258545, 0.35192885994911194, 0.6173939108848572, 0.4116193354129791, 0.34681227803230286, 0.3366564214229584, 0.30718058347702026, 0.4701167345046997, 0.47483521699905396, 0.39347487688064575, 0.36206990480422974, 0.44687753915786743, 0.4282834827899933, 0.5980253219604492, 0.5255855917930603, 0.5673026442527771, 0.34888601303100586, 0.35566964745521545, 0.3443540334701538, 0.2933204770088196, 0.3451060652732849, 0.5291972160339355, 0.30913156270980835, 0.2896312177181244, 0.3012586832046509, 0.35738420486450195, 0.35361531376838684, 0.32147902250289917, 0.28817325830459595, 0.37765228748321533, 0.44156426191329956, 0.37673419713974, 0.3114514648914337, 0.3805806636810303, 0.32459557056427, 0.3273971378803253, 0.3490051329135895, 0.422839492559433, 0.33451563119888306, 0.33483758568763733, 0.4393360912799835, 0.3124849200248718, 0.4114469289779663, 0.3682982325553894, 0.3674754798412323, 0.3943081498146057, 0.39800921082496643, 0.41333770751953125, 0.3911398649215698, 0.40210744738578796, 0.36725422739982605, 0.4069976806640625, 0.3998160660266876, 0.38370001316070557, 0.47709664702415466, 0.43956539034843445, 0.5180056691169739, 0.35370421409606934, 0.4759678542613983, 0.36268308758735657, 0.3677654564380646, 0.3959551751613617, 0.4062499403953552, 0.3815518319606781, 0.43728965520858765, 0.33726465702056885, 0.3941384553909302, 0.3629090487957001, 0.30963942408561707, 0.29394933581352234, 0.11910495907068253, 0.32715997099876404]
        pred_scores, pred_dist, anchor_points, stride_tensor = head_outs
        pred_bboxes = batch_distance2bbox(anchor_points, pred_dist)
        pred_bboxes *= stride_tensor
        if self.exclude_post_process:
            return paddle.concat(
                [pred_bboxes, pred_scores.transpose([0, 2, 1])], axis=-1), None
        else:
            # scale bbox to origin
            scale_y, scale_x = paddle.split(scale_factor, 2, axis=-1)
            scale_factor = paddle.concat(
                [scale_x, scale_y, scale_x, scale_y],
                axis=-1).reshape([-1, 1, 4])
            pred_bboxes /= scale_factor
            if self.exclude_nms:
                # `exclude_nms=True` just use in benchmark
                return pred_bboxes, pred_scores
            else:
                if is_teacher:
                    bbox_pred=[]
                    bbox_num=[]
                    for i in range(len(pred_bboxes)):
                        _pred, _num, _= self.nms(pred_bboxes[i].unsqueeze(0), pred_scores[i].unsqueeze(0))
                        bbox_pred.append(_pred)
                        bbox_num.append(_num)
                    # bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
                    return bbox_pred, bbox_num
                else:
                    b_mask=[]
                    # for j in range(1):
                    #     for i in range(80):
                    #         cls_thr=cls[i]
                    #         b_mask[j].append(pred_scores[j,i,:]>cls_thr)
                    #     b_mask[j]=paddle.concat([ _ for _ in b_mask[j]],axis=0).unsqueeze(0).reshape([80,-1])
                    # b_mask=b_mask[0].sum(0)>0
                    # if  b_mask.sum()==0:
                    #     pred=pred_scores.squeeze(0).max(0)
                    #     ids=paddle.argmax(pred_scores.squeeze(0).max(0))
                    #     b_mask[ids]=True
                    # pred_bboxes=pred_bboxes[b_mask.unsqueeze(0)]
                    # pred_scores=pred_scores.transpose([0,2,1])[b_mask.unsqueeze(0)].transpose([1,0])
                    # bbox_pred, bbox_num, _ = self.nms(pred_bboxes.unsqueeze(0), pred_scores.unsqueeze(0))
                    bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
                    # for j in range(len(bbox_pred)):
                    #     cls_thr=cls[bbox_pred[j][0]]
                    #     b_mask.append(bbox_pred[j][1]>cls_thr)    
                    # b_mask=paddle.concat([_ for _ in b_mask])       
                    # return bbox_pred[b_mask],b_mask.sum()
                    return bbox_pred, bbox_num


#先卡阈值再nms
                    # for j in range(1):
                    #     for i in range(80):
                    #         cls_thr=cls[i]
                    #         b_mask[j].append(pred_scores[j,i,:]>cls_thr)
                    #     b_mask[j]=paddle.concat([ _ for _ in b_mask[j]],axis=0).unsqueeze(0).reshape([80,-1])
                    # b_mask=b_mask[0].sum(0)>0
                    # if  b_mask.sum()==0:
                    #     pred=pred_scores.squeeze(0).max(0)
                    #     ids=paddle.argmax(pred_scores.squeeze(0).max(0))
                    #     b_mask[ids]=True
                    # pred_bboxes=pred_bboxes[b_mask.unsqueeze(0)]
                    # pred_scores=pred_scores.transpose([0,2,1])[b_mask.unsqueeze(0)].transpose([1,0])
                    # bbox_pred, bbox_num, _ = self.nms(pred_bboxes.unsqueeze(0), pred_scores.unsqueeze(0))