import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Normal, Constant

from ppdet.modeling.ops import get_static_shape
from ppdet.modeling.assigners.utils import generate_anchors_for_grid_cell
from ppdet.modeling.bbox_utils import bbox_center, batch_distance2bbox, bbox2distance

from ppdet.utils.checkpoint import load_pretrain_weight
from ppdet.core.workspace import register, create, load_config
from ppdet.utils.logger import setup_logger
from ppdet.slim.pgd_distill import PGDMask

logger = setup_logger(__name__)

eps = 1e-10


def get_single_level_center_point(featmap_size, stride,
                                    cell_offset=0):
    """
    Generate pixel centers of a single stage feature map.
    Args:
        featmap_size: height and width of the feature map
        stride: down sample stride of the feature map
    Returns:
        y and x of the center points
    """
    h, w = featmap_size
    x_range = (paddle.arange(w, dtype='float32') + cell_offset) * stride
    y_range = (paddle.arange(h, dtype='float32') + cell_offset) * stride
    y, x = paddle.meshgrid(y_range, x_range)
    y = y.flatten()
    x = x.flatten()
    return y, x


class PPYOLOEPGDMask(nn.Layer):
    def __init__(self, cls_alpha, cls_topk, reg_alpha, reg_topk):
        super(PPYOLOEPGDMask, self).__init__()
        self.pgd_cls_assign = PGDMask(cls_alpha, cls_topk)
        self.pgd_reg_assign = PGDMask(reg_alpha, reg_topk)
        self.cell_offset = 0.

    def get_pgd_mask(self, head, head_outs, inputs, fpn_feats, static_assign=True):
        self.head = head  # get model head
        # cls_score_list, reg_list, box_list, fpn_feats = head_outs
        cls_score_list, reg_dist_list, anchor_points, stride_tensor = head_outs # ppyoloe headouts
        box_list = batch_distance2bbox(anchor_points, reg_dist_list)
        box_list *= stride_tensor
        pred_bboxes = box_list.detach()

        num_levels = len(fpn_feats)
        batch_size = fpn_feats[0].shape[0]

        kd_value_cls_maps = [[] for _ in range(num_levels)]
        kd_value_reg_maps = [[] for _ in range(num_levels)]
        kd_backmaps = [[] for _ in range(num_levels)]
        gt_bboxes = inputs['gt_bbox']
        gt_labels = inputs['gt_class']

        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                fpn_feats, self.head.fpn_strides, self.head.grid_cell_scale,
                self.head.grid_cell_offset)
        
        num_per_level = num_anchors_list
        # head_outs
        # important
        anchors = paddle.tile(anchors, [batch_size,1,1])

        if static_assign is False:  # get center list from cls_score prediction
            center_list = []
            num_per_level = []
            for i, (fpn_feat,
                    stride) in enumerate(zip(fpn_feats, self.head.fpn_stride)):
                b, _, h, w = get_static_shape(fpn_feat)
                y, x = get_single_level_center_point(
                    [h, w], stride, cell_offset=self.cell_offset)
                center_points = paddle.stack([x, y], axis=-1)
                center_list.append(center_points)
                num_per_level.append(center_points.shape[0])

            center_list = paddle.concat(center_list, axis=0)
            center_list = paddle.tile(center_list, [batch_size,1,1]) 
        
        for i in range(batch_size):
            if static_assign:
                pgd_cls_map_each, bg_mask = self.pgd_cls_assign.pgw_anchor_based_assign(
                    anchors[i], cls_score_list[i], pred_bboxes[i], gt_bboxes[i],
                    gt_labels[i], None)
                pgd_reg_map_each, _ = self.pgd_reg_assign.pgw_anchor_based_assign(
                    anchors[i], cls_score_list[i], pred_bboxes[i], gt_bboxes[i],
                    gt_labels[i], None)
            else:
                # for anchor free
                pgd_cls_map_each, bg_mask = self.pgd_cls_assign.pgw_anchor_free_assign(
                    center_list[i], cls_score_list[i], pred_bboxes[i],
                    gt_bboxes[i], gt_labels[i], None)
                pgd_reg_map_each, _ = self.pgd_reg_assign.pgw_anchor_free_assign(
                    center_list[i], cls_score_list[i], pred_bboxes[i],
                    gt_bboxes[i], gt_labels[i], None)

            for l in range(len(num_per_level)):
                kd_value_cls_level = pgd_cls_map_each[sum(
                    num_per_level[:l]):sum(num_per_level[:(l + 1)])]
                kd_value_reg_level = pgd_reg_map_each[sum(
                    num_per_level[:l]):sum(num_per_level[:(l + 1)])]

                bg_mask_level = bg_mask[sum(num_per_level[:l]):sum(
                    num_per_level[:(l + 1)])]
                kd_value_cls_maps[l].append(kd_value_cls_level)
                kd_value_reg_maps[l].append(kd_value_reg_level)
                kd_backmaps[l].append(bg_mask_level)

        kd_value_cls_maps = [
            paddle.concat(
                x, axis=0) for x in kd_value_cls_maps
        ]
        kd_value_reg_maps = [
            paddle.concat(
                x, axis=0) for x in kd_value_reg_maps
        ]
        kd_backmaps = [paddle.concat(x, axis=0) for x in kd_backmaps]

        return kd_value_cls_maps, kd_value_reg_maps, kd_backmaps


class PPYOLOEPGDDistill(nn.Layer):
    def __init__(self, cfg, slim_cfg):
        super(PPYOLOEPGDDistill, self).__init__()

        self.is_inherit = False
        # build student model before load slim config
        self.student_model = create(cfg.architecture)
        self.arch = cfg.architecture
        stu_pretrain = cfg['pretrain_weights']
        slim_cfg = load_config(slim_cfg)
        self.teacher_cfg = slim_cfg
        self.loss_cfg = slim_cfg
        #tea_pretrain = cfg['pretrain_weights']

        self.teacher_model = create(self.teacher_cfg.architecture)
        self.teacher_model.eval()

        for param in self.teacher_model.parameters():
            param.trainable = False

        if 'pretrain_weights' in cfg and stu_pretrain:
            if self.is_inherit and 'pretrain_weights' in self.teacher_cfg and self.teacher_cfg.pretrain_weights:
                load_pretrain_weight(self.student_model,
                                     self.teacher_cfg.pretrain_weights)
                logger.debug(
                    "Inheriting! loading teacher weights to student model!")

            load_pretrain_weight(self.student_model, stu_pretrain)

        if 'pretrain_weights' in self.teacher_cfg and self.teacher_cfg.pretrain_weights:
            load_pretrain_weight(self.teacher_model,
                                 self.teacher_cfg.pretrain_weights)

        #build cls_loss and reg_loss
        self.distill_cls_loss, self.distill_reg_loss = self.build_loss(self.loss_cfg)

        #get mask parameters from config
        # topk take from student head. assigner
        cls_alpha = self.loss_cfg.distill_loss['cls_alpha']
        reg_alpha =  self.loss_cfg.distill_loss['reg_alpha']
        cls_topk = self.loss_cfg.distill_loss['cls_topk']
        reg_topk = self.loss_cfg.distill_loss['reg_topk']

        self.pgd_mask = PPYOLOEPGDMask(cls_alpha, cls_topk, reg_alpha,
                                         reg_topk)

        # build align layer
        _channels = [768, 384, 192]
        # _s, _m, _l, _x = 0.5, 0.75, 1.0, 1.25
        teacher_width_mult = self.loss_cfg['teacher_width_mult']
        student_width_mult = self.loss_cfg['student_width_mult']
        num_level = 3
        if student_width_mult != teacher_width_mult:
            self.cls_align_layer = nn.LayerList([
                nn.Conv2D(int(_channels[i] * student_width_mult), int(_channels[i] * teacher_width_mult), 1) for i in range(num_level)])
            self.reg_align_layer = nn.LayerList([
                nn.Conv2D(int(_channels[i] * student_width_mult), int(_channels[i] * teacher_width_mult), 1) for i in range(num_level)])
        else:
            self.cls_align_layer = None
            self.reg_align_layer = None

        self.is_norm = False

    def build_loss(self, cfg):
        #build cls_loss and reg_loss
        distill_cls_loss_list = []
        distill_reg_loss_list = []

        for loss_weight in cfg['distill_loss']['loss_weight']:
            cfg[cfg['distill_loss']['loss_cls']]['loss_weight'] = loss_weight
            distill_cls_loss = create(cfg['distill_loss']['loss_cls'])
            distill_reg_loss = create(cfg['distill_loss']['loss_reg'])
            distill_cls_loss_list.append(distill_cls_loss)
            distill_reg_loss_list.append(distill_reg_loss)
        return distill_cls_loss_list, distill_reg_loss_list

    def stu_forward_loss(self, stu_head, feats, targets):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, stu_head.fpn_strides, stu_head.grid_cell_scale,
                stu_head.grid_cell_offset)

        cls_score_list, reg_distri_list = [], []
        d_cls_list, d_reg_list = [], []
        for i, feat in enumerate(feats):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            # collect distill feature which is the feature before last conv
            d_cls_feat = stu_head.stem_cls[i](feat, avg_feat) + feat
            d_reg_feat = stu_head.stem_reg[i](feat, avg_feat)
            d_cls_list.append(d_cls_feat), d_reg_list.append(d_reg_feat)

            cls_logit = stu_head.pred_cls[i](d_cls_feat)
            reg_distri = stu_head.pred_reg[i](d_reg_feat)
            # cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
            #                              feat)
            # reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
            reg_distri_list.append(reg_distri.flatten(2).transpose([0, 2, 1]))
        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_distri_list = paddle.concat(reg_distri_list, axis=1)

        stu_loss = stu_head.get_loss([
            cls_score_list, reg_distri_list, anchors, anchor_points,
            num_anchors_list, stride_tensor
        ], targets)    

        return stu_loss, d_cls_list, d_reg_list
    
    def tea_forward_head(self, tea_head, feats, targets):
        # if tea_head.eval_size:
        #     anchor_points, stride_tensor = tea_head.anchor_points, tea_head.stride_tensor
        # else:
        anchor_points, stride_tensor = tea_head._generate_anchors(feats)
        cls_score_list, reg_dist_list = [], []
        d_cls_list, d_reg_list = [], []
        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            l = h * w
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            d_cls_feat = tea_head.stem_cls[i](feat, avg_feat) + feat
            d_reg_feat = tea_head.stem_reg[i](feat, avg_feat)
            d_cls_list.append(d_cls_feat)
            d_reg_list.append(d_reg_feat)
            cls_logit = tea_head.pred_cls[i](d_cls_feat)
            reg_dist = tea_head.pred_reg[i](d_reg_feat)
            # cls_logit = tea_head.pred_cls[i](tea_head.stem_cls[i](feat, avg_feat) +
            #                              feat)
            # reg_dist = tea_head.pred_reg[i](tea_head.stem_reg[i](feat, avg_feat))
            reg_dist = reg_dist.reshape([-1, 4, tea_head.reg_channels, l]).transpose(
                [0, 2, 3, 1])
            reg_dist = tea_head.proj_conv(F.softmax(reg_dist, axis=1)).squeeze(1)
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([b, tea_head.num_classes, l]))
            reg_dist_list.append(reg_dist)

        cls_score_list = paddle.concat(cls_score_list, axis=-1)
        reg_dist_list = paddle.concat(reg_dist_list, axis=1)
        # print("====>debug")
        # print("reg_dist_list.shape")
        tea_head_outs = [cls_score_list, reg_dist_list, anchor_points, stride_tensor]
        # for v in tea_head_outs:
        #     print(v.shape)
        return tea_head_outs, d_cls_list, d_reg_list

    def distill_loss(self, fpn_feats, s_cls_list, t_cls_list, s_reg_list, t_reg_list, tea_head, tea_head_outs, inputs, static_assign):
        #  get_pgd_mask(self, head, head_outs, inputs, fpn_feats, static_assign=True):
        kd_value_cls_maps, kd_value_reg_maps, kd_backmaps = self.pgd_mask.get_pgd_mask(
            tea_head, tea_head_outs, inputs, fpn_feats, static_assign)

        assert len(kd_value_cls_maps) == len(kd_value_reg_maps) == len(
            kd_backmaps), "ERROR"

        # build cls loss
        distill_cls_loss = {}
        distill_reg_loss = {}

        for idx, (s_cls_f, t_cls_f, cls_mask, bg_mask) in enumerate(zip(s_cls_list, t_cls_list, kd_value_cls_maps, kd_backmaps)):
            # print("cls_align===>, ", idx, s_cls_f.shape, t_cls_f.shape)
            if self.cls_align_layer is not None:
                s_cls_f = self.cls_align_layer[idx](s_cls_f)
            if self.is_norm is True:
                s_cls_f, t_cls_f = self.norm(s_cls_f), self.norm(t_cls_f)

            b, c, h, w = s_cls_f.shape
            distill_cls_loss[f'pgd_cls_{idx}'] =  self.distill_cls_loss[idx](
                s_cls_f, 
                t_cls_f, 
                cls_mask.reshape((b,1,h,w)), 
                bg_mask.reshape((b,1,h,w)).astype('float32')
                )
    
        for idx, (s_reg_f, t_reg_f, reg_mask, bg_mask) in enumerate(zip(s_reg_list, t_reg_list, kd_value_reg_maps, kd_backmaps)):
            # print("reg_align===>, ",idx,  s_reg_f.shape, t_reg_f.shape)
            if self.reg_align_layer is not None:
                s_reg_f = self.reg_align_layer[idx](s_reg_f)
            
            if self.is_norm is True:
                s_reg_f, t_reg_f = self.norm(s_reg_f), self.norm(t_reg_f)
            
            b, c, h, w = s_reg_f.shape
            distill_reg_loss[f'pgd_reg_{idx}'] = self.distill_reg_loss[idx](
                s_reg_f, 
                t_reg_f, 
                reg_mask.reshape((b,1,h,w))
                )
        
        return distill_cls_loss, distill_reg_loss

    def forward(self, inputs):
        if self.training:
            with paddle.no_grad():
                t_body_feats = self.teacher_model.backbone(inputs)
                t_neck_feats = self.teacher_model.neck(t_body_feats)
                t_head_outs, t_cls_list, t_reg_list = self.tea_forward_head(self.teacher_model.yolo_head, t_neck_feats, inputs)
            
            static_assign = bool(
                    inputs['epoch_id'] <
                    self.student_model.yolo_head.static_assigner_epoch)

            s_body_feats = self.student_model.backbone(inputs)
            s_neck_feats = self.student_model.neck(s_body_feats)
            stu_loss, s_cls_list, s_reg_list = self.stu_forward_loss(self.student_model.yolo_head, s_neck_feats, inputs)

            distill_cls_loss, distill_reg_loss = self.distill_loss(t_neck_feats, s_cls_list, t_cls_list, s_reg_list, t_reg_list, self.teacher_model.yolo_head, t_head_outs, inputs, static_assign)

            stu_loss['pgd_cls_loss'] = paddle.add_n(list(distill_cls_loss.values()))
            stu_loss['pgd_reg_loss'] = paddle.add_n(list(distill_reg_loss.values()))

            stu_loss['loss'] += stu_loss['pgd_cls_loss']  + stu_loss['pgd_reg_loss'] 
            return stu_loss
        else:
            body_feats = self.student_model.backbone(inputs)
            neck_feats = self.student_model.neck(body_feats)
            yolo_head_outs = self.student_model.yolo_head(neck_feats)
            if self.student_model.return_idx:
                _, bbox, bbox_num, _ = self.student_model.post_process(
                    yolo_head_outs,
                    self.student_model.yolo_head.mask_anchors)
            elif self.student_model.post_process is not None:
                bbox, bbox_num = self.student_model.post_process(
                    yolo_head_outs,
                    self.student_model.yolo_head.mask_anchors,
                    inputs['im_shape'], inputs['scale_factor'])
            else:
                bbox, bbox_num = self.student_model.yolo_head.post_process(
                    yolo_head_outs, inputs['scale_factor'])
            output = {'bbox': bbox, 'bbox_num': bbox_num}
            return output

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
