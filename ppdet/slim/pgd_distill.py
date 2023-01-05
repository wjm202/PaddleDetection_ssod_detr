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

logger = setup_logger(__name__)

eps = 1e-10


@register
class PGDClsLoss(nn.Layer):
    def __init__(self,
                 temp_s,
                 temp_c,
                 loss_weight,
                 alpha,
                 beta,
                 delta,
                 **kwargs,
                 ):
        super(PGDClsLoss, self).__init__()
        self.temp_s = temp_s
        self.temp_c = temp_c
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

    def forward(self,
                preds_S,
                preds_T,
                mask_fg,
                mask_bg,
                **kwargs):
        assert preds_S.shape[-2:] == preds_T.shape[-2:], 'the output dim of teacher and student differ'

        mask_bg = mask_bg / (mask_bg != 0).cast(dtype=preds_S.dtype).sum(axis=(2, 3), keepdim=True).clip(min=eps)
        mask_fg = mask_fg / (mask_fg != 0).cast(dtype=preds_S.dtype).sum(axis=(2, 3), keepdim=True).clip(min=eps)

        S_attention_t, C_attention_t = self.get_attention(preds_T)
        S_attention_s, C_attention_s = self.get_attention(preds_S)

        fg_loss, bg_loss = self.get_fea_loss(preds_S, preds_T,
                                             mask_fg, mask_bg,
                                             C_attention_s, C_attention_t,
                                             S_attention_s, S_attention_t)
        mask_loss = self.get_mask_loss(C_attention_s, C_attention_t,
                                       S_attention_s, S_attention_t)
        # print("PGD cls_loss: ", self.loss_weight * self.alpha * fg_loss.numpy(), self.loss_weight * self.beta * bg_loss.numpy(), self.delta * mask_loss.numpy())
        loss = self.loss_weight * self.alpha * fg_loss + \
               self.loss_weight * self.beta * bg_loss + \
               self.delta * mask_loss
        return loss

    def get_attention(self, preds):
        """ preds: Bs*C*W*H """
        N, C, H, W = preds.shape
        value = paddle.abs(preds)

        # Bs*W*H
        fea_map = value.mean(axis=1, keepdim=True)
        S_attention = (H * W * F.softmax((fea_map / self.temp_s).reshape((N, -1)), axis=1)).reshape((N, H, W))

        # Bs*C
        C_attention = C * F.softmax(value / self.temp_c, axis=1)

        return S_attention, C_attention

    def get_fea_loss(self, preds_S, preds_T, Mask_fg, Mask_bg, C_s, C_t, S_s, S_t):
        # 这一块很占显存，新增6-7G
        loss_mse = nn.MSELoss(reduction='sum')

        Mask_fg = Mask_fg.unsqueeze(axis=1)
        Mask_bg = Mask_bg.unsqueeze(axis=1)

        S_t = S_t.unsqueeze(axis=1)

        # 主要是这一块吧 好像也优化不了 大矩阵相乘
        fea_t = paddle.multiply(preds_T, paddle.sqrt(S_t))
        fea_t = paddle.multiply(fea_t, paddle.sqrt(C_t))
        fg_fea_t = paddle.multiply(fea_t, paddle.sqrt(Mask_fg))
        bg_fea_t = paddle.multiply(fea_t, paddle.sqrt(Mask_bg))

        fea_s = paddle.multiply(preds_S, paddle.sqrt(S_t))
        fea_s = paddle.multiply(fea_s, paddle.sqrt(C_t))
        fg_fea_s = paddle.multiply(fea_s, paddle.sqrt(Mask_fg))
        bg_fea_s = paddle.multiply(fea_s, paddle.sqrt(Mask_bg))

        fg_loss = loss_mse(fg_fea_s, fg_fea_t) / len(Mask_fg)
        bg_loss = loss_mse(bg_fea_s, bg_fea_t) / len(Mask_bg)

        return fg_loss, bg_loss

    def get_mask_loss(self, C_s, C_t, S_s, S_t):

        mask_loss = paddle.sum(paddle.abs((C_s - C_t))) / (C_s.shape[0] * C_s.shape[-2] * C_s.shape[-1]) \
                    + paddle.sum(paddle.abs((S_s - S_t))) / S_t.shape[0]

        return mask_loss


@register
class PGDRegLoss(nn.Layer):
    def __init__(self,
                 temp,
                 gamma,
                 delta,
                 **kwargs
                 ):
        super(PGDRegLoss, self).__init__()
        self.temp = temp
        self.gamma = gamma
        self.delta = delta

    def forward(self,
                preds_S,
                preds_T,
                mask_fg,
                **kwargs):
        
        assert preds_S.shape[-2:] == preds_T.shape[-2:], 'the output dim of teacher and student differ'
        
        N, C, H, W = preds_S.shape

        preds_S = preds_S.transpose((0, 2, 3, 1)).reshape([-1, C])
        preds_T = preds_T.transpose((0, 2, 3, 1)).reshape([-1, C])
        target_map = mask_fg.reshape([-1])

        # pos_inds = target_map != 0
        # preds_S = preds_S[pos_inds]
        # preds_T = preds_T[pos_inds]
        # target_w = target_map[pos_inds]

        # NOTE: to support backward
        pos_inds =  paddle.nonzero(target_map != 0).squeeze(1)
        preds_S = paddle.gather(preds_S, pos_inds, axis=0)
        preds_T = paddle.gather(preds_T, pos_inds, axis=0)
        target_w = paddle.gather(target_map, pos_inds, axis=0)


        if preds_S.numel() > 0:
            C_attention_s = C * F.softmax(paddle.abs(preds_S) / self.temp, axis=1)
            C_attention_t = C * F.softmax(paddle.abs(preds_T) / self.temp, axis=1)

            fea_loss = F.mse_loss(preds_S, preds_T, reduction='none') * C_attention_t
            fea_loss = (fea_loss.mean(axis=1) * target_w).mean()

            mask_loss = paddle.sum(paddle.abs((C_attention_s - C_attention_t))) / C_attention_s.shape[0]
            # print("pGD reg loss: ", self.gamma * fea_loss, self.delta * mask_loss)
            return self.gamma * fea_loss + self.delta * mask_loss

        else:
            return preds_S.sum() * 0.

def iou_similarity(box1, box2, eps=1e-10):
    """Calculate iou of box1 and box2
    Args:
        box1 (Tensor): box with the shape [M1, 4]
        box2 (Tensor): box with the shape [M2, 4]
    Return:
        iou (Tensor): iou between box1 and box2 with the shape [M1, M2]
    """
    box1 = box1.unsqueeze(1)  # [M1, 4] -> [M1, 1, 4]
    box2 = box2.unsqueeze(0)  # [M2, 4] -> [1, M2, 4]
    px1y1, px2y2 = box1[:, :, 0:2], box1[:, :, 2:4]
    gx1y1, gx2y2 = box2[:, :, 0:2], box2[:, :, 2:4]
    x1y1 = paddle.maximum(px1y1, gx1y1)
    x2y2 = paddle.minimum(px2y2, gx2y2)
    overlap = (x2y2 - x1y1).clip(0).prod(-1)
    area1 = (px2y2 - px1y1).clip(0).prod(-1)
    area2 = (gx2y2 - gx1y1).clip(0).prod(-1)
    union = area1 + area2 - overlap + eps
    return overlap / union


def repeat(x, repeats, axis=0):
    return paddle.concat([x] * repeats, axis=axis)


def mle_2d_gaussian_2(sample_data):
    """
    from : https://github.com/ChenhongyiYang/PGD/blob/HEAD/mmdet/models/dense_heads/pgd_heads/utils/pgw_anchor_based.py
    """

    data = sample_data + (paddle.randn(sample_data.shape) - 0.5) * 0.1
    miu = data.mean(axis=1, keepdim=True)  #[N, 1, 2]
    diff = (data - miu)[:, :, :, None]
    sigma = paddle.matmul(diff, diff.transpose([0, 1, 3, 2])).mean(axis=1)
    deter = sigma[:, 0, 0] * sigma[:, 1, 1] - sigma[:, 0, 1] * sigma[:, 1, 0]

    inverse = paddle.zeros_like(sigma)
    inverse[:, 0, 0] = sigma[:, 1, 1]
    inverse[:, 0, 1] = -1. * sigma[:, 0, 1]
    inverse[:, 1, 0] = -1. * sigma[:, 1, 0]
    inverse[:, 1, 1] = sigma[:, 0, 0]
    inverse /= (deter[:, None, None] + 1e-10)

    return miu, sigma, inverse, deter


class PGDMask(object):
    def __init__(self, alpha, topk):
        self.alpha = alpha
        self.topk = topk

    def pgw_anchor_free_assign(self,
                               points,
                               cls_scores,
                               bbox_preds,
                               gt_bboxes,
                               gt_labels,
                               gt_bboxes_ignore=None):
        bbox_preds = bbox_preds.detach()
        cls_scores = cls_scores.detach()

        num_gt, num_points = gt_bboxes.shape[0], points.shape[0]
        num_gt = 0
        for i in range(gt_bboxes.shape[0]):
            gt = gt_bboxes[i]
            if paddle.sum(paddle.abs(gt)) > 1e-5:
                num_gt += 1

        gt_bboxes = gt_bboxes[:num_gt]
        gt_labels = gt_labels[:num_gt]

        if num_gt == 0 or num_points == 0:
            return paddle.zeros((num_points, ))

        overlaps = iou_similarity(bbox_preds, gt_bboxes)
        cls_cost = paddle.index_select(cls_scores, gt_labels, axis=1)

        # quality score
        overlaps = cls_cost**(1 - self.alpha) * overlaps**self.alpha

        assigned_gt_inds = paddle.zeros(
            [cls_scores.shape[0]], dtype='int32'
        )  # overlaps.new_full((num_points, ), 0, dtype=torch.long)
        cx = repeat(points[:, 0].reshape([-1, 1]), num_gt, axis=-1)
        cy = repeat(points[:, 1].reshape([-1, 1]), num_gt, axis=-1)

        gx1 = repeat(
            paddle.reshape(gt_bboxes[:, 0], [1, -1]), num_points, axis=0)
        gy1 = repeat(
            paddle.reshape(gt_bboxes[:, 1], [1, -1]), num_points, axis=0)
        gx2 = repeat(
            paddle.reshape(gt_bboxes[:, 2], [1, -1]), num_points, axis=0)
        gy2 = repeat(
            paddle.reshape(gt_bboxes[:, 3], [1, -1]), num_points, axis=0)

        valid = ((cx - gx1) > eps) * ((cy - gy1) > eps) * ((gx2 - cx) > eps) * (
            (gy2 - cy) > eps)
        in_box = paddle.sum(valid, axis=1)
        back_mask = paddle.where(in_box < 1, paddle.to_tensor(1), paddle.to_tensor(0))
        topk = self.topk
        _, topk_idxs = overlaps.topk(
            topk, axis=0, largest=True)  # [topk, num_gt]
        candidate_idxs = topk_idxs

        points = points[:, ::-1]
        candidate_points = paddle.index_select(
            points, paddle.reshape(candidate_idxs, [-1])).reshape(
                [topk, num_gt, 2]).transpose([1, 0, 2])

        miu, sigma, inverse, deter = mle_2d_gaussian_2(candidate_points)

        pos_diff = (candidate_points - miu)[:, :,
                                            None, :]  # [num_gt, topk, 1, 2]
        candidate_w = paddle.exp(-0.5 * paddle.matmul(
            paddle.matmul(pos_diff, inverse[:, None, :, :]),
            pos_diff.transpose([0, 1, 3, 2])))  # [num_gt, topk]

        candidate_w = candidate_w.reshape([num_gt, topk]).transpose([1, 0])

        w = paddle.zeros_like(points[:, 0]).reshape([-1, 1])
        w = repeat(w, num_gt, axis=1)

        for i in range(num_gt):
            w[candidate_idxs[:, i], i] = candidate_w[:, i]

        w = w * valid
        w = paddle.max(w, axis=1)

        low_bound = 0.
        w[w < low_bound] = 0.

        return w, back_mask

    def pgw_anchor_based_assign(self,
                                anchors,
                                cls_scores,
                                bbox_preds,
                                gt_bboxes,
                                gt_labels,
                                bbox_levels=None,
                                ):

        bbox_preds = bbox_preds.detach()
        cls_scores = cls_scores.detach()

        if cls_scores.shape[0] == 80:
            cls_scores = cls_scores.transpose([1, 0])
        num_gt, num_bboxes = gt_bboxes.shape[0], anchors.shape[0]
        num_gt = 0
        for i in range(gt_bboxes.shape[0]):
            gt = gt_bboxes[i]
            if paddle.sum(paddle.abs(gt)) > 1e-5:
                num_gt += 1
        gt_bboxes = gt_bboxes[:num_gt]
        gt_labels = gt_labels[:num_gt]

        if num_gt == 0 or num_bboxes == 0:
            return paddle.zeros((num_bboxes, ))

        overlaps = iou_similarity(bbox_preds, gt_bboxes)  # [num_anchors,  num_gt]

        cls_cost = paddle.index_select(cls_scores, gt_labels, axis=1)

        # quality score
        overlaps = cls_cost**(1 - self.alpha) * overlaps**self.alpha

        assigned_gt_inds = paddle.zeros(
            [cls_scores.shape[0]], dtype='int32'
        )  # overlaps.new_full((num_points, ), 0, dtype=torch.long)

        anchor_cx = (anchors[:, 0] + anchors[:, 2]) / 2.0
        anchor_cy = (anchors[:, 1] + anchors[:, 3]) / 2.0

        _, topk_idxs = overlaps.topk(
            self.topk, axis=0, largest=True)  # [topk, num_gt]

        candidate_cx = paddle.index_select(
            anchor_cx, paddle.reshape(topk_idxs, [-1])).reshape(
                topk_idxs.shape)  # [topk, num_gt]
        candidate_cy = paddle.index_select(
            anchor_cy, paddle.reshape(
                topk_idxs, [-1])).reshape(topk_idxs.shape) #[self.topk, num_gt]

        candidate_pos = paddle.stack(
            (candidate_cx.transpose([1, 0]), candidate_cy.transpose([1, 0])),
            axis=-1)  # [num_gt, top_k, 2]

        if self.topk != 1:
            miu, sigma, inverse, deter = mle_2d_gaussian_2(candidate_pos)

        x1 = repeat(anchors[:, 0][:, None], num_gt, axis=1)  # [n_bbox, n_gt]
        y1 = repeat(anchors[:, 1][:, None], num_gt, axis=1)
        x2 = repeat(anchors[:, 2][:, None], num_gt, axis=1)
        y2 = repeat(anchors[:, 3][:, None], num_gt, axis=1)

        cx = (x1 + x2) * 0.5  # [n_bbox, n_gt]
        cy = (y1 + y2) * 0.5  # [n_bbox, n_gt]

        gx1 = repeat(
            paddle.reshape(gt_bboxes[:, 0], [1, -1]), num_bboxes, axis=0)
        gy1 = repeat(
            paddle.reshape(gt_bboxes[:, 1], [1, -1]), num_bboxes, axis=0)
        gx2 = repeat(
            paddle.reshape(gt_bboxes[:, 2], [1, -1]), num_bboxes, axis=0)
        gy2 = repeat(
            paddle.reshape(gt_bboxes[:, 3], [1, -1]), num_bboxes, axis=0)

        valid = ((cx - gx1) > eps) * ((cy - gy1) > eps) * ((gx2 - cx) > eps) * (
            (gy2 - cy) > eps)
        in_box = paddle.sum(valid, axis=1)
        back_mask = paddle.where(in_box < 1, paddle.to_tensor(1), paddle.to_tensor(0))

        pos_diff = (candidate_pos - miu)[:, :, None, :]  # [num_gt, topk, 1, 2]
        candidate_w = paddle.exp(-0.5 * paddle.matmul(
            paddle.matmul(pos_diff, inverse[:, None, :, :]),
            pos_diff.transpose([0, 1, 3, 2])))  # [num_gt, topk]

        candidate_w = candidate_w.reshape([num_gt, self.topk]).transpose([1, 0])

        w = paddle.zeros_like(anchor_cx).reshape([-1, 1])
        w = repeat(w, num_gt, axis=1)

        for i in range(num_gt):
            w[topk_idxs[:, i], i] = candidate_w[:, i]

        w = w * valid
        w = paddle.max(w, axis=1)
        # w[assigned_gt_inds == -1] = 0.
        low_bound = 0.
        w[w < low_bound] = 0.

        return w, back_mask


class PicodetV2PGDMask(nn.Layer):
    def __init__(self, cls_alpha, cls_topk, reg_alpha, reg_topk):
        super(PicodetV2PGDMask, self).__init__()
        self.pgd_cls_assign = PGDMask(cls_alpha, cls_topk)
        self.pgd_reg_assign = PGDMask(reg_alpha, reg_topk)

    def get_pgd_mask(self, head, head_outs, inputs, static_assign=True):
        self.head = head  # get model head
        cls_score_list, reg_list, box_list, fpn_feats = head_outs

        num_levels = len(fpn_feats)
        batch_size = fpn_feats[0].shape[0]

        kd_value_cls_maps = [[] for _ in range(num_levels)]
        kd_value_reg_maps = [[] for _ in range(num_levels)]
        kd_backmaps = [[] for _ in range(num_levels)]
        gt_bboxes = inputs['gt_bbox']
        gt_labels = inputs['gt_class']

        anchors, _, num_anchors_list, stride_tensor_list = generate_anchors_for_grid_cell(
            fpn_feats, self.head.fpn_stride, self.head.grid_cell_scale,
            self.head.cell_offset)
        num_per_level = num_anchors_list
        # head_outs
        pred_bboxes = box_list.detach() * stride_tensor_list  # important
        anchors = paddle.tile(anchors, [batch_size,1,1])

        if static_assign is False:  # get center list from cls_score prediction
            center_list = []
            num_per_level = []
            for i, (fpn_feat,
                    stride) in enumerate(zip(fpn_feats, self.head.fpn_stride)):
                b, _, h, w = get_static_shape(fpn_feat)
                y, x = self.head.get_single_level_center_point(
                    [h, w], stride, cell_offset=self.head.cell_offset)
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


class PicodetV2PGDDistill(nn.Layer):
    def __init__(self, cfg, slim_cfg):
        super(PicodetV2PGDDistill, self).__init__()

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

        self.loss_name = []
        for i in range(len(self.loss_cfg.distill_loss['loss_weight'])):
            self.loss_name.append('loss_name_' + str(i))

        self.pgd_mask = PicodetV2PGDMask(cls_alpha, cls_topk, reg_alpha,
                                         reg_topk)

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

    def forward_head(self, head, fpn_feats):
        cls_logit_list, reg_pred_list = [], []
        cls_score_list, reg_list, box_list = [], [], []

        for i, (fpn_feat,
                stride) in enumerate(zip(fpn_feats, head.fpn_stride)):
            b, _, h, w = get_static_shape(fpn_feat)
            # task decomposition
            conv_cls_feat, se_feat = head.conv_feat(fpn_feat, i)

            # NOTE: add
            cls_logit_list.append(se_feat)
            reg_pred_list.append(se_feat)

            cls_logit = head.head_cls_list[i](se_feat)
            reg_pred = head.head_reg_list[i](se_feat)
            if head.use_align_head: 
                cls_prob = F.sigmoid(head.cls_align[i](conv_cls_feat))
                cls_score = (F.sigmoid(cls_logit) * cls_prob + eps).sqrt() 
            else:
                cls_score = F.sigmoid(cls_logit)

            # # add
            # cls_logit_list.append(cls_score)
            # reg_pred_list.append(reg_pred)

            cls_score_out = cls_score.transpose([0, 2, 3, 1])
            bbox_pred = reg_pred.transpose([0, 2, 3, 1])
            b, cell_h, cell_w, _ = paddle.shape(cls_score_out) 
            y, x = head.get_single_level_center_point(
                [cell_h, cell_w], stride, cell_offset=head.cell_offset)            
            center_points = paddle.stack([x, y], axis=-1)
            cls_score_out = cls_score_out.reshape(
                [b, -1, head.cls_out_channels])
            bbox_pred = head.distribution_project(bbox_pred) * stride
            bbox_pred = bbox_pred.reshape([b, cell_h * cell_w, 4])
            bbox_pred = batch_distance2bbox(
                center_points, bbox_pred, max_shapes=None)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1])) #蒸馏用到的分类logit
            reg_list.append(reg_pred.flatten(2).transpose([0, 2, 1])) #蒸馏用到的回归logit
            box_list.append(bbox_pred / stride) #xyxy坐标映射回特征图，预测的bbox坐标

        cls_score_list = paddle.concat(cls_score_list, axis=1)
        box_list = paddle.concat(box_list, axis=1)
        reg_list = paddle.concat(reg_list, axis=1)

        head_outs = tuple((cls_score_list, reg_list, box_list, fpn_feats))
        return cls_logit_list, reg_pred_list, head_outs


    def get_distill_loss(self, stu_fpn_feats, tea_fpn_feats, stu_head, tea_head,
                         inputs, static_assign):
        
        # get student distill feature
        stu_cls_logit_list, stu_reg_pred_list, stu_head_outs = self.forward_head(stu_head, stu_fpn_feats)
        tea_cls_logit_list, tea_reg_pred_list, tea_head_outs = self.forward_head(tea_head, tea_fpn_feats)

        # loss gfl
        loss_gfl = stu_head.get_loss(stu_head_outs, inputs)

        # loss kd
        kd_value_cls_maps, kd_value_reg_maps, kd_backmaps = self.pgd_mask.get_pgd_mask(
            tea_head, tea_head_outs, inputs, static_assign)
        # kd_value_cls_maps: 每一层的cls mask，shape为[B*h*w] 大部分为0，少数几个kp区域有权值
        # kd_backmaps:每一层的cls 背景mask，shape为[B*h*w] 背景为1，前景为0，
        #             与kd_value_cls_maps不是互补关系，point不在gt box区域则为背景

        assert len(kd_value_cls_maps) == len(kd_value_reg_maps) == len(
            kd_backmaps), "ERROR"

        distill_cls_loss = {}
        distill_reg_loss = {}
        for i, (s_cls_f, t_cls_f, s_reg_f, t_reg_f, cls_mask, reg_mask,
                bg_mask) in enumerate(
                    zip(stu_cls_logit_list, tea_cls_logit_list,
                        stu_reg_pred_list, tea_reg_pred_list, kd_value_cls_maps,
                        kd_value_reg_maps, kd_backmaps)):

            b, c, h, w = s_cls_f.shape

            # 分类loss很占显存6-8G 翻倍了
            distill_cls_loss['cls_' + self.loss_name[i]] = self.distill_cls_loss[i](
                s_cls_f, 
                t_cls_f, 
                cls_mask.reshape((b,1,h,w)), 
                bg_mask.reshape((b,1,h,w)).astype('float32')
                )
            distill_reg_loss['reg_' + self.loss_name[i]] = self.distill_reg_loss[i](
                s_reg_f, 
                t_reg_f, 
                reg_mask.reshape((b,1,h,w)),
                )

        return distill_cls_loss, distill_reg_loss, loss_gfl

    def forward(self, inputs):
        if self.training:
            with paddle.no_grad():
                t_body_feats = self.teacher_model.backbone(inputs)
                t_neck_feats = self.teacher_model.neck(t_body_feats)

            s_body_feats = self.student_model.backbone(inputs)
            s_neck_feats = self.student_model.neck(s_body_feats)

            if self.arch == "PicoDet":
                # get GT loss
                loss = {}
                # get distill loss
                static_assign = bool(
                    inputs['epoch_id'] <
                    self.student_model.head.static_assigner_epoch)
                    
                pgd_cls_loss, pgd_reg_loss, loss_gfl = self.get_distill_loss(
                    s_neck_feats, t_neck_feats,
                    self.student_model.head, self.teacher_model.head,
                    inputs, static_assign)

                loss['pgd_cls_loss'] = 0.0001 * paddle.add_n(list(pgd_cls_loss.values()))
                loss['pgd_reg_loss'] = paddle.add_n(list(pgd_reg_loss.values()))
                loss.update(loss_gfl)
                
                total_loss = paddle.add_n(list(loss.values()))
                loss.update({'loss': total_loss})
            else:
                raise ValueError(f"Unsupported model {self.arch}")

            return loss
        else:
            body_feats = self.student_model.backbone(inputs)
            neck_feats = self.student_model.neck(body_feats)
            if self.arch == "PicoDet":
                head_outs = self.student_model.head(
                    neck_feats, self.student_model.export_post_process)
                scale_factor = inputs['scale_factor']
                bboxes, bbox_num = self.student_model.head.post_process(
                    head_outs,
                    scale_factor,
                    export_nms=self.student_model.export_nms)
                return {'bbox': bboxes, 'bbox_num': bbox_num}
            else:
                raise ValueError(f"Unsupported model {self.arch}")
