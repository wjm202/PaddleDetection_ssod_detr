# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved. 
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
from ppdet.core.workspace import register
from .roi_extractor import RoIAlign
from .bbox_head import BBoxHead, TwoFCHead
from ..bbox_utils import bbox2delta

__all__ = ['BBoxHeadLM', 'TwoFCHeadLM']


@register
class BBoxHeadLM(BBoxHead):
    __shared__ = ['num_classes']
    __inject__ = ['bbox_assigner', 'bbox_loss']
    """
    RCNN bbox head

    Args:
        head (nn.Layer): Extract feature in bbox head
        in_channel (int): Input channel after RoI extractor
        roi_extractor (object): The module of RoI Extractor
        bbox_assigner (object): The module of Box Assigner, label and sample the 
            box.
        with_pool (bool): Whether to use pooling for the RoI feature.
        num_classes (int): The number of classes
        bbox_weight (List[float]): The weight to get the decode box 
    """

    def __init__(self,
                 head,
                 in_channel,
                 roi_extractor=RoIAlign().__dict__,
                 bbox_assigner='BboxAssigner',
                 with_pool=False,
                 num_classes=80,
                 bbox_weight=[10., 10., 5., 5.],
                 bbox_loss=None):
        super(BBoxHeadLM, self).__init__(head, in_channel, roi_extractor, bbox_assigner,
                                        with_pool, num_classes, bbox_weight, bbox_loss)

    def forward_train_step1(self, body_feats=None, rois=None, rois_num=None, inputs=None):
        """
        body_feats (list[Tensor]): Feature maps from backbone
        rois (list[Tensor]): RoIs generated from RPN module
        rois_num (Tensor): The number of RoIs in each image
        inputs (dict{Tensor}): The ground-truth of image
        """
        sampling_results = self.bbox_assigner.assignlm(rois, rois_num, inputs)
        return sampling_results
    
    def forward_train_step2(self, body_feats, sampling_results, inputs):
        """
        body_feats (list[Tensor]): Feature maps from backbone
        sampling_results:
        inputs (dict{Tensor}): The ground-truth of image
        """
        losses = dict()
        rois = [paddle.concat([
                res['pos_bboxes'], res['ig_bboxes'], res['neg_bboxes']
                ] , axis=0)
                for res in sampling_results]

        flag = paddle.concat([paddle.concat([
            paddle.zeros_like(res['pos_bboxes'][:, -1]),
            paddle.ones_like(res['ig_bboxes'][:, -1]),
            paddle.zeros_like(res['neg_bboxes'][:, -1])], axis=0).astype('bool')
            for res in sampling_results])

        rois_num = paddle.to_tensor([i.shape[0] for i in rois], dtype='int32')
        rois_feat = self.roi_extractor(body_feats, rois, rois_num)
        bbox_feat = self.head(rois_feat)
        # preds, _ = self.student.bbox_head(feat, proposals, rois_num, None)
        scores = self.bbox_score(bbox_feat)
        deltas = self.bbox_delta(bbox_feat)

        bbox_targets = self.head.get_targets_lm(
            sampling_results, self.num_classes)
        
        loss_bbox = self.loss(scores, deltas, *bbox_targets)

        losses.update(loss_bbox)
        scores = scores[flag]
        return losses, scores

    def loss(self,
             scores,
             deltas,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        
        # bbox cls
        tgt_labels = labels
        valid_inds = paddle.nonzero(tgt_labels >= 0).flatten()
        if valid_inds.shape[0] == 0:
            losses['loss_cls'] = paddle.zeros([1], dtype='float32')
        else:
            tgt_labels = tgt_labels.cast('int64')
            tgt_labels.stop_gradient = True
            loss_bbox_cls = F.cross_entropy(input=scores, label=tgt_labels, reduction='none')
            loss_cls_ = loss_bbox_cls * label_weights
            losses['loss_cls'] = loss_cls_.sum() / max(paddle.sum(label_weights > 0).item(), 1.)
        
        # bbox reg
        bbox_pred = deltas
        if bbox_pred is not None:
            bg_class_ind = self.num_classes

            cls_agnostic_bbox_reg = bbox_pred.shape[1] == 4

            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                pos_inds = paddle.nonzero(pos_inds).flatten()
                if cls_agnostic_bbox_reg:
                    reg_delta = paddle.gather(bbox_pred, pos_inds)
                else:
                    fg_gt_classes = paddle.gather(tgt_labels, pos_inds)

                    reg_row_inds = paddle.arange(fg_gt_classes.shape[0]).unsqueeze(1)
                    reg_row_inds = paddle.tile(reg_row_inds, [1, 4]).reshape([-1, 1])

                    reg_col_inds = 4 * fg_gt_classes.unsqueeze(1) + paddle.arange(4)

                    reg_col_inds = reg_col_inds.reshape([-1, 1])
                    reg_inds = paddle.concat([reg_row_inds, reg_col_inds], axis=1)

                    reg_delta = paddle.gather(bbox_pred, pos_inds)
                    reg_delta = paddle.gather_nd(reg_delta, reg_inds).reshape([-1, 4])

                pred = reg_delta
                target = paddle.gather(bbox_targets, pos_inds)
                weight = paddle.gather(bbox_weights, pos_inds)
                avg_factor = bbox_targets.shape[0]
                target.stop_gradient = True
                assert pred.shape == target.shape and target.numel() > 0
                loss = paddle.abs(pred - target)
                if weight is not None:
                    loss = loss * weight
                    
                losses['loss_bbox'] = loss.sum() / avg_factor

            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses
        # paddle

        # cls_agnostic_bbox_reg = deltas.shape[1] == 4
        # fg_inds = paddle.nonzero(
        #     paddle.logical_and(tgt_labels >= 0, tgt_labels <
        #                     self.num_classes)).flatten()

        # if fg_inds.numel() == 0:
        #     losses['loss_bbox'] = paddle.zeros([1], dtype='float32')
        # else:
        #     if cls_agnostic_bbox_reg:
        #         reg_delta = paddle.gather(deltas, fg_inds)
        #     else:
        #         fg_gt_classes = paddle.gather(tgt_labels, fg_inds)

        #         reg_row_inds = paddle.arange(fg_gt_classes.shape[0]).unsqueeze(1)
        #         reg_row_inds = paddle.tile(reg_row_inds, [1, 4]).reshape([-1, 1])

        #         reg_col_inds = 4 * fg_gt_classes.unsqueeze(1) + paddle.arange(4)

        #         reg_col_inds = reg_col_inds.reshape([-1, 1])
        #         reg_inds = paddle.concat([reg_row_inds, reg_col_inds], axis=1)

        #         reg_delta = paddle.gather(deltas, fg_inds)
        #         reg_delta = paddle.gather_nd(reg_delta, reg_inds).reshape([-1, 4])
            
        #     rois = paddle.concat(rois) if len(rois) > 1 else rois[0]

        #     tgt_bboxes = bbox_targets
        #     tgt_bboxes = paddle.concat(tgt_bboxes) if len(
        #         tgt_bboxes) > 1 else tgt_bboxes[0]

        #     # reg_target = bbox2delta(rois, tgt_bboxes, bbox_weights)
        #     reg_target = paddle.gather(reg_target, fg_inds)
        #     reg_target.stop_gradient = True
            
        #     loss_bbox_reg = paddle.abs(reg_delta - reg_target).sum() / max(tgt_labels.shape[0], 1.0)
        #     losses['loss_bbox'] = loss_bbox_reg
        
        # if cls_score is not None:


@register
class TwoFCHeadLM(TwoFCHead):
    """
    RCNN bbox head with Two fc layers to extract feature

    Args:
        in_channel (int): Input channel which can be derived by from_config
        out_channel (int): Output channel
        resolution (int): Resolution of input feature map, default 7
    """

    def __init__(self,
                 in_channel=256, 
                 out_channel=1024, 
                 resolution=7,
                 bbox_w=[10., 10., 5., 5.],
                 ):
        super(TwoFCHeadLM, self).__init__(in_channel,
                                          out_channel,
                                          resolution)
        self.bbox_w = bbox_w
    def _get_target_single_lm(
            self, pos_bboxes, pos_gt_bboxes, pos_gt_labels, pos_reg_weight,  # positive
            ig_bboxes, ig_gt_bboxes, ig_gt_labels, ig_reg_weight,  # ignore
            neg_bboxes, num_classes, pos_weight=-1):
        num_pos = pos_bboxes.shape[0]
        num_ig = ig_bboxes.shape[0]
        num_neg = neg_bboxes.shape[0]
        num_samples = num_pos + num_neg + num_ig

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = paddle.ones([num_samples], dtype=pos_gt_labels.dtype) * num_classes
        label_weights = paddle.zeros([num_samples], dtype='float32')
        bbox_targets = paddle.zeros([num_samples, 4], dtype='float32')
        bbox_weights = paddle.zeros([num_samples, 4], dtype='float32')

        # reliable pseudo labels
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if pos_weight <= 0 else pos_weight
            label_weights[:num_pos] = pos_weight
            pos_bbox_targets = bbox2delta(
                pos_bboxes, pos_gt_bboxes, self.bbox_w)
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = pos_reg_weight.unsqueeze(1)
        # uncertain pseudo labels
        if num_ig > 0:
            labels[num_pos:num_ig + num_pos] = ig_gt_labels
            label_weights[num_pos:num_ig + num_pos] = ig_reg_weight

            ig_bbox_targets = bbox2delta(
                ig_bboxes, ig_gt_bboxes, self.bbox_w)
            bbox_targets[num_pos:num_pos + num_ig, :] = ig_bbox_targets
            bbox_weights[num_pos:num_pos + num_ig, :] = ig_reg_weight.unsqueeze(1)

        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets_lm(self,
                       sampling_results,
                        num_classes,
                       concat=True):
        labels = []
        label_weights = []
        bbox_targets = []
        bbox_weights = []
        for i in range(len(sampling_results)):
            labels_s, label_weights_s, bbox_targets_s, bbox_weights_s = self._get_target_single_lm(
                                    sampling_results[i]['pos_bboxes'], sampling_results[i]['pos_gt_bboxes'],
                                    sampling_results[i]['pos_gt_labels'], sampling_results[i]['pos_reg_weight'],
                                    sampling_results[i]['ig_bboxes'], sampling_results[i]['ig_gt_bboxes'],
                                    sampling_results[i]['ig_gt_labels'], sampling_results[i]['ig_reg_weight'],
                                    sampling_results[i]['neg_bboxes'], num_classes)
            labels.append(labels_s)
            label_weights.append(label_weights_s)
            bbox_targets.append(bbox_targets_s)
            bbox_weights.append(bbox_weights_s)
        if concat:
            labels = paddle.concat(labels, 0)
            label_weights = paddle.concat(label_weights, 0)
            bbox_targets = paddle.concat(bbox_targets, 0)
            bbox_weights = paddle.concat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights