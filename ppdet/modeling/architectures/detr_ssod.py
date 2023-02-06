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

from ppdet.core.workspace import register, create
from .meta_arch import BaseArch
from ppdet.data.reader import transform
import paddle
import os

import numpy as np
from operator import itemgetter
import paddle
import paddle.nn.functional as F
import paddle.distributed as dist
from paddle.fluid import framework
from ppdet.core.workspace import register, create
# from ppdet.data.reader import get_dist_info
# from ppdet.modeling.proposal_generator.target import label_box
from ppdet.modeling.bbox_utils import delta2bbox
from ppdet.data.transform.atss_assigner import bbox_overlaps
from ppdet.utils.logger import setup_logger

from .multi_stream_detector import MultiSteamDetector
logger = setup_logger(__name__)

__all__ = ['DETR_SSOD']
@register
class DETR_SSOD(MultiSteamDetector):
    def __init__(self, teacher, student, train_cfg=None, test_cfg=None):
        super(DETR_SSOD, self).__init__(
            dict(teacher=teacher, student=student),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("teacher")
            self.unsup_weight = self.train_cfg['unsup_weight']
            self._teacher = None
            self._student = None

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        teacher = create(cfg['teacher'])
        student = create(cfg['student'])
        train_cfg = cfg['train_cfg']
        test_cfg = cfg['test_cfg']
        return {
            'teacher': teacher,
            'student': student,
            'train_cfg': train_cfg,
            'test_cfg' : test_cfg
        }

    def forward_train(self, inputs, **kwargs):
        if dist.get_world_size() > 1 and self._teacher == None:
            self._teacher = self.teacher
            self.teacher = self.teacher._layers
            self._student = self.student
            self.student = self.student._layers

        data_sup_w, data_sup_s, data_unsup_w, data_unsup_s=inputs
        # loss
        loss = {}
        if "sup" in data_groups:
            gt_bboxes = data_groups["sup"]["gt_bbox"]
            sup_loss = self.student.forward(data_groups["sup"])
            sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
            loss.update(**sup_loss)

        if "unsup_student" in data_groups:
            unsup_loss = weighted_loss(
                self.foward_unsup_train(
                    data_groups["unsup_teacher"], data_groups["unsup_student"]
                ),
                weight=self.unsup_weight,
            )
            unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
            loss.update(**unsup_loss)
        
        loss.update({'loss': loss['sup_loss'] + loss.get('unsup_loss', 0)})

        if dist.get_world_size() > 1:
            if framework._dygraph_tracer(
            )._has_grad and self._teacher.grad_need_sync:
                self._teacher._reducer.prepare_for_backward(
                    list(self._teacher._find_varbase(loss)))
            if framework._dygraph_tracer(
            )._has_grad and self._student.grad_need_sync:
                self._student._reducer.prepare_for_backward(
                    list(self._student._find_varbase(loss)))
        
        return loss

    def foward_unsup_train(self, teacher_data, student_data):

        teacher_data['img_metas'] = []
        unsup_num = teacher_data["image"].shape[0]
        for i in range(unsup_num):
            tmp_dict = {}
            tmp_dict['tag'] = teacher_data['tag'][i]
            tmp_dict['batch_input_shape'] = teacher_data['batch_input_shape'][i]
            tmp_dict['scale_factor'] = teacher_data['scale_factor'][i]
            tmp_dict['transform_matrix'] = teacher_data['transform_matrix'][i]
            tmp_dict['img_shape'] = tuple((teacher_data['image'][i].shape))
            tmp_dict['img_shape'] = tmp_dict['img_shape'][1:] + (tmp_dict['img_shape'][0],)
            teacher_data['img_metas'].append(tmp_dict)
        
        student_data['img_metas'] = []
        for i in range(unsup_num):
            tmp_dict = {}
            tmp_dict['tag'] = student_data['tag'][i]
            tmp_dict['batch_input_shape'] = student_data['batch_input_shape'][i]
            tmp_dict['scale_factor'] = student_data['scale_factor'][i]
            tmp_dict['transform_matrix'] = student_data['transform_matrix'][i]
            tmp_dict['img_shape'] = tuple((student_data['image'][i].shape))
            tmp_dict['img_shape'] = tmp_dict['img_shape'][1:] + (tmp_dict['img_shape'][0],)
            student_data['img_metas'].append(tmp_dict)

        with paddle.no_grad():
            teacher_info = self.extract_teacher_info(
                teacher_data,
                teacher_data["image"],
                teacher_data["img_metas"],
                teacher_data["proposals"]
                if ("proposals" in teacher_data)
                and (teacher_data["proposals"] is not None)
                else None,
            )
        student_info = self.extract_student_info(student_data, 
            student_data["image"],
            student_data["img_metas"],
            student_data["proposals"]
            if ("proposals" in student_data)
            and (student_data["proposals"] is not None)
            else None,
        )

        return self.compute_pseudo_label_loss(student_info, teacher_info)

    def compute_pseudo_label_loss(self, student_info, teacher_info):                                 
        # teacher_info["transform_matrix"] = [matrix.astype('float32') for matrix in teacher_info["transform_matrix"]]
        # student_info["transform_matrix"] = [matrix.astype('float32') for matrix in student_info["transform_matrix"]]
        M = Transform2D.get_trans_mat(
            teacher_info["transform_matrix"], student_info["transform_matrix"]
        )
        # n,9
        pseudo_bboxes = self._transform_bbox(
            teacher_info["det_bboxes"],
            M,
            [meta["img_shape"] for meta in student_info["img_metas"]],
        )
        pseudo_labels = teacher_info["det_labels"]
        loss = {}
        rpn_loss, proposal_list, rois, rois_num = self.rpn_loss(
           student_info["rpn_out"],
           pseudo_bboxes,
           pseudo_labels,
           student_info["img_metas"],
           student_info=student_info,
        )
        loss.update(rpn_loss)
        if proposal_list is not None:
           student_info["proposals"] = proposal_list
        if self.train_cfg['use_teacher_proposal']:
            proposals = self._transform_bbox(
                teacher_info["proposals"],
                M,
                [meta["img_shape"] for meta in student_info["img_metas"]],
            )
        else:
            proposals = student_info["proposals"]

        loss.update(
            self.unsup_rcnn_cls_loss(
                student_info["backbone_feature"],
                student_info["img_metas"],
                proposals,
                pseudo_bboxes,
                pseudo_labels,
                teacher_info["transform_matrix"],
                student_info["transform_matrix"],
                teacher_info["img_metas"],
                teacher_info["backbone_feature"],
                rois, rois_num, 
                student_info=student_info,
                teacher_info=teacher_info,
            )
        )
        loss.update(
            self.unsup_rcnn_reg_loss(
                student_info["backbone_feature"],
                student_info["img_metas"],
                proposals,
                pseudo_bboxes,
                pseudo_labels,
                rois_num, 
                student_info=student_info,
            )
        )
        return loss

    def rpn_loss(
        self,
        rpn_out,
        pseudo_bboxes,
        pseudo_labels,
        img_metas,
        gt_bboxes_ignore=None,
        student_info=None,
        **kwargs,
    ): 
        # print("[DEBUG]: rpn_loss before: ", len(pseudo_bboxes))
        if hasattr(self.student, 'rpn_head'):
            gt_bboxes = []
            for bbox in pseudo_bboxes:
                if bbox.numel() > 0:
                    bbox, _, _ = filter_invalid(
                        bbox[:, :4],
                        score=bbox[:, 4],  # TODO: replace with foreground score, here is classification score,
                        thr=self.train_cfg['rpn_pseudo_threshold'],
                        min_size=self.train_cfg['min_pseduo_box_size'],
                    )
                else:
                    bbox = paddle.expand(paddle.to_tensor([])[:, None], (-1, 4))
                gt_bboxes.append(bbox)

            # print("[DEBUG]: rpn_loss after: ", len(gt_bboxes))
            # print("rpn_gt_num:", sum([len(bbox) for bbox in gt_bboxes]))
            
            rpn_feats = student_info["rpn_feats"]
            
            scores = rpn_out[0]
            deltas = rpn_out[1]
            anchors = self.student.rpn_head.anchor_generator(rpn_feats)
            inputs = student_info['student_data']
            inputs['gt_bbox'] = gt_bboxes
            rois, rois_num = self.student.rpn_head._gen_proposal(scores, deltas, anchors, inputs)

            losses = self.student.rpn_head.get_loss(scores, deltas, anchors, inputs)
            ######## maybe null error
            # valid_flg = [True if gt.numel()>0 else False for gt in gt_bboxes]
            # inputs['gt_bbox'] = [gt for gt in gt_bboxes if gt.numel() > 0]
            # inputs['im_shape'] = inputs['im_shape'][valid_flg]
            # rois_ = [paddle.expand(paddle.to_tensor([])[:, None], (-1, 4))] * len(valid_flg)
            # rois, rois_num = self.student.rpn_head._gen_proposal([f[valid_flg] for f in scores], 
            #                                                     [f[valid_flg] for f in deltas],
            #                                                     anchors,inputs)
            # rois_ =  [for roi in rois] 
                                                          
            # rois, rois_num = self.student.rpn_head._gen_proposal(scores, deltas, anchors, inputs)
            
            # recall
            # rois, rois_num, rpn_loss = self.student.rpn_head(student_info["backbone_feature"], student_info['student_data'])

            output_rois_prob = [paddle.clone(item) for item in self.student.rpn_head.output_rois_prob]
            proposal_list = [paddle.concat([_rois, _output_rois_prob.unsqueeze(axis=1)], axis=1) 
                                for (_rois, _output_rois_prob) in zip(rois, output_rois_prob)]

            flag_rpn = 1
            for i in range(len(student_info["img"])):
                bboxes = pseudo_bboxes[i][:, :4]
                log_image_with_boxes(
                    "rpn_student",
                    student_info["img"][i],
                    bboxes,
                    interval=500,
                    cnt = i,
                    flag = flag_rpn,
                )
                flag_rpn = 0

            return losses, proposal_list, rois, rois_num
        else:
            return {}, None

    def unsup_rcnn_cls_loss(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        teacher_transMat,
        student_transMat,
        teacher_img_metas,
        teacher_feat,
        rois, rois_num, 
        student_info=None,
        teacher_info=None,
        **kwargs,
    ):
        # print("[DEBUG]: unsup_rcnn_cls_loss before: ", len(pseudo_bboxes))
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            thr=self.train_cfg['cls_pseudo_threshold'],
        )
        # print("[DEBUG]: unsup_rcnn_cls_loss after: ", sum([len(bbox) for bbox in gt_bboxes]))

        inputs = student_info['student_data']
        inputs['gt_class'] = [paddle.unsqueeze(gt_label, axis=1) for gt_label in gt_labels]
        inputs['gt_bbox'] = gt_bboxes

        # rois list b,[512,4]  
        # rois_num tensor b 
        # targets :tgt_labels, tgt_bboxes, tgt_gt_inds 
        losses = dict()
        if sum([gt.shape[0] for gt in gt_labels]) > 0:
            rois = [pro[:, :4]for pro in proposal_list]
            rois, rois_num, targets = self.student.bbox_head.bbox_assigner(rois, rois_num, inputs)
            self.student.bbox_head.assigned_rois = (rois, rois_num)
            self.student.bbox_head.assigned_targets = targets

            M = Transform2D.get_trans_mat(student_transMat, teacher_transMat)
            aligned_proposals = self._transform_bbox(
                rois, # list b, [512,4]
                M, # list b, [3,3]
                [meta["img_shape"] for meta in teacher_img_metas],
            ) # list b, [512,4]
            with paddle.no_grad():   # preds list [0] [_,96] [1] [_,25]
                preds, _ = self.teacher.bbox_head(teacher_feat, aligned_proposals, rois_num, None)
                bboxes, bbox_num = self.teacher.bbox_post_process(preds, (aligned_proposals, rois_num),
                                                    teacher_info['im_shape'], teacher_info['scale_factor'], use_nms=False)
    
                bg_score = preds[1][:, -1]
                
                # data type to int32
                targets = list(targets)
                targets[0] = list(targets[0])
                targets[0] = [paddle.cast(target, dtype='int32') for target in targets[0]]
                assigned_label = paddle.concat(targets[0], axis=0)
                neg_inds = assigned_label == self.student.bbox_head.num_classes
                # loss weight
                weight = paddle.ones_like(paddle.concat(targets[0]) 
                                                        if len(targets[0]) > 1 
                                                        else targets[0][0], dtype='float32')
                weight[neg_inds] = bg_score[neg_inds].detach()

            rois_feat = self.student.bbox_head.roi_extractor(feat, rois, rois_num)
            bbox_feat = self.student.bbox_head.head(rois_feat)
            
            feat = bbox_feat
            scores = self.student.bbox_head.bbox_score(feat)
            deltas = self.student.bbox_head.bbox_delta(feat)

            # recall
            # loss = self.student.bbox_head.get_loss(scores, deltas, targets, rois,
            #                         self.student.bbox_head.bbox_weight)
            
            tgt_labels, tgt_bboxes, tgt_gt_inds = targets

            # bbox cls
            tgt_labels = paddle.concat(tgt_labels) if len(tgt_labels) > 1 else tgt_labels[0]

            valid_inds = paddle.nonzero(tgt_labels >= 0).flatten()
            if valid_inds.shape[0] == 0:
                losses['loss_cls'] = paddle.zeros([1], dtype='float32')
            else:
                tgt_labels = tgt_labels.cast('int64')
                tgt_labels.stop_gradient = True
                loss_bbox_cls = F.cross_entropy(input=scores, label=tgt_labels, reduction='none')
                loss_cls_ = loss_bbox_cls * weight
                losses['loss_cls'] = loss_cls_.sum() / max(paddle.sum(weight > 0).item(), 1.)

            # bbox reg
            # cls_agnostic_bbox_reg = deltas.shape[1] == 4
            # fg_inds = paddle.nonzero(
            #     paddle.logical_and(tgt_labels >= 0, tgt_labels <
            #                     self.student.bbox_head.num_classes)).flatten()

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
            #     tgt_bboxes = paddle.concat(tgt_bboxes) if len(
            #         tgt_bboxes) > 1 else tgt_bboxes[0]

            #     reg_target = bbox2delta(rois, tgt_bboxes, self.student.bbox_head.bbox_weight)
            #     reg_target = paddle.gather(reg_target, fg_inds)
            #     reg_target.stop_gradient = True
                
            #     loss_bbox_reg = paddle.abs(reg_delta - reg_target).sum() / max(tgt_labels.shape[0], 1.0)
            #     losses['loss_bbox'] = loss_bbox_reg
            # draw    
            flag_rcnn_cls = 1
            for i in range(len(student_info["img"])):
                if len(gt_bboxes[i]) > 0:
                    log_image_with_boxes(
                        "rcnn_cls",
                        student_info["img"][i],
                        gt_bboxes[i],
                        labels=gt_labels[i],
                        interval=500,
                        cnt = i,
                        flag = flag_rcnn_cls,
                    )
                    flag_rcnn_cls = 0

        else:
            losses['loss_cls'] = paddle.zeros([1], dtype='float32')
        
        return losses

    def unsup_rcnn_reg_loss(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        rois_num, 
        student_info=None,
        **kwargs,
    ):
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [-bbox[:, 5:].mean(axis=-1) if bbox.shape[0] > 0 else paddle.to_tensor([])
            for bbox in pseudo_bboxes],
            thr=-self.train_cfg['reg_pseudo_threshold'],
        )
        # print("[DEBUG]:before var 0.2:unsup_rcnn_reg_loss:", sum([len(bbox) for bbox in pseudo_bboxes]))
        # print("[DEBUG]:after var 0.2:unsup_rcnn_reg_loss:", sum([len(bbox) for bbox in gt_bboxes]))

        inputs = student_info['student_data']
        inputs['gt_class'] = [paddle.unsqueeze(gt_label, axis=1) for gt_label in gt_labels]
        inputs['gt_bbox'] = gt_bboxes
        if sum([gt.shape[0] for gt in gt_bboxes]) > 0:
            rois = [pro[:, :4]for pro in proposal_list]
            bbox_loss, _ = self.student.bbox_head(feat, rois, rois_num, inputs)
            loss_bbox = bbox_loss['loss_bbox_reg']  
            # draw    
            flag_rcnn_reg = 1
            for i in range(len(student_info["img"])):
                if len(gt_bboxes[i]) > 0:
                    log_image_with_boxes(
                        "rcnn_reg",
                        student_info["img"][i],
                        gt_bboxes[i],
                        labels=gt_labels[i],
                        interval=500,
                        cnt = i,
                        flag = flag_rcnn_reg,
                    )
                    flag_rcnn_reg = 0
        else:
            loss_bbox = paddle.zeros([1], dtype='float32')
        return {"loss_bbox": loss_bbox}

    def _transform_bbox(self, bboxes, trans_mat, max_shape):
        bboxes = Transform2D.transform_bboxes(bboxes, trans_mat, max_shape)
        return bboxes

    def extract_student_info(self, student_data, img, img_metas, proposals=None, **kwargs):
        student_info = {}
        student_info["img"] = img
        feat = self.student.backbone(student_data)
        if self.student.neck is not None:
            feat = self.student.neck(feat)
        student_info["backbone_feature"] = feat
        # rpn_head -> get rpn_out 
        rpn_feats = self.student.rpn_head.rpn_feat(feat)
        student_info["rpn_feats"] = rpn_feats
        scores = []
        deltas = []

        for rpn_feat in rpn_feats:
            rrs = self.student.rpn_head.rpn_rois_score(rpn_feat)
            rrd = self.student.rpn_head.rpn_rois_delta(rpn_feat)
            scores.append(rrs)
            deltas.append(rrd)
        
        rpn_out = []
        rpn_out.append(scores)
        rpn_out.append(deltas)
        student_info["rpn_out"] = rpn_out
        student_info['student_data'] = student_data

        student_info["img_metas"] = img_metas
        student_info["proposals"] = proposals
        student_info["transform_matrix"] = [
            paddle.to_tensor(meta["transform_matrix"], place=feat[0][0].place)
            for meta in img_metas
        ]
        return student_info

    def extract_teacher_info(self, teacher_data, img, img_metas, proposals=None, **kwargs):
        teacher_info = {}
        teacher_info["img"] = img
        feat = self.teacher.backbone(teacher_data)
        if self.teacher.neck is not None:
            feat = self.teacher.neck(feat)
        teacher_info["backbone_feature"] = feat

        if proposals is None:
            rois, rois_num, _ = self.teacher.rpn_head(feat, teacher_data)
            output_rois_prob = [paddle.clone(item).detach() for item in self.teacher.rpn_head.output_rois_prob]
            proposal_list = [paddle.concat([_rois, _output_rois_prob.unsqueeze(axis=1)], axis=1) 
                                for (_rois, _output_rois_prob) in zip(rois, output_rois_prob)]
        else:
            proposal_list = proposals
        # 1000
        teacher_info["proposals"] = proposal_list

        preds, _ = self.teacher.bbox_head(feat, rois, rois_num, None)

        im_shape = teacher_data['im_shape']
        teacher_info['im_shape'] = im_shape
        scale_factor = teacher_data['scale_factor']
        teacher_info['scale_factor'] = scale_factor
        bbox, bbox_num = self.teacher.bbox_post_process(preds, (rois, rois_num),
                                                im_shape, scale_factor)

        # rescale the prediction back to origin image
        # bboxes, bbox_pred, bbox_num = self.teacher.bbox_post_process.get_pred(
        #     bbox, bbox_num, im_shape, scale_factor)
        if bbox.numel() > 0:
            proposal_list = paddle.concat([bbox[:, 2:], bbox[:, 1:2]], axis=-1)
            proposal_list = proposal_list.split(tuple(np.array(bbox_num)), 0)
        else:
            proposal_list = [paddle.expand(paddle.to_tensor([])[:, None], (-1, 5))]
        
        proposal_label_list = paddle.cast(bbox[:, 0], np.int32)
        proposal_label_list = proposal_label_list.split(tuple(np.array(bbox_num)), 0)
            

        proposal_list = [paddle.to_tensor(p, place=feat[0].place) for p in proposal_list]
        proposal_label_list = [paddle.to_tensor(p, place=feat[0].place) for p in proposal_label_list]

        # filter invalid box roughly
        if isinstance(self.train_cfg['pseudo_label_initial_score_thr'], float):
            thr = self.train_cfg['pseudo_label_initial_score_thr']
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")
        # print("thr0.5 :",sum([len(bbox) for bbox in proposal_list]), "\tscore:",[proposal[:, -1] for proposal in proposal_list])
        
        proposal_list, proposal_label_list, _ = list(
            zip(
                *[
                    filter_invalid(
                        proposal,
                        proposal_label,
                        proposal[:, -1],
                        thr=thr,
                        min_size=self.train_cfg['min_pseduo_box_size'],
                    )
                    for proposal, proposal_label in zip(
                        proposal_list, proposal_label_list
                    )
                ]
            )
        )
        # print("thr0.5 after:",sum([len(bbox) for bbox in proposal_list]), "\tscore:",[proposal[:, -1] for proposal in proposal_list])
        det_bboxes = proposal_list
        valid_pro_flag = [proposal.numel() > 0 for proposal in proposal_list]
        if np.array(valid_pro_flag).sum() > 0:
            reg_unc = self.compute_uncertainty_with_aug(
                teacher_data, feat, img_metas, proposal_list, proposal_label_list
            )
            det_bboxes = [
                paddle.concat([bbox, unc], axis=-1) 
                if bbox.numel() > 0 else paddle.expand(paddle.to_tensor([])[:, None], (-1, 9))
                for bbox, unc in zip(det_bboxes, reg_unc)
            ]
        else:
            det_bboxes = len(img_metas) * [paddle.expand(paddle.to_tensor([])[:, None], (-1, 9))]

        teacher_info["det_bboxes"] = det_bboxes
        teacher_info["det_labels"] = proposal_label_list
        teacher_info["transform_matrix"] = [
            paddle.to_tensor(meta["transform_matrix"], place=feat[0][0].place)
            for meta in img_metas
        ]
        teacher_info["img_metas"] = img_metas

        # bboxes = {}
            
        # for i in range(len(teacher_info["img"])):
        #     bboxes['bbox'] = det_bboxes[i][:, :4]
        #     image = log_image_with_boxes(
        #         teacher_info["img"][i],
        #         bboxes,
        #     )
        #     save_name = "rpn_teacher_{}.jpg".format(i)
        #     image.save(save_name, quality=95)

        return teacher_info

    def compute_uncertainty_with_aug(
        self, teacher_data, feat, img_metas, proposal_list, proposal_label_list
    ):
        auged_proposal_list = self.aug_box(
            proposal_list, self.train_cfg['jitter_times'], self.train_cfg['jitter_scale']
        )
        # flatten
        auged_proposal_list = [
            paddle.reshape(auged, (-1, auged.shape[-1]))
            if auged.shape[1] > 0 else 
            paddle.expand(paddle.to_tensor([])[:, None], (-1, 5))
            for auged in auged_proposal_list
        ]
        rois_num = paddle.to_tensor([i.shape[0] for i in auged_proposal_list], dtype=np.int32)
        rois_num_new = rois_num[rois_num.nonzero()].flatten()

        flag_reg_unc = [proposal.shape[0] > 0 for proposal in auged_proposal_list] # list b

        auged_proposal_list = [proposal for proposal in auged_proposal_list if proposal.shape[0] > 0] 
        preds, _ = self.teacher.bbox_head([f[flag_reg_unc] for f in feat], 
                                            [aug_p[:, :4] for aug_p in auged_proposal_list], 
                                            rois_num_new, None)

        im_shape = teacher_data['im_shape']
        scale_factor = teacher_data['scale_factor']

        # recode null proposal 
        
        im_shape = im_shape[flag_reg_unc]
        scale_factor = scale_factor[flag_reg_unc]

        bboxes, bbox_num = self.teacher.bbox_post_process(preds, 
                                                        ([aug_p[:, :4] for aug_p in auged_proposal_list], 
                                                        rois_num_new),
                                                        im_shape, scale_factor, use_nms=False)

        bboxes = bboxes.split(tuple(np.array(rois_num[flag_reg_unc])), 0)

        new_proposal_label_list = []
        for proposal_label in proposal_label_list:
            if proposal_label.shape[0] > 0:
                new_proposal_label_list.append(proposal_label)
        proposal_label_list = new_proposal_label_list

        # rescale the prediction back to origin image
        # bboxes, bbox_pred, bbox_num = self.teacher.bbox_post_process.get_pred(
        #     bbox, bbox_num, im_shape, scale_factor)
        
        reg_channel = max([bbox.shape[-1] for bbox in bboxes]) // 4
        bboxes = [
            paddle.reshape(bbox, (self.train_cfg['jitter_times'], -1, bbox.shape[-1]))
            if bbox.numel() > 0
            else paddle.zeros([self.train_cfg['jitter_times'], 0, 4 * reg_channel])
            for bbox in bboxes
        ]

        box_unc = [bbox.std(axis=0) for bbox in bboxes]
        bboxes = [bbox.mean(axis=0) for bbox in bboxes]
        
        if reg_channel != 1:
            bboxes = [
                paddle.reshape(bbox, (bbox.shape[0], reg_channel, 4))[
                    paddle.arange(bbox.shape[0]), paddle.cast(label, np.int64)
                ] 
                for bbox, label in zip(bboxes, proposal_label_list)
            ]
            box_unc = [
                paddle.reshape(unc, (unc.shape[0], reg_channel, 4))[
                    paddle.arange(unc.shape[0]), paddle.cast(label, np.int64)
                ] 
                for unc, label in zip(box_unc, proposal_label_list)
            ]

        box_shape = [(bbox[:, 2:4] - bbox[:, :2]).clip(min=1.0) 
                        if len(bbox.shape)==2 
                        else (bbox.unsqueeze(axis=0)[:, 2:4] - bbox.unsqueeze(axis=0)[:, :2]).clip(min=1.0) 
                    for bbox in bboxes]
        # relative unc
        box_unc = [
            unc / paddle.reshape(paddle.expand(wh[:, None, :], (-1, 2, 2)), (-1, 4))
            if wh.numel() > 0
            else unc
            for unc, wh in zip(box_unc, box_shape)
        ]
        box_unc = paddle.concat(box_unc, axis=0)
        box_unc = box_unc.split(tuple(np.array(rois_num/10)), 0)

        return box_unc

    @staticmethod
    def aug_box(boxes, times=1, frac=0.06):
        def _aug_single(box):
            # random translate
            # TODO: random flip or something
            if box.shape[0] == 0:
                return paddle.expand(box[None, :, :], [times, -1, -1])
            box_scale = box[:, 2:4] - box[:, :2]
            box_scale = (
                paddle.reshape(paddle.expand(box_scale.clip(min=1)[:, None, :], [-1, 2, 2]), (-1, 4))
            )
            aug_scale = box_scale * frac  # [n,4]

            offset = (
                paddle.to_tensor(paddle.randn([times, box.shape[0], 4]), place=box.place)
                * aug_scale[None, ...]
            )
            new_box = paddle.expand(box.clone()[None, ...], (times, box.shape[0], -1))
            return paddle.concat(
                [new_box[:, :, :4].clone() + offset, new_box[:, :, 4:]], axis=-1
            )

        return [_aug_single(box) for box in boxes]

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


