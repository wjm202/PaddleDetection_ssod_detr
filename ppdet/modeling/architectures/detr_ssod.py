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
        self.momentum=0.9996
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
       
        self.update_ema_model(self.momentum)
        data_sup_w, data_sup_s, data_unsup_w, data_unsup_s=inputs
        if True:
            for k, v in data_sup_s.items():
                if k in ['epoch_id']:
                    continue
                if k in ['gt_class','gt_bbox','is_crowd']:
                    data_sup_s[k].extend(data_sup_w[k])
                else:
                    data_sup_s[k] = paddle.concat([v, data_sup_w[k]])
        loss = {}
        sup_loss = self.student.forward(data_sup_s)    
        sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
        loss.update(**sup_loss)   
        # unsup_loss = weighted_loss(
        #         self.foward_unsup_train(
        #             data_groups["unsup_teacher"], data_groups["unsup_student"]
        #         ),
        #         weight=self.unsup_weight,
        #     )
        unsup_loss =     self.foward_unsup_train(
                    data_unsup_w, data_unsup_s
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

        with paddle.no_grad():
           body_feats=self.teacher.backbone(teacher_data)
           pad_mask = teacher_data['pad_mask'] if self.training else None
           out_transformer = self.transformer(body_feats, pad_mask, teacher_data)
           preds = self.detr_head(out_transformer, body_feats)
           bbox, bbox_num = self.teacher.post_process(
                    preds, teacher_data['im_shape'], teacher_data['scale_factor'])
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



