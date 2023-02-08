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
from typing import KeysView

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
from ppdet.modeling.ssod_utils import filter_invalid,weighted_loss
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
        self.semi_start_iters=train_cfg['semi_start_iters']
        self.ema_start_iters=train_cfg['ema_start_iters']
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
        # print(self.teacher)
        data_sup_w, data_sup_s, data_unsup_w, data_unsup_s,iter_id=inputs
        if iter_id==self.ema_start_iters:
            self.update_ema_model(self.momentum==0)
        elif iter_id<self.ema_start_iters:
            pass
        elif iter_id>self.ema_start_iters:
            self.update_ema_model(self.momentum==0.99)
        if True:
            for k, v in data_sup_s.items():
                if k in ['epoch_id']:
                    continue
                if k in ['gt_class','gt_bbox','is_crowd']:
                    data_sup_s[k].extend(data_sup_w[k])
                else:
                    data_sup_s[k] = paddle.concat([v, data_sup_w[k]])
        loss = {}
        sup_loss = self.student.forward(data_sup_w)    
        sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
        loss.update(**sup_loss)   
        unsup_loss = weighted_loss(
                self.foward_unsup_train(
                        data_unsup_w, data_unsup_s
                    ),
                weight=self.unsup_weight,
            )
        if iter_id>self.semi_start_iters:
            unsup_loss =     self.foward_unsup_train(
                        data_unsup_w, data_unsup_s
                    )
            unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
            loss.update(**unsup_loss)     
            
            loss.update({'loss': loss['sup_loss'] + loss.get('unsup_loss', 0)})
        else:
            loss.update({'loss': loss['sup_loss']})
        # if dist.get_world_size() > 1:
        #     if framework._dygraph_tracer(
        #     )._has_grad and self._teacher.grad_need_sync:
        #         self._teacher._reducer.prepare_for_backward(
        #             list(self._teacher._find_varbase(loss)))
        #     if framework._dygraph_tracer(
        #     )._has_grad and self._student.grad_need_sync:
        #         self._student._reducer.prepare_for_backward(
        #             list(self._student._find_varbase(loss)))
        
        return loss

    def foward_unsup_train(self, teacher_data, student_data):

        with paddle.no_grad():
           body_feats=self.teacher.backbone(teacher_data)
           pad_mask = teacher_data['pad_mask'] if self.training else None
           out_transformer = self.teacher.transformer(body_feats, pad_mask, teacher_data)
           preds = self.teacher.detr_head(out_transformer, body_feats)
           bbox, bbox_num = self.teacher.post_process(
                    preds, teacher_data['im_shape'], teacher_data['scale_factor'])
           scale_factor=teacher_data['scale_factor'].flip(-1).tile([1, 2])
           bbox = paddle.split(bbox,list(bbox_num))
           for i in range (len(bbox)):
               bbox[i]=paddle.concat([bbox[i][:,0:2],bbox[i][:,2:]*scale_factor[i]],axis=-1)
           bbox=paddle.concat(bbox,axis=0)
        self.place=body_feats[0].place
        if bbox.numel() > 0:
            proposal_list = paddle.concat([bbox[:, 2:], bbox[:, 1:2]], axis=-1)
            proposal_list = proposal_list.split(tuple(np.array(bbox_num)), 0)
        else:
            proposal_list = [paddle.expand(paddle.to_tensor([])[:, None], (-1, 5),place=self.place)]
        
        proposal_label_list = paddle.cast(bbox[:, 0], np.int32)
        proposal_label_list = proposal_label_list.split(tuple(np.array(bbox_num)), 0)
            
        proposal_list = [paddle.to_tensor(p, place=self.place) for p in proposal_list]
        proposal_label_list = [paddle.to_tensor(p, place=self.place) for p in proposal_label_list]

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
        teacher_bboxes = proposal_list
        teacher_labels = proposal_label_list
        teacher_info=[teacher_bboxes,teacher_labels]
        student_info=student_data

        return self.compute_pseudo_label_loss(student_info, teacher_info)

    def compute_pseudo_label_loss(self, student_info, teacher_info):                                 

        pseudo_bboxes=list(teacher_info[0])
        pseudo_labels=list(teacher_info[1])
        student_data=student_info
        losses = dict()
        if sum([label.shape[0] for label in pseudo_labels]) > 0:
            no_empty=[]
            for i in range(len(pseudo_bboxes)):
                if pseudo_labels[i].shape[0]>0:
                    no_empty.append(i)
                pseudo_bboxes[i]=pseudo_bboxes[i][:,:4].numpy()
                pseudo_labels[i]=pseudo_labels[i].unsqueeze(-1).numpy()
            print('scale_factor')
            print(student_data[ 'scale_factor'])
            print('curr_iter')
            print(student_data[ 'curr_iter'])
            import copy
            f=copy.copy(student_data)
            for k in ['im_id', 'curr_iter', 'image', 'im_shape', 'scale_factor', 'pad_mask', 'epoch_id','gt_class','gt_bbox']:
                if k in 'gt_class':
                    gt =[]
                    for i in no_empty:
                        gt.append(pseudo_labels[i])
                    student_data.update({k:gt})
                elif k in 'gt_bbox':
                    gt =[]
                    for i in no_empty:
                        gt.append(pseudo_bboxes[i])
                    student_data.update({k:gt})
                elif k in ['epoch_id','im_id', 'curr_iter']:
                    continue
                else:
                    student_data[k]=paddle.to_tensor(paddle.index_select(student_data[k], paddle.to_tensor(no_empty), axis=0),place=self.place)
            print('**************************')
            print('scale_factor_after')
            if student_data[ 'scale_factor'].sum()==0:
                print(1)
            print(student_data[ 'scale_factor'])
            if student_data[ 'curr_iter']>10000:
                print('curr_iter_after')
            print(student_data[ 'curr_iter'])
            print('**************************')
            print(paddle.to_tensor(no_empty))
            print('**************************')
            print('**************************')
            print('**************************')
            print('**************************')

            student_data=self.normalize_box(student_data)
            losses=self.student(student_data)
        else:
            losses['loss'] = paddle.zeros([1], dtype='float32')
            losses['loss_class'] = paddle.zeros([1], dtype='float32')
            losses['loss_bbox'] = paddle.zeros([1], dtype='float32')
            losses['loss_giou'] = paddle.zeros([1], dtype='float32')
            losses['loss_class_aux'] = paddle.zeros([1], dtype='float32')
            losses['loss_bbox_aux'] = paddle.zeros([1], dtype='float32')
            losses['loss_giou_aux'] = paddle.zeros([1], dtype='float32')
        return losses



    def normalize_box(self,sample,):
        im = sample['image']
        if  'gt_bbox' in sample.keys():
            gt_bbox = sample['gt_bbox']
            gt_class = sample['gt_class']
            _, _, height, width, = im.shape
            for i in range(len(gt_bbox)):
                for j in range(gt_bbox[i].shape[0]):
                    gt_bbox[i][j][0] = gt_bbox[i][j][0] / width
                    gt_bbox[i][j][1] = gt_bbox[i][j][1] / height
                    gt_bbox[i][j][2] = gt_bbox[i][j][2] / width
                    gt_bbox[i][j][3] = gt_bbox[i][j][3] / height
                    gt_class[i]= paddle.to_tensor(gt_class[i],dtype=paddle.int32,place=self.place)
            sample['gt_bbox'] = gt_bbox
            sample['gt_class'] = gt_class
        if  'gt_bbox' in sample.keys():
            bbox = sample['gt_bbox']
            for i in range(len(bbox)):
                bbox[i][:, 2:4] = bbox[i][:, 2:4] - bbox[i][:, :2]
                bbox[i][:, :2] = bbox[i][:, :2] + bbox[i][:, 2:4] / 2.
                bbox[i]= paddle.to_tensor(bbox[i],dtype=paddle.float32,place=self.place)
            sample['gt_bbox'] = bbox

        
        return sample
    