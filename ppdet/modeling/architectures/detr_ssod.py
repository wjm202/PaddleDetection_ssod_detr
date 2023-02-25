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
from cProfile import label
from typing import KeysView
import copy

import cv2
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
            self.sup_weight = self.train_cfg['sup_weight']
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
        # # if isinstance(inputs,dict):
        # iter_id=inputs['iter_id']
        # # elif isinstance(inputs,list):
        iter_id=inputs[-1]
        print(iter_id)
        if iter_id==7300:
            print('***********************')
            print('******model quel*******')
            print('***********************')
            self.update_ema_model(momentum=0)
        elif iter_id>self.semi_start_iters:
            self.update_ema_model(momentum=self.momentum)
        # elif iter_id<self.semi_start_iters:
        #     self.update_ema_model(momentum=0)
        if iter_id>=self.semi_start_iters:
            if iter_id==self.semi_start_iters:
                print('***********************')
                print('******semi start*******')
                print('***********************')
            data_sup_w, data_sup_s, data_unsup_w, data_unsup_s,_=inputs
            data_list=[data_sup_w, data_sup_s,data_unsup_s]
            # import cv2
            # a= data_unsup_s['image'][0]
            # image = a.numpy()
            # img=np.uint8(image*255)
            # cv2.imwrite('3.jpg',img.transpose([1,2,0]))
            if data_list[0]['image'].shape[1] == 3:
                max_size = _max_by_axis([list(data['image'].shape[1:]) for data in data_list])
                batch_shape = [len(data_list)] + max_size
                b, c, h, w = batch_shape
                dtype = data_list[0]['image'].dtype
                tensor = paddle.zeros(batch_shape, dtype=dtype)
                mask = paddle.zeros((b, h, w), dtype=dtype)
                mask_af=[]
                pad_img_af=[]
                for img, pad_img,m in zip(data_list, tensor,mask):
                    pad_img[:, : img['image'].shape[2], : img['image'].shape[3]]=paddle.clone(img['image'].squeeze(0))
                    m[: img['image'].shape[2], :img['image'].shape[3]] = paddle.to_tensor(1.0)
                    pad_img_af.append(pad_img)
                    mask_af.append(m)
                mask_af=paddle.stack(mask_af,axis=0)
                pad_img_af=paddle.stack(pad_img_af,axis=0)
                data_student=copy.deepcopy(data_sup_w)
                data_student.update({'image':pad_img_af,'pad_mask':mask_af})
                for k in data_sup_w.keys():
                    if k in ['gt_class','gt_bbox','is_crowd']:
                            data_student[k]=data_sup_w[k]
                for k in data_sup_s.keys():
                    if k in ['gt_class','gt_bbox','is_crowd']:
                            data_student[k].extend(data_sup_s[k])
            else:
                print(data_list[0]['image'])
                raise ValueError('not supported')
            loss = {}
            unsup_loss =  self.foward_unsup_train(data_unsup_w, data_student,data_unsup_s)
            unsup_loss.update({
            'loss':
            paddle.add_n([v for k, v in unsup_loss.items() if 'log' not in k])
        })
            unsup_loss = { k: v for k, v in unsup_loss.items()}
            loss.update(**unsup_loss)     
            loss.update({'loss':  unsup_loss['loss']})                
                # loss.update({'loss': loss['sup_loss'] + self.unsup_weight*loss.get('unsup_loss', 0)})
        else:
            if iter_id==self.semi_start_iters-1:
                print('********************')
                print('******sup ing*******')
                print('********************')
            loss = {}
            sup_loss=self.student(inputs)
            sup_loss = {k: v for k, v in sup_loss.items()}
            loss.update(**sup_loss)      

        return loss

    def foward_unsup_train(self, teacher_data, student_data,data_unsup_s):

        with paddle.no_grad():
           body_feats=self.teacher.backbone(teacher_data)
           pad_mask = teacher_data['pad_mask'] if self.training else None
           out_transformer = self.teacher.transformer(body_feats, pad_mask, teacher_data)
           preds = self.teacher.detr_head(out_transformer, body_feats)
           bbox=preds[0].astype('float32')
           label=preds[1].argmax(-1).unsqueeze(-1).astype('float32')
           score=F.softmax(preds[1],axis=2).max(-1).unsqueeze(-1).astype('float32')

        proposal_list=paddle.concat([label,score,bbox],axis=-1)
        print(score.max())    

        if isinstance(self.train_cfg['pseudo_label_initial_score_thr'], float):
            thr = self.train_cfg['pseudo_label_initial_score_thr']
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.") 
        proposal_list, proposal_label_list = filter_invalid(
                                                                bbox[0,:],
                                                                label[0,:,0],
                                                                score[0,:,0],
                                                                thr=thr,
                                                                min_size=self.train_cfg['min_pseduo_box_size'],)
        
        teacher_bboxes = [proposal_list]
        teacher_labels = [proposal_label_list]
        assert len(teacher_bboxes)==len(teacher_labels)==1
        max_pixels=1000
        assert data_unsup_s['image'].shape[0]==len(teacher_bboxes)
        for i in range(data_unsup_s['image'].shape[0]):
            if teacher_bboxes[i].sum()==0:
                teacher_bboxes[i]=paddle.zeros([1,4])
            
            else:
                cur_one_tensor = paddle.to_tensor([1.0, 0.0, 0.0, 0.0])
                cur_one_tensor = cur_one_tensor
                cur_one_tensor = cur_one_tensor.tile([teacher_bboxes[i].shape[0], 1])
                if 'flipped' in teacher_data.keys() and teacher_data['flipped']:
                    original_boxes = paddle.abs(cur_one_tensor - teacher_bboxes[i])
                else:
                    original_boxes = teacher_bboxes[i]

                # if 'filpped' in records_unlabel_q.keys() and records_unlabel_q['filpped']:
                #     cur_boxes = cur_boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]).cuda() + torch.as_tensor([img_w, 0, img_w, 0]).cuda()

                original_boxes = bbox_cxcywh_to_xyxy(original_boxes)
                img_w = teacher_data['OriginalImageSize'][i][1]
                img_h = teacher_data['OriginalImageSize'][i][0]
                scale_fct = paddle.to_tensor([img_w, img_h, img_w, img_h])
                original_boxes = original_boxes * scale_fct
            
                cur_boxes = paddle.clone(original_boxes)
                cur_labels = paddle.clone(teacher_labels[i])
                if 'flipped' in  data_unsup_s.keys() and  data_unsup_s['flipped']:
                   cur_boxes = paddle.index_select(x=cur_boxes, index=paddle.to_tensor([2,1,0,3]), axis=1) * paddle.to_tensor([-1, 1, -1, 1]) + paddle.to_tensor([img_w, 0, img_w, 0])
  
                if data_unsup_s['RandomResize_times'][i] > 1:
                    rescaled_size1 = data_unsup_s['RandomResize_scale'][i][0]
                    
                    rescaled_size1 = get_size_with_aspect_ratio((img_w, img_h), rescaled_size1)
                    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_size1, (img_w, img_h)))
                    ratio_width, ratio_height = ratios
                    cur_boxes = cur_boxes * paddle.to_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
                    img_w = rescaled_size1[0]
                    img_h = rescaled_size1[1]

                
                    region = data_unsup_s['RandomSizeCrop'][i]
                    i1, j1, h, w = region
                    fields = ["labels", "area", "iscrowd"]
                    max_size = paddle.to_tensor([w, h], dtype='float32')
                    cropped_boxes = cur_boxes - paddle.to_tensor([j1, i1, j1, i1])
                    cropped_boxes = paddle.minimum(cropped_boxes.reshape([-1, 2, 2]), max_size)
                    cropped_boxes = cropped_boxes.clip(min=0)
                    area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(axis=1)
                    cur_boxes = cropped_boxes.reshape([-1, 4])
                    gt_before = copy.deepcopy(cur_boxes)
                    fields.append("boxes")
                    cropped_boxes = paddle.clone(cur_boxes)
                    cropped_boxes = cropped_boxes.reshape([-1, 2, 2])
                    keep = paddle.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], axis=1)
                    if False in keep:
                        print(1)
                    cur_boxes = cur_boxes[keep]
                    cur_labels = cur_labels[keep]
                    if cur_boxes.shape[0]!=0:
                        img_w = w
                        img_h = h
                        # random resize
                        rescaled_size2 = data_unsup_s['RandomResize_scale'][i][1]
                        rescaled_size2 = get_size_with_aspect_ratio((img_w, img_h), rescaled_size2, max_size=max_pixels)
                        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_size2, (img_w, img_h)))
                        ratio_width, ratio_height = ratios
                        cur_boxes = cur_boxes * paddle.to_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
                        gt_before = gt_before * paddle.to_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
                        img_w = rescaled_size2[0]
                        img_h = rescaled_size2[1]
                else:
                    # random resize
                    rescaled_size1 = data_unsup_s['RandomResize_scale'][i][0]
                    rescaled_size1 = get_size_with_aspect_ratio((img_w, img_h), rescaled_size1, max_size=max_pixels)
                    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_size1, (img_w, img_h)))
                    ratio_width, ratio_height = ratios
                    cur_boxes = cur_boxes * paddle.to_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
                    gt_before = copy.deepcopy(cur_boxes)
                    img_w = rescaled_size1[0]
                    img_h = rescaled_size1[1]

                # finally, deal with normalize part in deformable detr aug code
                if cur_boxes.shape[0]!=0:
                    gt = paddle.clone(cur_boxes)
                    cur_boxes = bbox_xyxy_to_cxcywh(cur_boxes)
                    
                    cur_boxes = cur_boxes / paddle.to_tensor([img_w, img_h, img_w, img_h], dtype=paddle.float32)   

                if 'RandomErasing1' in data_unsup_s.keys() and cur_boxes.shape[0]!=0:
                    region = data_unsup_s['RandomErasing1'][0]
                    i1, j1, h1, w1, _ = region
                    cur_boxes_xy = bbox_cxcywh_to_xyxy(cur_boxes)
                    i1 = i1 / img_h
                    j1 = j1 / img_w
                    h1 = h1 / img_h
                    w1 = w1 / img_w
                    keep = ~((cur_boxes_xy[:, 0] > j1) & (cur_boxes_xy[:, 1] > i1) & (cur_boxes_xy[:, 2] < j1 + w1) & (cur_boxes_xy[:, 3] < i1 + h1))
                    cur_boxes = cur_boxes[keep]
                    cur_labels = cur_labels[keep]
                    if cur_boxes.shape[0]==0:
                        cur_labels=paddle.zeros([0,1])

                if 'RandomErasing2' in data_unsup_s.keys() and cur_boxes.shape[0]!=0:
                    region = data_unsup_s['RandomErasing2'][0]
                    i1, j1, h1, w1, _ = region
                    cur_boxes_xy = bbox_cxcywh_to_xyxy(cur_boxes)
                    i1 = i1 / img_h
                    j1 = j1 / img_w
                    h1 = h1 / img_h
                    w1 = w1 / img_w
                    keep = ~((cur_boxes_xy[:, 0] > j1) & (cur_boxes_xy[:, 1] > i1) & (cur_boxes_xy[:, 2] < j1 + w1) & (cur_boxes_xy[:, 3] < i1 + h1))
                    cur_boxes = cur_boxes[keep]
                    cur_labels = cur_labels[keep]
                    if cur_boxes.shape[0]==0:
                        cur_labels=paddle.zeros([0,1])
                if 'RandomErasing3' in data_unsup_s.keys() and cur_boxes.shape[0]!=0:
                    region = data_unsup_s['RandomErasing3'][0]
                    i1, j1, h1, w1, _ = region
                    cur_boxes_xy = bbox_cxcywh_to_xyxy(cur_boxes)
                    i1 = i1 / img_h
                    j1 = j1 / img_w
                    h1 = h1 / img_h
                    w1 = w1 / img_w
                    keep = ~((cur_boxes_xy[:, 0] > j1) & (cur_boxes_xy[:, 1] > i1) & (cur_boxes_xy[:, 2] < j1 + w1) & (cur_boxes_xy[:, 3] < i1 + h1))
                    cur_boxes = cur_boxes[keep]
                    cur_labels = cur_labels[keep]
                    if cur_boxes.shape[0]==0:
                        cur_labels=paddle.zeros([0,1])
                # if 'keep' in locals().keys():
                #    print(keep)
                assert cur_boxes.shape[0] == cur_labels.shape[0]
        teacher_info=[teacher_bboxes,teacher_labels]
        student_info=student_data

        return self.compute_pseudo_label_loss(student_info, teacher_info)

    def compute_pseudo_label_loss(self, student_info, teacher_info):                                 

        pseudo_bboxes=list(teacher_info[0])
        pseudo_labels=list(teacher_info[1])
        student_data=student_info
        self.place=student_data['gt_bbox'][0].place
        # print(pseudo_labels)
        losses = dict()
        for i in range(len(pseudo_bboxes)):
            if pseudo_labels[i].shape[0]==0:
                pseudo_bboxes[i]=paddle.zeros([1,4])
                pseudo_labels[i]=paddle.zeros([1,1])
            else:
                pseudo_bboxes[i]=pseudo_bboxes[i][:,:4]
                pseudo_labels[i]=pseudo_labels[i].unsqueeze(-1)
        for i in range(len(pseudo_bboxes)):
            pseudo_labels[i]= paddle.to_tensor(pseudo_labels[i],dtype=paddle.int32,place=self.place)
            pseudo_bboxes[i]= paddle.to_tensor(pseudo_bboxes[i],dtype=paddle.float32,place=self.place)
        # print(pseudo_bboxes[0].shape[0])
        student_data['gt_bbox'].extend(pseudo_bboxes)  
        student_data['gt_class'].extend(pseudo_labels)
        
        # student_data.update(gt_bbox=pseudo_bboxes,gt_class=pseudo_labels)
        body_feats=self.student.backbone(student_data)
        pad_mask = student_data['pad_mask'] if self.training else None
        out_transformer = self.student.transformer(body_feats, pad_mask)
        losses = self.student.detr_head(out_transformer, body_feats, student_data)

            # losses['loss'] = paddle.zeros([1], dtype='float32')
            # losses['loss_class'] = paddle.zeros([1], dtype='float32')
            # losses['loss_bbox'] = paddle.zeros([1], dtype='float32')
            # losses['loss_giou'] = paddle.zeros([1], dtype='float32')
            # losses['loss_class_aux'] = paddle.zeros([1], dtype='float32')
            # losses['loss_bbox_aux'] = paddle.zeros([1], dtype='float32')
            # losses['loss_giou_aux'] = paddle.zeros([1], dtype='float32')
        return losses



    def normalize_box(self,sample,):
        im = sample['image']
        if  'gt_bbox' in sample.keys():
            gt_bbox = sample['gt_bbox']
            gt_class = sample['gt_class']
            # _, _, height, width, = im.shape
            for i in range(len(gt_bbox)):
                # for j in range(gt_bbox[i].shape[0]):
                #     gt_bbox[i][j][0] = gt_bbox[i][j][0] / width
                #     gt_bbox[i][j][1] = gt_bbox[i][j][1] / height
                #     gt_bbox[i][j][2] = gt_bbox[i][j][2] / width
                #     gt_bbox[i][j][3] = gt_bbox[i][j][3] / height
                    gt_class[i]= paddle.to_tensor(gt_class[i],dtype=paddle.int32,place=self.place)
            sample['gt_bbox'] = gt_bbox
            sample['gt_class'] = gt_class
        if  'gt_bbox' in sample.keys():
            bbox = sample['gt_bbox']
            for i in range(len(bbox)):
                # bbox[i][:, 2:4] = bbox[i][:, 2:4] - bbox[i][:, :2]
                # bbox[i][:, :2] = bbox[i][:, :2] + bbox[i][:, 2:4] / 2.
                bbox[i]= paddle.to_tensor(bbox[i],dtype=paddle.float32,place=self.place)
            sample['gt_bbox'] = bbox

        
        return sample


def bbox_cxcywh_to_xyxy(x):
    cxcy, wh = paddle.split(x, 2, axis=-1)
    return paddle.concat([cxcy - 0.5 * wh, cxcy + 0.5 * wh], axis=-1)


def bbox_xyxy_to_cxcywh(x):
    x1, y1, x2, y2 = x.split(4, axis=-1)
    return paddle.concat(
        [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)], axis=-1)

def get_size_with_aspect_ratio(image_size, size, max_size=None):
    w, h = image_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    if (w <= h and w == size) or (h <= w and h == size):
        return (w, h)

    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)

    return (ow, oh)

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes