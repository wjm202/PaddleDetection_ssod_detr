"""
LabelMatch
"""
import os
import numpy as np
from operator import itemgetter
import paddle
import paddle.nn.functional as F
import paddle.distributed as dist
from paddle.fluid import framework
from ppdet.core.workspace import register, create
from ppdet.data.reader import get_dist_info
from ppdet.modeling.proposal_generator.target import label_box
from ppdet.modeling.bbox_utils import delta2bbox
from ppdet.data.transform.atss_assigner import bbox_overlaps
from ppdet.utils.logger import setup_logger

from .multi_stream_detector import MultiSteamDetector
from .utils import Transform2D
from .logger import log_image_with_boxes
logger = setup_logger(__name__)

__all__ = [
    'LabelMatch'
]

@register
class LabelMatch(MultiSteamDetector):
    def __init__(self, teacher, student, cfg=None, train_cfg=None, test_cfg=None, **kwargs):
        super(LabelMatch, self).__init__(
            dict(teacher=teacher, student=student),
            train_cfg=train_cfg,
            test_cfg=test_cfg
        )
        self.debug = cfg.get('debug', False)
        self.interval = cfg.get('interval', 1000)
        # self.debug = True
        # self.interval = 1

        self.CLASSES = cfg.get('classes', [])
        self.num_classes = len(self.CLASSES)
        self.save_dir = cfg.get('save_dir', '')

        self.cur_iter = 0

        # hyper-parameter: fixed
        self.tpt = cfg.get('tpt', 0.5)
        self.tps = cfg.get('tps', 1.0)
        self.momentum = cfg.get('momentum', 0.996)
        self.weight_u = cfg.get('weight_u', 2.0)
        logger.info(f'set ema momentum is: {self.momentum}')
        # adat
        score_thr = cfg.get('score_thr', 0.9)  # if not use ACT, will use this hard thr
        self.cls_thr = [0.9 if self.debug else score_thr] * self.num_classes
        self.cls_thr_ig = [0.2 if self.debug else score_thr] * self.num_classes
        self.percent = cfg.get('percent', 0.2)

        # mining
        self.use_mining = cfg.get('use_mining', True)
        self.reliable_thr = cfg.get('reliable_thr', 0.8)
        self.reliable_iou = cfg.get('reliable_iou', 0.8)

        # analysis
        self.image_num = 0
        self.pseudo_num = np.zeros(self.num_classes)
        self.pseudo_num_ig = np.zeros(self.num_classes)
        self.pseudo_num_tp = np.zeros(self.num_classes)
        self.pseudo_num_gt = np.zeros(self.num_classes)
        self.pseudo_num_tp_ig = np.zeros(self.num_classes)
        self.pseudo_num_mining = np.zeros(self.num_classes)

        if train_cfg is not None:
            self.freeze("teacher")
            self._teacher = None
            self._student = None

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        teacher = create(cfg['teacher'])
        student = create(cfg['student'])
        train_cfg = cfg['train_cfg']
        test_cfg = cfg['test_cfg']
        cfg['classes'] = kwargs['classes']
        cfg['save_dir'] = kwargs['save_dir']
        return {
            'teacher': teacher,
            'student': student,
            'cfg': cfg,
            'train_cfg': train_cfg,
            'test_cfg' : test_cfg
        }

    def forward_train(self, inputs, **kwargs):
        # if dist.get_world_size() > 1 and self._teacher == None:
        #     self._teacher = self.teacher
        #     self.teacher = self.teacher._layers
        #     self._student = self.student
        #     self.student = self.student._layers

        imgs = inputs['image']
        img_metas = inputs
        img_metas.pop('epoch_id')

        self.update_ema_model(self.momentum)
        
        batch_input_shape = tuple(imgs[0].shape[-2:])
        img_metas['batch_input_shape'] = [batch_input_shape] * imgs.shape[0]

        tag_dict = {1: "sup", 2: "unsup_teacher", 3: "unsup_student"}
        new_tag = [tag_dict[tag] for tag in list(img_metas['tag'].numpy())]
        img_metas['tag'] = new_tag
        group_names = list(set(img_metas['tag']))

        sup_index = [i for i, tag in enumerate(new_tag) if tag == "sup"]
        unsup_teacher_index = [i for i, tag in enumerate(new_tag) if tag == "unsup_teacher"]
        unsup_student_index = [i for i, tag in enumerate(new_tag) if tag == "unsup_student"]
        
        data_groups = {}
        data_groups.update({
            'sup': {},
            'unsup_teacher': {},
            'unsup_student': {}
        })
        for k, v in img_metas.items():
            for tag_id, indexes in enumerate([sup_index, unsup_teacher_index, unsup_student_index]):
                tag = tag_dict[tag_id + 1]
                data_groups[tag][k] = [v[i] for i in indexes]
            
        for tag, info in data_groups.items():
            for k, v in info.items():
                if k in ['image', 'im_shape', 'scale_factor']:
                    data_groups[tag][k] = paddle.stack(data_groups[tag][k])
            # print(tag, info['image'].shape)

        # loss
        loss = {}
        # # ---------------------label data---------------------
        if "sup" in data_groups:
            gt_bboxes = data_groups["sup"]["gt_bbox"]
            sup_loss = self.student.forward(data_groups["sup"])
            sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
            sup_loss.pop('sup_loss')
            loss.update(**sup_loss)
            draw_flag = 1
            if self.debug and get_dist_info()[0] == 0:
                if self.save_dir!= '':
                    save_path = os.path.join(self.save_dir.replace("/weights", "").replace("/weight", ""), 'show', 'sup_data_gt')
                else:
                    save_path = 'show/sup_data_gt'
                for i in range(len(data_groups["sup"]["gt_class"])):
                    log_image_with_boxes(
                        save_path,
                        data_groups["sup"]["image"][i],
                        gt_bboxes[i],
                        labels=[self.CLASSES[item.item()] for item in data_groups["sup"]["gt_class"][i]],
                        interval=self.interval,
                        cnt = i,
                        flag = draw_flag)
                    draw_flag = 0
        # sup data
        sup_num = data_groups["sup"]["image"].shape[0]
        img_metas = []
        for i in range(sup_num):
            tmp_dict = {}
            tmp_dict['tag'] = data_groups["sup"]['tag'][i]
            tmp_dict['batch_input_shape'] = data_groups["sup"]['batch_input_shape'][i]
            tmp_dict['scale_factor'] = data_groups["sup"]['scale_factor'][i]
            tmp_dict['transform_matrix'] = data_groups["sup"]['transform_matrix'][i]
            tmp_dict['img_shape'] = tuple((data_groups["sup"]['image'][i].shape))
            tmp_dict['img_shape'] = tmp_dict['img_shape'][1:] + (tmp_dict['img_shape'][0],)
            img_metas.append(tmp_dict)

        # unsup data
        teacher_data = data_groups["unsup_teacher"]
        student_data = data_groups["unsup_student"]

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
        
        # 数据解析
        img = data_groups['sup']['image']

        img_unlabeled = teacher_data["image"]
        img_metas_unlabeled = teacher_data['img_metas']
        gt_bboxes_unlabeled = teacher_data['gt_bbox']
        gt_labels_unlabeled = teacher_data['gt_class']

        img_unlabeled_1 = student_data['image']
        img_metas_unlabeled_1 = student_data['img_metas']
        gt_bboxes_unlabeled_1 = student_data['gt_bbox']
        gt_labels_unlabeled_1 = student_data['gt_class']
        
        _, _, h, w = img_unlabeled_1.shape
        self.image_num += unsup_num
        self.cur_iter += 1
        self.analysis()  # record the information in the training

        # -------------------unlabeled data-------------------
        # create pseudo label
        # results = self.teacher(teacher_data)
        self.inputs = teacher_data
        body_feats = self.teacher.backbone(self.inputs)
        if self.teacher.neck is not None:
            body_feats = self.teacher.neck(body_feats)

        rois, rois_num, _ = self.teacher.rpn_head(body_feats, self.inputs)
        preds, _ = self.teacher.bbox_head(body_feats, rois, rois_num, None)

        im_shape = self.inputs['im_shape']
        scale_factor = self.inputs['scale_factor']
        bbox, bbox_num = self.teacher.bbox_post_process(preds, (rois, rois_num),
                                                im_shape, scale_factor)
        results = {'bbox': bbox, 'bbox_num': bbox_num}
        if self.debug and get_dist_info()[0] == 0:
            draw_flag = 1
            b_num = 0
            if self.save_dir!= '':
                save_path = os.path.join(self.save_dir.replace("/weights", "").replace("/weight", ""), 'show', 'unlabeled_teacher')
            else:
                save_path = 'show/unlabeled_teacher'
            for ii in range(img_unlabeled.shape[0]):
                temp = results['bbox'][b_num:results['bbox_num'][ii].item()+b_num,:].numpy()
                # temp = temp[temp[:,1]>0.1]
                lls = [self.CLASSES[int(tt[0])] for tt in temp]
                temp = paddle.to_tensor(temp[:, 2:])
                log_image_with_boxes(
                    save_path,
                    img_unlabeled[ii],
                    temp,
                    labels=lls,
                    interval=self.interval,
                    cnt = ii,
                    flag = draw_flag)
                draw_flag = 0
                b_num += results['bbox_num'][ii].item()
        bbox_results = [[np.empty([0, 5]) for j in range(self.num_classes)] for i in range(results['bbox_num'].shape[0])]
        bbox_num = [num.item() for num in results['bbox_num']]
        bbox_results_split = paddle.split(results['bbox'], num_or_sections=bbox_num, axis=0)
        for bid, res in enumerate(bbox_results_split):
            for i in range(res.shape[0]):
                bbox_results[bid][res[i][0]] = np.append(bbox_results[bid][res[i][0]], np.append(res[i][2:], res[i][1]).reshape(1, -1), axis=0)

        gt_bboxes_pred, gt_labels_pred, gt_bboxes_ig_pred, gt_labels_ig_pred = \
            self.create_pseudo_results(
                img_unlabeled_1, img_metas_unlabeled_1, bbox_results, teacher_data['transform_matrix'], student_data['transform_matrix'],
                gt_bboxes_unlabeled, gt_labels_unlabeled, img_metas_unlabeled # for analysis
            )
        # training on unlabeled data
        losses_unlabeled = self.training_unlabeled(
            teacher_data,
            student_data,
            img_unlabeled_1, img_metas_unlabeled_1, student_data['transform_matrix'],
            img_unlabeled, img_metas_unlabeled, teacher_data['transform_matrix'],
            gt_bboxes_pred, gt_labels_pred, gt_bboxes_ig_pred, gt_labels_ig_pred
        )
        # losses_unlabeled = self.parse_loss(losses_unlabeled)
        
        for key, val in losses_unlabeled.items():
            if key.find('loss') == -1:
                continue
            else:
                losses_unlabeled[key] = self.weight_u * val
        unsup_loss = {"unsup_" + k: v for k, v in losses_unlabeled.items()}
        loss.update(**unsup_loss)
        # losses.update({f'{key}_unlabeled': val for key, val in losses_unlabeled.items()})

        # extra info for analysis
        extra_info = {
            'pseudo_num': paddle.to_tensor([self.pseudo_num.sum() / (self.image_num + 1e-10)]),
            'pseudo_num_ig': paddle.to_tensor([self.pseudo_num_ig.sum() / (self.image_num+ 1e-10)]),
            'pseudo_num_mining': paddle.to_tensor([self.pseudo_num_mining.sum() / (self.image_num + 1e-10)]),
            'pseudo_num(acc)': paddle.to_tensor([self.pseudo_num_tp.sum() / (self.pseudo_num.sum() + 1e-10)]),
            'pseudo_num ig(acc)': paddle.to_tensor([self.pseudo_num_tp_ig.sum() / (self.pseudo_num_ig.sum() + 1e-10)]),
        }

        infos = {}
        infos.update(extra_info)
        infos.update(loss)
        total_loss = paddle.add_n(list(loss.values()))
        infos.update({'loss': total_loss})
        return infos

    # # ---------------------------------------------------------------------------------
    # # training on unlabeled data
    # # ---------------------------------------------------------------------------------
    def training_unlabeled(self, teacher_data, student_data, img, img_metas, bbox_transform,
                           img_t, img_metas_t, bbox_transform_t,
                           gt_bboxes, gt_labels, gt_bboxes_ig, gt_labels_ig):
        losses = dict()
        feat = self.student.backbone(student_data)
        if self.student.neck is not None:
            feat = self.student.neck(feat)
        # rpn_head -> get rpn_out 
        rpn_feats = self.student.rpn_head.rpn_feat(feat)
        scores = []
        deltas = []

        for rpn_feat in rpn_feats:
            rrs = self.student.rpn_head.rpn_rois_score(rpn_feat)
            rrd = self.student.rpn_head.rpn_rois_delta(rpn_feat)
            scores.append(rrs)
            deltas.append(rrd)

        # rpn loss
        # gt_bboxes_cmb = [paddle.concat(x=[a, b], axis=0) for a, b in zip(gt_bboxes, gt_bboxes_ig)]
        gt_bboxes_cmb = []
        for a, b in zip(gt_bboxes, gt_bboxes_ig):
            if a.shape[0] == 0 and b.shape[0] == 0:
                gt_bboxes_cmb.append(a)
            else:
                temp = paddle.concat(x=[a, b], axis=0) 
                gt_bboxes_cmb.append(temp)

        anchors = self.student.rpn_head.anchor_generator(rpn_feats)
        inputs = student_data.copy()
        # inputs['gt_bbox'] = [paddle.cast(gt, 'float32') for gt in gt_bboxes_cmb]
        inputs['gt_bbox'] = gt_bboxes_cmb
        inputs['gt_class'] = None
        rois, rois_num = self.student.rpn_head._gen_proposal(scores, deltas, anchors, inputs)
        rpn_losses = self.student.rpn_head.get_loss(scores, deltas, anchors, inputs)
        losses.update(rpn_losses)
        # proposal_list
        # output_rois_prob = [item.clone() for item in self.student.rpn_head.output_rois_prob]
        # proposal_list = [paddle.concat([_rois, _output_rois_prob.unsqueeze(axis=1)], axis=1) 
        #                     for (_rois, _output_rois_prob) in zip(rois, output_rois_prob)]

        # roi loss
        inputs['gt_bbox'] = gt_bboxes
        inputs['gt_class'] = [paddle.to_tensor(la.reshape(-1, 1), dtype='int32') for la in gt_labels]
        inputs['gt_bbox_ig'] = gt_bboxes_ig
        inputs['gt_class_ig'] = [paddle.to_tensor(la_ig.reshape(-1, 1), dtype='int32') for la_ig in gt_labels_ig]
        sampling_results = self.student.bbox_head.forward_train_step1(feat, rois, rois_num, inputs)
        
        if sum([gt.shape[0] for gt in gt_labels]) > 0:
            # rois = [pro[:, :4]for pro in proposal_list]
            rois_num = paddle.to_tensor([i.shape[0] for i in rois], dtype=np.int32)
            # rois, rois_num, targets = self.student.bbox_head.bbox_assigner(rois, rois_num, inputs)
            
            gt_classes = inputs['gt_class']
            is_crowd = inputs.get('is_crowd', None)

        # teacher model to get pred

        ig_boxes = [ig if ig.shape[0] == 0 and res['ig_bboxes'].shape[0] == 0 else
                    paddle.concat([ig, res['ig_bboxes']], axis=0) for ig, res in zip(gt_bboxes_ig, sampling_results)]
        ig_len = [len(ig) for ig in gt_bboxes_ig]

        Ms = Transform2D.get_trans_mat(bbox_transform, bbox_transform_t)
        draw_flag1 = 1
        for i in range(len(img_metas_t)):
            
            ig_boxes[i] = self._transform_bbox(
            paddle.to_tensor(ig_boxes[i], dtype='float32'), # list b, [512,4]
            Ms[i], # list b, [3,3]
            img_metas_t[i]["img_shape"],
            )
            if self.debug and get_dist_info()[0] == 0:
                if self.save_dir!= '':
                    save_path = os.path.join(self.save_dir.replace("/weights", "").replace("/weight", ""), 'show', 'unlabeled_student_trans_ig')
                else:
                    save_path = 'show/unlabeled_student_trans_ig'
                lab_ig = ['00000000000000000000_' + self.CLASSES[gt_labels_ig[i][i_l].item()] for i_l in range(ig_len[i])] + ['11' for i_l in range(ig_boxes[i].shape[0]-ig_len[i])]
                log_image_with_boxes(
                    save_path,
                    img_t[i],
                    ig_boxes[i],
                    labels=lab_ig,
                    interval=self.interval,
                    cnt = i,
                    flag = draw_flag1)
                draw_flag1 = 0
        ignore_boxes_t = [b[:l] for l, b in zip(ig_len, ig_boxes)]
        ig_boxes = [b[l:] for l, b in zip(ig_len, ig_boxes)]
        
        # ignore_boxes_t = gt_bboxes_ig
        # ig_boxes = [res['ig_bboxes'] for res in sampling_results]

        with paddle.no_grad():
            feat_t = self.teacher.backbone(teacher_data)
            if self.teacher.neck is not None:
                feat_t = self.teacher.neck(feat_t)
            if sum([ig_bbox.shape[0] for ig_bbox in ig_boxes]) > 0:
                det_bboxes_t, det_labels_t = self.simple_test_bboxes_base(
                    teacher_data, 
                    feat_t, img_metas_t, ig_boxes)
            else:
                det_bboxes_t = paddle.to_tensor([])
                det_labels_t = paddle.to_tensor([])
            cls_scores_t = [F.softmax(l / self.tpt, axis=-1) for l in det_labels_t]
            det_labels_t = [F.softmax(l, axis=-1) for l in det_labels_t]
        
        # mining
        draw_flag2 = 1
        for n, res in enumerate(sampling_results):
            for i in range(max(res['ig_assigned_gt_inds']) + 1 if len(res['ig_assigned_gt_inds']) > 0 else 0):
                flag = res['ig_assigned_gt_inds'] == i
                if flag.sum() < 1:
                    continue
                cls_cur = gt_labels_ig[n][i]
                if self.use_mining:
                    mean_iou = bbox_overlaps(ignore_boxes_t[n][i:i + 1].numpy(),
                                                              det_bboxes_t[n][flag].numpy()).mean()
                    mean_score = det_labels_t[n][flag].numpy()[:, cls_cur].mean()
                    if mean_iou >= self.reliable_iou and mean_score >= self.reliable_thr:
                        res['ig_reg_weight'][flag] = 1.0
                        self.pseudo_num_mining[cls_cur] += 1
                if self.debug and get_dist_info()[0] == 0:
                    if self.save_dir!= '':
                        save_path = os.path.join(self.save_dir.replace("/weights", "").replace("/weight", ""), 'show', 'unlabeled_student_mining')
                    else:
                        save_path = 'show/unlabeled_student_mining'
                    # 0：iggt 1：igpred 2：igdet
                    lab_ig = ['0000_'+ self.CLASSES[cls_cur]] + ['111' for i in range(flag.sum().item())]+ ['2' for i in range(flag.sum().item())]
                    log_image_with_boxes(
                        save_path,
                        img_t[n],
                        paddle.concat([ignore_boxes_t[n][i:i + 1] ,ig_boxes[n][flag], det_bboxes_t[n][flag]], axis=0),
                        labels=lab_ig,
                        interval=self.interval,
                        cnt = f"{n}_{i}",
                        flag = draw_flag2)
                    draw_flag2 = 0
        inputs['gt_bbox'] = gt_bboxes
        inputs['gt_class'] = [paddle.to_tensor(la.reshape(-1, 1), dtype='int32') for la in gt_labels]

        roi_losses, cls_scores = self.student.bbox_head.forward_train_step2(
                                feat, sampling_results, inputs)
        losses.update(roi_losses)

        # proposal based learning
        if len(cls_scores) > 0:
            cls_scores_t = paddle.concat(cls_scores_t, axis=0)
            assert len(cls_scores_t) == cls_scores.shape[0], "cls_scores_t not equal cls_scores"
            cls_scores = F.softmax(cls_scores / self.tps, axis=-1)
            weight = paddle.concat([1 - res['ig_reg_weight']  for res in sampling_results], axis=0)
            avg_factor = len(img_metas) * self.student.bbox_head.bbox_assigner.batch_size_per_im
            losses_cls_ig = (-cls_scores_t * paddle.log(cls_scores)).sum(-1)
            losses_cls_ig = (losses_cls_ig * weight).sum() / avg_factor
        else:
            losses_cls_ig = cls_scores.sum()  # 0
        losses.update({'losses_cls_ig': losses_cls_ig})
        return losses

    # # ---------------------------------------------------------------------------------
    # # create pseudo labels
    # # ---------------------------------------------------------------------------------
    def create_pseudo_results(self, img, img_metas, bbox_results, transform_m_t, transform_m_s,
                              gt_bboxes=None, gt_labels=None, gt_img_metas=None):
        """using dynamic score to create pseudo results"""
        gt_bboxes_pred, gt_labels_pred = [], []
        gt_bboxes_ig_pred, gt_labels_ig_pred = [], []
        _, _, h, w = img.shape
        use_gt = gt_bboxes is not None

        Ms = Transform2D.get_trans_mat(transform_m_t, transform_m_s)
        draw_flag = 1
        for b, result in enumerate(bbox_results):
            bboxes, labels = [], []
            bboxes_ig, labels_ig = [], []
            if use_gt:
                gt_bbox, gt_label = gt_bboxes[b].numpy(), gt_labels[b].numpy()
                gt_bbox_scale = self._transform_bbox(
                    paddle.to_tensor(gt_bbox, dtype='float32'), # list b, [512,4]
                    Ms[b], # list b, [3,3]
                    img_metas[b]["img_shape"],
                    )
            for cls, r in enumerate(result):
                label = cls * np.ones_like(r[:, 0], dtype=np.uint8)
                flag_pos = r[:, -1] >= self.cls_thr[cls]
                flag_ig = (r[:, -1] >= self.cls_thr_ig[cls]) & (~flag_pos)
                bboxes.append(r[flag_pos][:, :4])
                bboxes_ig.append(r[flag_ig][:, :4])
                labels.append(label[flag_pos])
                labels_ig.append(label[flag_ig])
                if use_gt and (gt_label == cls).sum() > 0 and len(bboxes[-1]) > 0:
                    temp = bboxes[-1].copy()
                    temp = self._transform_bbox(
                                paddle.to_tensor(temp, dtype='float32'), # list b, [512,4]
                                Ms[b], # list b, [3,3]
                                img_metas[b]["img_shape"],
                                )
                    overlap = self.bbox_overlaps(temp.numpy(), gt_bbox_scale[(gt_label == cls).flatten()].numpy())
                    iou = overlap.max(-1)
                    self.pseudo_num_tp[cls] += (iou > 0.5).sum()
                if use_gt and (gt_label == cls).sum() > 0 and len(bboxes_ig[-1]) > 0:
                    temp = bboxes_ig[-1].copy()
                    temp = self._transform_bbox(
                                paddle.to_tensor(temp, dtype='float32'), # list b, [512,4]
                                Ms[b], # list b, [3,3]
                                img_metas[b]["img_shape"],
                                )
                    overlap = self.bbox_overlaps(temp.numpy(), gt_bbox_scale[(gt_label == cls).flatten()].numpy())
                    iou = overlap.max(-1)
                    self.pseudo_num_tp_ig[cls] += (iou > 0.5).sum()
                self.pseudo_num_gt[cls] += (gt_label == cls).sum()
                self.pseudo_num[cls] += len(bboxes[-1])
                self.pseudo_num_ig[cls] += len(bboxes_ig[-1])
            bboxes = np.concatenate(bboxes)
            bboxes_ig = np.concatenate(bboxes_ig)
            bboxes_concat = np.r_[bboxes, bboxes_ig]
            labels = np.concatenate(labels)
            labels_ig = np.concatenate(labels_ig)

            bboxes_concat = self._transform_bbox(
            paddle.to_tensor(bboxes_concat, dtype='float32'), # list b, [512,4]
            Ms[b], # list b, [3,3]
            img_metas[b]["img_shape"],
            ) # list b, [512,4]
                # bboxes_concat, labels = BboxResize(bboxes_concat, labels)
            if isinstance(bboxes_concat, paddle.Tensor):
                bboxes_concat = bboxes_concat.numpy()
            bboxes, bboxes_ig = bboxes_concat[:len(bboxes)], bboxes_concat[len(bboxes):]
            bboxes = paddle.to_tensor(bboxes)
            bboxes_ig = paddle.to_tensor(bboxes_ig)

            if self.debug and get_dist_info()[0] == 0:
                if self.save_dir!= '':
                    save_path = os.path.join(self.save_dir.replace("/weights", "").replace("/weight", ""), 'show')
                else:
                    save_path = 'show'
                log_image_with_boxes(
                    os.path.join(save_path, "unlabeled_teacher_gt"),
                    img[b],
                    gt_bbox_scale,
                    labels=[self.CLASSES[ll]for lls in gt_label.tolist()  for ll in lls],
                    interval=self.interval,
                    cnt = b,
                    flag = draw_flag)
                log_image_with_boxes(
                    os.path.join(save_path, "unlabeled_teacher_trans_pos"),
                    img[b],
                    bboxes,
                    labels=[self.CLASSES[ll] for ll in labels.tolist()],
                    interval=self.interval,
                    cnt = b,
                    flag = draw_flag)
                log_image_with_boxes(
                    os.path.join(save_path, "unlabeled_teacher_trans_ig"),
                    img[b],
                    bboxes_ig,
                    labels=[self.CLASSES[ll] for ll in labels_ig.tolist()],
                    interval=self.interval,
                    cnt = b,
                    flag = draw_flag)
                draw_flag = 0
            gt_bboxes_pred.append(bboxes)
            gt_labels_pred.append(labels)
            gt_bboxes_ig_pred.append(bboxes_ig)
            gt_labels_ig_pred.append(labels_ig)
        return gt_bboxes_pred, gt_labels_pred, gt_bboxes_ig_pred, gt_labels_ig_pred

    # # -----------------------------analysis function------------------------------
    def analysis(self):
        if self.cur_iter % 500 == 0 and get_dist_info()[0] == 0:
            info = ' '.join([f'{b / (a + 1e-10):.2f}({a}-{cls})' for cls, a, b
                             in zip(self.CLASSES, self.pseudo_num, self.pseudo_num_tp)])
            info_ig = ' '.join([f'{b / (a + 1e-10):.2f}({a}-{cls})' for cls, a, b
                                in zip(self.CLASSES, self.pseudo_num_ig, self.pseudo_num_tp_ig)])
            info_gt = ' '.join([f'{a}' for a in self.pseudo_num_gt])
            logger.info(f'pseudo pos: {info}')
            logger.info(f'pseudo ig: {info_ig}')
            logger.info(f'pseudo gt: {info_gt}')
            if self.use_mining:
                info_mining = ' '.join([f'{a}' for a in self.pseudo_num_mining])
                logger.info(f'pseudo mining: {info_mining}')
                  
    def _transform_bbox(self, bboxes, trans_mat, max_shape):
        bboxes = Transform2D.transform_bboxes(bboxes, trans_mat, max_shape)
        return bboxes


    def simple_test_bboxes_base(self, teacher_data, feat, img_metas, proposals):
        rois = proposals
        rois_num = paddle.to_tensor([i.shape[0] for i in proposals], dtype=np.int32)
        
        rois_feat = self.teacher.bbox_head.roi_extractor(feat, rois, rois_num)
        bbox_feat = self.teacher.bbox_head.head(rois_feat)
        # preds, _ = self.teacher.bbox_head(feat, proposals, rois_num, None)
        scores = self.teacher.bbox_head.bbox_score(bbox_feat)
        deltas = self.teacher.bbox_head.bbox_delta(bbox_feat)
        
        preds = (deltas, scores)
        bboxes, bbox_num = self.teacher.bbox_post_process(preds, (proposals, rois_num),
                                            teacher_data['im_shape'], teacher_data['scale_factor'], use_nms=False)
        max_idxs = F.softmax(scores)[:, :-1].argmax(axis=1) 
        bboxes = bboxes.reshape((bboxes.shape[0], -1, 4))
        bboxes = paddle.concat([bboxes[i, max_idxs[i], :].reshape((1,4)) for i in range(bboxes.shape[0])])
        bboxes = bboxes.split(tuple(np.array(rois_num)), 0)
        # cls_agnostic_bbox_reg = bbox_pred.shape[1] == 4


        bboxes = [
            paddle.reshape(bbox, (-1, 4))
            if bbox.numel() > 0
            else paddle.zeros([0, 4])
            for bbox in bboxes
        ]
        det_labels = scores.split(tuple(np.array(rois_num)), 0) 
        return bboxes, det_labels

    def bbox_overlaps(
                  self,
                  bboxes1,
                  bboxes2,
                  mode='iou',
                  eps=1e-6,
                  use_legacy_coordinate=False):
        """Calculate the ious between each bbox of bboxes1 and bboxes2.
        Args:
            bboxes1 (ndarray): Shape (n, 4)
            bboxes2 (ndarray): Shape (k, 4)
            mode (str): IOU (intersection over union) or IOF (intersection
                over foreground)
            use_legacy_coordinate (bool): Whether to use coordinate system in
                mmdet v1.x. which means width, height should be
                calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
                Note when function is used in `VOCDataset`, it should be
                True to align with the official implementation
                `http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar`
                Default: False.
        Returns:
            ious (ndarray): Shape (n, k)
        """

        assert mode in ['iou', 'iof']
        if not use_legacy_coordinate:
            extra_length = 0.
        else:
            extra_length = 1.
        bboxes1 = bboxes1.astype(np.float32)
        bboxes2 = bboxes2.astype(np.float32)
        rows = bboxes1.shape[0]
        cols = bboxes2.shape[0]
        ious = np.zeros((rows, cols), dtype=np.float32)
        if rows * cols == 0:
            return ious
        exchange = False
        if bboxes1.shape[0] > bboxes2.shape[0]:
            bboxes1, bboxes2 = bboxes2, bboxes1
            ious = np.zeros((cols, rows), dtype=np.float32)
            exchange = True
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + extra_length) * (
            bboxes1[:, 3] - bboxes1[:, 1] + extra_length)
        area2 = (bboxes2[:, 2] - bboxes2[:, 0] + extra_length) * (
            bboxes2[:, 3] - bboxes2[:, 1] + extra_length)
        for i in range(bboxes1.shape[0]):
            x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
            y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
            x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
            y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
            overlap = np.maximum(x_end - x_start + extra_length, 0) * np.maximum(
                y_end - y_start + extra_length, 0)
            if mode == 'iou':
                union = area1[i] + area2 - overlap
            else:
                union = area1[i] if not exchange else area2
            union = np.maximum(union, eps)
            ious[i, :] = overlap / union
        if exchange:
            ious = ious.T
        return ious