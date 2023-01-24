from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import datetime
import six
import copy
import json
import shutil
import os.path as osp
import numpy as np

import paddle
import paddle.distributed as dist
from paddle.nn.layer.norm import BatchNorm2D
from ppdet.engine.callbacks import Callback
from ppdet.utils.logger import setup_logger
from ppdet.core.workspace import create
from ppdet.data.reader import get_dist_info
logger = setup_logger(__name__)

class LabelMatchCallback(Callback):
    def __init__(self, trainer):
        super(LabelMatchCallback, self).__init__(trainer)
        rank, world_size = get_dist_info()
        cfg = trainer.cfg

        dataset = create('TrainDatasetLM')()
        dataloader = create('LMReader')(
            dataset, 0)
        assert dataloader.shuffle == True, "LMReader data need shuffle"
        boxes_per_image_gt, cls_ratio_gt = self.get_data_info(os.path.join(cfg.TrainDataset.dataset_dir, cfg.TrainDataset.anno_path))
        eval_cfg = cfg.get('evaluation', {})
        manual_prior = cfg.get('min_thr', None)
        if manual_prior:  # manual setting the boxes_per_image and cls_ratio
            boxes_per_image_gt = manual_prior.get('boxes_per_image', boxes_per_image_gt)
            cls_ratio_gt = manual_prior.get('cls_ratio', cls_ratio_gt)
        min_thr = cfg.get('min_thr', 0.05)
        if dataset.manual_length != None and dataset.manual_length < len(dataset):
            potential_positive = dataset.manual_length * boxes_per_image_gt * cls_ratio_gt # list unlabeled每类可能pos的数量=data的长度*平均每张的bbox*每类的比例
        else:
            potential_positive = len(dataset) * boxes_per_image_gt * cls_ratio_gt
        update_interval = cfg['update_interval']
        self.eval_callback = LabelMatchEvalCallback(
            self.model, self.CLASSES, dataloader, potential_positive, boxes_per_image_gt, cls_ratio_gt, min_thr, interval=update_interval, **eval_cfg)

    def get_data_info(self, json_file):
        """get information from labeled data"""
        with open(json_file, 'r') as f:
            info = json.load(f)
        id2cls = {}
        
        self.CLASSES = []
        for cat_id, cat_info in enumerate(info['categories']):
            self.CLASSES.append(cat_info['name'])
        
        total_image = len(info['images'])
        for value in info['categories']:
            id2cls[value['id']] = self.CLASSES.index(value['name'])
        cls_num = [0] * len(self.CLASSES)
        for value in info['annotations']:
            cls_num[id2cls[value['category_id']]] += 1
        cls_num = [max(c, 1) for c in cls_num] # for some cls not select, we set it 1 rather than 0
        total_boxes = sum(cls_num)
        cls_ratio_gt = np.array([c / total_boxes for c in cls_num])
        boxes_per_image_gt = total_boxes / total_image
        info = ' '.join([f'({v:.4f}-{self.CLASSES[i]})' for i, v in enumerate(cls_ratio_gt)])
        logger.info(f'boxes per image (label data): {boxes_per_image_gt}')
        logger.info(f'class ratio (label data): {info}')
        return boxes_per_image_gt, cls_ratio_gt


    def on_epoch_begin(self, status):
        self.eval_callback.on_epoch_begin(status)

    def on_epoch_end(self, status):
        self.eval_callback.on_epoch_end(status)

    def on_step_begin(self, status):
        self.eval_callback.on_step_begin(status)

    def on_step_end(self, status):
        self.eval_callback.on_step_end(status)

    
class LabelMatchEvalCallback(Callback):
    def __init__(self,
                model,
                classes,
                dataloader,
                potential_positive,
                boxes_per_image_gt,
                cls_ratio_gt,
                min_thr,
                interval,
                broadcast_bn_buffer=True,
                **eval_kwargs
                ):
        super().__init__(model)
        self.model = model
        self.dst_root = None
        self.initial_epoch_flag = True

        self.potential_positive = potential_positive
        self.boxes_per_image_gt = boxes_per_image_gt
        self.cls_ratio_gt = cls_ratio_gt
        self.min_thr = min_thr
        self.dataloader = dataloader
        self.interval = interval
        self.CLASSES = classes
        self.broadcast_bn_buffer = broadcast_bn_buffer

    def on_epoch_begin(self, status):
        if not self.initial_epoch_flag:
            return
        interval_temp = self.interval
        self.interval = 1
        self.on_step_end(status)            
        self.initial_epoch_flag = False
        self.interval = interval_temp
        self.model.boxes_per_image_gt = self.boxes_per_image_gt
        self.model.cls_ratio_gt = self.cls_ratio_gt

    def on_step_end(self, status):
        mode = status['mode']
        if mode == 'train' and self.every_n_iters(status['iter_id'], self.interval):
            self.update_cls_thr()

    def do_evaluation(self):
        results = []
        with paddle.no_grad():
            self.model.ema.model.eval()
            if dist.get_world_size() < 2 or dist.get_rank() == 0:
                logger.info("update cls_thr using dataset number is: {}".format(len(self.dataloader) * dist.get_world_size()))
            for step_id, data in enumerate(self.dataloader):
                outs = self.model.ema.model(data)
                if dist.get_world_size() < 2 or dist.get_rank() == 0:
                    if (step_id + 1) % 100 == 0:
                        logger.info("update cls_thr eval iter: {}".format(step_id + 1))
                results.append(outs['bbox'][:,:2])
        results = paddle.concat(x=results, axis=0)
        print('result length is', results.shape)
        return results

    def _broadcast_bn_buffer(self, model):
        if self.broadcast_bn_buffer:
            if dist.get_world_size() >= 2:
                for key, val in model.named_sublayers():
                    if isinstance(val, BatchNorm2D):
                        dist.broadcast(val._variance, 0)
                        dist.broadcast(val._mean, 0)
            

    def update_cls_thr(self):
        if self.broadcast_bn_buffer:
            self._broadcast_bn_buffer(self.model.ema.model)
        results = self.do_evaluation()
        results_len = paddle.to_tensor([results.shape[0]], dtype='int32')
        if dist.get_world_size() >= 2:
            total_len = []
            paddle.distributed.all_gather(total_len, results_len)
        else:
            total_len = [results_len]
        logger.info("dataset after gather total len is {}".format(total_len))
        max_len = max([l.item() for l in total_len])
        if results_len.item() < max_len:
            pad_zeros = paddle.zeros([max_len - results_len.item(), 2], dtype=results.dtype) 
            results = paddle.concat(x=[results, pad_zeros], axis=0)
        if dist.get_world_size() >= 2:
            total_results = []
            paddle.distributed.all_gather(total_results, results)
        else:
            total_results = [results]
        results_tensor = []
        for idx in range(len(total_results)):
            len_res, res = total_len[idx], total_results[idx]
            results_tensor.append(res[:len_res])
        results_tensor = paddle.concat(x=results_tensor, axis=0)
        # print(len(total_results))
        percent = 0.4 # percent as positive
        cls_thr, cls_thr_ig = self.eval_score_thr(results_tensor, percent)
        self.model.model.cls_thr = cls_thr
        self.model.model.cls_thr_ig = cls_thr_ig

    def eval_score_thr(self, results, percent):
        score_list = [[] for _ in self.CLASSES]
        for r in results:
            cls = int(r[0].item())
            score_list[cls].append(r[1].item())
        score_list = [np.array(c) for c in score_list]
        score_list = [np.zeros(1) if len(c) == 0 else np.sort(c)[::-1] for c in score_list]
        cls_thr = [0] * len(self.CLASSES)
        cls_thr_ig = [0] * len(self.CLASSES)
        for i, score in enumerate(score_list):
            cls_thr[i] = max(0.05, score[min(int(self.potential_positive[i] * percent), len(score) - 1)])# reliable pseudo label
            cls_thr_ig[i] = max(self.min_thr, score[min(int(self.potential_positive[i]), len(score) - 1)])

        logger.info(f'current percent: {percent}')
        info = ' '.join([f'({v:.2f}-{self.CLASSES[i]})' for i, v in enumerate(cls_thr)])
        logger.info(f'update score thr (positive): {info}')
        info = ' '.join([f'({v:.2f}-{self.CLASSES[i]})' for i, v in enumerate(cls_thr_ig)])
        logger.info(f'update score thr (ignore): {info}')
        return cls_thr, cls_thr_ig

    def every_n_iters(self, iter_id, n):
        return (iter_id + 1) % n == 0 if n > 0 else False