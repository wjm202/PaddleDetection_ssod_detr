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
        percent = self.model.cfg['percent'] # percent as positive
        cls_thr, cls_thr_ig = self.eval_score_thr(results_tensor, percent)
        self.model.cls_thr = cls_thr
        self.model.cls_thr_ig = cls_thr_ig

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
        #cls_thr=[0.9189905524253845, 0.7669094800949097, 0.8273994326591492, 0.8715341091156006, 0.9220973253250122, 0.9523329734802246, 0.9498604536056519, 0.7825229167938232, 0.697105348110199, 0.723545491695404, 0.9506539106369019, 0.9730640649795532, 0.8821449875831604, 0.715083122253418, 0.7480572462081909, 0.9156230688095093, 0.9115221500396729, 0.9176708459854126, 0.8364088535308838, 0.8600541949272156, 0.9407229423522949, 0.957146406173706, 0.9568769335746765, 0.9556859731674194, 0.6102584004402161, 0.7861237525939941, 0.5747857093811035, 0.7598803639411926, 0.70057612657547, 0.9206396341323853, 0.5912915468215942, 0.6034570336341858, 0.8555730581283569, 0.7819466590881348, 0.8224555850028992, 0.8670865297317505, 0.8467326164245605, 0.7857649326324463, 0.9125673174858093, 0.7608055472373962, 0.7900867462158203, 0.8142127990722656, 0.57862389087677, 0.5232635736465454, 0.5680792331695557, 0.8037459254264832, 0.5418198704719543, 0.5144788026809692, 0.7467670440673828, 0.5772063732147217, 0.7377564907073975, 0.6024068593978882, 0.6298245787620544, 0.9120379090309143, 0.7962210178375244, 0.7550150752067566, 0.7592353820800781, 0.8480515480041504, 0.728065550327301, 0.8784624338150024, 0.8028295040130615, 0.9323474764823914, 0.9312314987182617, 0.9519029855728149, 0.9121245741844177, 0.6786559820175171, 0.925177276134491, 0.7597594261169434, 0.8943276405334473, 0.8377682566642761, 0.4730987846851349, 0.769322395324707, 0.9388293027877808, 0.4868394434452057, 0.9101532101631165, 0.7329792380332947, 0.5713301301002502, 0.8498570919036865, 0.1338973492383957, 0.4976862967014313]
        #cls_thr_ig=[0.37371402978897095, 0.3277507722377777, 0.39911168813705444, 0.3840101361274719, 0.3036014139652252, 0.4430861473083496, 0.40308064222335815, 0.4173223674297333, 0.3281969726085663, 0.35154831409454346, 0.5697162747383118, 0.41473516821861267, 0.34061887860298157, 0.33561649918556213, 0.3038831949234009, 0.47113677859306335, 0.4643377959728241, 0.398902028799057, 0.3629454970359802, 0.44687753915786743, 0.44427862763404846, 0.5980253219604492, 0.5230969190597534, 0.5721573233604431, 0.3483468294143677, 0.3498336970806122, 0.34177374839782715, 0.29273492097854614, 0.3500933051109314, 0.5205440521240234, 0.3118540048599243, 0.2871639132499695, 0.3061806261539459, 0.353799045085907, 0.35361531376838684, 0.33585503697395325, 0.29187971353530884, 0.3799827992916107, 0.438271164894104, 0.3759513199329376, 0.3151155114173889, 0.38356730341911316, 0.3237382471561432, 0.327974796295166, 0.3478938937187195, 0.42712393403053284, 0.33204054832458496, 0.3332289159297943, 0.4326817989349365, 0.31126073002815247, 0.41370904445648193, 0.3721049427986145, 0.3621525168418884, 0.39720600843429565, 0.39649468660354614, 0.4109834134578705, 0.3888798654079437, 0.39187222719192505, 0.36997562646865845, 0.40865156054496765, 0.39888158440589905, 0.3816637396812439, 0.4658812880516052, 0.4406881034374237, 0.5229347944259644, 0.35291755199432373, 0.47773781418800354, 0.36428120732307434, 0.36336109042167664, 0.39560091495513916, 0.38673704862594604, 0.3790462911128998, 0.43921712040901184, 0.33765771985054016, 0.4001680612564087, 0.3727276921272278, 0.30678045749664307, 0.29774510860443115, 0.11861539632081985, 0.330264151096344]
        # for i in range(len(self.CLASSES)):
        #     cls_thr[i]=cls_thr[i]*0.5+cls_thr_ig[i]*0.5
        
        logger.info(f'current percent: {percent}')
        info = ' '.join([f'({v:.2f}-{self.CLASSES[i]})' for i, v in enumerate(cls_thr)])
        logger.info(f'update score thr (positive): {info}')
        info = ' '.join([f'({v:.2f}-{self.CLASSES[i]})' for i, v in enumerate(cls_thr_ig)])
        logger.info(f'update score thr (ignore): {info}')
        return cls_thr, cls_thr_ig

    def every_n_iters(self, iter_id, n):
        return (iter_id + 1) % n == 0 if n > 0 else False