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

import os
import traceback
import six
import sys
if sys.version_info >= (3, 0):
    pass
else:
    pass
import numpy as np
import paddle
import paddle.nn.functional as F

from paddle.io import DataLoader, DistributedBatchSampler
from .utils import default_collate_fn

from ppdet.core.workspace import register
from . import transform
from .shm_utils import _get_shared_memory_size_in_M

from ppdet.utils.logger import setup_logger
logger = setup_logger('reader')

MAIN_PID = os.getpid()


class Compose(object):
    def __init__(self, transforms, num_classes=80):
        self.transforms = transforms
        self.transforms_cls = []
        if self.transforms is None or self.transforms == []:
            self.transforms_cls = []
        else:
            for t in self.transforms:
                for k, v in t.items():
                    op_cls = getattr(transform, k)
                    f = op_cls(**v)
                    if hasattr(f, 'num_classes'):
                        f.num_classes = num_classes

                    self.transforms_cls.append(f)

    def __call__(self, data):
        for f in self.transforms_cls:
            try:
                data = f(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.warning("fail to map sample transform [{}] "
                               "with error: {} and stack:\n{}".format(
                                   f, e, str(stack_info)))
                raise e

        return data


class BatchCompose(Compose):
    def __init__(self, transforms, num_classes=80, collate_batch=True):
        super(BatchCompose, self).__init__(transforms, num_classes)
        self.collate_batch = collate_batch

    def __call__(self, data):
        for f in self.transforms_cls:
            try:
                data = f(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.warning("fail to map batch transform [{}] "
                               "with error: {} and stack:\n{}".format(
                                   f, e, str(stack_info)))
                raise e

        # remove keys which is not needed by model
        extra_key = ['flipped']
        for k in extra_key:
            for sample in data:
                if k in sample:
                    sample.pop(k)

        # batch data, if user-define batch function needed
        # use user-defined here
        if self.collate_batch:
            batch_data = default_collate_fn(data)
        else:
            batch_data = {}
            for k in data[0].keys():
                tmp_data = []
                for i in range(len(data)):
                    tmp_data.append(data[i][k])
                if not 'gt_' in k and not 'is_crowd' in k and not 'difficult' in k:
                    tmp_data = np.stack(tmp_data, axis=0)
                batch_data[k] = tmp_data
        return batch_data


class BatchComposeUnSup(Compose):
    def __init__(self, transforms, num_classes=80, collate_batch=True):
        super(BatchComposeUnSup, self).__init__(transforms, num_classes)
        self.collate_batch = collate_batch

    def __call__(self, data):
        for f in self.transforms_cls:
            try:
                data = f(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.warning("fail to map batch transform [{}] "
                               "with error: {} and stack:\n{}".format(
                                   f, e, str(stack_info)))
                raise e

        # remove keys which is not needed by model
        extra_key = ['flipped']
        for k in extra_key:
            for sample in data:
                if k in sample:
                    sample.pop(k)

        # batch data, if user-define batch function needed
        # use user-defined here

        return data


class BatchComposeSemi(Compose):
    def __init__(self, transforms, num_classes=80):
        self.transforms = transforms
        self.transforms_cls = []
        for t in self.transforms:
            for k, v in t.items():
                op_cls = getattr(transform, k)
                f = op_cls(**v)
                if hasattr(f, 'num_classes'):
                    f.num_classes = num_classes
                self.transforms_cls.append(f)

    def __call__(self, data):
        for f in self.transforms_cls:
            data = f(data)
        from ppdet.data.utils import default_collate_fn
        batch_data = default_collate_fn(data)
        for k, v in batch_data.items():
            batch_data[k] = paddle.to_tensor(v)
        return batch_data


def SupAugmentation(data_ori, sample_aug_lists, sup_batch_aug_lists, num_classes=80):
    sampleAug = Compose(sample_aug_lists)
    batchAug = BatchComposeSemi(sup_batch_aug_lists, num_classes)
    data_aug = data_ori
    sample_imgs = []
    # only support image transforms now
    for i in range(len(data_ori)):
        sample = {}
        # sample['image'] = data_ori[i]['image'].numpy().transpose(
        #     (1, 2, 0))  # [c, h, w] -》 [h, w, c]
        sample['image'] = data_ori[i]['image'].numpy()
        sample['im_shape'] = data_ori[i]['im_shape'].numpy()
        sample['scale_factor'] = data_ori[i]['scale_factor'].numpy()
        sample['is_crowd'] = data_ori[i]['is_crowd'].numpy()
        sample['gt_bbox'] = data_ori[i]['gt_bbox'].numpy()
        sample['gt_class'] = data_ori[i]['gt_class'].numpy()

        sample = sampleAug(sample)
        sample_imgs.append(sample)

    data_aug = sample_imgs
    data_aug = batchAug(data_aug)
    return data_aug


def UnSupAugmentation(data_ori, sample_aug_lists, unsup_batch_aug_lists, num_classes=80):
    sampleAug = Compose(sample_aug_lists)
    batchAug = BatchComposeSemi(unsup_batch_aug_lists, num_classes)
    data_aug = data_ori
    sample_imgs = []
    # only support image transforms now
    for i in range(len(data_ori)):
        sample = {}
        # sample['image'] = data_ori[i]['image'].numpy().transpose(
        #     (1, 2, 0))  # [c, h, w] -》 [h, w, c]
        sample['image'] = data_ori[i]['image'].numpy()
        sample['im_shape'] = data_ori[i]['im_shape'].numpy()
        sample['scale_factor'] = data_ori[i]['scale_factor'].numpy()

        sample = sampleAug(sample)
        sample_imgs.append(sample)

    data_aug = sample_imgs
    data_aug = batchAug(data_aug)
    return data_aug


def align_weak_strong_shape(data_weak, data_strong):
    max_shape_x = max(data_strong['image'].shape[2], data_weak['image'].shape[2])
    max_shape_y = max(data_strong['image'].shape[3], data_weak['image'].shape[3])

    scale_x_s = max_shape_x / data_strong['image'].shape[2]
    scale_y_s = max_shape_y / data_strong['image'].shape[3]
    scale_x_w = max_shape_x / data_weak['image'].shape[2]
    scale_y_w = max_shape_y / data_weak['image'].shape[3]
    target_size = [max_shape_x, max_shape_y]

    if scale_x_s != 1 or scale_y_s != 1:
        data_strong['image'] = F.interpolate(
            data_strong['image'],
            size=target_size,
            mode='bilinear',
            align_corners=False)
        if 'gt_bbox' in data_strong:
            gt_bboxes = data_strong['gt_bbox']
            for i in range(len(gt_bboxes)):
                if len(gt_bboxes[i]) > 0:
                    gt_bboxes[i][:, 0::2] = gt_bboxes[i][:, 0::2] * scale_x_s
                    gt_bboxes[i][:, 1::2] = gt_bboxes[i][:, 1::2] * scale_y_s
            data_strong['gt_bbox'] = gt_bboxes
    
    if scale_x_w != 1 or scale_y_w != 1:
        data_weak['image'] = F.interpolate(
            data_weak['image'],
            size=target_size,
            mode='bilinear',
            align_corners=False)
        if 'gt_bbox' in data_weak:
            gt_bboxes = data_weak['gt_bbox']
            for i in range(len(gt_bboxes)):
                if len(gt_bboxes[i]) > 0:
                    gt_bboxes[i][:, 0::2] = gt_bboxes[i][:, 0::2] * scale_x_w
                    gt_bboxes[i][:, 1::2] = gt_bboxes[i][:, 1::2] * scale_y_w
            data_weak['gt_bbox'] = gt_bboxes

    return data_weak, data_strong


class BaseDataLoader(object):
    """
    Base DataLoader implementation for detection models

    Args:
        sample_transforms (list): a list of transforms to perform
                                  on each sample
        batch_transforms (list): a list of transforms to perform
                                 on batch
        batch_size (int): batch size for batch collating, default 1.
        shuffle (bool): whether to shuffle samples
        drop_last (bool): whether to drop the last incomplete,
                          default False
        num_classes (int): class number of dataset, default 80
        collate_batch (bool): whether to collate batch in dataloader.
            If set to True, the samples will collate into batch according
            to the batch size. Otherwise, the ground-truth will not collate,
            which is used when the number of ground-truch is different in 
            samples.
        use_shared_memory (bool): whether to use shared memory to
                accelerate data loading, enable this only if you
                are sure that the shared memory size of your OS
                is larger than memory cost of input datas of model.
                Note that shared memory will be automatically
                disabled if the shared memory of OS is less than
                1G, which is not enough for detection models.
                Default False.
    """

    def __init__(self,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 num_classes=80,
                 collate_batch=True,
                 use_shared_memory=False,
                 **kwargs):
        # sample transform
        self._sample_transforms = Compose(
            sample_transforms, num_classes=num_classes)

        # batch transfrom 
        self._batch_transforms = BatchCompose(batch_transforms, num_classes,
                                              collate_batch)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.use_shared_memory = use_shared_memory
        self.kwargs = kwargs

    def __call__(self,
                 dataset,
                 worker_num,
                 batch_sampler=None,
                 return_list=False):
        self.dataset = dataset
        self.dataset.check_or_download_dataset()
        self.dataset.parse_dataset()
        # get data
        self.dataset.set_transform(self._sample_transforms)
        # set kwargs
        self.dataset.set_kwargs(**self.kwargs)
        # batch sampler
        if batch_sampler is None:
            self._batch_sampler = DistributedBatchSampler(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                drop_last=self.drop_last)
        else:
            self._batch_sampler = batch_sampler

        # DataLoader do not start sub-process in Windows and Mac
        # system, do not need to use shared memory
        use_shared_memory = self.use_shared_memory and \
                            sys.platform not in ['win32', 'darwin']
        # check whether shared memory size is bigger than 1G(1024M)
        if use_shared_memory:
            shm_size = _get_shared_memory_size_in_M()
            if shm_size is not None and shm_size < 1024.:
                logger.warning("Shared memory size is less than 1G, "
                               "disable shared_memory in DataLoader")
                use_shared_memory = False

        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_sampler=self._batch_sampler,
            collate_fn=self._batch_transforms,
            num_workers=worker_num,
            return_list=return_list,
            use_shared_memory=use_shared_memory)
        self.loader = iter(self.dataloader)

        return self

    def __len__(self):
        return len(self._batch_sampler)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.loader)
        except StopIteration:
            self.loader = iter(self.dataloader)
            six.reraise(*sys.exc_info())

    def next(self):
        # python2 compatibility
        return self.__next__()


@register
class TrainReader(BaseDataLoader):
    __shared__ = ['num_classes']

    def __init__(self,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=True,
                 drop_last=True,
                 num_classes=80,
                 collate_batch=True,
                 **kwargs):
        super(TrainReader, self).__init__(sample_transforms, batch_transforms,
                                          batch_size, shuffle, drop_last,
                                          num_classes, collate_batch, **kwargs)


@register
class SupTrainReader(BaseDataLoader):
    __shared__ = ['num_classes', 'fuse_normalize']

    def __init__(self,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=True,
                 drop_last=True,
                 num_classes=80,
                 collate_batch=True,
                 fuse_normalize=False,
                 use_shared_memory=False,
                 **kwargs):
        super(SupTrainReader, self).__init__(sample_transforms, batch_transforms,
                                          batch_size, shuffle, drop_last,
                                          num_classes, collate_batch, fuse_normalize, **kwargs)
        if fuse_normalize:
            sample_transforms_ = []
            batch_transforms_ = []
            for t in sample_transforms:
                for k, v in t.items():
                    if k == 'NormalizeImage':
                        continue
                    sample_transforms_.append(t)
            for t in batch_transforms:
                for k, v in t.items():
                    if k == 'NormalizeImage':
                        continue
                    batch_transforms_.append(t)
        else:
            sample_transforms_ = sample_transforms
            batch_transforms_ = batch_transforms

        # sample transform
        self._sample_transforms = Compose(
            sample_transforms_, num_classes=num_classes)
        # batch transfrom 
        self._batch_transforms = BatchComposeUnSup(batch_transforms_, num_classes,
                                              collate_batch)

@register
class UnsupTrainReader(BaseDataLoader):
    __shared__ = ['num_classes', 'fuse_normalize']

    def __init__(self,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=True,
                 drop_last=True,
                 num_classes=80,
                 collate_batch=True,
                 fuse_normalize=False,
                 use_shared_memory=False,
                 **kwargs):
        super(UnsupTrainReader, self).__init__(sample_transforms, batch_transforms,
                                          batch_size, shuffle, drop_last,
                                          num_classes, collate_batch, fuse_normalize, **kwargs)
        if fuse_normalize:
            sample_transforms_ = []
            batch_transforms_ = []
            for t in sample_transforms:
                for k, v in t.items():
                    if k == 'NormalizeImage':
                        continue
                    sample_transforms_.append(t)
            for t in batch_transforms:
                for k, v in t.items():
                    if k == 'NormalizeImage':
                        continue
                    batch_transforms_.append(t)
        else:
            sample_transforms_ = sample_transforms
            batch_transforms_ = batch_transforms

        # sample transform
        self._sample_transforms = Compose(
            sample_transforms_, num_classes=num_classes)
        # batch transfrom 
        self._batch_transforms = BatchComposeUnSup(batch_transforms_, num_classes,
                                              collate_batch)

@register
class EvalReader(BaseDataLoader):
    __shared__ = ['num_classes']

    def __init__(self,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 drop_last=True,
                 num_classes=80,
                 **kwargs):
        super(EvalReader, self).__init__(sample_transforms, batch_transforms,
                                         batch_size, shuffle, drop_last,
                                         num_classes, **kwargs)


@register
class TestReader(BaseDataLoader):
    __shared__ = ['num_classes']

    def __init__(self,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 num_classes=80,
                 **kwargs):
        super(TestReader, self).__init__(sample_transforms, batch_transforms,
                                         batch_size, shuffle, drop_last,
                                         num_classes, **kwargs)


@register
class EvalMOTReader(BaseDataLoader):
    __shared__ = ['num_classes']

    def __init__(self,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 num_classes=1,
                 **kwargs):
        super(EvalMOTReader, self).__init__(sample_transforms, batch_transforms,
                                            batch_size, shuffle, drop_last,
                                            num_classes, **kwargs)


@register
class TestMOTReader(BaseDataLoader):
    __shared__ = ['num_classes']

    def __init__(self,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 num_classes=1,
                 **kwargs):
        super(TestMOTReader, self).__init__(sample_transforms, batch_transforms,
                                            batch_size, shuffle, drop_last,
                                            num_classes, **kwargs)
