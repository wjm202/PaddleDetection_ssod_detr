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

import math
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
import paddle.distributed as dist
from paddle.io import DataLoader, DistributedBatchSampler, BatchSampler
from paddle.fluid.dataloader.collate import default_collate_fn
#from .utils import default_collate_fn

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
        extra_key = ['h', 'w', 'flipped']
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


class DistributedBatchSampler1(BatchSampler):
    
    def __init__(self,
                 dataset,
                 batch_size,
                 num_replicas=None,
                 rank=None,
                 shuffle=False,
                 drop_last=False):
        self.dataset = dataset

        assert isinstance(batch_size, int) and batch_size > 0, \
                "batch_size should be a positive integer"
        self.batch_size = batch_size
        assert isinstance(shuffle, bool), \
                "shuffle should be a boolean value"
        self.shuffle = shuffle
        assert isinstance(drop_last, bool), \
                "drop_last should be a boolean number"

        from paddle.fluid.dygraph.parallel import ParallelEnv

        if num_replicas is not None:
            assert isinstance(num_replicas, int) and num_replicas > 0, \
                    "num_replicas should be a positive integer"
            self.nranks = num_replicas
        else:
            self.nranks = ParallelEnv().nranks

        if rank is not None:
            assert isinstance(rank, int) and rank >= 0, \
                    "rank should be a non-negative integer"
            self.local_rank = rank
        else:
            self.local_rank = ParallelEnv().local_rank

        self.drop_last = drop_last
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.nranks))
        self.total_size = self.num_samples * self.nranks

    def __iter__(self):
        num_samples = len(self.dataset)
        indices = np.arange(num_samples).tolist()
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        if self.shuffle:
            np.random.RandomState(self.epoch).shuffle(indices)
            self.epoch += 1

        # subsample
        def _get_indices_by_batch_size(indices):
            subsampled_indices = []
            last_batch_size = self.total_size % (self.batch_size * self.nranks)
            assert last_batch_size % self.nranks == 0
            last_local_batch_size = last_batch_size // self.nranks

            for i in range(self.local_rank * self.batch_size,
                           len(indices) - last_batch_size,
                           self.batch_size * self.nranks):
                subsampled_indices.extend(indices[i:i + self.batch_size])

            indices = indices[len(indices) - last_batch_size:]
            subsampled_indices.extend(indices[
                self.local_rank * last_local_batch_size:(
                    self.local_rank + 1) * last_local_batch_size])
            return subsampled_indices

        if self.nranks > 1:
            indices = _get_indices_by_batch_size(indices)

        assert len(indices) == self.num_samples
        _sample_iter = iter(indices)

        batch_indices = []
        for idx in _sample_iter:
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                yield batch_indices
                batch_indices = []
        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices

    def __len__(self):
        num_samples = self.num_samples
        num_samples += int(not self.drop_last) * (self.batch_size - 1)
        return num_samples // self.batch_size

    def set_epoch(self, epoch):
        """
        Sets the epoch number. When :attr:`shuffle=True`, this number is used
        as seeds of random numbers. By default, users may not set this, all
        replicas (workers) use a different random ordering for each epoch.
        If set same number at each epoch, this sampler will yield the same
        ordering at all epoches.

        Arguments:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class DistributedBatchLMSampler1(BatchSampler):
    
    def __init__(self,
                 dataset,
                 batch_size,
                 num_replicas=None,
                 rank=None,
                 shuffle=False,
                 drop_last=False,
                 ):
        self.dataset = dataset

        assert isinstance(batch_size, int) and batch_size > 0, \
                "batch_size should be a positive integer"
        self.batch_size = batch_size
        assert isinstance(shuffle, bool), \
                "shuffle should be a boolean value"
        self.shuffle = shuffle
        assert isinstance(drop_last, bool), \
                "drop_last should be a boolean number"

        from paddle.fluid.dygraph.parallel import ParallelEnv

        if num_replicas is not None:
            assert isinstance(num_replicas, int) and num_replicas > 0, \
                    "num_replicas should be a positive integer"
            self.nranks = num_replicas
        else:
            self.nranks = ParallelEnv().nranks

        if rank is not None:
            assert isinstance(rank, int) and rank >= 0, \
                    "rank should be a non-negative integer"
            self.local_rank = rank
        else:
            self.local_rank = ParallelEnv().local_rank

        self.drop_last = drop_last
        self.epoch = 0
        if self.dataset.manual_length != None and self.dataset.manual_length < len(self.dataset):
            manual_length = self.dataset.manual_length
        else:
            manual_length = len(self.dataset)
        self.manual_length = manual_length
        self.num_samples = int(math.ceil(self.manual_length * 1.0 / self.nranks))
        self.total_size = self.num_samples * self.nranks

    def __iter__(self):
        num_samples = self.manual_length
        if self.manual_length < len(self.dataset):
            indices = np.random.RandomState(self.epoch).choice(np.arange(len(self.dataset)), num_samples).tolist()
        else:
            indices = np.arange(num_samples).tolist()
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        if self.shuffle:
            np.random.RandomState(self.epoch).shuffle(indices)
            self.epoch += 1

        # subsample
        def _get_indices_by_batch_size(indices):
            subsampled_indices = []
            last_batch_size = self.total_size % (self.batch_size * self.nranks)
            assert last_batch_size % self.nranks == 0
            last_local_batch_size = last_batch_size // self.nranks

            for i in range(self.local_rank * self.batch_size,
                           len(indices) - last_batch_size,
                           self.batch_size * self.nranks):
                subsampled_indices.extend(indices[i:i + self.batch_size])

            indices = indices[len(indices) - last_batch_size:]
            subsampled_indices.extend(indices[
                self.local_rank * last_local_batch_size:(
                    self.local_rank + 1) * last_local_batch_size])
            return subsampled_indices

        if self.nranks > 1:
            indices = _get_indices_by_batch_size(indices)

        assert len(indices) == self.num_samples
        _sample_iter = iter(indices)

        batch_indices = []
        for idx in _sample_iter:
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                yield batch_indices
                batch_indices = []
        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices

    def __len__(self):
        num_samples = self.num_samples
        num_samples += int(not self.drop_last) * (self.batch_size - 1)
        return num_samples // self.batch_size

    def set_epoch(self, epoch):
        """
        Sets the epoch number. When :attr:`shuffle=True`, this number is used
        as seeds of random numbers. By default, users may not set this, all
        replicas (workers) use a different random ordering for each epoch.
        If set same number at each epoch, this sampler will yield the same
        ordering at all epoches.

        Arguments:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class DistributedBatchEqualSampler(BatchSampler):
    """
    Sample data from different datasets by sample ratio
    """
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r
    def __init__(self,
                 dataset,
                 batch_size,
                 num_replicas=None,
                 rank=None,
                 shuffle=False,
                 drop_last=False,
                 epoch_length=1250,
                 sample_ratio=[1,3]):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.cumulative_sizes = self.cumsum(self.dataset)
        # [len(dataset[idx].roidbs) for idx in range(len(self.dataset))]
        print(self.cumulative_sizes)
        # decide the frequency to sample each kind of datasets
        if not isinstance(sample_ratio, list):
            sample_ratio = [sample_ratio] * len(self.cumulative_sizes)
        self.sample_ratio = sample_ratio
        self.sample_ratio = [
            int(sr / min(self.sample_ratio)) for sr in self.sample_ratio
        ]
        self.epoch_length = epoch_length

        assert isinstance(batch_size, int) and batch_size > 0, \
                "batch_size should be a positive integer"
        self.batch_size = batch_size
        assert isinstance(shuffle, bool), \
                "shuffle should be a boolean value"
        self.shuffle = shuffle
        assert isinstance(drop_last, bool), \
                "drop_last should be a boolean number"

        from paddle.fluid.dygraph.parallel import ParallelEnv

        if num_replicas is not None:
            assert isinstance(num_replicas, int) and num_replicas > 0, \
                    "num_replicas should be a positive integer"
            self.nranks = num_replicas
        else:
            self.nranks = ParallelEnv().nranks

        if rank is not None:
            assert isinstance(rank, int) and rank >= 0, \
                    "rank should be a non-negative integer"
            self.local_rank = rank
        else:
            self.local_rank = ParallelEnv().local_rank

        self.drop_last = drop_last
        self.epoch = 0
        self.num_samples = self.cumulative_sizes[-1]
        # int(math.ceil(len(self.dataset) * 1.0 / self.nranks))
        self.total_size = self.num_samples * self.nranks

    def __iter__(self):
        paddle.seed(self.epoch)
        indices = []
        cumulative_index = np.arange(0, sum(self.cumulative_sizes))
        indice_per_dataset = []
        cumulative_sizes = [0] + self.cumulative_sizes
        for i in range(len(self.cumulative_sizes)):
            indice_per_dataset.append(cumulative_index[cumulative_sizes[i]: cumulative_sizes[i + 1]])

        shuffled_indice_per_dataset = [
            s[np.random.permutation(int(s.shape[0])).tolist()]
            for s in indice_per_dataset
        ]    

        total_indice = []
        batch_idx = 0
        while batch_idx < self.epoch_length * self.num_replicas:
            # [0.25, 0.75]
            ratio = [x / sum(self.sample_ratio) for x in self.sample_ratio]

            # num of each dataset
            ratio = [int(r * self.batch_size) for r in ratio]
            ratio[-1] = self.batch_size - sum(ratio[:-1])
            assert ratio[-1] > 0, "ratio[-1] <= 0, sample_ratio set error"
            selected = []
            
            for j in range(len(shuffled_indice_per_dataset)):
                if len(shuffled_indice_per_dataset[j]) < ratio[j]: 
                    shuffled_indice_per_dataset[j] = np.concatenate(
                        (
                            shuffled_indice_per_dataset[j],
                            indice_per_dataset[j][
                                list(
                                    paddle.randperm(
                                        int(indice_per_dataset[j].shape[0])
                                    ).numpy()
                                )
                            ],
                        )
                    ) 
                selected.append(shuffled_indice_per_dataset[j][: ratio[j]])   
                shuffled_indice_per_dataset[j] = shuffled_indice_per_dataset[j][
                ratio[j]:
                ]
            selected = np.concatenate(selected)
            
            total_indice.append(selected)
            batch_idx += 1 
        
        indices = np.concatenate(total_indice)
        indices = [
            indices[j]
            for i in list(
                np.random.permutation(
                    len(indices) // self.batch_size
                )
            )
            for j in range(
                i * self.batch_size,
                (i + 1) * self.batch_size,
            )
        ]   
        
        offset = len(self) * self.local_rank
        # print(offset, self.rank, len(self), len(indices), indices[:10], indices[-10:])
        indices = indices[offset: offset + len(self)]  
        print(offset, self.local_rank, len(self), len(indices), indices[:10], indices[-10:])
        
        assert len(indices) == len(self)                                          

        _sample_iter = iter(indices)

        batch_indices = []
        for idx in _sample_iter:
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                yield batch_indices
                batch_indices = []
        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices

    def __len__(self):
        return self.epoch_length * self.batch_size
        
        # num_samples = self.num_samples
        # num_samples += int(not self.drop_last) * (self.batch_size - 1)
        # return num_samples // self.batch_size

    def set_epoch(self, epoch):
        """
        Sets the epoch number. When :attr:`shuffle=True`, this number is used
        as seeds of random numbers. By default, users may not set this, all
        replicas (workers) use a different random ordering for each epoch.
        If set same number at each epoch, this sampler will yield the same
        ordering at all epoches.
        """
        self.epoch = epoch


class DistributedBatchSemiSampler(BatchSampler):
    """
    Sample data from different datasets by sample ratio
    """
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r
    def __init__(self,
                 dataset,
                 batch_size,
                 num_replicas=None,
                 rank=None,
                 shuffle=False,
                 drop_last=False,
                 epoch_length=1250,
                 sample_ratio=[1,3]):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.cumulative_sizes = self.cumsum(self.dataset)
        # [len(dataset[idx].roidbs) for idx in range(len(self.dataset))]
        print(self.cumulative_sizes)
        # decide the frequency to sample each kind of datasets
        if not isinstance(sample_ratio, list):
            sample_ratio = [sample_ratio] * len(self.cumulative_sizes)
        self.sample_ratio = sample_ratio
        self.sample_ratio = [
            int(sr / min(self.sample_ratio)) for sr in self.sample_ratio
        ]
        self.epoch_length = epoch_length

        assert isinstance(batch_size, int) and batch_size > 0, \
                "batch_size should be a positive integer"
        self.batch_size = batch_size
        assert isinstance(shuffle, bool), \
                "shuffle should be a boolean value"
        self.shuffle = shuffle
        assert isinstance(drop_last, bool), \
                "drop_last should be a boolean number"

        from paddle.fluid.dygraph.parallel import ParallelEnv

        if num_replicas is not None:
            assert isinstance(num_replicas, int) and num_replicas > 0, \
                    "num_replicas should be a positive integer"
            self.nranks = num_replicas
        else:
            self.nranks = ParallelEnv().nranks

        if rank is not None:
            assert isinstance(rank, int) and rank >= 0, \
                    "rank should be a non-negative integer"
            self.local_rank = rank
        else:
            self.local_rank = ParallelEnv().local_rank

        self.drop_last = drop_last
        self.epoch = 0
        self.num_samples = self.cumulative_sizes[-1]
        # int(math.ceil(len(self.dataset) * 1.0 / self.nranks))
        self.total_size = self.num_samples * self.nranks

    def __iter__(self):
        paddle.seed(self.epoch)
        indices = []
        cumulative_index = np.arange(0, sum(self.cumulative_sizes))
        indice_per_dataset = []
        cumulative_sizes = [0] + self.cumulative_sizes
        for i in range(len(self.cumulative_sizes)):
            indice_per_dataset.append(cumulative_index[cumulative_sizes[i]: cumulative_sizes[i + 1]])

        shuffled_indice_per_dataset = [
            s[np.random.permutation(int(s.shape[0])).tolist()]
            for s in indice_per_dataset
        ]    

        total_indice = []
        batch_idx = 0
        while batch_idx < self.epoch_length * self.num_replicas:
            # [0.25, 0.75]
            ratio = [x / sum(self.sample_ratio) for x in self.sample_ratio]
            # array([0, 1])
            index_tmp = np.arange(len(ratio))
            # array([1, 1, 1, 1, 0, 1, 1, 1])
            indicator = np.random.choice(a=list(index_tmp), size=self.batch_size, replace=True, p=ratio)
            unique, counts = np.unique(indicator, return_counts=True)  
            ratio = [0] * len(shuffled_indice_per_dataset)
            for u, c in zip(unique, counts):
                ratio[u] = c
            
            assert len(ratio) == 2, "Only two set is suppoted"
            if ratio[0] == 0:
                ratio[0] = 1
                ratio[1] -= 1
            elif ratio[1] == 0:
                ratio[1] = 1
                ratio[0] -= 1

            ratio = [r / sum(ratio) for r in ratio]

            # num of each dataset
            ratio = [int(r * self.batch_size) for r in ratio]
            ratio[-1] = self.batch_size - sum(ratio[:-1])
            selected = []
            
            for j in range(len(shuffled_indice_per_dataset)):
                if len(shuffled_indice_per_dataset[j]) < ratio[j]: 
                    shuffled_indice_per_dataset[j] = np.concatenate(
                        (
                            shuffled_indice_per_dataset[j],
                            indice_per_dataset[j][
                                list(
                                    paddle.randperm(
                                        int(indice_per_dataset[j].shape[0])
                                    ).numpy()
                                )
                            ],
                        )
                    ) 
                selected.append(shuffled_indice_per_dataset[j][: ratio[j]])   
                shuffled_indice_per_dataset[j] = shuffled_indice_per_dataset[j][
                ratio[j]:
                ]
            selected = np.concatenate(selected)
            
            total_indice.append(selected)
            batch_idx += 1 
        
        indices = np.concatenate(total_indice)
        indices = [
            indices[j]
            for i in list(
                np.random.permutation(
                    len(indices) // self.batch_size
                )
            )
            for j in range(
                i * self.batch_size,
                (i + 1) * self.batch_size,
            )
        ]   
        
        offset = len(self) * self.local_rank
        # print(offset, self.rank, len(self), len(indices), indices[:10], indices[-10:])
        indices = indices[offset: offset + len(self)]  
        print(offset, self.local_rank, len(self), len(indices), indices[:10], indices[-10:])
        
        assert len(indices) == len(self)                                          

        _sample_iter = iter(indices)

        batch_indices = []
        for idx in _sample_iter:
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                yield batch_indices
                batch_indices = []
        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices

    def __len__(self):
        return self.epoch_length * self.batch_size
        
        # num_samples = self.num_samples
        # num_samples += int(not self.drop_last) * (self.batch_size - 1)
        # return num_samples // self.batch_size

    def set_epoch(self, epoch):
        """
        Sets the epoch number. When :attr:`shuffle=True`, this number is used
        as seeds of random numbers. By default, users may not set this, all
        replicas (workers) use a different random ordering for each epoch.
        If set same number at each epoch, this sampler will yield the same
        ordering at all epoches.
        """
        self.epoch = epoch


class Semi_BaseDataLoader(object):
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
                 unsup_tea_transforms=[],
                 unsup_stu_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 num_classes=80,
                 collate_batch=True,
                 use_shared_memory=False,
                 epoch_length=1250, 
                 sample_ratio=[1, 3],
                 **kwargs):
        # sample transform
        self._sample_transforms = Compose(
            sample_transforms, num_classes=num_classes)
        # unsup teacher transform
        self._unsup_tea_transforms = Compose(unsup_tea_transforms, num_classes=num_classes)
        # unsup student transform
        self._unsup_stu_transforms = Compose(unsup_stu_transforms, num_classes=num_classes)

        # batch transfrom 
        self._batch_transforms = BatchCompose(batch_transforms, num_classes,
                                              collate_batch)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.use_shared_memory = use_shared_memory
        self.epoch_length = epoch_length
        self.sample_ratio = sample_ratio
        self.kwargs = kwargs

    def __call__(self,
                 dataset,
                 worker_num,
                 batch_sampler=None,
                 return_list=False):
        self.dataset = dataset
        for idx in range(len(self.dataset.datasets)):
            # get data
            if idx == 0:
                self.dataset.datasets[idx].set_transform([self._sample_transforms])
            elif idx == 1:
                self.dataset.datasets[idx].set_transform([self._unsup_tea_transforms, self._unsup_stu_transforms])
        # set kwargs
            self.dataset.datasets[idx].set_kwargs(**self.kwargs)
        # batch sampler
        if batch_sampler is None:
            self._batch_sampler = DistributedBatchSemiSampler(
                self.dataset.datasets,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                drop_last=self.drop_last,
                epoch_length=self.epoch_length,
                sample_ratio=self.sample_ratio
                )
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


class Semi_EqualDataLoader(object):
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
                 unsup_tea_transforms=[],
                 unsup_stu_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 num_classes=80,
                 collate_batch=True,
                 use_shared_memory=False,
                 epoch_length=1250, 
                 sample_ratio=[1, 3],
                 **kwargs):
        # sample transform
        self._sample_transforms = Compose(
            sample_transforms, num_classes=num_classes)
        # unsup teacher transform
        self._unsup_tea_transforms = Compose(unsup_tea_transforms, num_classes=num_classes)
        # unsup student transform
        self._unsup_stu_transforms = Compose(unsup_stu_transforms, num_classes=num_classes)

        # batch transfrom 
        self._batch_transforms = BatchCompose(batch_transforms, num_classes,
                                              collate_batch)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.use_shared_memory = use_shared_memory
        self.epoch_length = epoch_length
        self.sample_ratio = sample_ratio
        self.kwargs = kwargs

    def __call__(self,
                 dataset,
                 worker_num,
                 batch_sampler=None,
                 return_list=False):
        self.dataset = dataset
        for idx in range(len(self.dataset.datasets)):
            # get data
            if idx == 0:
                self.dataset.datasets[idx].set_transform([self._sample_transforms])
            elif idx == 1:
                self.dataset.datasets[idx].set_transform([self._unsup_tea_transforms, self._unsup_stu_transforms])
        # set kwargs
            self.dataset.datasets[idx].set_kwargs(**self.kwargs)
        # batch sampler
        if batch_sampler is None:
            self._batch_sampler = DistributedBatchEqualSampler(
                self.dataset.datasets,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                drop_last=self.drop_last,
                epoch_length=self.epoch_length,
                sample_ratio=self.sample_ratio
                )
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
                 batch_size=2,
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
            self._batch_sampler = DistributedBatchSampler1(
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
class Semi_TrainReader(Semi_BaseDataLoader):
    __shared__ = ['num_classes']

    def __init__(self,
                 sample_transforms=[],
                 unsup_tea_transforms=[],
                 unsup_stu_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=True,
                 drop_last=True,
                 num_classes=80,
                 collate_batch=True,
                 **kwargs):
        super(Semi_TrainReader, self).__init__(sample_transforms, unsup_tea_transforms, unsup_stu_transforms, 
                                            batch_transforms, batch_size, shuffle, drop_last,
                                          num_classes, collate_batch, **kwargs)


@register
class Semi_EvalReader(BaseDataLoader):
    __shared__ = ['num_classes']

    def __init__(self,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 drop_last=True,
                 num_classes=80,
                 **kwargs):
        super(Semi_EvalReader, self).__init__(sample_transforms, batch_transforms,
                                         batch_size, shuffle, drop_last,
                                         num_classes, **kwargs)



@register
class Semi_TrainEqualReader(Semi_EqualDataLoader):
    __shared__ = ['num_classes']

    def __init__(self,
                 sample_transforms=[],
                 unsup_tea_transforms=[],
                 unsup_stu_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=True,
                 drop_last=True,
                 num_classes=80,
                 collate_batch=True,
                 **kwargs):
        super(Semi_TrainEqualReader, self).__init__(sample_transforms, unsup_tea_transforms, unsup_stu_transforms, 
                                            batch_transforms, batch_size, shuffle, drop_last,
                                          num_classes, collate_batch, **kwargs)


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


def get_dist_info():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    return rank, world_size


@register
class LMReader(BaseDataLoader):
    __shared__ = ['num_classes']

    def __init__(self,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 num_classes=80,
                 **kwargs):
        super(LMReader, self).__init__(sample_transforms, batch_transforms,
                                         batch_size, shuffle, drop_last,
                                         num_classes, **kwargs)

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
            self._batch_sampler = DistributedBatchLMSampler1(
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
