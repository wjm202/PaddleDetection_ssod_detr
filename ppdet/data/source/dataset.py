# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved. 
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
import numpy as np
import bisect

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence
from paddle.io import Dataset, IterableDataset
from ppdet.core.workspace import register, serializable
from ppdet.utils.download import get_dataset_path
import copy
from ppdet.data import source


class _ConcatDataset(Dataset):
    """Dataset as a concatenation of multiple datasets.
    This class is useful to assemble different existing datasets.
    Args:
        datasets(list of Dataset): List of datasets to be composed.
    Returns:
        Dataset: A Dataset which composes fields of multiple datasets.
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        self.datasets = list(datasets)
        for idx in range(len(self.datasets)):
            self.datasets[idx].check_or_download_dataset()
            self.datasets[idx].parse_dataset()
        assert len(self.datasets) > 0, "input datasets shoule not be empty"
        for i, dataset in enumerate(self.datasets):
            assert isinstance(dataset, Dataset), \
                    "each input dataset should be paddle.io.Dataset"
            assert not isinstance(dataset, IterableDataset), \
                    "paddle.io.IterableDataset not supported"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    # def __getitem__(self, idx):
        # sample = []
        # for dataset in self.datasets:
        #     sample.extend(to_list(dataset[idx]))
        # return tuple(sample)

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        # sup 
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


class SemiDataset(_ConcatDataset):
    """Wrapper for semisupervised od."""
    def __init__(self, sup, unsup, **kwargs):
        super().__init__([sup, unsup], **kwargs)
    
    @property
    def sup(self):
        return self.datasets[0]

    @property
    def unsup(self):
        return self.datasets[1]
    
    def set_epoch(self, epoch_id):
        self._epoch = epoch_id
   

@serializable
class DetDataset(Dataset):
    """
    Load detection dataset.

    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str): annotation file path.
        data_fields (list): key name of data dictionary, at least have 'image'.
        sample_num (int): number of samples to load, -1 means all.
        use_default_label (bool): whether to load default label list.
    """

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 data_fields=['image'],
                 sample_num=-1,
                 use_default_label=None,
                 **kwargs):
        super(DetDataset, self).__init__()
        self.dataset_dir = dataset_dir if dataset_dir is not None else ''
        self.anno_path = anno_path
        self.image_dir = image_dir if image_dir is not None else ''
        self.data_fields = data_fields
        self.sample_num = sample_num
        self.use_default_label = use_default_label
        self._epoch = 0
        self._curr_iter = 0

    def __len__(self, ):
        return len(self.roidbs)

    def __call__(self, *args, **kwargs):
        return self

    def __getitem__(self, idx):
        # data batch
        roidb = copy.deepcopy(self.roidbs[idx])
        if self.mixup_epoch == 0 or self._epoch < self.mixup_epoch:
            n = len(self.roidbs)
            idx = np.random.randint(n)
            roidb = [roidb, copy.deepcopy(self.roidbs[idx])]
        elif self.cutmix_epoch == 0 or self._epoch < self.cutmix_epoch:
            n = len(self.roidbs)
            idx = np.random.randint(n)
            roidb = [roidb, copy.deepcopy(self.roidbs[idx])]
        elif self.mosaic_epoch == 0 or self._epoch < self.mosaic_epoch:
            n = len(self.roidbs)
            roidb = [roidb, ] + [
                copy.deepcopy(self.roidbs[np.random.randint(n)])
                for _ in range(4)
            ]
        if isinstance(roidb, Sequence):
            for r in roidb:
                r['curr_iter'] = self._curr_iter
        else:
            roidb['curr_iter'] = self._curr_iter
        self._curr_iter += 1
        return self.transform(roidb)

    def check_or_download_dataset(self):
        self.dataset_dir = get_dataset_path(self.dataset_dir, self.anno_path,
                                            self.image_dir)

    def set_kwargs(self, **kwargs):
        self.mixup_epoch = kwargs.get('mixup_epoch', -1)
        self.cutmix_epoch = kwargs.get('cutmix_epoch', -1)
        self.mosaic_epoch = kwargs.get('mosaic_epoch', -1)

    def set_transform(self, transform):
        self.transform = transform

    def set_epoch(self, epoch_id):
        self._epoch = epoch_id

    def parse_dataset(self, ):
        raise NotImplementedError(
            "Need to implement parse_dataset method of Dataset")

    def get_anno(self):
        if self.anno_path is None:
            return
        return os.path.join(self.dataset_dir, self.anno_path)


def _is_valid_file(f, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    return f.lower().endswith(extensions)


def _make_dataset(dir):
    dir = os.path.expanduser(dir)
    if not os.path.isdir(dir):
        raise ('{} should be a dir'.format(dir))
    images = []
    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if _is_valid_file(path):
                images.append(path)
    return images


@register
@serializable
class ImageFolder(DetDataset):
    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 sample_num=-1,
                 use_default_label=None,
                 **kwargs):
        super(ImageFolder, self).__init__(
            dataset_dir,
            image_dir,
            anno_path,
            sample_num=sample_num,
            use_default_label=use_default_label)
        self._imid2path = {}
        self.roidbs = None
        self.sample_num = sample_num

    def check_or_download_dataset(self):
        return

    def get_anno(self):
        if self.anno_path is None:
            return
        if self.dataset_dir:
            return os.path.join(self.dataset_dir, self.anno_path)
        else:
            return self.anno_path

    def parse_dataset(self, ):
        if not self.roidbs:
            self.roidbs = self._load_images()

    def _parse(self):
        image_dir = self.image_dir
        if not isinstance(image_dir, Sequence):
            image_dir = [image_dir]
        images = []
        for im_dir in image_dir:
            if os.path.isdir(im_dir):
                im_dir = os.path.join(self.dataset_dir, im_dir)
                images.extend(_make_dataset(im_dir))
            elif os.path.isfile(im_dir) and _is_valid_file(im_dir):
                images.append(im_dir)
        return images

    def _load_images(self):
        images = self._parse()
        ct = 0
        records = []
        for image in images:
            assert image != '' and os.path.isfile(image), \
                    "Image {} not found".format(image)
            if self.sample_num > 0 and ct >= self.sample_num:
                break
            rec = {'im_id': np.array([ct]), 'im_file': image}
            self._imid2path[ct] = image
            ct += 1
            records.append(rec)
        assert len(records) > 0, "No image file found"
        return records

    def get_imid2path(self):
        return self._imid2path

    def set_images(self, images):
        self.image_dir = images
        self.roidbs = self._load_images()


@register
class CommonDataset(object):
    def __init__(self, **dataset_args):
        super(CommonDataset, self).__init__()
        dataset_args = copy.deepcopy(dataset_args)
        type = dataset_args.pop("name")
        self.dataset = getattr(source, type)(**dataset_args)

    def __call__(self):
        return self.dataset


@register
class TrainDataset(CommonDataset):
    pass


@register
class EvalMOTDataset(CommonDataset):
    pass


@register
class TestMOTDataset(CommonDataset):
    pass


@register
class EvalDataset(CommonDataset):
    pass


@register
class TestDataset(CommonDataset):
    pass



@serializable
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`ConcatDataset`, but
    concat the group flag for image aspect ratio.

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
        separate_eval (bool): Whether to evaluate the results
            separately if it is used as validation dataset.
            Defaults to True.
    """

    def __init__(self, datasets, separate_eval=True):
        super(ConcatDataset, self).__init__()

        assert len(datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore
        self.datasets = list(datasets)
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

        self.CLASSES = datasets[0].CLASSES
        self.PALETTE = getattr(datasets[0], 'PALETTE', None)
        self.separate_eval = separate_eval
        if not separate_eval:
            if any([isinstance(ds, CocoDataset) for ds in datasets]):
                raise NotImplementedError(
                    'Evaluating concatenated CocoDataset as a whole is not'
                    ' supported! Please set "separate_eval=True"')
            elif len(set([type(ds) for ds in datasets])) != 1:
                raise NotImplementedError(
                    'All the datasets should have same types')

        if hasattr(datasets[0], 'flag'):
            flags = []
            for i in range(0, len(datasets)):
                flags.append(datasets[i].flag)
            self.flag = np.concatenate(flags)
    
    def get_cat_ids(self, idx):
        """Get category ids of concatenated dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    'absolute value of index should not exceed dataset length')
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_cat_ids(sample_idx)

    def get_ann_info(self, idx):
        """Get annotation of concatenated dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    'absolute value of index should not exceed dataset length')
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_ann_info(sample_idx)

    def evaluate(self, results, logger=None, **kwargs):
        """Evaluate the results.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str: float]: AP results of the total dataset or each separate
            dataset if `self.separate_eval=True`.
        """
        assert len(results) == self.cumulative_sizes[-1], \
            ('Dataset and results have different sizes: '
             f'{self.cumulative_sizes[-1]} v.s. {len(results)}')

        # Check whether all the datasets support evaluation
        for dataset in self.datasets:
            assert hasattr(dataset, 'evaluate'), \
                f'{type(dataset)} does not implement evaluate function'

        if self.separate_eval:
            dataset_idx = -1
            total_eval_results = dict()
            for size, dataset in zip(self.cumulative_sizes, self.datasets):
                start_idx = 0 if dataset_idx == -1 else \
                    self.cumulative_sizes[dataset_idx]
                end_idx = self.cumulative_sizes[dataset_idx + 1]

                results_per_dataset = results[start_idx:end_idx]
                print_log(
                    f'\nEvaluateing {dataset.ann_file} with '
                    f'{len(results_per_dataset)} images now',
                    logger=logger)

                eval_results_per_dataset = dataset.evaluate(
                    results_per_dataset, logger=logger, **kwargs)
                dataset_idx += 1
                for k, v in eval_results_per_dataset.items():
                    total_eval_results.update({f'{dataset_idx}_{k}': v})

            return total_eval_results
        elif any([isinstance(ds, CocoDataset) for ds in self.datasets]):
            raise NotImplementedError(
                'Evaluating concatenated CocoDataset as a whole is not'
                ' supported! Please set "separate_eval=True"')
        elif len(set([type(ds) for ds in self.datasets])) != 1:
            raise NotImplementedError(
                'All the datasets should have same types')
        else:
            original_data_infos = self.datasets[0].data_infos
            self.datasets[0].data_infos = sum(
                [dataset.data_infos for dataset in self.datasets], [])
            eval_results = self.datasets[0].evaluate(
                results, logger=logger, **kwargs)
            self.datasets[0].data_infos = original_data_infos
            return eval_results