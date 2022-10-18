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

__all__ = ['DenseTeacher']


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
            data = f(data)
        return data


@register
class DenseTeacher(BaseArch):
    __category__ = 'architecture'
    """
    DenseTeacher network, see 
    Args:
        teacher (object): teacher detector model instance
        student (object): student detector model instance
    """

    def __init__(self,
                 model='FCOS',
                 train_cfg=None,
                 test_cfg=None,
                 strongAug=[]):
        super(DenseTeacher, self).__init__()
        self.model = model
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.strongAug = Compose(strongAug)

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        model = create(cfg['model'])
        return {'model': model, }

    def _forward(self):
        return True

    def strong_augmentatin(self, data_weak):
        data_strong = data_weak
        sample_imgs = []
        # only support image transforms now
        for i in range(data_weak['image'].shape[0]):
            sample = {}
            sample['image'] = data_weak['image'][i].numpy().transpose(
                (1, 2, 0))  # [c, h, w] -》 [h, w, c]
            sample = self.strongAug(sample)
            sample['image'] = paddle.to_tensor(sample['image'].transpose((
                2, 0, 1))).unsqueeze(0)
            sample_imgs.append(sample['image'])
        data_strong['image'] = paddle.concat(sample_imgs, 0)
        return data_strong

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()
