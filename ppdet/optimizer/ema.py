# # Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
import weakref
from copy import deepcopy


class ModelEMA(object):
    def __init__(self, model, decay=0.9996, ema_decay_type='ssl'):
        self.step = 0
        self.epoch = 0
        self.decay = decay
        self.state_dict = dict()
        # self.state_dict -> teacher
        for k, v in model.student.state_dict().items():
            self.state_dict[k] = v
        self.ema_decay_type = ema_decay_type

        self._model_state = {
            k: weakref.ref(p)
            for k, p in model.student.state_dict().items()
        }

    def update(self, model=None, decay=None):

        if decay is None:
            decay = self.decay
        if model is not None:
            model_dict = model.student.state_dict()
        else:
            model_dict = {k: p() for k, p in self._model_state.items()}
            assert all(
                [v is not None for _, v in model_dict.items()]), 'python gc.'

        for k, v in self.state_dict.items():  # teacher
            v = decay * v + (1 - decay) * model_dict[k]  # student
            v.stop_gradient = True
            self.state_dict[k] = v

    def apply(self):
        if self.step == 0:
            return self.state_dict
        state_dict = dict()
        for k, v in self.state_dict.items():
            v.stop_gradient = True
            state_dict[k] = v
        return state_dict
