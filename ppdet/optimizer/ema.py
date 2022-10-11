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

    # """
    # Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    # Keep a moving average of everything in the model state_dict (parameters and buffers).
    # This is intended to allow functionality like
    # https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    # A smoothed version of the weights is necessary for some training schemes to perform well.
    # This class is sensitive where it is initialized in the sequence of model init,
    # GPU assignment and distributed training wrappers.
    # """

    # def __init__(self, model, decay=0.9996):
    #     """
    #     Args:
    #         model (nn.Module): model to apply EMA.
    #         decay (float): ema decay reate.
    #     """
    #     # Create EMA(FP32)
    #     self.model = model
    #     self.decay = decay

    # def update(self, model, decay=None):
    #     if decay is None:
    #         decay = self.decay
    #     # Update EMA parameters
    #     with paddle.no_grad():
    #         msd = model.student.state_dict()  #传参
    #         # model state_dict
    #         for k, v in model.teacher.state_dict().items():
    #             print(paddle.is_floating_point(v))
    #             if paddle.is_floating_point(v):
    #                 v *= decay
    #                 v += (1.0 - decay) * msd[k].detach()
    #                 v.stop_gradient = True
    #                 model.teacher.state_dict()[k]=v

    # def resume(self, state_dict, step=0):
    #     for k, v in state_dict.items():
    #         if k in self.model.state_dict():
    #             if self.model.state_dict()[k].dtype == v.dtype:
    #                 self.model.state_dict()[k] = v
    #             else:
    #                 self.model.state_dict()[k] = v.astype(self.model.state_dict(
    #                 )[k].dtype)
    #     self.step = step

    # def burn_up(self, model, decay=0): #wjm10.10
    #     with paddle.no_grad():
    #         msd = model.student.state_dict()  #传参
    #         # model state_dict
    #         for k, v in model.teacher.state_dict().items():
    #             if paddle.is_floating_point(v):
    #                 v *= decay
    #                 v += (1.0 - decay) * msd[k].detach()
    #                 v.stop_gradient = True
    #                 model.teacher.state_dict()[k]=v
    #         return model.teacher.state_dict()
