from copy import deepcopy
from types import TracebackType

import paddle

__all__ = ["meanteacher"]


class meanteacher(object):
    """
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model=None, decay=0.9996):
        """
        Args:
            model (nn.Module): model to apply EMA.
            decay (float): ema decay reate.
        """
        self.model = deepcopy(model)
        self.decay = decay

    def update(self, model, decay=None):
        if decay is None:
            decay = self.decay
        # Update EMA parameters

        with paddle.no_grad():
            state = {}
            msd = model.state_dict()  # model state_dict
            for k, v in self.model.state_dict().items():
                if paddle.is_floating_point(v):
                    v *= decay
                    v += (1.0 - decay) * msd[k].detach()

                state[k] = v
            self.model.set_state_dict(state)

    def resume(self, state_dict, step=0):
        state = {}
        msd = state_dict  # model state_dict
        for k, v in self.model.state_dict().items():
            if paddle.is_floating_point(v):
                v = msd[k].detach()
                v.stop_gradient = False
            state[k] = v
        self.model.set_state_dict(state)
        self.step = step
