import warnings
from collections import Counter, Mapping, Sequence
from numbers import Number
from typing import List, Optional, Tuple, Union, Dict
from functools import partial

import numpy as np
import paddle
from six.moves import map, zip


_step_counter = Counter()

def sequence_mul(obj, multiplier):
    if isinstance(obj, Sequence):
        return [o * multiplier for o in obj]
    else:
        return obj * multiplier

def is_match(word, word_list):
    for keyword in word_list:
        if keyword in word:
            return True
    return False

def weighted_loss(loss: dict, weight, ignore_keys=[], warmup=0):
    if len(loss) == 0:
        return {}
    _step_counter["weight"] += 1
    lambda_weight = (
        lambda x: x * (_step_counter["weight"] - 1) / warmup
        if _step_counter["weight"] <= warmup
        else x
    )
    if isinstance(weight, Mapping):
        for k, v in weight.items():
            for name, loss_item in loss.items():
                if (k in name) and ("loss" in name):
                    loss[name] = sequence_mul(loss[name], lambda_weight(v))
    elif isinstance(weight, Number):
        for name, loss_item in loss.items():
            if "loss" in name:
                if not is_match(name, ignore_keys):
                    loss[name] = sequence_mul(loss[name], lambda_weight(weight))
                else:
                    loss[name] = sequence_mul(loss[name], 0.0)
    else:
        raise NotImplementedError()

    total_loss = paddle.add_n(list(loss.values()))
    loss.update({'loss': total_loss})
    return loss

def filter_invalid(bbox, label=None, score=None, mask=None, thr=0.0, min_size=0):
    if score.numel() > 0:
        # valid = score > thr
        valid = score >= thr
        if valid.shape[0] == 1 :
            bbox = bbox if valid.item() else paddle.expand(paddle.to_tensor([])[:, None], (-1, 4))
        else:
            bbox = bbox[valid]

        if label is not None:
            if valid.shape[0] == 1 :
                label = label if valid.item() else paddle.to_tensor([])
            else:
                label = label[valid]
        # bbox = bbox[valid]
        # if label is not None:
        #     label = label[valid]
        if mask is not None:
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
    if min_size is not None and bbox.shape[0] > 0:
        bw = bbox[:, 2] - bbox[:, 0]
        bh = bbox[:, 3] - bbox[:, 1]
        valid = (bw > min_size) & (bh > min_size)

        if valid.shape[0] == 1 :
            bbox = bbox if valid.item() else paddle.expand(paddle.to_tensor([])[:, None], (-1, 4))
        else:
            bbox = bbox[valid]

        if label is not None:
            if valid.shape[0] == 1 :
                label = label if valid.item() else paddle.to_tensor([])
            else:
                label = label[valid]
            
        if mask is not None:
            mask = BitmapMasks(mask.masks[valid.cpu().numpy()], mask.height, mask.width)
    return bbox, label, mask


def bbox2points(box):
    min_x, min_y, max_x, max_y = paddle.split(box[:, :4], [1, 1, 1, 1], axis=1)

    return paddle.reshape(
            paddle.concat(
                [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y], axis=1
            ), (-1, 2)
     ) # n*4,2


def points2bbox(point, max_w, max_h):
    point = paddle.reshape(point, (-1, 4, 2))
    if point.shape[0] > 0:
        min_xy = paddle.min(point, axis=1) # n,2
        max_xy = paddle.max(point, axis=1)
        xmin = paddle.clip(min_xy[:, 0], min=0, max=max_w)
        ymin = paddle.clip(min_xy[:, 1], min=0, max=max_h)
        xmax = paddle.clip(max_xy[:, 0], min=0, max=max_w)
        ymax = paddle.clip(max_xy[:, 1], min=0, max=max_h)
        min_xy = paddle.stack([xmin, ymin], axis=1)
        max_xy = paddle.stack([xmax, ymax], axis=1)
        return paddle.concat([min_xy, max_xy], axis=1)  # n,4
    else:
        return point.new_zeros(0, 4)


class Transform2D:
    @staticmethod
    def transform_bboxes(bbox, M, out_shape):
        if isinstance(bbox, Sequence):
            assert len(bbox) == len(M)
            return [
                Transform2D.transform_bboxes(b, m, o)
                for b, m, o in zip(bbox, M, out_shape)
            ]
        else:
            if bbox.shape[0] == 0:
                return bbox
            score = None
            if bbox.shape[1] > 4:
                score = bbox[:, 4:]
            points = bbox2points(bbox[:, :4])
            points = paddle.concat(
                [points, paddle.ones([points.shape[0], 1])], axis=1
            )  # n*4,3
            points = paddle.matmul(M, points.t()).t()
            points = points[:, :2] / points[:, 2:3]
            bbox = points2bbox(points, out_shape[1], out_shape[0])
            if score is not None:
                return paddle.concat([bbox, score], axis=1)
            return bbox

    @staticmethod
    def transform_image(img, M, out_shape):
        if isinstance(img, Sequence):
            assert len(img) == len(M)
            return [
                Transform2D.transform_image(b, m, shape)
                for b, m, shape in zip(img, M, out_shape)
            ]
        else:
            if img.dim() == 2:
                img = img[None, None, ...]
            elif img.dim() == 3:
                img = img[None, ...]

            return (
                warp_affine(img.float(), M[None, ...], out_shape, mode="nearest")
                .squeeze()
                .to(img.dtype)
            )

    @staticmethod
    def get_trans_mat(a, b):
        a = [matrix.astype('float32') for matrix in a]
        b = [matrix.astype('float32') for matrix in b]
        return [bt @ at.inverse() for bt, at in zip(b, a)]


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))