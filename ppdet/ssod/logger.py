import logging
import os
import sys
from collections import Counter
from typing import Tuple
import cv2

import numpy as np
from ppdet.utils.visualizer import draw_bbox
from PIL import Image, ImageDraw

import paddle
import paddle.distributed as dist
from ppdet.utils.colormap import colormap

_log_counter = Counter()


def _find_caller():
    frame = sys._getframe(2)
    while frame:
        code = frame.f_code
        if os.path.join("utils", "logger.") not in code.co_filename:
            mod_name = frame.f_globals["__name__"]
            if mod_name == "__main__":
                mod_name = r"ssod"
            return mod_name, (code.co_filename, frame.f_lineno, code.co_name)
        frame = frame.f_back

def imdenormalize(img, mean, std, to_bgr=True):
    assert img.dtype != np.uint8
    mean = mean.reshape(1, -1).astype(np.float64)
    std = std.reshape(1, -1).astype(np.float64)
    img = cv2.multiply(img, std)  # make a copy
    cv2.add(img, mean, img)  # inplace
    if to_bgr:
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)  # inplace
    return img


def log_image_with_boxes(
    tag,
    image, 
    bboxes,
    labels=None,
    filename=None,
    class_names=None,
    interval=500,
    cnt=0,
    flag=1):
    """
    Draw bbox on image
    """
    rank = dist.get_rank()
    if rank != 0:
        return
    _, key = _find_caller()
    if flag:
        _log_counter[key] += 1
    if not (interval == 1 or _log_counter[key] % interval == 1):
        return
    if filename is None:
        filename = f"{_log_counter[key]}_{cnt}.jpg"
    if labels is None:
        # labels = (paddle.zeros([bboxes.shape[0]])).cpu().detach().numpy()
        class_names = ["foreground"]

    mean = np.array([0.485,0.456,0.406], dtype=np.float32)
    std = np.array([0.229, 0.224,0.225], dtype=np.float32)
    img_np = image.detach().cpu().numpy().transpose((1, 2, 0)).astype(np.float32)
    img_nor = imdenormalize(img_np, mean, std, to_bgr=True) 
    image = Image.fromarray(np.uint8((img_nor * 255)))

    draw = ImageDraw.Draw(image)
    
    catid2color = {}
    color_list = colormap(rgb=True)[:40]    
    for ii, bbox in enumerate(bboxes):
        color = tuple(color_list[1])
        # draw bbox
        if len(bbox) == 4:
            # draw bbox
            xmin, ymin, xmax, ymax = bbox
            # xmax = xmin + w
            # ymax = ymin + h
            draw.line(
                [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
                 (xmin, ymin)],
                width=2,
                fill=color)
        elif len(bbox) == 8:
            x1, y1, x2, y2, x3, y3, x4, y4 = bbox
            draw.line(
                [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)],
                width=2,
                fill=color)
            xmin = min(x1, x2, x3, x4)
            ymin = min(y1, y2, y3, y4)
        else:
            logger.error('the shape of bbox must be [M, 4] or [M, 8]!')

        # draw label
        # text = "foreground"
        if isinstance(labels, list):
            if isinstance(labels[ii], str):
                class_names = labels[ii]
            else:
                class_names = str(labels[ii])
        elif isinstance(labels, paddle.Tensor):
            class_names = str(labels[ii].numpy()[0])

        tw, th = draw.textsize(class_names)
        draw.rectangle(
            [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
        draw.text((xmin + 1, ymin - th), class_names, fill=(255, 255, 255))
    
    root_dir = os.environ.get("WORK_DIR")
    
    if root_dir != None:
        out_file=os.path.join(root_dir, tag)
    else:
        out_file=os.path.join(tag)
    save_name = _get_save_image_name(out_file, filename)
    image.save(save_name, quality=95)


def _get_save_image_name(output_dir, filename):
        """
        Get save image name from source image path.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        name, ext = os.path.splitext(filename)
        return os.path.join(output_dir, "{}".format(name)) + ext

def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger.

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name="mmdet.ssod", log_file=log_file, log_level=log_level)
    logger.propagate = False
    return logger


def convert_box(tag, boxes, box_labels, class_labels, std, scores=None):
    if isinstance(std, int):
        std = [std, std]
    if len(std) != 4:
        std = std[::-1] * 2
    std = boxes.new_tensor(std).reshape(1, 4)
    wandb_box = {}
    boxes = boxes / std
    boxes = boxes.detach().cpu().numpy().tolist()
    box_labels = box_labels.detach().cpu().numpy().tolist()
    class_labels = {k: class_labels[k] for k in range(len(class_labels))}
    wandb_box["class_labels"] = class_labels
    assert len(boxes) == len(box_labels)
    if scores is not None:
        scores = scores.detach().cpu().numpy().tolist()
        box_data = [
            dict(
                position=dict(minX=box[0], minY=box[1], maxX=box[2], maxY=box[3]),
                class_id=label,
                scores=dict(cls=scores[i]),
            )
            for i, (box, label) in enumerate(zip(boxes, box_labels))
        ]
    else:
        box_data = [
            dict(
                position=dict(minX=box[0], minY=box[1], maxX=box[2], maxY=box[3]),
                class_id=label,
            )
            for i, (box, label) in enumerate(zip(boxes, box_labels))
        ]

    wandb_box["box_data"] = box_data
    return {tag: wandb.data_types.BoundingBoxes2D(wandb_box, tag)}


def imdenormalize(img, mean, std, to_bgr=True):
    assert img.dtype != np.uint8
    mean = mean.reshape(1, -1).astype(np.float64)
    std = std.reshape(1, -1).astype(np.float64)
    img = cv2.multiply(img, std)  # make a copy
    cv2.add(img, mean, img)  # inplace
    if to_bgr:
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)  # inplace
    return img

def color_transform(img_tensor, mean, std, to_rgb=False):
    img_np = img_tensor.detach().cpu().numpy().transpose((1, 2, 0)).astype(np.float32)
    return imdenormalize(img_np, mean, std, to_bgr=not to_rgb)

