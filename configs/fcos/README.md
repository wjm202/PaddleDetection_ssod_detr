# FCOS (Fully Convolutional One-Stage Object Detection)

## Model Zoo on COCO

| 骨架网络             | 网络类型       | 每张GPU图片个数 | 学习率策略 |推理时间(fps) | Box AP |                           下载                          | 配置文件 |
| :------------------- | :------------- | :-----: | :-----: | :------------: | :-----: | :-----------------------------------------------------: | :-----: |
| ResNet50-FPN    | FCOS           |    2    |   1x      |     ----     |  39.6  | [download](https://paddledet.bj.bcebos.com/models/fcos_r50_fpn_1x_coco.pdparams) | [config](configs/fcos/fcos_r50_fpn_1x_coco.yml) |
| ResNet50-FPN    | FCOS + DCN       |    2    |   1x      |     ----     |  44.3  | [download](https://paddledet.bj.bcebos.com/models/fcos_dcn_r50_fpn_1x_coco.pdparams) | [config](configs/fcos/fcos_dcn_r50_fpn_1x_coco.yml) |
| ResNet50-FPN    | FCOS + multiscale_train    |    2    |   2x      |     ----     |  41.8  | [download](https://paddledet.bj.bcebos.com/models/fcos_r50_fpn_multiscale_2x_coco.pdparams) | [config](configs/fcos/fcos_r50_fpn_multiscale_2x_coco.yml) |
| ResNet50-FPN    | FCOS + multiscale_train + iou    |    2    |   2x      |     ----     |  42.6  | [download](https://paddledet.bj.bcebos.com/models/fcos_r50_fpn_iou_multiscale_2x_coco.pdparams) | [config](configs/fcos/fcos_r50_fpn_iou_multiscale_2x_coco.yml) |

**注意:**

- `fcos_r50_fpn_1x` 在8卡 Tesla v100 上训练只需8.5小时。
- `+ iou` 与原版版本 FCOS 相比，表示不使用 `centerness` 而是使用 `iou` 来参与计算loss。


## Citations
```
@inproceedings{tian2019fcos,
  title   =  {{FCOS}: Fully Convolutional One-Stage Object Detection},
  author  =  {Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  booktitle =  {Proc. Int. Conf. Computer Vision (ICCV)},
  year    =  {2019}
}
```
