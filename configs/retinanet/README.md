# RetinaNet (Focal Loss for Dense Object Detection)

## Model Zoo

| 骨架网络         | 网络类型        | 每张GPU图片个数 | 学习率策略 |推理时间(fps)| Box AP |  下载  | 配置文件 |
| :------------- | :------------- | :-----------: | :------: | :--------: |:-----: | :----: | :----: |
| ResNet50-FPN   | RetinaNet      |    2          |   1x     |    -       |  37.3  | [下载链接](https://paddledet.bj.bcebos.com/models/retinanet_r50_fpn_1x_coco.pdparams) | [配置文件](./retinanet_r50_fpn_1x_coco.yml) |


## Citations
```
@inproceedings{lin2017focal,
  title={Focal loss for dense object detection},
  author={Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  year={2017}
}
```
