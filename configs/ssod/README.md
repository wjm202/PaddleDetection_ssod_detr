简体中文 | [English](README_en.md)

# Semi-Supervised Object Detection (SSOD) 半监督目标检测

## 内容
- [简介](#简介)
- [模型库](#模型库)
- [数据集准备](#数据集准备)
- [配置半监督检测](#配置半监督检测)
    - [全局配置](#全局配置)
    - [配置半监督检测器](#配置半监督检测器)
    - [配置半监督训练集](#配置半监督训练集)
    - [配置半监督数据增强](#配置半监督数据增强)
    - [其他配置](#其他配置)
- [使用说明](#使用说明)
    - [训练](#训练)
    - [评估](#评估)
    - [预测](#预测)
    - [部署](#部署)
- [引用](#引用)

## 简介
半监督目标检测(SSOD)是**同时使用有标注数据和无标注数据**进行训练的目标检测，既可以极大地节省标注成本，也可以充分利用无标注数据进一步提高检测精度。


## 模型库

### [Baseline](baseline)

**纯监督数据**模型的训练和模型库，请参照[Baseline](baseline)；



## 数据集准备

半监督目标检测**同时需要有标注数据和无标注数据**，且无标注数据量一般**远多于有标注数据量**。
对于COCO数据集一般有两种常规设置：

（1）抽取部分比例的原始训练集`train2017`作为标注数据和无标注数据；

从`train2017`中按固定百分比（1%、2%、5%、10%等）抽取，由于抽取方法会对半监督训练的结果影响较大，所以采用五折交叉验证来评估。运行数据集划分制作的脚本如下：
```bash
python tools/gen_semi_coco.py
```
会按照 1%、2%、5%、10% 的监督数据比例来划分`train2017`全集，为了交叉验证每一种划分会随机重复5次，生成的半监督标注文件如下：
- 标注数据集标注：`instances_train2017.{fold}@{percent}.json`
- 无标注数据集标注：`instances_train2017.{fold}@{percent}-unlabeled.json`
其中，`fold` 表示交叉验证，`percent` 表示有标注数据的百分比。

（2）使用全量原始训练集`train2017`作为有标注数据 和 全量原始无标签图片集`unlabeled2017`作为无标注数据；


### 下载链接

PaddleDetection团队提供了COCO数据集全部的标注文件，请下载并解压存放至对应目录:

```shell
# 下载COCO全量数据集图片和标注
# 包括 train2017, val2017, annotations
wget https://bj.bcebos.com/v1/paddledet/data/coco.tar

# 下载PaddleDetection团队整理的COCO部分比例数据的标注文件
wget https://bj.bcebos.com/v1/paddledet/data/coco/semi_annotations.zip

# unlabeled2017是可选，如果不需要训‘full’则无需下载
# 下载COCO全量 unlabeled 无标注数据集
wget https://bj.bcebos.com/v1/paddledet/data/coco/unlabeled2017.zip
wget https://bj.bcebos.com/v1/paddledet/data/coco/image_info_unlabeled2017.zip
# 下载转换完的 unlabeled2017 无标注json文件
wget https://bj.bcebos.com/v1/paddledet/data/coco/instances_unlabeled2017.zip
```

如果需要用到COCO全量unlabeled无标注数据集，需要将原版的`image_info_unlabeled2017.json`进行格式转换，运行以下代码:

<details>
<summary> COCO unlabeled 标注转换代码：</summary>

```python
import json
anns_train = json.load(open('annotations/instances_train2017.json', 'r'))
anns_unlabeled = json.load(open('annotations/image_info_unlabeled2017.json', 'r'))
unlabeled_json = {
  'images': anns_unlabeled['images'],
  'annotations': [],
  'categories': anns_train['categories'],
}
path = 'annotations/instances_unlabeled2017.json'
with open(path, 'w') as f:
  json.dump(unlabeled_json, f)
```

</details>


<details open>
<summary> 解压后的数据集目录如下：</summary>

```
PaddleDetection
├── dataset
│   ├── coco
│   │   ├── annotations
│   │   │   ├── instances_train2017.json
│   │   │   ├── instances_unlabeled2017.json
│   │   │   ├── instances_val2017.json
│   │   ├── semi_annotations
│   │   │   ├── instances_train2017.1@1.json
│   │   │   ├── instances_train2017.1@1-unlabeled.json
│   │   │   ├── instances_train2017.1@2.json
│   │   │   ├── instances_train2017.1@2-unlabeled.json
│   │   │   ├── instances_train2017.1@5.json
│   │   │   ├── instances_train2017.1@5-unlabeled.json
│   │   │   ├── instances_train2017.1@10.json
│   │   │   ├── instances_train2017.1@10-unlabeled.json
│   │   ├── train2017
│   │   ├── unlabeled2017
│   │   ├── val2017
```

</details>

## 配置半监督检测

配置半监督检测，需要基于选用的**基础检测器**的配置文件，如：

```python
_BASE_: [
  '../../ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml',
  '../_base_/coco_detection_percent_10.yml',
]
log_iter: 20
snapshot_epoch: 2
epochs: &epochs 240
weights: output/denseteacher_ppyoloe_plus_crn_s_coco_semi010/model_final
```

并依次做出如下5点改动：

### 预训练和warmup配置

```python
### pretrain and warmup config, choose one and coment another
pretrain_weights: https://bj.bcebos.com/v1/paddledet/models/ssod/ppyoloe_plus_crn_s_80e_coco_sup010.pdparams # mAP=35.3
semi_start_iters: 0
ema_start_iters: 0
use_warmup: &use_warmup False

# pretrain_weights: https://bj.bcebos.com/v1/paddledet/models/pretrained/ppyoloe_crn_s_obj365_pretrained.pdparams
# semi_start_iters: 5000
# ema_start_iters: 3000
# use_warmup: &use_warmup True
```

### 全局配置

需要在配置文件中添加如下全局配置：

```python
### global config
use_simple_ema: True
ema_decay: 0.9996
ssod_method: DenseTeacher
DenseTeacher:
  train_cfg:
    sup_weight: 1.0
    unsup_weight: 1.0
    loss_weight: {distill_loss_cls: 1.0, distill_loss_iou: 2.5, distill_loss_dfl: 0.5}
    concat_sup_data: True
    suppress: linear
    ratio: 0.01 #
    gamma: 2.0
  test_cfg:
    inference_on: teacher
```

### 配置半监督模型

以 `DenseTeacher` 为例，选择 `fcos_r50_fpn_1x_coco` 作为 `基础检测器` 进行半监督训练，**teacher网络的结构和student网络的结构均为基础检测器作为**，在半监督中**teacher和student网络必须是完全相同的模型结构**：

```python
SSOD: DenseTeacher
DenseTeacher:
  train_cfg:
    ratio: 0.01
    sup_weight: 1.0
    unsup_weight: 1.0
    suppress: linear
    loss_weight: {distill_loss_cls: 4.0, distill_loss_box: 1.0, distill_loss_quality: 1.0}
    gamma: 2.0
  test_cfg:
    #inference_on: teacher
    inference_on: student
  weakAug:
    - NormalizeImage: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], is_scale: true}
  strongAug:
    - AugmentationUTStrong: {}
    - NormalizeImage: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], is_scale: true}
  sup_batch_transforms:
    - Permute: {}
    - PadBatch: {pad_to_stride: 32}
    - Gt2FCOSTarget:
        object_sizes_boundary: [64, 128, 256, 512]
        center_sampling_radius: 1.5
        downsample_ratios: [8, 16, 32, 64, 128]
        norm_reg_targets: True
  unsup_batch_transforms:
    - Permute: {}
    - PadBatch: {pad_to_stride: 32}

architecture: FCOS
pretrain_weights: https://paddledet.bj.bcebos.com/models/pretrained/ResNet50_cos_pretrained.pdparams
```

此外，我们也支持其他检测模型进行半监督训练，比如，`PP-YOLOE`，示例如下：

```python
_BASE_: [
  '../../ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml',
]
weights: output/dt_semi_010_ppyoloe_plus_crn_l_80e_coco/model_final

SSOD: DenseTeacher
DenseTeacher:
  train_cfg:
    ratio: 0.01
    sup_weight: 1.0
    unsup_weight: 1.0
    suppress: linear
    loss_weight: {distill_loss_cls: 4.0, distill_loss_box: 1.0}
    gamma: 2.0
  test_cfg:
    #inference_on: teacher
    inference_on: student
  weakAug:
    - NormalizeImage: {mean: [0., 0., 0.], std: [1., 1., 1.], is_scale: true, norm_type: none}
  strongAug:
    - AugmentationUTStrong: {}
    - NormalizeImage: {mean: [0., 0., 0.], std: [1., 1., 1.], is_scale: true, norm_type: none}
  sup_batch_transforms:
    - Permute: {}
    - PadGT: {}
  unsup_batch_transforms:
    - Permute: {}

pretrain_weights: /paddle/ppyoloe_plus_crn_s_coco_sup010_353.pdparams
architecture: YOLOv3
norm_type: bn #sync_bn !!!
```

### 配置半监督训练集

构建半监督数据集，需要同时配置监督数据集`TrainDataset`和无监督数据集`UnsupTrainDataset`的路径，**注意必须选用`SemiCOCODataSet`类而不是`COCODataSet`类**，如以下所示:

COCO-train2017部分比例：

```python
### dataset config
metric: COCO
num_classes: 80
TrainDataset:
  !SemiCOCODataSet
    image_dir: train2017
    anno_path: semi_annotations/instances_train2017.1@10.json
    dataset_dir: dataset/coco
    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']

UnsupTrainDataset:
  !SemiCOCODataSet
    image_dir: train2017
    anno_path: semi_annotations/instances_train2017.1@10-unlabeled.json
    dataset_dir: dataset/coco
    data_fields: ['image']
    supervised: False
```

或者 COCO-train2017全量：

```python
### dataset config
metric: COCO
num_classes: 80
TrainDataset:
  !SemiCOCODataSet
    image_dir: train2017
    anno_path: annotations/instances_train2017.json
    dataset_dir: dataset/coco
    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']

UnsupTrainDataset:
  !SemiCOCODataSet
    image_dir: unlabeled2017
    anno_path: annotations/image_info_unlabeled2017.json
    dataset_dir: dataset/coco
    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']
    supervised: False
```

验证集`EvalDataset`和测试集`TestDataset`配置一般不需要改变，且还是采用`COCODataSet`类。


### 配置Reader和数据增强

构建半监督训练集的Reader，需要在原先`TrainReader`的基础上，新增加`weak_aug`,`strong_aug`，`sup_batch_transforms`和`unsup_batch_transforms`，注意如果有`NormalizeImage`，需要单独从`sample_transforms`中抽出来放在`weak_aug`和`strong_aug`中。`sample_transforms`为公用的基础数据增强，完整的弱数据增强为``sample_transforms + weak_aug`，完整的强数据增强为`sample_transforms + strong_aug`。如以下所示:

原纯监督模型的`TrainReader`：
```python
TrainReader:
  sample_transforms:
    - Decode: {}
    - RandomDistort: {}
    - RandomExpand: {fill_value: [123.675, 116.28, 103.53]}
    - RandomCrop: {}
    - RandomFlip: {}
  batch_transforms:
    - BatchRandomResize: {target_size: [320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768], random_size: True, random_interp: True, keep_ratio: False}
    - NormalizeImage: {mean: [0., 0., 0.], std: [1., 1., 1.], norm_type: none}
    - Permute: {}
    - PadGT: {}
  batch_size: 8
  shuffle: true
  drop_last: true
  use_shared_memory: true
  collate_batch: true

```

更改后的半监督TrainReader：

```python
SemiTrainReader:
  sample_transforms:
    - Decode: {}
    - RandomDistort: {}
    - RandomExpand: {fill_value: [123.675, 116.28, 103.53]}
    - RandomFlip: {}
    - RandomCrop: {} # unsup will be fake gt_boxes
  weak_aug:
    - NormalizeImage: {mean: [0., 0., 0.], std: [1., 1., 1.], is_scale: true, norm_type: none}
  strong_aug:
    - StrongAugImage: {transforms: [
        RandomColorJitter: {prob: 0.8, brightness: 0.4, contrast: 0.4, saturation: 0.4, hue: 0.1},
        RandomErasingCrop: {},
        RandomGaussianBlur: {prob: 0.5, sigma: [0.1, 2.0]},
        RandomGrayscale: {prob: 0.2},
      ]}
    - NormalizeImage: {mean: [0., 0., 0.], std: [1., 1., 1.], is_scale: true, norm_type: none}
  sup_batch_transforms:
    - BatchRandomResize: {target_size: [320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768], random_size: True, random_interp: True, keep_ratio: False}
    - Permute: {}
    - PadGT: {}
  unsup_batch_transforms:
    - BatchRandomResize: {target_size: [320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768], random_size: True, random_interp: True, keep_ratio: False}
    - Permute: {}
  sup_batch_size: 8
  unsup_batch_size: 8
  shuffle: True
  drop_last: True
  collate_batch: True
```

### 模型配置

如果没有特殊改动，还是继承自基础检测器里的模型配置

### 其他配置

训练epoch数需要和全量数据训练时换算总iter数保持一致，如全量训练12epoch(换算共约180k个iter)，则10%监督数据的半监督训练，总epoch数需要为120epoch。，示例如下：

```python
### other config
epoch: *epochs
LearningRate:
  base_lr: 0.01
  schedulers:
  - !CosineDecay
    max_epochs: *epochs
    use_warmup: *use_warmup
  - !LinearWarmup
    start_factor: 0.001
    epochs: 3

OptimizerBuilder:
  optimizer:
    momentum: 0.9
    type: Momentum
  regularizer:
    factor: 0.0005 # dt-fcos 0.0001
    type: L2
  clip_grad_by_norm: 1.0 # dt-fcos clip_grad_by_value
```


## 使用说明

仅训练时需要特别配置，评估、预测、部署均按基础检测器的配置文件即可。

### 训练

```
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c ssod/denseteacher/dt_semi_010_fcos_r50_fpn_1x_coco.yml
python -m paddle.distributed.launch --log_dir=denseteacher_fcos/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ssod/denseteacher/dt_semi_010_fcos_r50_fpn_1x_coco.yml --eval
```

### 评估

CUDA_VISIBLE_DEVICES=0 python3.7 tools/eval.py -c ${config} -o weights=${weights}

### 预测

#CUDA_VISIBLE_DEVICES=7 python3.7 tools/infer.py -c ${config} -o weights=${weights} --infer_img=demo/000000014439_640x640.jpg

### 部署

部署只需要基础检测器，只需保留`_BASE_`和`dataset config`，其余配置部分均注释掉，即当做基础检测器去部署使用。


## 引用

```
@article{xu2021end,
  title={End-to-End Semi-Supervised Object Detection with Soft Teacher},
  author={Xu, Mengde and Zhang, Zheng and Hu, Han and Wang, Jianfeng and Wang, Lijuan and Wei, Fangyun and Bai, Xiang and Liu, Zicheng},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```
