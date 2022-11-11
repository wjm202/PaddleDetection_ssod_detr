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
PaddleDetection团队提供的SSOD方案包括以下方法，并且支持适配PP-YOLOE系列模型。

- SSOD
  - [DenseTeacher](denseteacher/)
  - [SoftTeacher]()
  - [LabelMatch]()


## 模型库

### [Baseline](baseline)

**纯监督数据**模型的训练和模型库，请参照[Baseline](baseline)；


### [DenseTeacher](denseteacher)

|      模型       |   基础检测器             |   监督数据比例   |  mAP<sup>val<br>0.5:0.95 |  模型下载  |   配置文件   |
| :------------: | :---------------------: | :-----------: | :---------------: |:-----------: | :---------------: |
| DenseTeacher   |   [FCOS](../fcos)  |      5%       |       -        | [download]() | [config](denseteacher/dt_semi_005_fcos_r50_fpn_1x_coco.yml) |
| DenseTeacher   |   [FCOS](../fcos)  |      10%      |       -        | [download]() | [config](denseteacher/dt_semi_010_fcos_r50_fpn_1x_coco.yml) |
| DenseTeacher   |   [FCOS](../fcos)  |      full     |       -        | [download]() | [config](denseteacher/dt_semi_full_fcos_r50_fpn_1x_coco.yml) |

**注意:**
- 以上模型验证集均为`val2017`全量；
- COCO监督数据比例不为`full`的数据集，监督数据和无监督数据均为从`train2017`中抽取的部分，`semi_010`表示抽取10%作为监督数据，在剩余90%的`train2017`抽取无监督数据；`semi_005`表示抽取5%作为监督数据，在剩余95%中抽取无监督数据；一般训练时监督和无监督数据的比例为`1:1`；
- `full`表示全部使用`train2017`全量作为监督数据，使用`unlabeled2017`全量作为无监督数据；
- 抽部分百分比的监督数据，即比例不为`full`的数据集，精度会因**抽取的数据的不同**而有0.5mAP之多的差异；


## 数据集准备

### 常规设置

COCO数据集上半监督目标检测有两种常规设置：

（1）使用部分比例的`train2017`：监督数据和无监督数据均从`train2017`中按固定百分比1%，2%，5% 和 10%）抽取。考虑到抽取方法的不同会对半监督训练的结果影响较大，所以采用五折交叉验证来评估。运行数据集划分制作的脚本如下：
```bash
python tools/gen_semi_coco.py
```
会按照 1%，2%，5% 和 10% 的监督数据比例来划分 `train2017`全集，为了交叉验证每一种划分会随机重复5次，生成的半监督标注文件如下：
- 监督数据集标注：`instances_train2017.{fold}@{percent}.json`
- 无监督数据集标注：`instances_train2017.{fold}@{percent}-unlabeled.json`
其中，`fold` 表示交叉验证，`percent` 表示有监督数据的百分比。

（2）使用全量`train2017`监督数据 和 全量`unlabeled2017`无监督数据：

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


<details>
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
  '../../fcos/fcos_r50_fpn_1x_coco.yml',
]
weights: output/dt_semi_010_fcos_r50_fpn_coco/model_final
```

并依次做出如下5点改动：

### 全局配置

需要在配置文件中添加如下全局配置：

```python
### global config
semi_supervised: True # 必须设置为True
semi_start_steps: 5000 # 自己设定
use_ema: True # 必须设置为True
use_meanteacher: True # 必须设置为True
ema_decay: 0.9996
ema_decay_type: None
ema_start_steps: 3000 # 自己设定
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


### 配置半监督数据增强

构建半监督训练集的数据增强，需要拆解原先`TrainReader`的`batch_transforms`为监督和无监督部分，且配置在全局中。注意如果有`NormalizeImage`，需要单独从`sample_transforms`中抽出来。如以下所示:

原纯监督模型的`TrainReader`：
```python
TrainReader:
  sample_transforms:
    - Decode: {}
    - RandomResize: {target_size: [[640, 1333], [672, 1333], [704, 1333], [736, 1333], [768, 1333], [800, 1333]], keep_ratio: True, interp: 1}
    - NormalizeImage: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], is_scale: True}
    - RandomFlip: {}
  batch_transforms:
    - Permute: {}
    - PadBatch: {pad_to_stride: 32}
    - Gt2FCOSTarget:
        object_sizes_boundary: [64, 128, 256, 512]
        center_sampling_radius: 1.5
        downsample_ratios: [8, 16, 32, 64, 128]
        norm_reg_targets: True
  batch_size: 2
  shuffle: True
  drop_last: True

```

半监督的Reader配置：

```python
SSOD: DenseTeacher
DenseTeacher:
  train_cfg: xx
  test_cfg: xx
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

worker_num: 2
SupTrainReader:
  sample_transforms:
    - Decode: {}
    - RandomResize: {target_size: [[640, 1333], [672, 1333], [704, 1333], [736, 1333], [768, 1333], [800, 1333]], keep_ratio: True, interp: 1}
    - RandomFlip: {}
  batch_size: 2
  shuffle: True
  drop_last: True

UnsupTrainReader:
  sample_transforms:
    - Decode: {}
    - RandomResize: {target_size: [[640, 1333], [672, 1333], [704, 1333], [736, 1333], [768, 1333], [800, 1333]], keep_ratio: True, interp: 1}
    - RandomFlip: {}
  batch_size: 2
  shuffle: True
  drop_last: False
```

### 其他配置

训练epoch数需要和全量数据训练时换算总iter数保持一致，如全量训练12epoch(换算共约180k个iter)，则10%监督数据的半监督训练，总epoch数需要为120epoch。，示例如下：

```python
### other config
epoch: 240
LearningRate:
  base_lr: 0.001
  schedulers:
  - !PiecewiseDecay
    gamma: 0.1
    milestones: [240]
  - !LinearWarmup
    start_factor: 0.001
    steps: 1000
```

优化器的配置需要更改，示例如下：

```python
OptimizerBuilder:
  optimizer:
    momentum: 0.9
    type: Momentum
  regularizer:
    factor: 0.0001
    type: L2
  clip_grad_by_norm: 1.0
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
