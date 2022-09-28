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
半监督目标检测(SSOD)是**同时使用有标注数据和无标注数据**进行训练的目标检测，既可以极大地节省标注成本，也可以充分利用无标注数据进一步提高精度。
PaddleDetection团队提供了基于[DenseTeacher](denseteacher)的一套SSOD方案，并且支持适配PP-YOLOE系列模型。

## 模型库

### [DenseTeacher](denseteacher)

|      模型       |   基础检测器             |  Supervision   |  mAP<sup>val<br>0.5:0.95 |  模型下载  |   配置文件   |
| :------------: | :---------------------: | :-----------: | :---------------: |:-----------: | :---------------: |
| DenseTeacher   |   [fcos_r50_fpn_1x_coco](../fcos/fcos_r50_fpn_1x_coco.yml)  |      1%       |       -        | [download]() | [config](denseteacher/dt_semi_001_fcos_r50_fpn_1x_coco.yml) |
| DenseTeacher   |   [fcos_r50_fpn_1x_coco](../fcos/fcos_r50_fpn_1x_coco.yml)  |      5%       |       -        | [download]() | [config](denseteacher/dt_semi_005_fcos_r50_fpn_1x_coco.yml) |
| DenseTeacher   |   [fcos_r50_fpn_1x_coco](../fcos/fcos_r50_fpn_1x_coco.yml)  |      10%      |       -        | [download]() | [config](denseteacher/dt_semi_010_fcos_r50_fpn_1x_coco.yml) |
| DenseTeacher   |   [fcos_r50_fpn_1x_coco](../fcos/fcos_r50_fpn_1x_coco.yml)  |      full     |       -        | [download]() | [config](denseteacher/dt_semi_full_fcos_r50_fpn_1x_coco.yml) |

**注意:**
- 若训练**纯监督数据**的模型，请参照**基础检测器的配置文件**，只需**修改对应数据集标注路径**即可；


## 数据集准备

PaddleDetection团队提供了COCO数据集全部的标注文件，请下载并解压存放至对应目录:

```shell
# 下载命令
wget ...
```

<details>
<summary> 解压后的数据集目录如下：</summary>

```
PaddleDetection
├── dataset
│   ├── coco
│   │   ├── annotations
│   │   │   ├── image_info_unlabeled2017.json
│   │   │   ├── instances_train2017.json
│   │   │   ├── instances_val2017.json
│   │   ├── semi_anns
│   │   │   ├── instances_train2017.1@1.json
│   │   │   ├── instances_train2017.1@1-unlabeled.json
│   │   │   ├── instances_train2017.1@2.json
│   │   │   ├── instances_train2017.1@2-unlabeled.json
│   │   │   ├── instances_train2017.1@5.json
│   │   │   ├── instances_train2017.1@5-unlabeled.json
│   │   │   ├── instances_train2017.1@10.json
│   │   │   ├── instances_train2017.1@10-unlabeled.json
│   │   │   ├── instances_train2017.2@1.json
│   │   │   ├── instances_train2017.2@1-unlabeled.json
│   │   ├── train2017
│   │   ├── unlabeled2017
│   │   ├── val2017
```

</details>

########## from mmdet
半监督目标检测在 COCO 数据集上有两种比较通用的实验设置：

（1）将 `train2017` 按照固定百分比（1%，2%，5% 和 10%）划分出一部分数据作为标签数据集，剩余的训练集数据作为无标签数据集，同时考虑划分不同的训练集数据作为标签数据集对半监督训练的结果影响较大，所以采用五折交叉验证来评估算法性能。我们提供了数据集划分脚本：
```shell
python tools/misc/split_coco.py
```
该脚本默认会按照 1%，2%，5% 和 10% 的标签数据占比划分 `train2017`，每一种划分会随机重复 5 次，用于交叉验证。生成的半监督标注文件名称格式如下：
- 标签数据集标注名称格式：`instances_train2017.{fold}@{percent}.json`
- 无标签数据集名称标注：`instances_train2017.{fold}@{percent}-unlabeled.json`
其中，`fold` 用于交叉验证，`percent` 表示标签数据的占比。 

（2）将 `train2017` 作为标签数据集，`unlabeled2017` 作为无标签数据集。可直接下载PaddleDetection团队提供的`instances_unlabeled2017.json`。
```
wget ....
```
########## from mmdet



## 配置半监督检测

配置半监督检测，需要基于选用的**基础检测器**的配置文件，依次做出如下5点改动：

### 全局配置

需要在配置文件中添加如下全局配置：

```python
semi_supervised: True # 必须设置为True
semi_start_steps: 5000 # 自己设定

use_ema: True # 必须设置为True
ema_decay: 0.9996
ema_decay_type: None
ema_start_steps: 3000 # 自己设定
```

### 配置半监督模型

以 `DenseTeacher` 为例，选择 `fcos_r50_fpn_1x_coco` 作为 `基础检测器` 进行半监督训练：

```python
_BASE_: [
  '../fcos/fcos_r50_fpn_1x_coco.yml',
]
weights: output/dt_semi_010_fcos_r50_fpn_1x_coco/model_final

use_ema: True
ema_decay: 0.9996
ema_decay_type: None
ema_start_steps: 3000

semi_supervised: True
semi_start_steps: 5
semi_distill:
  ratio: 0.01
  sup_weight: 1.0
  unsup_weight: 1.0
  suppress: 'linear'
  loss_weight: {cls: 4.0, reg: 1.0, ctn: 1.0}
  gamma: 2.0
```

此外，我们也支持其他检测模型进行半监督训练，比如，`PP-YOLOE`，示例如下：

```python

```

### 配置半监督训练集

构建半监督数据集，需要同时配置有标注数据集`TrainDataset`和无标注数据集`UnsupTrainDataset`的路径，**注意必须选用`SemiCOCODataSet`类而不是`COCODataSet`类**，如以下所示:

```python
TrainDataset:
  !SemiCOCODataSet
    image_dir: train2017
    anno_path: annotations/instances_train2017.json
    dataset_dir: dataset/coco
    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']
    sup_file: 'dataset/coco/coco_supervision.txt'
    sup_percentage: 10.0
    sup_seed: 1

UnsupTrainDataset:
  !SemiCOCODataSet
    image_dir: train2017
    anno_path: annotations/instances_train2017.json
    dataset_dir: dataset/coco
    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']
    sup_file: 'dataset/coco/coco_supervision.txt'
    sup_percentage: 10.0
    sup_seed: 1
    supervised: False
```
验证集`EvalDataset`和测试集`TestDataset`配置一般不需要改变，且还是采用`COCODataSet`类。


### 配置半监督数据增强

构建半监督训练集的数据增强，需要同时配置有标注数据和无标注数据的数据增强，如以下所示:

```python
SupTrainReader:
  sample_transforms:
    - Decode: {}
    - Resize: {target_size: [260, 480], keep_ratio: true, interp: 1}
    - NormalizeImage: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], is_scale: true}
    - RandomFlip: {}
  batch_transforms:
    - Permute: {}
    - PadBatch: {pad_to_stride: 128}
    - Gt2FCOSTarget:
        object_sizes_boundary: [64, 128, 256, 512]
        center_sampling_radius: 1.5
        downsample_ratios: [8, 16, 32, 64, 128]
        norm_reg_targets: True
  batch_size: 2
  shuffle: true
  drop_last: true


UnsupTrainReader:
  sample_transforms:
    - Decode: {}
    - Resize: {target_size: [260, 480], keep_ratio: true, interp: 1}
    - NormalizeImage: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], is_scale: true}
    - RandomFlip: {}
  batch_transforms:
    - Permute: {}
    - PadBatch: {pad_to_stride: 128}
  batch_size: 2
  shuffle: true
  drop_last: true
```




### 其他配置

训练epoch数需要和全量数据训练时换算总iter数保持一致，如全量训练12epoch(换算共约180k个iter)，则10%监督数据的半监督训练，总epoch数需要为120epoch。，示例如下：
```python

```

优化器的配置需要更改，示例如下：
```python

```


## 使用说明

仅训练时需要特别配置，评估、预测、部署均按基础检测器的配置文件即可。

### 训练


### 评估


### 预测


### 部署



## 引用

```
@article{xu2021end,
  title={End-to-End Semi-Supervised Object Detection with Soft Teacher},
  author={Xu, Mengde and Zhang, Zheng and Hu, Han and Wang, Jianfeng and Wang, Lijuan and Wei, Fangyun and Bai, Xiang and Liu, Zicheng},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```
