# Supervised Baseline 纯监督模型基线

## COCO数据集模型库

### [FCOS](../../fcos)

|  基础模型          |    监督数据比例   |  mAP<sup>val<br>0.5:0.95 |  模型下载  |   配置文件   |
| :---------------: | :-------------: | :---------------------: |:--------: | :---------: |
| FCOS ResNet50-FPN |        5%       |       21.3        | [download]() | [config](fcos_r50_fpn_coco_sup005.yml) |
| FCOS ResNet50-FPN |        10%      |       26.3        | [download]() | [config](fcos_r50_fpn_coco_sup010.yml) |
| FCOS ResNet50-FPN |        full     |       42.6        | [download](https://paddledet.bj.bcebos.com/models/fcos_r50_fpn_iou_multiscale_2x_coco.pdparams) | [config](../../fcos/fcos_r50_fpn_iou_multiscale_2x_coco.yml) |

### [PP-YOLOE+](../../ppyoloe)

|  基础模型          |    监督数据比例   |  mAP<sup>val<br>0.5:0.95 |  模型下载  |   配置文件   |
| :---------------: | :-------------: | :---------------------: |:--------: | :---------: |
| PP-YOLOE+_s       |        5%      |        32.8       | [download]() | [config](ppyoloe_plus_crn_s_coco_sup005.yml) |
| PP-YOLOE+_s       |        10%      |       35.3       | [download]() | [config](ppyoloe_plus_crn_s_coco_sup010.yml) |
| PP-YOLOE+_s       |        full     |       43.7       | [download](https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_s_80e_coco.pdparams) | [config](../../ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml) |


**注意:**
- COCO监督数据集的训练集为从train2017中抽取的部分，`sup010`表示抽取10%的监督数据训练，`sup005`表示抽取5%，`full`表示全部train2017，验证集均为val2017全量；
