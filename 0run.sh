export FLAGS_allocator_strategy=auto_growth
model_type=ssod/denseteacher
#job_name=dt_semi_010_fcos_r50_fpn_1x_coco_load
job_name=dt_semi_010_ppyoloe_plus_crn_s_coco_load
#job_name=dt_semi_010_faster_rcnn_r50_fpn_coco_load
#job_name=dt_semi_010_retinanet_r50_fpn_coco_load

config=configs/${model_type}/${job_name}.yml
log_dir=log_dir/${job_name}
weights=output/${job_name}/model_final.pdparams

# 1. training
#CUDA_VISIBLE_DEVICES=2 python3.7 tools/train.py -c ${config} # -r ${weights}
python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} #--eval

# 2. eval
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/eval.py -c ${config} -o weights=https://paddledet.bj.bcebos.com/models/${job_name}.pdparams
#CUDA_VISIBLE_DEVICES=2 python3.7 tools/eval.py -c ${config} -o weights=${weights}
