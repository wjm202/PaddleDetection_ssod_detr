export FLAGS_allocator_strategy=auto_growth

model_type=ssod/baseline
#job_name=fcos_r50_fpn_coco_sup010
job_name=ppyoloe_plus_crn_s_coco_sup010
#job_name=faster_rcnn_r50_fpn_coco_sup010
#job_name=retinanet_r50_fpn_coco_sup010

config=configs/${model_type}/${job_name}.yml
log_dir=log_dir/${job_name}
weights=output/dt_semi_010_ppyoloe_plus_crn_s_coco_load/0_.pdparams # ln -s 0.pdema 0_.pdparams
#weights=/paddle/fcos_r50_fpn_coco_sup010_26.3.pdparams
#weights=/paddle/ppyoloe_plus_crn_s_coco_sup010_353.pdparams
#weights=/paddle/faster_rcnn_r50_fpn_coco_sup010_25.6.pdparams
#weights=/paddle/retinanet_r50_fpn_coco_sup010_23.6.pdparams

# 1. training
#CUDA_VISIBLE_DEVICES=6 python3.7 tools/train.py -c ${config} # -r ${weights}
#python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} #--eval #-r output/fcos_r50_fpn_multiscale_2x_coco/3 #--eval

# 2. eval
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/eval.py -c ${config} -o weights=https://paddledet.bj.bcebos.com/models/${job_name}.pdparams
CUDA_VISIBLE_DEVICES=2 python3.7 tools/eval.py -c ${config} -o weights=${weights}
