export FLAGS_allocator_strategy=auto_growth
model_type=ssod/denseteacher
#job_name=denseteacher_ppyoloe_plus_crn_s_coco_semi010_load
#job_name=denseteacher_ppyoloe_plus_crn_s_coco_semi010
job_name=denseteacher_ppyoloe_plus_crn_s_coco_semi005_load
#job_name=denseteacher_ppyoloe_plus_crn_s_coco_semi005

config=configs/${model_type}/${job_name}.yml
log_dir=log_dir/${job_name}
weights=output/denseteacher_ppyoloe_plus_crn_s_coco_semi005_load/1_.pdparams
#weights=~/.cache/paddle/weights/ppyoloe_plus_crn_s_80e_coco_sup005.pdparams

# 1. training
#CUDA_VISIBLE_DEVICES=5 python3.7 tools/train.py -c ${config} # -r ${weights}
python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --eval

# 2. eval
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/eval.py -c ${config} -o weights=https://paddledet.bj.bcebos.com/models/${job_name}.pdparams
#CUDA_VISIBLE_DEVICES=2 python3.7 tools/eval.py -c ${config} -o weights=${weights}
