export FLAGS_allocator_strategy=auto_growth

model_type=faster_rcnn/labelmatching
job_name=labelmatching
config=configs/${model_type}/${job_name}.yml
log_dir=log_dir/${job_name}
#weights=/paddle/weights/{job_name}.pdparams
weights=https://paddledet.bj.bcebos.com/models/${job_name}.pdparams

# 1. training
#CUDA_VISIBLE_DEVICES=3 python3.7 tools/train.py -c ${config} --semi_train --eval # --amp
python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --eval --semi_train # --amp
