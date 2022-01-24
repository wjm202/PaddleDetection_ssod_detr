#python3.7 dygraph_print.py -c configs/yolox/yolox_darknet53_coco.yml 2>&1 | tee ppdet_yolox_darknet53_print.txt

export FLAGS_allocator_strategy=auto_growth
model_type=yolox
job_name=byte_x_mot_ablation
config=configs/${model_type}/${job_name}.yml
log_dir=log_dir/${job_name}

# 1. training
CUDA_VISIBLE_DEVICES=7 python3.7 tools/train.py -c ${config} --eval --fp16 -r ../../byte_x_mot_ablation_paddle.pdparams
#python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --eval --fp16 -r output/byte_x_mot_ablation/0 # &> ${job_name}.log &

# 2. eval
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/eval.py -c ${config} -o weights=https://paddledet.bj.bcebos.com/models/${job_name}.pdparams
#CUDA_VISIBLE_DEVICES=6 python3.7 tools/eval.py -c ${config} -o weights=output/byte_x_mot_ablation/best_model.pdparams
#CUDA_VISIBLE_DEVICES=5 python3.7 tools/eval.py -c ${config} -o weights=../../byte_x_mot_ablation_paddle_rgb.pdparams #../../../paddle_${job_name}.pdparams

# 3. tools infer
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/infer.py -c ${config} -o weights=https://paddledet.bj.bcebos.com/models/${job_name}.pdparams --infer_img=demo/000000014439.jpg
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/infer.py -c ${config} -o weights=../byte_x_mot_ablation_paddle.pdparams --infer_img=demo/000000014439.jpg

# 4.export model
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/export_model.py -c ${config} -o weights=https://paddledet.bj.bcebos.com/models/${job_name}.pdparams
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/export_model.py -c ${config} -o weights=../byte_x_mot_ablation_paddle.pdparams

# 5. deploy infer
#CUDA_VISIBLE_DEVICES=0 python3.7 deploy/python/infer.py --model_dir=output_inference/${job_name} --video_file=test.mp4 --device=GPU

# 6. deploy speed
#CUDA_VISIBLE_DEVICES=0 python3.7 deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000329219.jpg --device=GPU #--run_benchmark=True #--trt_max_shape=640 --trt_min_shape=640 --trt_opt_shape=640 --run_mode=trt_fp16
#CUDA_VISIBLE_DEVICES=0 python3.7 deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439.jpg --device=GPU #--run_benchmark=True #--trt_max_shape=640 --trt_min_shape=640 --trt_opt_shape=640 --run_mode=trt_fp16
