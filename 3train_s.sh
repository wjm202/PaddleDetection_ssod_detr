export FLAGS_allocator_strategy=auto_growth
name=s
model_type=visdrone
job_name=ppyoloe_p2_crn_l_80e_visdrone
config=configs/${model_type}/${job_name}.yml
log_dir=log_dir/${job_name}

weight_path=https://paddledet.bj.bcebos.com/models/ppyoloe_p2_crn_l_80e_visdrone.pdparams

# 1. training
#CUDA_VISIBLE_DEVICES=7 python3.7 tools/train.py -c ${config} --amp # -r ${weight_path} 
#python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --eval --amp #-r output/yolov5_s_coco/best_model #--fp16 # &> ${job_name}.log &

# 2. eval
CUDA_VISIBLE_DEVICES=1 python3.7 tools/eval.py -c ${config} -o weights=${weight_path}  # im8 59.6
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/eval.py -c ${config} -o weights=output/${job_name}/8.pdparams
#CUDA_VISIBLE_DEVICES=2 python3.7 tools/eval.py -c ${config} -o weights=../paddle_yolov5_fuse/paddle_yolov5_s_coco.pdparams #./${job_name}_final_fromori_noneedrgb.pdparams
# im8 56.7

#CUDA_VISIBLE_DEVICES=1 python3.7 tools/eval.py -c ${config} -o weights=${weight_path} #--classwise #./${job_name}_final_fromori_noneedrgb.pdparams
# im8 59.6

# 3. tools infer
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/infer.py -c ${config} -o weights=https://paddledet.bj.bcebos.com/models/${job_name}.pdparams --infer_img=demo/000000014439.jpg
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/infer.py -c ${config} -o weights=./${job_name}_final_fromori_noneedrgb.pdparams --infer_img=demo/000000014439.jpg --draw_threshold=0.25
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/infer.py -c ${config} -o weights=new_yolov5_s.pdparams --infer_img=demo/000000014439.jpg --draw_threshold=0.25

#CUDA_VISIBLE_DEVICES=0 python3.7 tools/infer.py -c ${config} -o weights=${weight_path} --infer_img=demo/000000570688.jpg #--draw_threshold=0.5


# 4.export model
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/export_model.py -c ${config} -o weights=https://paddledet.bj.bcebos.com/models/${job_name}.pdparams
#CUDA_VISIBLE_DEVICES=5 python3.7 tools/export_model.py -c ${config} -o weights=${weight_path}

# 5. deploy infer
#CUDA_VISIBLE_DEVICES=5 python3.7 deploy/python/infer.py --model_dir=output_inference/${job_name} --video_file=test.mp4 --device=GPU

# 6. deploy speed
#CUDA_VISIBLE_DEVICES=5 python3.7 deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000570688.jpg --device=GPU #--run_benchmark=True #--trt_max_shape=640 --trt_min_shape=640 --trt_opt_shape=640 --run_mode=trt_fp16
#CUDA_VISIBLE_DEVICES=5 python3.7 deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000570688.jpg --device=GPU --run_benchmark=True # --trt_max_shape=640 --trt_min_shape=640 --trt_opt_shape=640 --run_mode=trt_fp16
