export FLAGS_allocator_strategy=auto_growth
model_type=ssod/baseline
job_name=fcos_r50_fpn_multiscale_2x_coco
config=configs/${model_type}/${job_name}.yml
log_dir=log_dir/${job_name}
#weights=/paddle/unicorn/PaddleDetection_semi/cvpods_fcos_10k.pdparams # 41.1 21.9
#weights=/paddle/unicorn/PaddleDetection_semi/cvpods_fcos.pdparams # 52.1 51.5
weights=/paddle/unicorn/PaddleDetection_semi/cvpods_fcos_5k.pdparams

# 1. training
#CUDA_VISIBLE_DEVICES=5 python3.7 tools/train.py -c ${config} -r ${weights}
python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} -r ${weights} #output/fcos_r50_fpn_multiscale_2x_coco/3 #--eval

# 2. eval
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/eval.py -c ${config} -o weights=https://paddledet.bj.bcebos.com/models/${job_name}.pdparams
#CUDA_VISIBLE_DEVICES=3 python3.7 tools/eval.py -c ${config} -o weights=${weights} # 

# 3. tools infer
#CUDA_VISIBLE_DEVICES=7 python3.7 tools/infer.py -c ${config} -o weights=https://paddledet.bj.bcebos.com/models/${job_name}.pdparams --infer_img=demo/000000014439.jpg
#CUDA_VISIBLE_DEVICES=7 python3.7 tools/infer.py -c ${config} -o weights=${weights} --infer_img=demo/000000014439_640x640.jpg

# 4.导出模型
#CUDA_VISIBLE_DEVICES=1 python3.7 tools/export_model.py -c ${config} -o weights=${weights} #exclude_nms=True trt=True

# 5.部署预测
#CUDA_VISIBLE_DEVICES=1 python3.7 deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439_640x640.jpg --device=GPU
#CUDA_VISIBLE_DEVICES=1 python3.7 deploy/python/infer.py --model_dir=output_inference/yolox_l_300e_coco/ --image_file=demo/000000014439_640x640.jpg --device=GPU

# 6.部署测速
#CUDA_VISIBLE_DEVICES=1 python3.7 deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439_640x640.jpg --device=GPU --run_benchmark=True # --run_mode=trt_fp16

# 7.onnx导出
#paddle2onnx --model_dir output_inference/${job_name} --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 12 --save_file ${job_name}.onnx
