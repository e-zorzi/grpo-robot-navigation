export CUDA_VISIBLE_DEVICES=0 
cd /scratch/ezorzi/vllm_dir && uv run vllm serve Qwen/Qwen3-30B-A3B --attention-backend FLASH_ATTN --max-model-len 10000 --gpu-memory-utilization 0.53 --quantization bitsandbytes &
sleep 60 &&
cd /scratch/ezorzi/grpo-robot-navigation && uv run main.py --model_name Qwen/Qwen2.5-VL-7B-Instruct --per_device_train_batch_size 32 --no-save_artifacts &
sleep 120 && unset CUDA_VISIBLE_DEVICES
