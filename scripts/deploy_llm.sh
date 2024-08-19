
# model_dir=/hy-tmp/qwen/merged_model_an
model_dir=output/qwen2-7b-instruct/lora/sft-merged
served_model_name=Qwen2-7B-Instruct-lora

python -m vllm.entrypoints.openai.api_server \
    --model $model_dir  \
    --served-model-name $served_model_name \
    --max-model-len 4096 \
    --enable-prefix-caching \
    --use-v2-block-manager \
    --gpu-memory-utilization 0.95 \
    --enable-chunked-prefill True