python src/baseline2_main.py \
    --model Qwen2-72B-Instruct-test \
    --task train \
    --threads 50

python src/baseline2_main.py \
    --model Qwen2-72B-Instruct-test \
    --task test \
    --data-path data/round1_test_data.jsonl \
    --threads 50
