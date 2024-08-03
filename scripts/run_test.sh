threads=100
model=Qwen2-72B-Instruct-test

python src/baseline2_main.py \
    --model $model \
    --task train \
    --threads $threads

python src/baseline2_main.py \
    --model $model \
    --task test \
    --data-path data/round1_test_data.jsonl \
    --threads $threads
