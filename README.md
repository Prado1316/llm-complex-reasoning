# Change Log

-   2024年08月08日22:46:26 从round1_train_data.jsonl构造推理数据集,筛选出答案和label一致的数据 -> `ticoAg/llm-complex-reasoning-train-qwen2-72b-instruct-correct`
-   2024 年 08 月 03 日 00:31:33 预测流程代码优化
-   2024 年 08 月 01 日 12:58:47 初始化仓库 from dw task3 baseline2

# Usage

1. 项目根创建.env 文件
2. 补充`OPENAI_BASE_URL`和`OPENAI_API_KEY`环境变量
3. 启动执行脚本

# ToDo

-   [x] 集成训练框架
-   [ ] 预测后选项修复机制
-   [x] 当前代码可用性 debug

# Usage

```sh
pip install -r requirements.txt
```
# 启动训练
## 使用LLaMA-Factory
```sh
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ..
export USE_MODELSCOPE_HUB=1
llamafactory-cli train scripts/examples/qwen2_0.5b_lora_sft.yaml
```
更多示例: https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/README.md

## 使用baseline
```sh
python src/train.py --model_name_or_path qwen/Qwen2-0___5B-Instruct --num_proc 8 --lr_scheduler_type cosine --bf16 --per_device_train_batch_size 1
```