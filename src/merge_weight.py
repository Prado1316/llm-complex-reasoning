import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./qwen/Qwen2-7B-Instruct/")
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./merged_model")
    return parser.parse_args()


def _setup_all():
    global tokenizer, model
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.mode_path, trust_remote_code=True)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        args.mode_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).eval()

    # 加载lora权重
    model = PeftModel.from_pretrained(model, model_id=args.lora_path)


def _make_infer_example():
    system_prompt = (
        "# 角色定义\n"
        "你是一个逻辑推理专家，擅长解决逻辑推理问题。\n"
        "# 任务要求\n"
        "1. 以下是一个逻辑推理的题目，形式为单项选择题\n"
        "2. 所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假\n"
        "3. 所有的定义，认知都来自于题目, 除此之外你不知道任何信息\n"
        "4. 禁止基于题目外任何信息对推理过程/结果做任何修改\n"
        "5. 请你根据提供的信息，逐步分析问题并在最后一行输出答案\n"
        "6. 最后一行的格式为`答案是: (你认为的正确答案)`\n"
    )

    user_input = (
        "在一个学术环境中，有多名学生（男生和女生），他们注册在不同的课程中。其中一些课程与学生的注册信息如下所示：\n"
        "- João、Antônio、Carlos、Luísa、Maria 和 Isabel 是学校的学生。\n"
        "- 课程包括 LEI，LSIRC 和 LSIG。\n"
        "- 学生有时会注册某些课程的特定科目。\n"
        "- João 注册了至少一门课程，而 Antonío、Carlos、Luísa 和 Isabel 没有注册任何课程。\n"
        "- Maria 注册了科目 FP。\n"
        "根据以上信息，回答以下选择题：\n"
        "### 问题:\n"
        "选择题 1：\n"
        "哪些学生有注册任何课程？\n"
        "A. João\n"
        "B. Antonío\n"
        "C. Carlos\n"
        "D. Luísa\n"
    )

    inputs = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda")

    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs["input_ids"].shape[1] :]
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))


args = parse_args()

_setup_all()

_make_infer_example()

# 模型合并存储
merged_model = model.merge_and_unload()
# 将权重保存为safetensors格式的权重, 且每个权重文件最大不超过2GB(2048MB)
merged_model.save_pretrained(
    args.output_path, max_shard_size="2048MB", safe_serialization=True
)

# !cp ./qwen/Qwen2-7B-Instruct/tokenizer.json ./merged_model_an/
