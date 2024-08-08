import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

mode_path = "./qwen/Qwen2-7B-Instruct/"
lora_path = "./output/Qwen2_instruct_lora_an/checkpoint-100"  # 这里改称你的 lora 输出对应 checkpoint 地址

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    mode_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
).eval()

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path)

prompt = """你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请逐步分析问题并在最后一行输出答案，最后一行的格式为"答案是：A"。题目如下：\n\n### 题目:\n假设您需要构建一个二叉搜索树，其中每个节点或者是一个空的节点（称为"空节点"），或者是一个包含一个整数值和两个子树的节点（称为"数值节点"）。以下是构建这棵树的规则：\n\n1. 树中不存在重复的元素。\n2. 对于每个数值节点，其左子树的所有值都小于该节点的值，其右子树的所有值都大于该节点的值。\n3. 插入一个新值到一个"空节点"时，该"空节点"会被一个包含新值的新的数值节点取代。\n4. 插入一个已存在的数值将不会改变树。\n\n请基于以上规则，回答以下选择题：\n\n### 问题:\n选择题 1：\n给定一个空的二叉搜索树，插入下列数字: [5, 9, 2, 10, 11, 3]，下面哪个选项正确描述了结果树的结构？\nA. tree(5, tree(2, tree(3, nil, nil), nil), tree(9, tree(10, nil, nil), tree(11, nil, nil)))\nB. tree(5, tree(2, nil, tree(3, nil, nil)), tree(9, nil, tree(10, nil, tree(11, nil, nil))))\nC. tree(5, tree(3, tree(2, nil, nil), nil), tree(9, nil, tree(10, tree(11, nil, nil), nil)))\nD. tree(5, nil, tree(2, nil, tree(3, nil, nil)), tree(9, tree(11, nil, nil), tree(10, nil, nil)))"""
inputs = tokenizer.apply_chat_template(
    [
        {"role": "user", "content": "你是一个逻辑推理专家，擅长解决逻辑推理问题。"},
        {"role": "user", "content": prompt},
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


# 模型合并存储

new_model_directory = "./merged_model_an"
merged_model = model.merge_and_unload()
# 将权重保存为safetensors格式的权重, 且每个权重文件最大不超过2GB(2048MB)
merged_model.save_pretrained(
    new_model_directory, max_shard_size="2048MB", safe_serialization=True
)

# !cp ./qwen/Qwen2-7B-Instruct/tokenizer.json ./merged_model_an/
