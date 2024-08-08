import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    Trainer,
    TrainingArguments,
)


def process_func(example):
    MAX_LENGTH = 1800  # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n你是一个逻辑推理专家，擅长解决逻辑推理问题。<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = (
        instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    )
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )  # 因为eos token咱们也是要关注的所以 补充为1
    labels = (
        [-100] * len(instruction["input_ids"])
        + response["input_ids"]
        + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


if __name__ == "__main__":
    df = pd.read_json("data/train.json")
    ds = Dataset.from_pandas(df)

    tokenizer = AutoTokenizer.from_pretrained(
        "./qwen/Qwen2-7B-Instruct", use_fast=False, trust_remote_code=True
    )

    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

    print(tokenizer.decode(tokenized_id[0]["input_ids"]))
    print(
        tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[1]["labels"])))
    )

    model = AutoModelForCausalLM.from_pretrained(
        "./qwen/Qwen2-7B-Instruct", device_map="auto", torch_dtype=torch.bfloat16
    )
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        inference_mode=False,  # 训练模式
        r=8,  # Lora 秩
        lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1,  # Dropout 比例
    )

    trainer_args = TrainingArguments(
        output_dir="./output/Qwen2_instruct_lora",
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=3,
        save_steps=0.1,  # 为了快速演示，这里设置10，建议你设置成100
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        # report_to="wandb",
        # run_name="llm_complex_reasoning",
    )

    model = get_peft_model(model, config)

    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    torch.backends.cuda.enable_mem_efficient_sdp(False)

    trainer.train()
