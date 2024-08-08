# -*- encoding: utf-8 -*-
"""
@Time    :   2024-08-08 22:50:07
@desc    :   lora训练简版
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse

import torch
from datasets import load_dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    HfArgumentParser,
    SchedulerType,
    Trainer,
    TrainingArguments,
)


@dataclass
class Arguments:
    dataset: str = field(
        default="ticoAg/llm-complex-reasoning-train-qwen2-72b-instruct-correct",
        metadata={"help": "Huggingface dataset repo"},
    )
    model_name_or_path: str = field(
        default="Qwen/Qwen2-7B-Instruct",
        metadata={"help": "Huggingface model repo"},
    )
    output_dir: str = field(
        default="./output/Qwen2_instruct_lora",
        metadata={"help": "Output lora weight & config dir path"},
    )
    report_to: Union[None, str, List[str]] = field(
        default=None,
        metadata={"help": "wandb or tensorboard"},
    )
    run_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional descriptor for the run. Notably used for wandb, mlflow and comet logging."
        },
    )
    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )
    num_train_epochs: float = field(
        default=3.0, metadata={"help": "Total number of training epochs to perform."}
    )
    learning_rate: float = field(
        default=1e-4, metadata={"help": "The initial learning rate for AdamW."}
    )
    save_steps: float = field(
        default=500,
        metadata={
            "help": (
                "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    lr_scheduler_type: Union[SchedulerType, str] = field(
        default="linear",
        metadata={"help": "The scheduler type to use."},
    )
    bf16: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change."
            )
        },
    )
    num_proc: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of processes for multiprocessing. By default it doesn't use multiprocessing."
        },
    )
    lora_rank: int = field(default=8, metadata={"help": "Lora attention dimension"})
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."},
    )
    max_model_length: int = field(
        default=4096,
        metadata={"help": "The maximum model length."},
    )


def process_func(example):

    input_ids, attention_mask, labels = [], [], []
    messages = example["messages"]
    system, user_input, reply = (
        messages[0]["content"],
        messages[1]["content"],
        messages[2]["content"],
    )
    instruction = tokenizer(
        f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{reply}", add_special_tokens=False)
    input_ids = (
        instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    )
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = (
        [-100] * len(instruction["input_ids"])
        + response["input_ids"]
        + [tokenizer.pad_token_id]
    )
    if len(input_ids) > args.max_model_length:  # 做一个截断
        input_ids = input_ids[: args.max_model_length]
        attention_mask = attention_mask[: args.max_model_length]
        labels = labels[: args.max_model_length]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _get_lora_config() -> LoraConfig:
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
        r=args.lora_rank,  # Lora 秩
        lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.2,  # Dropout 比例
    )
    return config


def _get_trainer_args() -> TrainingArguments:
    trainer_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=10,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to=args.report_to,
        run_name=args.run_name,
        bf16=args.bf16,
        optim="adamw_hf",
        lr_scheduler_type=args.lr_scheduler_type,
    )
    return trainer_args


def _parse_args() -> argparse.ArgumentParser:
    (args,) = HfArgumentParser(Arguments).parse_args_into_dataclasses()
    logger.info(args)
    return args


def _setup_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=True, trust_remote_code=True
    )
    return tokenizer


def _setup_model_for_train() -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
    return model


def _setup_for_all():
    global args
    global tokenizer
    global model
    global tokenized_dataset
    args = _parse_args()
    ds = load_dataset(args.dataset)
    cached_files_path = ".cache/train_tokenized/train.arrow"
    if not Path(cached_files_path).parent.exists():
        Path(cached_files_path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer = _setup_tokenizer()
    tokenized_dataset = ds.map(
        process_func,
        batched=False,
        num_proc=args.num_proc,
        load_from_cache_file=True,
        cache_file_names={"train": cached_files_path},
    )["train"]
    print(tokenizer.decode(tokenized_dataset[0]["input_ids"]))
    print(
        tokenizer.decode(
            list(filter(lambda x: x != -100, tokenized_dataset[1]["labels"]))
        )
    )
    model = _setup_model_for_train()


if __name__ == "__main__":
    _setup_for_all()

    lora_config = _get_lora_config()
    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    trainer_args = _get_trainer_args()
    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    torch.backends.cuda.enable_mem_efficient_sdp(False)

    trainer.train()
