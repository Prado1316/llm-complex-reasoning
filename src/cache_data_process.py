# -*- encoding: utf-8 -*-
"""
@Time    :   2024-08-08 20:25:34
@desc    :   从cache数据里构建数据集
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""

import datetime
import re
from ast import Dict
from collections import defaultdict
from pathlib import Path

import jsonlines
from loguru import logger


def _filter_test_data(data: Dict) -> Dict:
    """取出key不包含`test`字符串的数据

    - Args:
        data: Dict: {"round1_train_data_022": {"problem": "题目描述", "questions": [], "id": "round1_train_data_022"}}
    """
    return {k: v for k, v in data.items() if "test" not in k}


def _extract_raw_data():

    _cache_data = defaultdict()
    for item in list(jsonlines.open(cache_path, "r")):
        if not _cache_data.get(item["id"]):
            _cache_data[item["id"]] = item
        else:
            for _id, question in enumerate(item["questions"]):
                if question.get(_model_id):
                    _cache_data[item["id"]]["questions"][_id] = question
    submit_data = [
        {
            "id": item["id"],
            "questions": [
                {"answer": _item.get("answer", "A")} for _item in item["questions"]
            ],
        }
        for item in _cache_data.values()
    ]

    submit_filename = "submit_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    submit_file_path = Path("submit", f"{submit_filename}.jsonl")
    if not submit_file_path.exists():
        submit_file_path.parent.mkdir(parents=True, exist_ok=True)
    jsonlines.open(submit_file_path, "w").write_all(submit_data)
    exit()
    return _filter_test_data(_cache_data)


def _classify_ds(data):
    """提取正确和错误的数据"""
    _ds = []
    _ds_n = []
    ans_pattern = re.compile(r"答案是: (.)", re.S)
    for _id, metadata in data.items():
        for _q_idx, questionItem in enumerate(metadata["questions"]):
            正确选项 = questionItem["answer"]
            for _a_idx, answerItem in enumerate(questionItem[_model_id]["answers"]):
                splited_msg = answerItem["args"]["messages"][1]["content"].split(
                    "### 题目:\n"
                )
                system = splited_msg[0].replace(
                    "`答案是：A`\n\n", "`答案是: (你认为的正确答案)`"
                )
                query = splited_msg[1]
                reply = (
                    answerItem["text"]
                    .replace("答案是 ", "答案是: ")
                    .replace("答案是：", "答案是: ")
                )
                choices = ans_pattern.findall(reply)[-1]
                reply_splited = reply.split("\n")[:-1] + [f"答案是: {choices}"]
                reply = "\n".join(reply_splited)
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": reply},
                ]
                ds_id = f"{_id}-{_q_idx}-{_a_idx}"
                if choices == 正确选项:
                    _ds.append({"id": ds_id, "messages": messages})
                else:
                    logger.info(f"正确选项: {正确选项}\n错误选项推理过程:{reply[:300]}")
                    _ds_n.append({"id": ds_id, "messages": messages})
    logger.success(f"正确选项数量: {len(_ds)}, 错误选项数量: {len(_ds_n)}")
    return _ds, _ds_n


if __name__ == "__main__":
    # _model_id = "_Qwen2-72B-Instruct-test"
    _model_id = "_Qwen2-7B-Instruct-lora"
    # cache_path = Path(".cache", "req_cache.jsonl")
    cache_path = Path(".cache", "req_cache_Qwen2-7B-Instruct-lora_test.jsonl")
    data = _extract_raw_data()
    _ds, _ds_n = _classify_ds(data)
    jsonlines.open(Path(".cache", "train.jsonl"), "w").write_all(_ds)
    jsonlines.open(Path(".cache", "no_train.jsonl"), "w").write_all(_ds_n)
