# -*- encoding: utf-8 -*-
"""
@Time    :   2024-08-08 20:25:34
@desc    :   从cache数据里构建数据集
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""

import re
from ast import Dict
from collections import defaultdict
from pathlib import Path

import jsonlines


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
    return _filter_test_data(_cache_data)


def _extract_correct_data(data):
    """提取正确数据"""
    ds = []
    _ds_n = []
    ans_pattern = re.compile(r"答案是：(.)", re.S)
    for _id, metadata in data.items():
        for _q_idx, questionItem in enumerate(metadata["questions"]):
            正确选项 = questionItem["answer"]
            for _a_idx, answerItem in enumerate(questionItem[_model_id]["answers"]):
                splited_msg = answerItem["args"]["messages"][1]["content"].split(
                    "### 题目:\n"
                )
                system = splited_msg[0].replace("`答案是：A`\n\n", "答案是：")
                query = splited_msg[1] + "### 答案:\n"
                reply = answerItem["text"]
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": reply},
                ]
                choices = ans_pattern.findall(reply)[-1]
                ds_id = f"{_id}-{_q_idx}-{_a_idx}"
                if choices == 正确选项:
                    ds.append({"id": ds_id, "messages": messages})
                else:
                    print(f"正确选项: {正确选项}\n错误选项推理过程:{reply}")
                    _ds_n.append({"id": ds_id, "messages": messages})
    ...


if __name__ == "__main__":
    _model_id = "_Qwen2-72B-Instruct-test"
    cache_path = Path(".cache", "req_cache.jsonl")
    data = _extract_raw_data()
    data = _extract_correct_data(data)
