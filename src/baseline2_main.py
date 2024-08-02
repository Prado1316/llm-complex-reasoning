import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Union

import jsonlines
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from tqdm import tqdm

from data.prompts.single_infer import single_prompt

load_dotenv()

logger.remove()  # 移除默认的控制台输出
logger.add(
    "logs/app_{time:YYYY-MM-DD}.log",
    level="INFO",
    rotation="00:00",
    retention="10 days",
    compression="zip",
)


def call_openai_like_api(messages: List[Dict], **kwargs):
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
    )
    completion = client.chat.completions.create(
        model=args.model, messages=messages, **kwargs
    )
    return completion.choices[0].message.content


def api_retry(**gen_kwargs):
    max_retries = 5
    retry_delay = 60
    attempts = 0
    while attempts < max_retries:
        try:
            return call_openai_like_api(**gen_kwargs)
        except Exception as e:
            attempts += 1
            if attempts < max_retries:
                logger.warning(
                    f"Attempt {attempts} failed for args: {gen_kwargs}. Retrying in {retry_delay} seconds..."
                )
                time.sleep(retry_delay)
            else:
                logger.error(
                    f"All {max_retries} attempts failed for args: {gen_kwargs}. Error: {e}"
                )
                raise


def compose_messages(
    system: str = "You are a helpful assistant.", prompt: str = ""
) -> List[Dict]:
    return [
        {
            "role": "system",
            "content": system,
        },
        {"role": "user", "content": prompt},
    ]


def prepare_gen_kwargs(problem, question, options) -> Dict:
    options = "\n".join(f"{'ABCDEFG'[i]}. {o}" for i, o in enumerate(options))
    prompt = single_prompt.format(problem=problem, question=question, options=options)
    messages = compose_messages(prompt=prompt)
    return {
        "model": args.model,
        "messages": messages,
        "repetition_penalty": 1.05,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
    }


def extract(input_text):
    # ans_pattern = re.compile(r"(.)", re.S)

    ans_pattern = re.compile(r"答案是：(.)", re.S)
    problems = ans_pattern.findall(input_text)
    if not problems:
        return random.choice(["A", "B", "c", "D"])
    return problems[0]


def most_frequent_char(ext_rep):
    # 创建一个字典来存储每个字符的出现次数
    frequency = {char: 0 for char in ext_rep}

    # 增加每个字符的出现次数
    for char in ext_rep:
        frequency[char] += 1

    # 找到出现次数最多的字符
    most_frequent = max(frequency, key=frequency.get)

    return most_frequent


def send_a_group_req(gen_kwargs):
    res, res1, res2 = (
        api_retry(**gen_kwargs),
        api_retry(**gen_kwargs),
        api_retry(**gen_kwargs),
    )
    return res, res1, res2


def process_datas(datas):
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_data = {}
        lens = 0
        for data in tqdm(datas, desc="Submitting tasks", total=len(datas)):
            problem = data["problem"]
            for id, question in enumerate(data["questions"]):
                gen_kwargs = prepare_gen_kwargs(
                    problem,
                    question["question"],
                    question["options"],
                )
                future = executor.submit(send_a_group_req, gen_kwargs)
                future_data[future] = (data, id)
                lens += 1
        for future in tqdm(
            as_completed(future_data), total=lens, desc="Processing tasks"
        ):
            data, problem_id = future_data[future][0], future_data[future][1]
            try:
                res, res1, res2 = future.result()
                ext_rep = extract(res), extract(res1), extract(res2)
                ans = most_frequent_char(ext_rep)
                data["questions"][problem_id]["answer"] = ans
                results.append(data)
            except Exception as e:
                logger.error(f"Failed to process text: {data}. Error: {e}")

    return results


def get_api_info():
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
    )
    models = client.models.list()
    if args.model not in models:
        logger.error(f"{args.model} is not in {models}")
        exit(1)
    else:
        logger.info(f"Current api support model: {models}")


def main():
    if os.path.exists(args.output_path):
        return [json.loads(i) for i in open(args.output_path, "r")]
    datas = [json.loads(i) for i in jsonlines.open(args.data_path, "r")]
    return_list = process_datas(datas)
    print(f"All tasks: {len(return_list)} finished!")
    return return_list


def evaluate(data):
    pse = 0
    cnt = 0
    tot = 0
    for task in data:
        for question in task["questions"]:
            if args.model in question:
                tot += 1
                cnt += question[args.model] == question["answer"]
            else:
                pse += 1

    print(cnt, tot, cnt / tot, pse)


def has_complete_answer(questions):
    # 这里假设完整答案的判断逻辑是：每个question都有一个'answer'键
    for question in questions:
        if "answer" not in question:
            return False
    return True


def filter_problems(data):
    result = []
    problem_set = set()

    for item in data:
        # print('处理的item' ,item)
        problem = item["problem"]
        if problem in problem_set:
            # 找到已存在的字典
            for existing_item in result:
                if existing_item["problem"] == problem:
                    # 如果当前字典有完整答案，替换已存在的字典
                    if has_complete_answer(item["questions"]):
                        existing_item["questions"] = item["questions"]
                        existing_item["id"] = item["id"]
                    break
        else:
            # 如果当前字典有完整答案，添加到结果列表
            if has_complete_answer(item["questions"]):
                result.append(item)
                problem_set.add(problem)

    return result


def find_missing_ids(dict_list):
    # 提取所有序号
    extracted_ids = {int(d["id"][-3:]) for d in dict_list}

    # 创建0-500的序号集合
    all_ids = set(range(500))

    # 找出缺失的序号
    missing_ids = all_ids - extracted_ids

    return sorted(missing_ids)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Run task type, train: predict+evaluate, test: predict only",
    )
    parser.add_argument("--model", type=str, required=True, default="Qwen2-7b-Instruct")
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("OPENAI_BASE_URL"),
        required=True,
        help="Openai like api base",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY", "EMPTY"),
        help="Openai like api key",
    )
    parser.add_argument(
        "--data-path-to-predict",
        type=Union[str, Path],
        default=Path(__file__).parents[1].joinpath("data", "round1_train_data.json"),
        help="Data to predict file path.",
    )
    parser.add_argument(
        "--output-path",
        type=Union[str, Path],
        default=Path(__file__).parents[1].joinpath("output", "upload.json"),
        help="File to upload path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    return_list = main()
    return_list = filter_problems(return_list)
    sorted_data = sorted(return_list, key=lambda x: int(str(x["id"])[-3:]))

    dict_list = sorted_data

    missing_ids = find_missing_ids(dict_list)
    print("缺失的序号:", missing_ids)

    with open(args.data_path) as reader:
        for id, line in enumerate(reader):
            if id in missing_ids:
                sample = json.loads(line)
                for question in sample["questions"]:
                    question["answer"] = "A"
                sorted_data.append(sample)
    sorted_data = sorted(sorted_data, key=lambda x: int(str(x["id"])[-3:]))

    with open("upload.jsonl", "w") as writer:
        for sample in sorted_data:
            writer.write(json.dumps(sample, ensure_ascii=False))
            writer.write("\n")

    if args.task == "train":
        evaluate(sorted_data)
