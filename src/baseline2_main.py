# -*- encoding: utf-8 -*-
"""
@Time    :   2024-08-03 23:20:13
@desc    :   执行api调用模型测试流程
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import json5
import jsonlines
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from requests import Session
from tqdm import tqdm

sys.path.append(Path(__file__).parents[1].as_posix())
from data.prompts.single_infer import single_prompt, system_prompt

load_dotenv()

logger.add(
    "logs/app_{time:YYYY-MM-DD}.log",
    level="INFO",
    rotation="00:00",
    retention="10 days",
    compression="zip",
)


def call_openai_like_api(**kwargs):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + args.api_key,
    }
    try:
        # client = OpenAI()
        # resp = client.chat.completions.create(**kwargs)
        # return resp.choices[0].message.content
        with Session() as session:
            url = args.base_url + "/chat/completions"
            resp = session.post(url, headers=headers, json=kwargs).text
            resp = json5.loads(resp)
        return resp["choices"][0]["message"]["content"]
    except Exception as err:
        logger.error(err)


def api_retry(**gen_kwargs):
    max_retries = 5
    retry_delay = 5
    attempts = 0
    while attempts < max_retries:
        try:
            # 避免同时密集并发造成的网关压力
            # time.sleep(random.uniform(0, retry_delay))
            return call_openai_like_api(**gen_kwargs)
        except Exception as e:
            attempts += 1
            if attempts < max_retries:
                logger.error(f"Error: {repr(e)}")
                logger.warning(
                    f"Attempt {attempts} failed for args: {gen_kwargs}. Retrying in {retry_delay} seconds..."
                )
                time.sleep(retry_delay)
            else:
                logger.error(
                    f"All {max_retries} attempts failed for args: {gen_kwargs}. Error: {e}"
                )
                raise


def compose_messages(system: str = system_prompt, prompt: str = "") -> List[Dict]:
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
        # "repetition_penalty": 1.05,
        "temperature": 0.7,
        "top_p": 0.8,
        # "top_k": 20,
    }


def most_frequent_char(choices: List[str]) -> str:
    frequency = {char: 0 for char in choices}
    for char in choices:
        frequency[char] += 1
    most_frequent = max(frequency, key=frequency.get)
    return most_frequent


def _extract_votes_answer(input_texts: List[str]) -> str:
    ans_pattern = re.compile(r"答案是: (.)", re.S)
    choices = []
    true_choices = []
    for text in input_texts:
        problems = ans_pattern.findall(text)
        if not problems or problems[0] not in ["A", "B", "C", "D", "E"]:
            choices.append(random.choice(["A", "B", "C", "D"]))
            logger.warning(f"No answer can be extracted: {text}")
            true_choices.append("None")
        else:
            choices.append(problems[0])
            true_choices.append(problems[0])
    ans = most_frequent_char(choices)
    return ans, true_choices


def send_a_group_req(gen_kwargs, members: int = 5) -> List[str]:
    resps = [api_retry(**gen_kwargs) for _ in range(members)]
    return resps


def process_datas(datas):
    results = list(jsonlines.open(args.cache_path, "r"))
    _model_id = "_" + args.model
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        _cache_data = defaultdict()
        for item in list(jsonlines.open(args.cache_path, "r")):
            if not _cache_data.get(item["id"]):
                _cache_data[item["id"]] = item
            else:
                for _id, question in enumerate(item["questions"]):
                    if question.get(_model_id):
                        _cache_data[item["id"]]["questions"][_id] = question
        future_data = {}
        lens = 0
        for data in tqdm(datas, desc="Submitting tasks", total=len(datas)):
            problem = data["problem"]
            for id, question in enumerate(data["questions"]):
                # 对于已有数据 跳过
                if _cache_data.get(data["id"]) and _cache_data[data["id"]].get(
                    "questions"
                )[id].get(_model_id):
                    continue
                gen_kwargs = prepare_gen_kwargs(
                    problem,
                    question["question"],
                    question["options"],
                )
                future = executor.submit(send_a_group_req, gen_kwargs)
                future_data[future] = (data, id, gen_kwargs)
                lens += 1
        for future in tqdm(
            as_completed(future_data), total=lens, desc="Processing tasks"
        ):
            data, problem_id, gen_kwargs = (
                future_data[future][0],
                future_data[future][1],
                future_data[future][2],
            )
            data["questions"][problem_id][_model_id] = {}
            try:
                resps = future.result()
                votes_answer, true_choices = _extract_votes_answer(resps)
                if args.task == "train":
                    data["questions"][problem_id][args.model] = votes_answer
                else:
                    data["questions"][problem_id]["answer"] = votes_answer

                res_items = [
                    {
                        "text": resps[i],
                        "correct": true_choices[i]
                        == data["questions"][problem_id].get("answer"),
                        "args": gen_kwargs,
                    }
                    for i in range(len(resps))
                ]
                data["questions"][problem_id][_model_id]["ans"] = votes_answer
                data["questions"][problem_id][_model_id]["answers"] = res_items

                with jsonlines.open(args.cache_path, "a") as f:
                    f.write(data)
                results.append(data)
            except Exception as e:
                logger.error(f"Failed to process text: {data}. Error: {e}")
    for data in results:
        for question in data["questions"]:
            if question.get(_model_id):
                del question[_model_id]
    return results


def get_api_info():
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
    )
    models = [model_card.id for model_card in client.models.list().data]
    if args.model not in models:
        logger.error(f"{args.model} is not in {models}")
        exit(1)
    else:
        logger.info(f"Current api support model: {models}")


def main():
    logger.info(f"Run task {args.task}")
    if os.path.exists(args.output_path):
        return [json.loads(i) for i in open(args.output_path, "r")]
    datas = [i for i in jsonlines.open(args.data_path, "r")]
    if args.samples > 0:
        datas = datas[: args.samples]
    get_api_info()
    return_list = process_datas(datas)
    print(f"All tasks: {len(return_list)} finished!")
    return return_list


def _evaluate(data):
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

    logger.info(
        f"统计结果:\n答对 {cnt}\n总答题 {tot}\n答题正确率 {cnt / tot}\n未答题 {pse}"
    )


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
        help="Openai like api base",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY", "EMPTY"),
        help="Openai like api key",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=Path(__file__).parents[1].joinpath("data", "round1_train_data.jsonl"),
        help="Data to predict file path.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=Path(__file__).parents[1].joinpath("output", "upload.jsonl"),
        help="File to upload path.",
    )
    parser.add_argument("--threads", type=int, default=16, help="Threads to use.")
    parser.add_argument("--samples", type=int, default=-1, help="Samples to use.")
    args = parser.parse_args()
    os.environ["OPENAI_BASE_URL"] = args.base_url
    os.environ["OPENAI_API_KEY"] = args.api_key
    if not isinstance(args.output_path, Path):
        args.output_path = Path(args.output_path)
    if not args.output_path.parent.exists():
        args.output_path.parent.mkdir(parents=True)
    curr_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    model_str_path = args.model.replace("/", "_")
    args.output_path = args.output_path.parent.joinpath(
        f"{args.task}_{args.output_path.stem}_{model_str_path}_{curr_time}{args.output_path.suffix}"
    )
    logger.info(f"Output path redirected to: {args.output_path.as_posix()}")
    if not isinstance(args.data_path, Path):
        args.data_path = Path(args.data_path)
    args.cache_path = (
        Path(__file__)
        .parents[1]
        .joinpath(".cache", f"req_cache_{args.model}_{args.task}.jsonl")
    )
    if not args.cache_path.parent.exists():
        args.cache_path.parent.mkdir(parents=True)
    if not args.cache_path.exists():
        args.cache_path.touch()
    return args


if __name__ == "__main__":
    args = parse_args()
    return_list = main()
    return_list = filter_problems(return_list)
    sorted_data = sorted(return_list, key=lambda x: int(str(x["id"])[-3:]))

    dict_list = sorted_data

    missing_ids = find_missing_ids(dict_list)
    print("缺失的序号:", missing_ids)

    with open(args.data_path, encoding="utf-8") as reader:
        for id, line in enumerate(reader):
            if id in missing_ids:
                sample = json.loads(line)
                for question in sample["questions"]:
                    question["answer"] = "A"
                sorted_data.append(sample)
    sorted_data = sorted(sorted_data, key=lambda x: int(str(x["id"])[-3:]))

    jsonlines.open(args.output_path, "w").write_all(sorted_data)

    if args.task == "train":
        _evaluate(sorted_data)
