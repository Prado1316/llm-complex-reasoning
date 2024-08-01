import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from openai import OpenAI
from tqdm import tqdm

logger.remove()  # 移除默认的控制台输出
logger.add(
    "logs/app_{time:YYYY-MM-DD}.log",
    level="INFO",
    rotation="00:00",
    retention="10 days",
    compression="zip",
)

MODEL_NAME = "Qwen2-7B-Instruct-lora"


def call_openai_like_api(MODEL_NAME, query):
    # 这里采用dashscope的api调用模型推理，通过http传输的json封装返回结果

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="sk-xxx",  # 随便填写，只是为了通过接口参数校验
    )
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            # {'role':'system','content':'你是一个解决推理任务的专家，你需要分析出问题中的每个实体以及响应关系。然后根据问题一步步推理出结果。并且给出正确的结论。'},
            {"role": "user", "content": query}
        ],
    )
    return completion.choices[0].message.content


def api_retry(MODEL_NAME, query):
    max_retries = 5
    retry_delay = 60  # in seconds
    attempts = 0
    while attempts < max_retries:
        try:
            return call_openai_like_api(MODEL_NAME, query)
        except Exception as e:
            attempts += 1
            if attempts < max_retries:
                logger.warning(
                    f"Attempt {attempts} failed for text: {query}. Retrying in {retry_delay} seconds..."
                )
                time.sleep(retry_delay)
            else:
                logger.error(
                    f"All {max_retries} attempts failed for text: {query}. Error: {e}"
                )
                raise


# 这里定义了prompt推理模版


def get_prompt(problem, question, options):

    options = "\n".join(f"{'ABCDEFG'[i]}. {o}" for i, o in enumerate(options))

    prompt = f"""你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请逐步分析问题并在最后一行输出答案，最后一行的格式为"答案是：A"。题目如下：

### 题目:
{problem}

### 问题:
{question}
{options}
"""
    # print(prompt)
    return prompt


# 这里使用extract抽取模获得抽取的结果


def extract(input_text):
    # ans_pattern = re.compile(r"(.)", re.S)

    ans_pattern = re.compile(r"答案是：(.)", re.S)
    problems = ans_pattern.findall(input_text)
    # print(problems)
    if problems == []:
        return "A"
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


def send_a_group_req(prompt):
    res, res1, res2 = (
        api_retry(MODEL_NAME, prompt),
        api_retry(MODEL_NAME, prompt),
        api_retry(MODEL_NAME, prompt),
    )
    return res, res1, res2


def process_datas(datas):
    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_data = {}
        lens = 0
        for data in tqdm(datas, desc="Submitting tasks", total=len(datas)):
            problem = data["problem"]
            for id, question in enumerate(data["questions"]):
                prompt = get_prompt(
                    problem,
                    question["question"],
                    question["options"],
                )
                future = executor.submit(send_a_group_req, prompt)
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


def main(ifn, ofn):
    if os.path.exists(ofn):
        pass
    data = []
    # 按行读取数据
    with open(ifn) as reader:
        for line in reader:
            sample = json.loads(line)
            data.append(sample)
    datas = data
    # print(data)
    # 均匀地分成多个数据集
    return_list = process_datas(datas)
    print(len(return_list))
    print("All tasks finished!")
    return return_list


def evaluate(ofn):
    data = []
    with open(ofn) as reader:
        for line in reader:
            sample = json.loads(line)
            data.append(sample)

    pse = 0
    cnt = 0
    tot = 0
    for task in data:
        for question in task["questions"]:

            if MODEL_NAME in question:
                tot += 1
                cnt += question[MODEL_NAME] == question["answer"]
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


if __name__ == "__main__":
    return_list = main("data/round1_test_data.jsonl", "upload.jsonl")
    return_list = filter_problems(return_list)
    sorted_data = sorted(return_list, key=lambda x: int(str(x["id"])[-3:]))
    print(sorted_data)
    # 示例字典列表
    dict_list = sorted_data

    # 找出缺失的序号
    missing_ids = find_missing_ids(dict_list)
    print("缺失的序号:", missing_ids)

    data = []
    with open("data/round1_test_data.jsonl") as reader:
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

    evaluate(sorted_data)
