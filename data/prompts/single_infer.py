# -*- encoding: utf-8 -*-
"""
@Time    :   2024-08-02 23:58:43
@desc    :   单次纯推理prompt
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""


single_infer_task3_baseline2_ver1 = """你是一个逻辑推理专家，擅长帮人解决逻辑推理问题
# 解题要求
1. 题目类型为逻辑推理，形式为单项选择题。
2. 所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。
3. 请逐步分析问题并在最后一行输出答案
4. 最后一行的格式为: `\n答案是：A`

### 题目:
{problem}

### 问题:
{question}
{options}"""

# 强调事实完全来自于题目,但似乎作用没有很大
single_infer_ver2 = """# 角色定义
你是一个逻辑推理专家，擅长解决逻辑推理问题。
# 任务要求
1. 以下是一个逻辑推理的题目，形式为单项选择题
2. 所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假
3. 所有的定义，认知都来自于题目, 除此之外你不知道任何信息
4. 禁止基于题目外任何信息对推理过程/结果做任何修改
5. 请你根据提供的信息，逐步分析问题并在最后一行输出答案
6. 最后一行的格式为`答案是：A`

### 题目:
{problem}

### 问题:
{question}
{options}"""

single_prompt = single_infer_ver2
