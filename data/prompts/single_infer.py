# -*- encoding: utf-8 -*-
"""
@Time    :   2024-08-02 23:58:43
@desc    :   单次纯推理prompt
@Author  :   ticoAg
@Contact :   1627635056@qq.com
"""


single_infer_task3_baselline2_ver1 = """你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请逐步分析问题并在最后一行输出答案，最后一行的格式为: `\n答案是：A`。题目如下：

### 题目:
{problem}

### 问题:
{question}
{options}
"""

single_prompt = single_infer_task3_baselline2_ver1
