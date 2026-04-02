import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import faiss
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch

import sys
import io
import os

if sys.platform == "win32":
    os.system('chcp 65001 > nul')
    
# 强制标准输出使用UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 模拟数据集
qna_set=[
    {
        "question": "print() 函数的作用是什么?",
        "answer": "用于在屏幕或控制台输出指定的内容（如文本、数字等）。"
    },
    {
        "question": "什么是循环?",
        "answer": "循环是一种编程结构，用于重复执行某段代码，直到满足特定条件，例如 for 和 while 循环。"
    },
    {
        "question": "if-else语句的作用是什么?",
        "answer": "它用于条件判断，根据条件的真假执行不同的代码分支。"
    },
    {
        "question": "== 和 = 的区别是什么?",
        "answer": "== 是比较运算符，判断是否相等。= 是赋值运算符，给变量赋值。"
    },
    {
        "question": "break的作用是什么?",
        "answer": "break 直接退出循环。"
    },
    {
        "question": "if 语句的作用是什么?",
        "answer": "根据条件判断决定是否执行某段代码。"
    },
    {
        "question": "如何写注释?",
        "answer": "Python 用 #。java 和 C 用 //。HTML 用 <!-- -->。"
    },
    {
        "question": "什么是函数或方法?",
        "answer": "函数 (或方法) 是一段可重复调用的代码块, 用于执行特定任务, 提高代码复用性。"
    },
    {
        "question": "什么是操作系统?",
        "answer": "操作系统是管理计算机硬件和软件资源的系统软件, 为用户和应用程序提供基础服务, 例如Windows、Linux和macOS。"
    },
    {
        "question": "CPU的主要功能是什么?",
        "answer": "CPU (中央处理器) 是计算机的大脑, 负责执行程序指令、进行算术和逻辑运算。"
    }
]

# 初始化Sentence-BERT模型用于生成嵌入
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 将问题和答案分别取出，以list形式
questions = [pair["question"] for pair in qna_set]
answers = [pair["answer"] for pair in qna_set]

# 使用Sentence-BERT生成嵌入
question_embeddings = model.encode(questions)   # 将原始文本转换为语义向量（嵌入向量）

# 创建FAISS索引并添加向量（对问题）
dimension = question_embeddings.shape[1]     # 获取嵌入维度
index = faiss.IndexFlatL2(dimension)         # 创建L2距离索引 - 初始化索引
index.add(question_embeddings)               # 添加文档向量

# 对问题进行匹配
def ask_question(query):
    query_embedding = model.encode([query])

    distances, indices = index.search(query_embedding, 1)   # 找最相似的1个问题
    similarity = distances[0][0]                            # 获取第一个查询的最相似结果的距离
    best_match_index = indices[0][0]                        # 第一个查询的最相似结果的索引
    print("相似度距离: ", similarity)

    if similarity < 7:
        return answers[best_match_index]
    else:
        return "抱歉，我找不到相关问题的答案。"


print("用户: 如何注释? \n回答: ", ask_question("如何注释?"), "\n")  
print("用户: 循环语句怎么写? \n回答: ", ask_question("循环语句怎么写?"), "\n")     
print("用户: 简单解释下什么是操作系统? \n回答: ", ask_question("简单解释下什么是操作系统?"), "\n")
print("用户: ==的作用是什么? \n回答: ", ask_question("==的作用是什么?"), "\n") 