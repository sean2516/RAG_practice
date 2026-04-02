import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import faiss
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch

import sys
import io

# 修复标准输出的编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 初始化嵌入模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 初始化生成模型
generator_model_path = "D:\\model\\DeepSeek-R1-Distill-Qwen-1.5B"
generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_path)
generator_model = AutoModelForCausalLM.from_pretrained(generator_model_path)

# 添加padding token
if generator_tokenizer.pad_token is None:
    generator_tokenizer.pad_token = generator_tokenizer.eos_token

# 模拟数据集
qna_set=[
    {
        "question": "print() 函数的作用是什么?",
        "answer": "用于在屏幕或控制台输出指定的内容（如文本、数字等）。"
    },
    {
        "question": "循环是什么?",
        "answer": "循环是一种编程结构，用于重复执行某段代码，直到满足特定条件，例如 for 和 while 循环。"
    },
    {
        "question": "if-else语句的作用是什么?",
        "answer": "它用于条件判断，根据条件的真假执行不同的代码分支。"
    },
    {
        "question": "在Java中, == 和 = 的区别是什么?",
        "answer": "== 是比较运算符，判断是否相等。= 是赋值运算符，给变量赋值。"
    },
    {
        "question": "编程语言当中的break的作用是什么?",
        "answer": "break 直接退出循环。"
    },
    {
        "question": "if 语句的作用是什么?",
        "answer": "根据条件判断决定是否执行某段代码。"
    },
    {
        "question": "如何在Python, java和C语言中写注释?",
        "answer": "Python 用 #。java 和 C 用 //。"
    },
    {
        "question": "编程语言当中的函数或方法是什么?",
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

# 将问题和答案分别取出，以list形式
questions = [pair["question"] for pair in qna_set]
answers = [pair["answer"] for pair in qna_set]

# 生成嵌入, 创建FAISS索引并添加向量
question_embeddings = model.encode(questions)
dimension = question_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(question_embeddings)
index.add(question_embeddings)

# 用gpt2根据原本的问题和答案，生成回答
def generate_answers(que, org_answer):
    prompt = f"原来的问题: {que}\n原来的回答: {org_answer}\n优化后的新回答(用中文):"
    input_ids = generator_tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate enhanced answer
    with torch.no_grad(): 
        output = generator_model.generate(
            input_ids,
            max_length=100,         
            temperature=0.5,            #多样性
            top_k=50,                   #限制每一步只从概率最高的K个token中选择
            top_p=0.5,                  #动态选择概率累加超过p的最小token集合
            repetition_penalty=1.1,
            num_return_sequences=1,     #只返回1个生成结果
            pad_token_id=generator_tokenizer.eos_token_id,
        )
    
    new_answer = generator_tokenizer.decode(output[0], skip_special_tokens=True)
    #new_answer = new_answer.replace(prompt, "").strip()

    return new_answer


# 对问题进行匹配
def ask_question(query):
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, 1)  
    #print("距离：", distances[0][0])
    #print("回答：", answers[indices[0][0]])

    if distances[0][0] > 0.65:
        answer = answers[indices[0][0]]
        #return answer
        return generate_answers(query, answer)
    else:
        return "抱歉, 我无法找到该问题的答案。"
    

# 测试
print("用户: 如何在Python中写注释?")
print(ask_question("如何在Python中写注释?"), "\n")

print("用户：如何编写循环语句?")
print(ask_question("如何编写循环语句?"), "\n")

print("用户: 操作系统是什么?")
print(ask_question("操作系统是什么?"), "\n")

print("用户: == 在java当中是作什么的?")
print(ask_question("== 在java当中是作什么的?"), "\n")

del model
del generator_model
