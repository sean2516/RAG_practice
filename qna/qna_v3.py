import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import faiss
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch

import sys
import io
import os
import ast

#if sys.platform == "win32":
#    os.system('chcp 65001 > nul')
    
# 强制标准输出使用UTF-8
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# 模拟数据集
'''
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
'''

qna_set = [
    {
        "question": "What is the purpose of the print() function?", 
        "answer": "Used to output the specified content (such as text, numbers, etc.) on the screen or console."
    }, 
    { 
        "question": "What is a loop?", 
        "answer": "A loop is a programming construct used to repeatedly execute a piece of code until a specific condition is met, such as for and while loops."
    }, 
    { 
        "question": "What is the purpose of an if-else statement?", 
        "answer": "It is used for conditional judgment, where different branches of code are executed depending on whether the condition is true or false."
    }, 
    { 
        "question": "What is the difference between == and = in Java?", 
        "answer": "== is a comparison operator that determines if values are equal. = is the assignment operator, assigning a value to a variable."
    }, 
    { 
        "question": "In a loop, what is the purpose of break?", 
        "answer": "break exits the loop directly."
    }, 
    { 
        "question": "What is the purpose of an if statement?", 
        "answer": "Determines whether to execute a piece of code based on a conditional judgment."
    }, 
    { 
        "question": "How to write comments in Python, Java and C?", 
        "answer": "Python uses #. Java and C use //."
    }, 
    { 
        "question": "What is a function or method in a programming language?", 
        "answer": "A function (or method) is a block of code that can be called repeatedly to perform a specific task, increasing code reusability."
    }, 
    { 
        "question": "What is an operating system?", 
        "answer": "An operating system is system software that manages computer hardware and software resources, providing basic services to users and applications."
    }, 
    { 
        "question": "What is the main function of a CPU?", 
        "answer": "The CPU (Central Processing Unit) executes program instructions and performs arithmetic and logical operations."
    } 
]


# 初始化Sentence-BERT模型用于生成嵌入
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 生成模型
generator_model_name="gpt2"
generator_tokenizer=AutoTokenizer.from_pretrained(generator_model_name)
generator_model=AutoModelForCausalLM.from_pretrained(generator_model_name)



'''
df = pd.read_csv('qa_with_embeddings.csv', encoding='utf-8')

df['embedding'] = df['embedding'].apply(ast.literal_eval)  # 解析字符串列表
embeddings = np.vstack(df['embedding'].values)  # 转为二维数组

# 重建FAISS索引
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype('float32'))

# 获取问答对
qna_set = df[['question', 'answer']].to_dict('records')
'''


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
    
    # print("相似度距离: ", similarity)

    if similarity < 7:
        answer =  answers[best_match_index]
    else:
        answer = "Sorry, I cannot find the answer to the question."
    

    prompt=f"Question: {answer} Answer: "
    input_ids = generator_tokenizer.encode(prompt, return_tensors="pt")
    
    
    with torch.no_grad():
        output = generator_model.generate(
            input_ids,
            max_length = 100,
            temperature = 0.7,
            top_k=50
        )

    report = generator_tokenizer.decode(output[0], skip_special_tokens=True)
    
    return report


#print("用户: 如何注释? \n回答: ", ask_question("如何注释?"), "\n")  
#print("用户: 循环语句怎么写? \n回答: ", ask_question("循环语句怎么写?"), "\n")     
#print("用户: 简单解释下什么是操作系统? \n回答: ", ask_question("简单解释下什么是操作系统?"), "\n")
#print("用户: ==的作用是什么? \n回答: ", ask_question("==的作用是什么?"), "\n") 

print("User: How to comment? \n answer: ", ask_question("How to comment?") , "\n") 
print("User: How to write a loop statement? \n answer: ", ask_question("How to write a loop statement?") , "\n") 
print("User: What is operating system is? \nAnswer: ", ask_question("What is an operating system?") , "\n") 
print("User: What does == do? \n Answer: ", ask_question("What does == do?") , "\n") 