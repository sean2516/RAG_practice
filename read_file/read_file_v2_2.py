import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from flask import Flask, request, jsonify, json
import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import nltk
import markdown
import re

import sys
import io

# 修改标准输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


print("开始准备读取文件数据......\n")

file_name = "input.txt"

with open(file_name, "r", encoding="utf-8") as f:
    text = f.read()

#标题与内容分类
def to_document(text):
    lines = text.split('\n')
    document = []
    section = {"title": "", "content": []}
    
    for line in lines:
        # 检测标题 - 正则表达式
        if re.match(r'^#{3}\s+.+', line.strip()): 
            if section["content"]:  
                document.append(section)

            tittle_text = re.sub(r'^#+\s*', '', line).strip()
            section = {"title": f"{tittle_text}", "content": []}
        else:
            if line.strip():
                section["content"].append(line.strip())
    
    if section["content"]:
        document.append(section)
    
    return document


# 转换为data frame
df = pd.DataFrame(to_document(text))

df["content"] = df["content"].apply("\n".join)
df["content"] = (df["content"]
                .str.replace(r'[\*\-]{1,}', ' ', regex=True)
                .str.replace(r'\s+', ' ', regex=True)        
                .str.strip()                                 
                .str.lower()
                )    

#print(df,"\n")
#print(df["title"], "\n")
#print(df["content"], "\n")

# 加载嵌入模型
model_name = "nghuyong/ernie-3.0-base-zh"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# 向量化
def generate_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    embeddings = model(**inputs).last_hidden_state[:, 0, :].detach().numpy()
    return embeddings.flatten()

def input_embedding(input):
    input_2 = tokenizer(input, return_tensors="pt", truncation=True)
    input_embeddings = model(**input_2).last_hidden_state.mean(dim=1).detach().numpy()
    return input_embeddings


"""
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)  # 转换为 (1, dimension)
"""
    

# 生成嵌入向量
df["embedding"] = df["title"].apply(generate_embedding)
embedding_dim = df["embedding"].iloc[0].shape[0]
index = faiss.IndexFlatL2(embedding_dim)
embeddings = np.array(df["embedding"].tolist()).astype("float32")
#print(embeddings, "\n")
index.add(embeddings)


# 初始化生成模型
model_path = "D:\\model\\DeepSeek-R1-Distill-Qwen-1.5B"
generator_tokenizer = AutoTokenizer.from_pretrained(model_path)
generator_model = AutoModelForCausalLM.from_pretrained(model_path)



# 定义生成回答的函数
def generate_answer(question, relate_context):
    prompt = f"职工互助保障项目基本保障条款的问答: \n\n 问题: {question} \n\n 相关的条款信息: {relate_context}  \n\n 整理信息并优化后的回答: "
    inputs = generator_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    
    # 使用生成模型生成回答
    with torch.no_grad():
        output = generator_model.generate(
            inputs["input_ids"],
            max_length=500,         
            temperature=0.5,            #多样性
            top_k=50,                   #限制每一步只从概率最高的K个token中选择
            top_p=0.5,                  #动态选择概率累加超过p的最小token集合
            repetition_penalty=1.1,
            num_return_sequences=1,     #返回1个生成结果
            pad_token_id=generator_tokenizer.eos_token_id,
            do_sample=True 
        )
    
    answer = generator_tokenizer.decode(output[0], skip_special_tokens=True)
    return answer



def search(query):
    print("问题：", query)
    query_embedding = generate_embedding([query]).reshape(1,-1).astype("float32")
    #faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, 15)
    
    result = []
    for idx, dist in zip(indices[0], distances[0]):
        print("相似度:", dist)
        print(f"匹配内容: {df.iloc[idx]['title']}\n")
        if dist > 0.6:
            result.append(df.iloc[idx]['title'])
    
    if result:
        return result
    else:
        "抱歉, 我无法找到与之相关的协议信息。"



# 调用生成函数生成回答
answer = search("参保的费用是多少?")
print("生成的回答：", answer)


