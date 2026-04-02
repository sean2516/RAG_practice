import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from flask import Flask, request, jsonify, json
import faiss
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch
import sys
import io

# 修改标准输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("开始准备读取文件数据......\n")
# 准备数据
# 读取txt文件
document = []
with open('input.txt', 'r', encoding='utf-8') as file:
    for line in file:
        document.append(line.strip()) 
#print(document)

# 转换为data frame - 这里指定列名'text'
df = pd.DataFrame(document, columns=['text'])

# 清洗数据：去除多余空白
df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True).str.strip().str.lower()

# 打印清洗后的数据
#print("清洗后的数据：\n", df)

# 加载预训练的嵌入模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def generate_embedding(text):
    embedding = model.encode(text, convert_to_tensor=False)
    return np.array(embedding)

# 生成嵌入向量
df["embedding"] = df["text"].apply(generate_embedding)

# print("向量形状: ", df["embedding"].iloc[0].shape)

# 查看嵌入向量
# print(df["embedding"])

embeddings = np.vstack(df["embedding"].values)
#print("嵌入矩阵形状:", embeddings.shape)
#np.save("embeddings.npy", embeddings)

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)

faiss.normalize_L2(embeddings)
index.add(embeddings)


# 初始化生成模型
model_path = "D:\\model\\DeepSeek-R1-Distill-Qwen-1.5B"
generator_tokenizer = AutoTokenizer.from_pretrained(model_path)
generator_model = AutoModelForCausalLM.from_pretrained(model_path)

# 添加padding token
if generator_tokenizer.pad_token is None:
    generator_tokenizer.pad_token = generator_tokenizer.eos_token


# 定义生成回答的函数
def generate_answer(question, relate_context):
    prompt = f"DeepSeek的用户协议问答系统: \n\n 问题: {question} \n\n 相关的用户协议信息: {relate_context}  \n\n 整理信息并优化后的回答: "
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
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, 3)  


    relate_info = []    #将所有的相关的回答添加进list

    for idx, dist in zip(indices[0], distances[0]):
        if dist > 0.6:
            result_text = df.iloc[idx]['text']
            relate_info.append(result_text)
    
    if not relate_info:
        return "抱歉, 我无法找到与之相关的用户协议信息。"
    else:
        merge_text = "\n".join(relate_info) if relate_info else ""  #list不是空的话，将内容合并成一个字符串，和问题一起输入到生成模型，生成回答
        return generate_answer(query, merge_text)


# 测试
test_context="Deepseek的产品服务有什么?"

# 调用生成函数生成回答
answer = search(test_context)
print("生成的回答：", answer)
