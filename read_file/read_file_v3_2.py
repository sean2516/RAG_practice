import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

#from flask import Flask, request, jsonify, json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch

import sys
import io
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib

# 修改标准输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


print("开始准备读取文件数据......\n")

file_name = "input.txt"


#清理文本
def clean_string(text):
    text = text.replace("\\", "")                # 移除所有反斜杠
    text = text.replace("#", " ")                # 把所有 # 替换为空格
    text = text.replace(". .", ".")              # 把 ". ." 替换为 "."
    text = re.sub(r'\s\s+', ' ', text)           # 多个连续空白字符替换为单个空格
    text = re.sub(r'(\r\n|\n|\r)', ' ', text)    # 所有换行符替换为空格
    return text.strip()    

# py版 text-loader.js
class TextLoader:
    def __init__(self, text, chunk_size=300, chunk_overlap=0):
        self.text = text
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.unique_id = f"TextLoader_{hashlib.md5(text.encode()).hexdigest()}"
    
    def get_chunks(self):
        chunker = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        
        #文本清理
        cleaned_text = clean_string(self.text)
        chunks = chunker.split_text(cleaned_text)
        
        result = []
        for chunk in chunks:
            result.append({
                "pageContent": chunk,
                "metadata": {
                    "type": "TextLoader",
                    "source": self.text[:50] + "..." if len(self.text) > 50 else self.text,
                    "textId": self.unique_id,
                }
            })
        return result


# 读取文件，根据板块分段
def read_file(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        text = f.read()
    
    # 使用 TextLoader 分块
    loader = TextLoader(text, chunk_size=300, chunk_overlap=0)
    chunks = loader.get_chunks()
    
    # 提取 pageContent
    section = [chunk["pageContent"] for chunk in chunks]
    return section

section = read_file(file_name)
#print("分块结果：")
#print(section) 


# 嵌入模型，生成嵌入向量
#model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model = SentenceTransformer('moka-ai/m3e-base')
embeddings = model.encode(section, convert_to_tensor=False)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  
embeddings = np.array(embeddings).astype('float32')
index.add(embeddings)


# 初始化生成模型
model_path = "D:\\model\\DeepSeek-R1-Distill-Qwen-1.5B"
generator_tokenizer = AutoTokenizer.from_pretrained(model_path)
generator_model = AutoModelForCausalLM.from_pretrained(model_path)



# 定义生成回答的函数
def generate_answer(question, relate_context):

    prompt = (
        "Please answer the question based on the reference materials.\n\n"
        f"## My question is:\n\n{question}\n\n"
        f"## Reference Materials:\n\n {relate_context} \n\n"
        "Please respond in the same language as the user's question.\n"
    )

    inputs = generator_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        return_attention_mask=True
    )
    
    with torch.no_grad():
        output = generator_model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=2000,
            temperature=0.5,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            num_return_sequences=1,
            do_sample=True,
            pad_token_id=generator_tokenizer.eos_token_id
        )
    answer = generator_tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

def search(query):
    #print("问题：", query)
   
    query_embedding = model.encode([query])
    # 返回最相似的3个结果
    distances, indices = index.search(query_embedding, 6)

    result = []
    print("最相似的部分索引:", indices)
    print("距离:", distances)

    for idx in indices[0]:
        result.append(section[idx])
    
    return generate_answer(query, result)
    #return result



# 调用生成函数生成回答
answer = search("什么是上海职工互助保障项目？")
print("生成的回答：", answer)


