import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

#from flask import Flask, request, jsonify, json
import requests
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
import time
import json

from flask import Flask, request, jsonify


app = Flask(__name__)


# 修改标准输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


#print("开始准备读取文件数据......\n")

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
model = SentenceTransformer('BAAI/bge-m3')
embeddings = model.encode(section, convert_to_tensor=False)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  
embeddings = np.array(embeddings).astype('float32')
index.add(embeddings)


# 调用ollama的生成模型，进行问答
def generate_answer(question, relate_context):

    contexts = "\n\n".join(relate_context)
    
    prompt = (
        "Please answer the question based on the reference materials.\n\n"
        f"## My question is:\n\n{question}\n\n"
        f"## Reference Materials:\n\n {contexts} \n\n"
        "Please respond in the same language as the user's question.\n"
    )
    

    """
    prompt = (
        "请根据引用的材料回答问题.\n\n"
        f"## 我的问题:\n\n{question}\n\n"
        f"## 引用材料:\n\n {contexts} \n\n"
    )
    """
    
    ollama_url = "http://localhost:11434/v1/chat/completions"

    headers = {"Content-Type": "application/json"}

    payload = {
        "model": "deepseek-r1:7b",
        "messages": [
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "temperature": 0.5,
        "top_p": 0.9,
        "stream": True
    }

    response = requests.post(ollama_url, headers=headers, json=payload, stream=True)

    if response.status_code == 200:
        answer = ""

        
        print("-" * 50)
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                
                if line_str.startswith('data: '):
                    data_str = line_str[6:]
                    
                    if data_str == '[DONE]':
                        break
                    
                    try:
                        data = json.loads(data_str)

                        content = data["choices"][0]["delta"]["content"]
                        if content:
                            print(content, end='', flush=True)
                            answer += content

                        """
                        if "choices" in data and len(data["choices"]) > 0:
                            choice = data["choices"][0]
                            
                            if "delta" in choice and "content" in choice["delta"]:
                                content = choice["delta"]["content"]

                                if content:
                                    print(content, end='', flush=True)
                                    answer += content
                        """
                    except json.JSONDecodeError:
                        continue
        
        print("\n" + "-" * 50)
        
        return answer
       
    else:
        print("Ollama 调用失败:", response.text)
        return ""


def search(query):
    #print("问题：", query)
   
    query_embedding = model.encode([query])
    # 返回最相似的6个结果
    distances, indices = index.search(query_embedding, 6)

    result = []
    print("最相似的部分索引:", indices)
    print("距离:", distances)

    for idx in indices[0]:
        result.append(section[idx])
    
    return generate_answer(query, result)
    #return result


@app.route('/ask', methods=['POST'])
def handle_ask():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "请求数据为空"}), 400
        
        question = data.get("question")
        
        if not question:
            return jsonify({"error": "问题不能为空"}), 400
        
        print(f"收到问题: {question}")
        final_answer = search(question)
        
        return jsonify({
            "question": question,
            "answer": final_answer
        })
        
    except Exception as e:
        print(f"处理请求时出错: {str(e)}")
        return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500

@app.route('/test', methods=['GET'])
def connect_test():
    return "后端运行成功"

if __name__ == '__main__':
    # 设置本地后端地址和端口
    host = '127.0.0.1'  # 本地地址
    port = 5000         # 端口号
    
    print(f"启动后端服务...")
    print(f"服务地址: http://{host}:{port}")
    print(f"测试连接: http://{host}:{port}/test")
    print(f"问答接口: http://{host}:{port}/ask")
    print("-" * 50)
    
    # 启动Flask应用
    app.run(host=host, port=port, debug=True)

