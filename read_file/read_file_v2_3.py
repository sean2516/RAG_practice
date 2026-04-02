import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

#from flask import Flask, request, jsonify, json
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel


import sys
import io

# 修改标准输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


print("开始准备读取文件数据......\n")

file_name = "input.txt"

def read_file(file_name, separator="---"):
    with open(file_name, "r", encoding="utf-8") as f:
        text = f.read()
    parts = [p.strip() for p in text.split(separator) if p.strip()]
    return parts

text_parts = read_file(file_name)
    
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

embeddings = model.encode(text_parts, convert_to_tensor=False)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # 使用L2距离（欧氏距离）
index.add(embeddings)

"""
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

"""

def search(query):
    print("问题：", query)
   
    query_embedding = model.encode([query])
    k = 3  # 返回最相似的3个结果
    distances, indices = index.search(query_embedding, k)

    print("最相似的部分索引:", indices)
    print("距离:", distances)

    for idx in indices[0]:
        print(f"结果 {idx}: {text_parts[idx]}")



# 调用生成函数生成回答
answer = search("参保的费用是多少?")
print("生成的回答：", answer)


