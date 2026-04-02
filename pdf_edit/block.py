import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import fitz  # PyMuPDF
import io
import sys
import hashlib
import re
import requests
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


# 修改标准输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def get_all_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def split_structured(text):
    # 以“附件X”、“第X条”、“第X章”、“总则”等为分段点
    pattern = re.compile(
        r'(第[一二三四五六七八九十百]+条\n|总则\n|附表\d+\n)', re.MULTILINE)
    # 以两个及以上换行也分段
    paragraphs = re.split(r'\n{2,}', text)
    result = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        splits = pattern.split(para)
        temp = ""
        for s in splits:
            if pattern.match(s):
                if temp:
                    result.append(temp.strip())
                temp = s
            else:
                temp += s
        if temp:
            result.append(temp.strip())
    # 去除过短的段落
    result = [p for p in result if len(p) > 10]
    return result


def clean_text(texts):

    ollama_url = "http://localhost:11434/v1/chat/completions"

    headers = {"Content-Type": "application/json"}
    prompt = (
        "请对以下从PDF中提取的文本进行重新整理和排序, 只允许调整顺序和结构，不允许删减、添加或改写任何内容。不要生成任何新的内容，不要总结，不要扩写。"
        "请确保所有原始信息都被保留。如果不确定顺序，请保持原有顺序。\n\n"
        "输出格式为：每一段独立，顺序合理。"
        "需要处理的文本：\n"
        f"{texts}\n"
    )

    payload = {
        "model": "deepseek-r1:7b",
        "messages": [
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "temperature": 0.0,
        "top_p": 0.7,
        "stream": False
    }

    response = requests.post(ollama_url, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        else:
            return str(data)
    else:
        print("Ollama 调用失败:", response.text)
        return ""
    

texts = get_all_text("output.pdf")

with open('output.txt', 'w', encoding='utf-8') as file:
    file.write(texts)

cleaned_texts = clean_text(texts)

with open('AI_cleaned_output.txt', 'w', encoding='utf-8') as file:
    file.write(cleaned_texts)

"""
segments = split_structured(texts)

with open("blocks.txt", "w", encoding="utf-8") as f:
    for i, seg in enumerate(segments):
        f.write(f"=== 分段 {i+1} ===\n")
        f.write(seg)
        f.write("\n\n")  # 用空行分隔
"""   