import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import fitz  # PyMuPDF
import io
import sys
import hashlib
import re
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

def clean_text(text):
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def split_paragraphs(text):
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    return paragraphs

raw_text = get_all_text("output.pdf")

with open('output.txt', 'w', encoding='utf-8') as file:
    file.write(raw_text)


"""
cleaned_text = clean_text(raw_text)
paragraphs = split_paragraphs(cleaned_text)

# 语义分块
embeddings = HuggingFaceEmbeddings(
    model_name='BAAI/bge-m3',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_amount=0.1  # 更敏感
)

documents = chunker.create_documents(paragraphs)

for i, doc in enumerate(documents):
    print(f"=== 分段 {i+1} ===")
    print(doc.page_content)
    print("\n")
"""