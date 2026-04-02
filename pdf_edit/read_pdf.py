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

import pymupdf
import fitz
from collections import defaultdict

# 修改标准输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def check_pdf_headers_footers(pdf_path):

    doc = fitz.open(pdf_path)

    pages = len(doc)
    header_texts = set()
    footer_texts = set()
    
    for i in range(pages):
        page = doc[i]
        text_blocks = page.get_text("blocks")
        
        for block in text_blocks:
            text = block[4].strip() # 文本内容
            y0 = block[1]           # 文本顶部位置
            
            # 页眉检测(顶部10%区域)
            if y0 < page.rect.height * 0.1:
                if text: header_texts.add(text)
            
            # 页脚检测(底部9%区域)
            elif y0 > page.rect.height * 0.91:
                if text: footer_texts.add(text)
    
    doc.close()
    
    # 判断结果
    has_header = len(header_texts) > 0
    has_footer = len(footer_texts) > 0
    
    return has_footer, footer_texts

# 使用示例
has_footer, footer_texts = check_pdf_headers_footers("input.pdf")

print(f"页尾页码打印: {footer_texts}")
