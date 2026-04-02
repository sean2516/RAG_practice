import os
import sys
import fitz  # PyMuPDF
import nltk
from unstructured.partition.auto import partition
from unstructured.chunking.basic import chunk_elements
from unstructured.chunking.title import chunk_by_title

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["PATH"] = r"D:\poppler-24.08.0\Library\bin" + os.pathsep + os.environ["PATH"]
os.environ["PATH"] = r"D:\Tesseract-OCR" + os.pathsep + os.environ["PATH"]

def ensure_nltk_data():
    for pkg, path in [
        ('punkt', 'tokenizers/punkt'),
        ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
        ('stopwords', 'corpora/stopwords')
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg)

def main():
    #ensure_nltk_data()
    if not os.path.exists("output.pdf"):
        print("output.pdf 文件不存在！")
        sys.exit(1)
    print("开始分块")
    try:
        elements = partition(
            filename="output.pdf",
            languages=["chi_sim"],
            strategy="hi_res"  # 或 "fast"、"auto"，具体看 unstructured 版本支持
        )
    except Exception as e:
        print(f"分块失败: {e}")
        sys.exit(1)
    #print(f"分块完成，共 {len(elements)} 个分块")

    chunks = chunk_by_title(elements)
    for chunk in chunks:
        print(chunk.text)
        print("--------------------------------")
    
      

    
    
"""
    # 示例：合并同类项的简单后处理
    merged = []
    last_cat = None
    last_text = ""
    for element in elements:
        if element.category == last_cat:
            last_text += element.text
        else:
            if last_cat is not None:
                merged.append((last_cat, last_text))
            last_cat = element.category
            last_text = element.text
    if last_cat is not None:
        merged.append((last_cat, last_text))

    with open("output_segments_merged.txt", "w", encoding="utf-8") as f:
        for cat, text in merged:
            f.write(f"{cat}: {text}\n")
"""

if __name__ == "__main__":
    main()
