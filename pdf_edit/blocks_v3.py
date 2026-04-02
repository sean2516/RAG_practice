import os
import sys
import fitz  # PyMuPDF
import nltk
from unstructured.partition.auto import partition

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
        elements = partition(filename="output.pdf", languages=["chi_sim"])
    except Exception as e:
        print(f"分块失败: {e}")
        sys.exit(1)
    print(f"分块完成，共 {len(elements)} 个分块")
    for element in elements[:5]:
        print(f"{element.category}: {element.text}")

if __name__ == "__main__":
    main()