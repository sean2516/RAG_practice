import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["PATH"] = r"D:\poppler-24.08.0\Library\bin;" + os.environ["PATH"]
os.environ["PATH"] = r"D:\Tesseract-OCR;" + os.environ["PATH"]

import fitz  # PyMuPDF
import nltk
from unstructured.partition.auto import partition

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

print("开始分块")
elements = partition(filename="output.pdf", languages=["chi_sim"])
print("分块完成")

for element in elements[:5]:
    print(f"{element.category}: {element.text}")