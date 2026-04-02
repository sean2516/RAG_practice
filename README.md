# RAG_practice

---

## RAG：结合语言模型和信息检索技术

---

## 嵌入模型

paraphrase-multilingual-minilm-l12-v2    - 已被bge-m3代替
m3e-base                                                       - 已被bge-m3代替
bge-m3

---

## 生成模型

deepseek-r1:1.5b        - 已被deepseek-r1:7b代替
deepseek-r1:7b
llama3.1:8b                  - 还未测试其效果

---

## Rerank - 重排模型

bge-rerank-v2-m3

---

## Python 依赖：

由于国内的网络等因素，建议添加pip下载/安装时添加镜像

### Faiss

由Facebook AI Research开发，擅长处理大规模向量数据，尤其适用于高维空间的相似度搜索。
专为大规模向量相似度检索而设计的高效库。其主要原理是将高纬向量压缩和索引，从而大幅度降低检索时间的复杂度。
Faiss的核心包括向量索引（Indexing）和向量压缩（Quantization）

开发流程通常包括以下几个步骤：
1.向量生成：将文本数据向量化，例如使用Transformrs中的预训练模型。
2.索引创建：创建索引并添加向量数据
3.相似度检索：使用已创建的索引对查询向量进行相似度检索，返回最相似的向量或其对应的原始数据

---

### Transformers

由Hugging Face开发，提供了多种预训练的Transformer模型，可以实现语言理解，文本生成等任务。

开发流程通常包括以下几个步骤：
1.模型加载：从HF模型库中加载所需的模型
2.数据预处理：将输入输出通过分词器处理成模型接受的输入格式
3.模型推理：将预处理后的数据输入模型，得到预测结果

PS：
1.由于国内的网络问题，需设置是HF的镜像地址：

```
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

2.在read_file_v4版本，改使用Ollama调用生成模型后，不再使用使用Transformers

---

### Sentence_transformers

用于访问、使用嵌入和重新排序器模型：

```
Model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
```

PS：
1.由于国内的网络问题，需设置HF的镜像地址才能够调用模型

```
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

---

### Torch

一个开源的 Python 机器学习库，主要用于深度学习任务。（感觉其实是设置生成模型的各种参数......也许还其他的作用还没有参透吧）

注：在read_file_v4版本，改使用Ollama调用生成模型后，不再使用使用Torch

---

### Flask

轻量级的Web框架，用于构建 RESTful API 或 Web 应用
创建python文件的调用接口，路由

---

### Requests

发送 HTTP 请求的库，用与API交互
向本地ollama发送并接收，调用生成模型

---

### Json

用于 JSON 数据的编码与解码，解析 API 请求/响应的 JSON 数据

---

### Hashlib

提供常见哈希算法（如 SHA256、MD5）的库。数据完整性校验（文件哈希）

---

### Nltk

提供用于文本处理的工具，自然语言处理。
大模型RAG应用开发 - 构建智能生成系统（作者：凌峰）：“NLTK以丰富的NLP功能集和资源库，适合深入的文本分析和自然语言研究，常见的功能分词，词性标注，命名实体类别，情感分析等。NLTK还包含多种语言库和词典工具，在RAG系统的文本预处理，语法分析，情感分析等有着广分应用”
punkt：分词
averaged_perceptron_tagger：词性标注（如名词、动词、形容词等）
stopwords：去除停用词，移除常见但无实际意义的词
Wordent：词形还原

---

### Langchian

用于构建基于大语言模型（LLM）的应用程序框架
分块，上下文

---

### Fitz / PyMupDF

处理PDF文件（打开，删除，创建）
支持高精度文本提取，图像提取、PDF 渲染、OCR 协同（需结合 Tesseract），处理加密或复杂版式 PDF 更高效。

---

### Unstructured

从 PDF 等文档中提取结构化数据：如标题、段落、表格。
自动化清理和结构化文档内容

---

## Pdf处理工具

### Poppler - 24.08.0

开源 PDF 渲染引擎；将 PDF 渲染为图像
尝试用于移除页眉，页码；清理文本
地址：https://poppler.freedesktop.org/

---

### Tesseract ORC

对PDF进行文字识别，从 PDF 扫描件或图像中提取文本
在下载过程中，好像是不选择就会下载全部的语言处理插件。若不需要的话，跳过并仅自动安装需要的语言处理插件，以节省时间，但要注意识别。
地址：https://digi.bib.uni-mannheim.de/tesseract/

---

## 模型部署工具

### Ollama

本地模型
PS：在尝试用dify连接时，疑似docker的ip的地址关系连接失败，改用流动硅基代替。在卸载docker，使用WSL + docker engine驱动替代运行dify后再次次尝试连接，但依旧失败。

---

### Xinference

类似于ollama，可以部署本地大模型的平台。在dify连接ollama失败后，在Dify的插件列表中搜索找到，尝试下载安装，但不知到底是什么原因无法运行，最后决定放弃，使用硅基流动代替。【失败原因依旧不明】

---

### 硅基流动

通过API密匙调用模型；仅使用免费的模型。被用于Dify的模型调用

---

## RAG相关

### RAGflow

基于RAG的开源人工智能框架。
尝试搭建基于deepseek-r1的本地知识库系统，但使用失败。
失败原因暂且不明 - 好像是无法接入模型，或知识库的文件解析/分块失败 - 有待确认

---

### Cherry Studio

用于搭建本地的个人知识库，和知识库问道。
根据源代码，似乎都是根据字符数量分块
官网地址：https://www.cherry-ai.com/

---

### Dify

大模型应用平台 - 构建AI工作流程。
模板：Question Classifier + Knowledge + Chatbot
后续尝试改用wsl2 + docker engine来运行。由于wsl和windows的权限冲突关系，需要将dify的文件位置放到wsl2的系统里面，才能够正常运行。
官网地址：https://github.com/langgenius/dify/

- 待确认：好像是要打开某个文件设置镜像地址

运行指令和步骤：
1.cmd进入到dify/docker 路径
2.cp .env.example .env
    - windows系统为：copy .env.example .env。
3.docker compose up -d
4.打开浏览器进入http://localhost

---

## 图数据库和框架:

### Neo4j

图数据库。有上传文件，自动读取文件并生成关系的功能/网站，但仅支持连接云端版的Neo4j。
官网地址：https://neo4j.com/

---

### Light RAG

可通过上传文件，自动生成关系图。自带RAG功能，需要在运行service前，提前在源代码里设置模型的调用。
Github 地址：https://github.com/HKUDS/LightRAG/blob/main/README-zh.md

---

## 其他：

### Text-loader.js

在浏览了cherry studio的源代码后发现的txt分块代码。
来源：https://www.npmjs.com/package/@cherrystudio/embedjs?activeTab=code

---

## 迭代记录

### qna 系列

#### qna.py

根据《大模型RAG应用开发 - 构建智能生成系统》（作者：凌峰）的chapter4的示例代码为原型，根据需求模仿并改造而来。该版本仅有向量检索功能，非完整的RAG。
嵌入模型：paraphrase-multilingual-MiniLM-L12-v2
无生成模型
嵌入模型通过sentence_transformer下载
通过设定模拟数据集qna_set设置10个问答，格式为：

```
qna_set=[
    {
        "question": "......",
        "answer": "......"
    }
]
```

通过将数据集中的问题和答案分别取出，以list保存后，通过嵌入模型和faiss将问题向量化（生成嵌入，创建FAISS索引并添加向量）。
在ask_question()接收到用户的提问后，将该提问向量化并进行向量检索，取最相似的一个问题的对应的回答返回。

---

#### qna_v2.py

以qna.py为基础，将相似度计算方式从IndexFlatIP（相似度匹配）改成IndexFlatL2（距离匹配）进行比较，移除了normalize_L2。
顺带优化了一下代码结构，移除不需要的代码和注解

---

#### qna_v3.py

以v2为基础，根据《大模型RAG应用开发 - 构建智能生成系统》的示例，正式加入并使用生成模型（gpt2），prompt（文本提示 - 生成模型的提示词）。对torch的正式使用（之前的版本/示例有引入，但没有使用）
注：根据示例进行的模仿和学习，v3版本的主要目的是先确保新加入的内容能够顺利运行。在第一次测试运行后，考虑到效率和内存，并且该项目对质量暂时还没有太高的需求，最后决定使用并关闭梯度计算（torch.no_grad）
出于模型对语言理解的考虑，将问题集的文本从简体中文换成英文

---

#### qna_v4.py

嵌入模型：all-mpnet-base-v2
生成模型：DeepSeek-R1-Distill-Qwen-1.5B

嵌入模型通过sentence_transformer下载
生成模型直接下载到本地，直接通过路径调用

在v3版本的基础上将嵌入和生成模型改成可以理解简体中文的版本，使用DeepSeek模型还有一点是因为其目前在国内的使用度高。优化生成模型的提示词(Prompt)，并加入更多的生成模型的阀值

```
prompt = f"原来的问题: {que}\n原来的回答: {org_answer}\n优化后的新回答:"
```

在ask_question()接收到用户的提问后，将该提问向量化并进行向量检索，根据匹配方式的不同，取最相似的一个问题的对应的回答后，通过generate_answers()将【原来的问题】，【原来的问题】和【提示词】一起发送给生成模型，获取生成答案后返回。

---

#### qna_v5.py

嵌入模型：paraphrase-multilingual-minilm-l12-v2
生成模型：Deepseek-R1-Distill-Qwen-1.5B

嵌入模型通过sentence_transformer下载
生成模型直接下载到本地，直接通过路径调用

在v5版本的基础上将相似度计算方式从IndexFlatL2（距离匹配）改回到IndexFlatIP（相似度匹配）
在ask_question()接收到用户的提问后，将该提问向量化并进行向量检索，根据匹配方式的不同，取最相似的一个问题的对应的回答后，通过generate_answers()将【用户的问题】，【匹配的答案】和【提示词】一起发送给生成模型，获取生成答案后返回。

---

#### qna_v5.2.py

在整理文件中找到，原名qna_v5.py，从记录的最后修改时间和内容来看是qna_v5.py的后续修改版本，现重命名为qna_v5.2.py

在v5版本的基础上添加了两行标准输出编码修复代码，确保输出结果文本为utf-8。该代码是在某次测试运行中，输出文本为乱码后添加修复。
注：在后续的开发中就没遇到过输出乱码的情况，但该版本还是作为案例留下，以应对后续开发中可能会再次遇到的文本乱码问题

---

#### qna_v6.py

在v5版本的基础上添加了Flask，可通过postman，端口路径 ”ask_qna” 对其进行调用，进行问答

---

### read_file 系列

#### read_file

以qna_v5为基础，移除了模拟数据集qna_set，添加了txt文件读取。
将每一行字段向量化

---

#### read_file_v2

以read_file为基础，尝试使用正则表达式来识别标题进行分块。
由于qna系列的设计架构，需要将分块后的section转换为data frame，并进行文本清理

结论：由于输入文件的结构规范问题，用标题符号（#）的数量来分块效果不佳
想法：重新设计分块标识符

---

#### Read_file_v2_2

以read_file_v2为基础，暂时停用了生成功能，主要用于测如何更好地进行分块，以及提升检索效率。修改了to_document中识别各种标题符号的正则表达式，以更好地识别内容，进行分块。

结论：依旧不行
想法：继续重新设计分块标识符。考虑重新整理文件的结构，使其能够被更好地分块

---

#### Read_file_v2_3

以read_file_v2为基础，移除了data frame，简化结构和功能，只保留必要的核心部分。依旧是停用生成功能。

想法：考虑到md文件的分割线为”---”，能否用它来进行分块。
结论：可行，但依然会产生无效块（标题等会被单独分块，有些的匹配度高，但是这些块无法提供任何的内容）

---

#### Read_file_v3

以read_file_v2_3为基础，正式恢复生成功能。
结论：可以运行，生成的内容也大差不差。但没试过用更加深层和不同层面的问题进行测试，也没有考虑结果的准确程度

---

#### Read_file_v3_1

在研究了cherry studio的运行和其源代码后，以Read_file_v3为基础，根据其源代码的内容和调用的用于txt文件分块的text-loader.js 文件，添加了分块和文本清理的步骤。不过，text-loader.js似乎是通过字符数量进行分块，可能会导致关键的语义错误，上下文衔接等问题......有待测试其效果。
后记：当时为什么不先试一下NLTK......忘了......

---

#### Read_file_v3_2

根据cherry studio的源代码，在Read_file_v3_1的基础上，修改了发送给生成模型的提示词

---

#### Read_file_v4

以Read_file_v3_2为基础，将嵌入模型替换为【bge-m3】，并调用ollama的生成模型（不再使用Transformers）
ollama的地址为“http://localhost:11434/v1/chat/completions”，由RAGflow提供。

由于是要发送到ollama，并接收从Ollama返回的（一次性）输出，引入requests依赖来实现发送和接收。

一次性返回的格式：

```
{
    "id": "chatcmpl-992",
    "object": "chat.completion",
    "created": 1755223172,
    "model": "deepseek-r1:7b",
    "system_fingerprint": "fp_ollama",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "<think>\n\n</think>\n\n你好！很高兴见到你，有什么我可以帮忙的吗？"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 5,
        "completion_tokens": 17,
        "total_tokens": 22
    }
}
```

因此，获得返回的答案的代码为：

```
data = response.json()
return data["choices"][0]["message"]["content"]
```

---

#### Read_file_v5

在read_file_v4的基础上，将ollama返回和打印的输出改为的流式输出。

通过在payload最后添加【"stream": True】来让ollama返回流式输出。

流式输出的返回格式与之前的一次性返回的格式不同。
Python文件接收的流式输出的返回格式为：

```
data: [DONE]
data: {"id":"chatcmpl-93", "object":……}
┆┆
data: 
    {
        "id":"chatcmpl-93",
        "object":"chat.completion.chunk",
        "created":1755223565,
        "model":"deepseek-r1:7b",
        "system_fingerprint":"fp_ollama",
        "choices":
            [
                {
                    "index":0,
                    "delta":
                    {
                        "role":"assistant",
                        "content":"\u003cthink\u003e"
                    },
                    "finish_reason":null
                }
            ]
    }
```

```
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
```

这里通过for循环获得返回的每一条信息，每一行信息前有【data: 】作为前缀，占6个字符，并且每行data之间会有一行空行（\n）将data分开。

这里使用【line_str.startswith('data: ')】来辨别是否是数据还是空行，然后通过【data_str = line_str[6:]】来移除接收的【data: 】字段。

[DONE]表示返回结束，因此当【data_str】为【[DONE]】时，便可停止循环。通过接收的信息，我们得知每一次返回的回答的字符的位置在【content】。因此，获取contenrt的代码为：

```
data = json.loads(data_str)
content = data["choices"][0]["delta"]["content"]
```

在获取content后，将其添加到answer里面。
注意，这里要将返回的每一行的字符decode为【utf-8】，否则是返回乱码。

---

#### Read_file_v6

在Read_file_v5的基础上添加了Flask，可通过localhost端口路径【/ask】对其进行调用，进行问答，并可以接收用户的输入。

---

#### Read_file_v7

仅是在Read_file_v6的基础上重新调整一下代码结构

---

### PDF_edit 系列

#### Read_pdf

PDF_edit 系列的第一个文件，目的是尝试能否在python中打开并读取pdf。
通过fitz / PyMuPDF 依赖，成功打开pdf文件，并识别和打印找到的页眉和页码

---

#### Remove_footer

在Read_pdf文件的基础上，开始尝试移除页码。在确认页码的存在后，使用Rect和add_react_annot使用白色的方块将页码覆盖，另存为output.pdf

---

#### Get_text

在使用Remove_footer.py将pdf文件的页码去除后，以得到的处理后的output.pdf文件为输入文件，尝试使用fitz / PyMuPDF的【.get_text()】提取出所有的文本。
结论：文本提取成功，并且没有页码

尝试使用re依赖对提取的文本进行清理和分割
结果：【没有记录】
记录：分块部分的代码已被标注。根据源码的标注内容，疑似尝试使用HuggingFaceEmbeddings来进行语义分块，但结果不明，原因未知（已不记得）

---

#### Block

在Get_text的基础上重新设计分段点（附件一，第一条，第一章，总则等）后，试图通过连接ollama的生成模型【deepseek-r1:7b】对提取的内容进行二次分类和分块

提示词：

```
"请对以下从PDF中提取的文本进行重新整理和排序, 只允许调整顺序和结构，不允许删减、添加或改写任何内容。不要生成任何新的内容，不要总结，不要扩写。"
        "请确保所有原始信息都被保留。如果不确定顺序，请保持原有顺序。\n\n"
        "输出格式为：每一段独立，顺序合理。"
        "需要处理的文本：\n"
```

结果：乱七八糟，返回的根本不是分块内容，而是对话。
结论：不行！目前看来，生成模型不适合进行分块识别......或是提示词的问题？下次要不要试一下重排模型

---

#### Block_v2

尝试使用nltk，unstructured.partition，Poppler和Tesseract-OCR对pdf内容进行分块

Nltk下载：
punkt：分词
averaged_perceptron_tagger：词性标注（如名词、动词、形容词等）
stopwords：停用词过滤，移除常见但无实际意义的词

---

#### Block_v3

仅是在block_v2的基础上添加处理过程当中的打印检查，以监视错误的发生位置，包括：流程提示，分块数量计算，和分块内容和类别预览

---

#### Block_v4

在block_v3的技术上引用【from unstructured.chunking.title import chunk_by_title】依赖，尝试根据pdf的title进行更一步的分块。

曾尝试过使用【from unstructured.chunking.basic import chunk_elements】，根据特殊字符的方法来分块，但这类想法已在前面尝试过，所以改用chunk_by_title替代
注：【部分记录缺失】；后续将分块内容写入txt的代码已被改写为Block_v5的部分，无法找到修改前的记录。

---

#### Block_v5

在block_v4的基础上，尝试在分块后合并同类项以达到更好的分块效果，但后面的似乎还是有点问题。

想法：

- 进行测试的输出文件其实还包括5个附件，第一篇文章的分块效果其实还可以。并且之前对原pdf的处理只是去除了页码，并没有去除后续附件的页眉（附件一，附件二，附件三等）。
- 先尝试手动分页，将输入文件当中的每一个附件分为独立的文件，然后将这6个文件单独进行分块，是否能够达到更好的分块效果。
