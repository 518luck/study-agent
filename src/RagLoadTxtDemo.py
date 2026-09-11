"""
【案例】用 TextLoader 加载纯文本（TXT）为 Document 列表

对应教程章节：第 19 章 - RAG 检索增强生成 → 2、RAG 文本处理核心知识

知识点速览：
- 文档加载器负责把各种格式的文件读成 LangChain 的 Document；每个 Document 有 page_content（正文）和 metadata（如 source 路径）。
- TextLoader 用于纯文本（.txt），需指定文件路径和编码（如 utf-8）；load() 返回 List[Document]，多行文本通常合并为一个 Document。
- TXT 是最容易入门的加载场景，但“能加载”不等于“适合直接检索”：真实 RAG 中通常仍要继续切块，再做向量化与入库。
- 后续可接文本分割器、嵌入模型与向量库，完成 RAG 的「加载 → 分割 → 向量化 → 存储」流程。
"""

# pip install langchain_community
from langchain_community.document_loaders import TextLoader

file_path = "assets/sample.txt"
encoding = "utf-8"

# load() 为 BaseLoader 统一接口，返回 List[Document]
docs = TextLoader(file_path, encoding).load()

print(docs)
"""
【输出示例】以下为实跑的真实输出（正文较长，中间用 …… 省略）

[Document(metadata={'source': 'assets/sample.txt'}, page_content='LangChain 是一个用于构建基于大语言模型（LLM）应用的开发框架，……\n\n在 RAG（检索增强生成）场景中，LangChain 把整条链路拆成几个可替换的组件：……\n\n文档加载器是这条链路的起点。……\n\n文本分割器解决的是“一次能塞进模型的内容有限”这个问题。……\n\n向量库是检索的基础设施。……\n')]

两点观察：
- 文件里的 5 个段落被合并成了 1 个 Document（TextLoader 不做切块），段落间的换行 \n\n 原样保留在 page_content 里；
  要切成多个小块，得在下一步接文本分割器。
- metadata 里的 source 就是你传入的路径（相对路径会原样记录）。

运行方式：本项目约定从根目录执行 `uv run python -u src/RagLoadTxtDemo.py`，
此时 "assets/sample.txt" 指向项目根目录下的 assets/ 目录；换个工作目录运行就会找不到文件。
"""
