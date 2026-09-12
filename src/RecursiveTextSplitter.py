"""
【案例】用 RecursiveCharacterTextSplitter 分割纯文本（split_text + create_documents）

对应教程章节：第 19 章 - RAG 检索增强生成 → 2、RAG 文本处理核心知识

知识点速览：
- 大文档需先切块再向量化：避免超长上下文、控制 token 成本；即使模型支持长上下文，也不代表把整篇文档直接塞进去就是更好的 RAG 方案。
- RecursiveCharacterTextSplitter 按字符递归切，尽量保持语义完整，是通用文本场景里最常见的入门分割器。
- chunk_size：单块最大长度（按 length_function 计算，默认 len 即字符数）；chunk_overlap：相邻块重叠字符数，常用 size 的 10%～20%。
- split_text(content)：把字符串切成字符串列表；create_documents(texts)：把字符串列表转成 Document 列表（或直接用 split_documents 处理 Document）。
- 重叠部分会重复出现在相邻块中，总字符数会大于原文，这不是 bug，而是为了减少「半句话被截断」的问题。
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 待分割的原文
content = (
    "大模型RAG（检索增强生成）是一种结合生成模型与外部知识检索的技术，通过从大规模文档或数据库中检索相关信息，"
    "辅助生成模型以提升回答的准确性和相关性。其核心流程包括用户输入查询、系统检索相关知识、"
    "生成模型基于检索结果生成内容，并输出最终答案。RAG的优势在于能够弥补生成模型的知识盲区，"
    "提供更准确、实时和可解释的输出，广泛应用于问答系统、内容生成、客服、教育和企业领域。"
    "然而，其也面临依赖高质量知识库、可能的响应延迟、较高的维护成本以及数据隐私等挑战。"
)

# 2. 分割器：块大小 100 字符，重叠 30 字符，长度按 len（字符数）计算
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, chunk_overlap=30, length_function=len
)

# 3. 先切成字符串列表
splitter_texts = text_splitter.split_text(content)

# 4. 再转成 Document 列表（便于后续与向量库、检索器对接）
splitter_documents = text_splitter.create_documents(splitter_texts)

print(f"原始文本大小：{len(content)}")
print(f"分割文档数量：{len(splitter_documents)}")
for splitter_document in splitter_documents:
    print(
        f"文档片段大小：{len(splitter_document.page_content)},文档内容：{splitter_document.page_content}"
    )

"""
【输出示例】
"""
"""
原始文本大小：225

分割文档数量：3

文档片段大小：100,文档内容：大模型RAG（检索增强生成）是一种结合生成模型与外部知识检索的技术，通过从大规模文档或数据库中检索相关信息，辅助生成模型以提升回答的准确性和相关性。其核心流程包括用户输入查询、系统检索相关知识、生成模

文档片段大小：100,文档内容：相关性。其核心流程包括用户输入查询、系统检索相关知识、生成模型基于检索结果生成内容，并输出最终答案。RAG的优势在于能够弥补生成模型的知识盲区，提供更准确、实时和可解释的输出，广泛应用于问答系统、内容

文档片段大小：85,文档内容：区，提供更准确、实时和可解释的输出，广泛应用于问答系统、内容生成、客服、教育和企业领域。然而，其也面临依赖高质量知识库、可能的响应延迟、较高的维护成本以及数据隐私等挑战。
"""

"""
验证总字符的逻辑（并非简单相加）
同学们可能会疑惑：100+100+85=285，比原始 225 多了 60，why?
这是因为重叠部分被重复计算了，实际原始文本的有效内容被完整覆盖，且无丢失：
第 1 块和第 2 块的重叠：30 字符（重复计算 1 次）
第 2 块和第 3 块的重叠：30 字符（重复计算 1 次）
总重复计算：60 字符 → 285 - 60 = 225（和原始文本长度一致）

这正是分割器设计 chunk_overlap 的目的：
以 “重复计算重叠部分” 为代价，保证每个文本块的语义完整性，避免分割切断上下文。
"""
