import re
from datetime import datetime
from operator import itemgetter
from typing import Any

from farlog import getLogger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .templates import discuss_prompt_template

logger = getLogger("funpaper")


def initialize_discussion_chain(txt_file: str, llm: ChatOpenAI) -> Any:
    """构建基于向量检索（RAG）的分段讨论生成链。

    加载论文纯文本、切分为片段并建立 Chroma 向量索引，返回一个 LangChain
    Runnable：输入当前小节计划和上一段对话，检索相关片段作为附加上下文，
    生成本小节的对话文本。

    Args:
        txt_file: 论文纯文本文件路径（由 `parse_pdf` 生成）。
        llm: 用于生成对话的 LangChain LLM 实例。

    Returns:
        一个可 `.invoke({"section_plan": ..., "previous_dialogue": ...})`
        的 LangChain Runnable，输出本小节对话文本（str）。
    """
    # 加载、切分并索引论文内容
    loader = TextLoader(txt_file, encoding="UTF-8")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    # 基于相关片段检索并生成对话
    retriever = vectorstore.as_retriever()

    def format_docs(docs: list) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    discuss_rag_chain = (
        {
            "additional_context": itemgetter("section_plan") | retriever | format_docs,
            "section_plan": itemgetter("section_plan"),
            "previous_dialogue": itemgetter("previous_dialogue"),
        }
        | discuss_prompt_template
        | llm
        | StrOutputParser()
    )
    return discuss_rag_chain


def parse_pdf(pdf_path: str, output_path: str) -> str:
    """抽取 PDF 正文，截取到"Conclusion"小节结束为止，写入文本文件。

    Args:
        pdf_path: 输入 PDF 文件路径。
        output_path: 抽取结果的输出文本文件路径。

    Returns:
        `output_path`（便于链式调用）。
    """
    pdf_reader = PdfReader(pdf_path)

    # 抽取 PDF 正文
    extracted_text = []
    collecting = True

    for page in pdf_reader.pages:
        text = page.extract_text()
        if text and collecting:
            extracted_text.append(text)

            # 检查结束条件："Conclusion" 之后的部分
            if "Conclusion" in text:
                conclusion_start = text.index("Conclusion")
                extracted_text.append(text[conclusion_start:])
                collecting = False  # "Conclusion" 小节之后停止收集

    # 拼接所有已收集文本
    final_text_to_section_after_conclusion = "\n".join(extracted_text)

    # 保存为 .txt 文件
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(final_text_to_section_after_conclusion)

    return output_path


def get_head(pdf_path: str) -> str:
    """抽取 PDF 开头到"Introduction"小节之前的内容。

    Args:
        pdf_path: 输入 PDF 文件路径。

    Returns:
        拼接后的文本内容。
    """
    # 加载 PDF 文件
    pdf_reader = PdfReader(pdf_path)

    # 抽取从开头到 "Introduction" 之前的内容
    extracted_text = []
    collecting = True

    for page in pdf_reader.pages:
        text = page.extract_text()
        if text and collecting:
            # 一旦找到 "Introduction" 就停止收集
            if "Introduction" in text:
                introduction_index = text.index("Introduction")
                extracted_text.append(
                    text[:introduction_index]
                )  # 只收集 "Introduction" 之前的内容
                break
            else:
                extracted_text.append(text)

    # 拼接已收集文本并返回
    return "\n".join(extracted_text)


def generate_script(pdf_path: str, chains: dict, llm: ChatOpenAI) -> str:
    """从论文 PDF 生成完整的三人访谈式播客脚本。

    Args:
        pdf_path: 论文 PDF 文件路径。
        chains: 包含 `plan_script_chain`/`initial_dialogue_chain`/
            `enhance_chain` 三个 LangChain Runnable 的字典。
        llm: 用于分段讨论生成链（`initialize_discussion_chain`）的 LLM 实例。

    Returns:
        润色后的最终播客脚本文本。
    """
    start_time = datetime.now()
    # 第一步：解析 PDF 文件
    txt_file = f"text_paper_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
    txt_file = parse_pdf(pdf_path, txt_file)
    with open(txt_file, "r", encoding="utf-8") as file:
        paper = file.read()
    plan = chains["plan_script_chain"].invoke({"paper": paper})
    logger.info("plan generated")

    # 第二步：按大纲逐段生成播客脚本
    script = ""
    # 生成开场对话
    initial_dialogue = chains["initial_dialogue_chain"].invoke(
        {"paper_head": get_head(pdf_path)}
    )

    script += initial_dialogue
    actual_script = initial_dialogue
    discuss_rag_chain = initialize_discussion_chain(txt_file, llm)
    for section in plan:
        section_script = discuss_rag_chain.invoke(
            {"section_plan": section, "previous_dialogue": actual_script}
        )
        script += section_script
        actual_script = section_script
    enhanced_script = chains["enhance_chain"].invoke({"draft_script": script})
    end_time = datetime.now()
    logger.info(f"Time taken: {end_time - start_time}")
    logger.info("final script generated")
    return enhanced_script


def parse_script_plan(ai_message: AIMessage) -> list:
    """把 LLM 返回的大纲文本（标题 + 分级小节 + 要点）解析为小节字符串列表。

    Args:
        ai_message: `plan_prompt` 生成链返回的 `AIMessage`，`content` 为
            Markdown 风格的大纲文本（首行为标题，随后是若干 `#` 小节标题
            及 `- ` 要点）。

    Returns:
        每个元素为一个小节（标题 + 其下所有要点拼接成的字符串）的列表。
    """
    # 初始化小节列表
    sections = []
    current_section = []

    # 按行拆分，跳过第一行（标题）
    lines = ai_message.content.strip().splitlines()
    lines = lines[1:]  # 跳过第一行（标题）

    # 匹配任意级别标题和要点的正则
    header_pattern = re.compile(r"^#+\s")  # 匹配任意数量 # 开头的标题
    bullet_pattern = re.compile(r"^- ")  # 匹配以 "- " 开头的要点

    # 逐行解析，从标题之后的第一个小节标题开始
    for line in lines:
        if header_pattern.match(line):
            # 遇到新标题时，把上一个小节（若存在）加入结果
            if current_section:
                sections.append(" ".join(current_section))
                current_section = []
            # 用该标题开始一个新小节
            current_section.append(line.strip())
        elif bullet_pattern.match(line):
            # 把要点行追加到当前小节
            current_section.append(line.strip())

    # 追加最后一个小节（若存在）
    if current_section:
        sections.append(" ".join(current_section))

    return sections
