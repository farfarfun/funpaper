from pathlib import Path

import click
from farlog import getLogger
from funai.llm import get_model
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from .audio_gen import generate_podcast
from .script import PodcastChains, generate_script, parse_script_plan
from .templates import enhance_prompt, initial_dialogue_prompt, plan_prompt

logger = getLogger("funpaper")


def paper_to_podcast(pdf_path: str) -> None:
    """将一篇论文 PDF 转换为播客音频文件。

    依次调用 `generate_script` 生成三人访谈脚本，再调用 `generate_podcast`
    合成并合并语音，最终产物落盘在当前目录的 `podcast_<时间戳>/` 及
    `podcast_<时间戳>.mp3`。

    Args:
        pdf_path: 论文 PDF 文件路径。

    Returns:
        None。
    """

    path = Path(pdf_path)
    if not path.is_file():
        raise click.ClickException(f"PDF 文件不存在或不是普通文件：{path}")

    client = get_model("deepseek")
    llm = ChatOpenAI(model="deepseek-chat")

    # 构造各阶段处理链
    chains: PodcastChains = {
        "plan_script_chain": plan_prompt | llm | parse_script_plan,
        "initial_dialogue_chain": initial_dialogue_prompt | llm | StrOutputParser(),
        "enhance_chain": enhance_prompt | llm | StrOutputParser(),
    }

    # 第一步：从 PDF 生成播客脚本
    logger.info("开始生成播客脚本")
    script = generate_script(pdf_path, chains, llm)
    logger.info("播客脚本生成完成")

    logger.info("开始生成播客音频文件")
    # 第二步：生成并合并播客音频
    generate_podcast(script, client)
    logger.info("播客生成完成")


def funpaper() -> None:
    """CLI 主入口，注册 `funpaper podcast --pdf_path ...` 命令。"""

    @click.group()
    def cli():
        pass

    @cli.command()
    @click.option("--pdf_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="论文地址")
    def podcast(pdf_path: str) -> None:
        paper_to_podcast(pdf_path)

    cli()
