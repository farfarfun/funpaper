import click
from farlog import getLogger
from funai.llm import get_model
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from .audio_gen import generate_podcast
from .script import generate_script, parse_script_plan
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

    client = get_model("deepseek")
    llm = ChatOpenAI(model="deepseek-chat")

    # chains
    chains = {
        "plan_script_chain": plan_prompt | llm | parse_script_plan,
        "initial_dialogue_chain": initial_dialogue_prompt | llm | StrOutputParser(),
        "enhance_chain": enhance_prompt | llm | StrOutputParser(),
    }

    # Step 1: Generate the podcast script from the PDF
    logger.info("Generating podcast script...")
    script = generate_script(pdf_path, chains, llm)
    logger.info("Podcast script generation complete!")

    logger.info("Generating podcast audio files...")
    # Step 2: Generate the podcast audio files and merge them
    generate_podcast(script, client)
    logger.info("Podcast generation complete!")


def funpaper() -> None:
    """CLI 主入口，注册 `funpaper podcast --pdf_path ...` 命令。"""

    @click.group()
    def cli():
        pass

    @cli.command()
    @click.option("--pdf_path", type=str, help="论文地址")
    def podcast(pdf_path: str) -> None:
        paper_to_podcast(pdf_path)

    cli()
