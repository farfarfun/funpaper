import click
from funai.llm import get_model
from funutil import getLogger
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from .audio_gen import generate_podcast
from .script import generate_script, parse_script_plan
from .templates import enhance_prompt, initial_dialogue_prompt, plan_prompt

logger = getLogger("funpaper")


def paper_to_podcast(pdf_path):
    """论文生成语音"""

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


def funpaper():
    """主入口函数"""

    @click.group()
    def cli():
        pass

    @cli.command()
    @click.option("--pdf_path", type=str, help="论文地址")
    def podcast(pdf_path):
        paper_to_podcast(pdf_path)

    cli()
