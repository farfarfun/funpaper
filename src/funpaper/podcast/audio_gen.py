import datetime
import glob
import os
import re
from typing import Any

from farlog import getLogger
from pydub import AudioSegment

logger = getLogger("funpaper")


def generate_host(text: str, client: Any, output_dir: str) -> Any:
    """用 `alloy` 音色合成主持人（Host）台词的语音文件。

    Args:
        text: 待合成的台词文本。
        client: OpenAI 客户端，需支持 `audio.speech.create`。
        output_dir: 输出目录（相对当前工作目录）。

    Returns:
        `stream_to_file` 写文件后的返回值（依赖 OpenAI SDK 版本，通常为 None）。
    """
    now = int(datetime.datetime.now().timestamp())
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
    )
    return response.stream_to_file(f"./{output_dir}/host_{now}.mp3")


def generate_expert(text: str, client: Any, output_dir: str) -> Any:
    """用 `fable` 音色合成专家（Expert）台词的语音文件。

    Args:
        text: 待合成的台词文本。
        client: OpenAI 客户端，需支持 `audio.speech.create`。
        output_dir: 输出目录（相对当前工作目录）。

    Returns:
        `stream_to_file` 写文件后的返回值（依赖 OpenAI SDK 版本，通常为 None）。
    """
    now = int(datetime.datetime.now().timestamp())
    response = client.audio.speech.create(
        model="tts-1",
        voice="fable",
        input=text,
    )
    return response.stream_to_file(f"./{output_dir}/expert_{now}.mp3")


def generate_learner(text: str, client: Any, output_dir: str) -> Any:
    """用 `nova` 音色合成学习者（Learner）台词的语音文件。

    Args:
        text: 待合成的台词文本。
        client: OpenAI 客户端，需支持 `audio.speech.create`。
        output_dir: 输出目录（相对当前工作目录）。

    Returns:
        `stream_to_file` 写文件后的返回值（依赖 OpenAI SDK 版本，通常为 None）。
    """
    now = int(datetime.datetime.now().timestamp())
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text,
    )
    return response.stream_to_file(f"./{output_dir}/learner_{now}.mp3")


def merge_mp3_files(directory_path: str, output_file: str) -> None:
    """将目录下的多个 mp3 片段按文件名中的时间戳排序后合并为一个文件。

    Args:
        directory_path: 存放待合并 mp3 片段的目录（相对当前工作目录）。
        output_file: 合并后输出的 mp3 文件路径。

    Returns:
        None。
    """
    # 找出目录下所有 .mp3 文件
    mp3_files = [os.path.basename(x) for x in glob.glob(f"./{directory_path}/*.mp3")]

    # 按文件名中提取的时间戳排序
    sorted_files = sorted(mp3_files, key=lambda x: re.search(r"(\d{10})", x).group(0))
    # 初始化一个空的 AudioSegment 用于合并
    merged_audio = AudioSegment.empty()

    # 按时间顺序依次合并每个 mp3 文件
    for file in sorted_files:
        audio = AudioSegment.from_mp3(f"./{directory_path}/{file}")
        merged_audio += audio

    # 导出最终合并后的音频
    merged_audio.export(output_file, format="mp3")
    logger.info(f"Merged file saved as {output_file}")


def generate_podcast(script: str, client: Any) -> None:
    """解析播客脚本文本，按角色分别合成语音并合并为一个完整 mp3 文件。

    Args:
        script: 包含 `Host:`/`Learner:`/`Expert:` 台词标记的完整脚本文本。
        client: OpenAI 客户端，透传给各角色的 TTS 合成函数。

    Returns:
        None。
    """
    # 创建一个新目录用于存放音频文件
    output_dir = f"podcast_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    os.mkdir(output_dir)
    # 用正则捕获 "Speaker: Text"
    lines = re.findall(
        r"(Host|Learner|Expert):\s*(.*?)(?=(Host|Learner|Expert|$))", script, re.DOTALL
    )

    for speaker, text, _ in lines:
        # 去除多余空格和换行
        text = text.strip()

        # 分发到对应角色的合成函数
        if speaker == "Host":
            generate_host(text, client, output_dir)
        elif speaker == "Learner":
            generate_learner(text, client, output_dir)
        elif speaker == "Expert":
            generate_expert(text, client, output_dir)

    # 合并音频文件为一个完整播客
    now = int(datetime.datetime.now().timestamp())
    merge_mp3_files(output_dir, f"podcast_{now}.mp3")
