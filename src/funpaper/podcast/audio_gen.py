import datetime
import glob
import itertools
import os
import re
import uuid

from farlog import getLogger
from openai import OpenAI
from pydub import AudioSegment

logger = getLogger("funpaper")

# 同一播客内各语音片段的单调递增序号：仅靠秒级时间戳命名时，同一角色在同一秒内
# 生成多段台词会产生同名文件、后一段覆盖前一段，最终合并音频会丢词。序号与时间戳
# 一起构成文件名，既保证同一进程内全局唯一，又保留 merge_mp3_files 排序所需的
# 严格递增键（序号本身就反映真实的生成/台词顺序）。
_segment_seq = itertools.count()


def _segment_filename(role: str, output_dir: str) -> str:
    """生成某角色语音片段的唯一文件名（时间戳 + 单调序号，避免同秒覆盖）。"""
    now = int(datetime.datetime.now().timestamp())
    seq = next(_segment_seq)
    return f"./{output_dir}/{role}_{now}_{seq:06d}.mp3"


def generate_host(text: str, client: OpenAI, output_dir: str) -> None:
    """用 `alloy` 音色合成主持人（Host）台词的语音文件。

    Args:
        text: 待合成的台词文本。
        client: OpenAI 客户端，需支持 `audio.speech.create`。
        output_dir: 输出目录（相对当前工作目录）。

    Returns:
        None。
    """
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
    )
    response.stream_to_file(_segment_filename("host", output_dir))


def generate_expert(text: str, client: OpenAI, output_dir: str) -> None:
    """用 `fable` 音色合成专家（Expert）台词的语音文件。

    Args:
        text: 待合成的台词文本。
        client: OpenAI 客户端，需支持 `audio.speech.create`。
        output_dir: 输出目录（相对当前工作目录）。

    Returns:
        None。
    """
    response = client.audio.speech.create(
        model="tts-1",
        voice="fable",
        input=text,
    )
    response.stream_to_file(_segment_filename("expert", output_dir))


def generate_learner(text: str, client: OpenAI, output_dir: str) -> None:
    """用 `nova` 音色合成学习者（Learner）台词的语音文件。

    Args:
        text: 待合成的台词文本。
        client: OpenAI 客户端，需支持 `audio.speech.create`。
        output_dir: 输出目录（相对当前工作目录）。

    Returns:
        None。
    """
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text,
    )
    response.stream_to_file(_segment_filename("learner", output_dir))


def merge_mp3_files(directory_path: str, output_file: str) -> None:
    """将目录下的多个 mp3 片段按文件名中的单调序号排序后合并为一个文件。

    Args:
        directory_path: 存放待合并 mp3 片段的目录（相对当前工作目录）。
        output_file: 合并后输出的 mp3 文件路径。

    Returns:
        None。
    """
    # 查找目录下所有 .mp3 文件
    mp3_files = [os.path.basename(x) for x in glob.glob(f"./{directory_path}/*.mp3")]

    # 按文件名末尾的单调序号排序（而非秒级时间戳：同一秒内生成的多段语音时间戳
    # 相同，若仅按时间戳排序会退化为不确定的文件系统遍历顺序，丢失真实的台词
    # 先后关系）。序号由 `_segment_filename` 写入，格式为 `..._<序号6位>.mp3`。
    sorted_files = sorted(
        mp3_files, key=lambda x: re.search(r"_(\d+)\.mp3$", x).group(1).zfill(20)
    )
    # 初始化空音频片段用于合并
    merged_audio = AudioSegment.empty()

    # 按时间顺序依次合并每个 mp3 文件
    for file in sorted_files:
        audio = AudioSegment.from_mp3(f"./{directory_path}/{file}")
        merged_audio += audio

    # 导出最终合并的音频
    merged_audio.export(output_file, format="mp3")
    logger.info(f"Merged file saved as {output_file}")


def generate_podcast(script: str, client: OpenAI) -> None:
    """解析播客脚本文本，按角色分别合成语音并合并为一个完整 mp3 文件。

    Args:
        script: 包含 `Host:`/`Learner:`/`Expert:` 台词标记的完整脚本文本。
        client: OpenAI 客户端，透传给各角色的 TTS 合成函数。

    Returns:
        None。
    """
    # 创建一个新目录用于存放音频文件；附加短随机后缀避免同一秒内重复调用时
    # 目录名冲突（纯秒级时间戳在高频调用场景下会撞名，`os.mkdir` 直接抛
    # FileExistsError）。
    output_dir = (
        f"podcast_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        f"_{uuid.uuid4().hex[:6]}"
    )
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
