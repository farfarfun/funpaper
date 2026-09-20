"""烟雾测试（smoke tests）

funpaper 是一个"论文 PDF -> LangChain 生成播客脚本 -> TTS 合成语音"的命令行工具。
本测试套件的目标是在完全离线（不发起任何真实网络/LLM/TTS 调用）的前提下，验证：

1. 顶层包及各子模块可以正常 import；
2. PDF 解析相关的纯逻辑函数（parse_pdf / get_head / parse_script_plan）在真实样例
   PDF（仓库自带的 paper/2017/*.pdf）上工作正常；
3. 涉及真实 LLM（DeepSeek / OpenAI via langchain-openai）和 TTS（OpenAI audio.speech）
   调用的函数，使用 unittest.mock 打桩后可以被安全地调用一次，验证其编排逻辑
   （而不验证真实模型输出）；
4. CLI 入口 `funpaper` 的 `--help` 能正常退出。

不修复业务逻辑 bug；如果某个函数无法在不改动源码的情况下被安全 mock 测试，则显式
skip 并说明原因。
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PDF = (
    REPO_ROOT
    / "paper"
    / "2017"
    / "DeepFM- A Factorization-Machine based Neural Network for CTR Prediction.pdf"
)


# ---------------------------------------------------------------------------
# 1. import 测试
# ---------------------------------------------------------------------------


def test_import_top_level_package():
    import funpaper  # noqa: F401


def test_import_podcast_subpackage():
    import funpaper.podcast  # noqa: F401


def test_import_templates_module():
    import funpaper.podcast.templates  # noqa: F401


def test_import_script_module():
    import funpaper.podcast.script  # noqa: F401


def test_import_audio_gen_module():
    import funpaper.podcast.audio_gen  # noqa: F401


def test_import_command_module():
    # command.py 顶层 import 了 click / funai / funutil / langchain_openai 等，
    # 全部只在 import 阶段做符号绑定，不会发起真实网络调用。
    import funpaper.podcast.command  # noqa: F401


# ---------------------------------------------------------------------------
# 2. 纯逻辑 / 本地逻辑：Prompt 模板
# ---------------------------------------------------------------------------


def test_templates_are_chat_prompt_templates():
    from langchain_core.prompts import ChatPromptTemplate

    from funpaper.podcast import templates

    for name in (
        "plan_prompt",
        "discuss_prompt_template",
        "initial_dialogue_prompt",
        "enhance_prompt",
    ):
        prompt = getattr(templates, name)
        assert isinstance(prompt, ChatPromptTemplate)


def test_plan_prompt_can_be_formatted_locally():
    """模板渲染是纯字符串操作，不涉及网络调用。"""
    from funpaper.podcast.templates import plan_prompt

    rendered = plan_prompt.format(paper="hello world, this is a tiny fake paper.")
    assert "hello world" in rendered


def test_discuss_prompt_template_can_be_formatted_locally():
    from funpaper.podcast.templates import discuss_prompt_template

    rendered = discuss_prompt_template.format(
        section_plan="# Section 1", previous_dialogue="Host: hi", additional_context="ctx"
    )
    assert "# Section 1" in rendered


# ---------------------------------------------------------------------------
# 3. PDF 解析逻辑：使用仓库自带的真实样例 PDF
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="样例 PDF 缺失，跳过 PDF 解析测试")
def test_parse_pdf_extracts_text_from_real_pdf(tmp_path):
    from funpaper.podcast.script import parse_pdf

    output_path = tmp_path / "extracted.txt"
    result_path = parse_pdf(str(SAMPLE_PDF), str(output_path))

    assert result_path == str(output_path)
    content = output_path.read_text(encoding="utf-8")
    # 论文标题应当出现在抽取结果里
    assert "DeepFM" in content
    assert len(content) > 100


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="样例 PDF 缺失，跳过 PDF 解析测试")
def test_get_head_extracts_intro_section_from_real_pdf():
    from funpaper.podcast.script import get_head

    head = get_head(str(SAMPLE_PDF))
    assert isinstance(head, str)
    assert "DeepFM" in head


def test_parse_script_plan_is_pure_logic():
    """parse_script_plan 只是对 AIMessage.content 做字符串解析，不涉及任何网络调用。"""
    from langchain_core.messages import AIMessage

    from funpaper.podcast.script import parse_script_plan

    fake_message = AIMessage(
        content=(
            "# Title: Demo Podcast\n"
            "# Section 1: Intro\n"
            "- point a\n"
            "- point b\n"
            "# Section 2: Body\n"
            "- point c\n"
        )
    )

    sections = parse_script_plan(fake_message)

    assert sections == [
        "# Section 1: Intro - point a - point b",
        "# Section 2: Body - point c",
    ]


# ---------------------------------------------------------------------------
# 4. TTS 音频生成：mock 掉真实 OpenAI TTS 调用
# ---------------------------------------------------------------------------


def _make_fake_tts_client():
    """构造一个假的 OpenAI 客户端，client.audio.speech.create(...) 返回一个
    带 stream_to_file 方法的假响应对象，不发起任何真实网络请求。"""
    fake_client = MagicMock()
    fake_response = MagicMock()
    fake_client.audio.speech.create.return_value = fake_response
    return fake_client, fake_response


def test_generate_host_calls_tts_with_expected_voice(tmp_path, monkeypatch):
    from funpaper.podcast.audio_gen import generate_host

    monkeypatch.chdir(tmp_path)
    fake_client, fake_response = _make_fake_tts_client()

    generate_host("hello from host", fake_client, "out_dir")

    fake_client.audio.speech.create.assert_called_once()
    _, kwargs = fake_client.audio.speech.create.call_args
    assert kwargs["voice"] == "alloy"
    assert kwargs["input"] == "hello from host"
    fake_response.stream_to_file.assert_called_once()


def test_generate_expert_calls_tts_with_expected_voice(tmp_path, monkeypatch):
    from funpaper.podcast.audio_gen import generate_expert

    monkeypatch.chdir(tmp_path)
    fake_client, fake_response = _make_fake_tts_client()

    generate_expert("hello from expert", fake_client, "out_dir")

    fake_client.audio.speech.create.assert_called_once()
    _, kwargs = fake_client.audio.speech.create.call_args
    assert kwargs["voice"] == "fable"
    fake_response.stream_to_file.assert_called_once()


def test_generate_learner_calls_tts_with_expected_voice(tmp_path, monkeypatch):
    from funpaper.podcast.audio_gen import generate_learner

    monkeypatch.chdir(tmp_path)
    fake_client, fake_response = _make_fake_tts_client()

    generate_learner("hello from learner", fake_client, "out_dir")

    fake_client.audio.speech.create.assert_called_once()
    _, kwargs = fake_client.audio.speech.create.call_args
    assert kwargs["voice"] == "nova"
    fake_response.stream_to_file.assert_called_once()


def test_generate_podcast_dispatches_speakers_without_real_tts_or_merge(
    tmp_path, monkeypatch
):
    """generate_podcast 会用正则从脚本文本里切出 Host/Learner/Expert 的台词，
    分别调用对应的 TTS 生成函数，最后合并 mp3。这里把三个 TTS 生成函数以及
    merge_mp3_files 都换成假实现，只验证编排/调度逻辑，不触碰真实网络或
    真实音频文件（ffmpeg 在 CI/沙箱环境里也不一定可用）。"""
    monkeypatch.chdir(tmp_path)

    script = (
        "Host: welcome to the show\n"
        "Learner: what is this paper about\n"
        "Expert: let me explain\n"
    )

    with patch("funpaper.podcast.audio_gen.generate_host") as mock_host, patch(
        "funpaper.podcast.audio_gen.generate_expert"
    ) as mock_expert, patch(
        "funpaper.podcast.audio_gen.generate_learner"
    ) as mock_learner, patch(
        "funpaper.podcast.audio_gen.merge_mp3_files"
    ) as mock_merge:
        from funpaper.podcast.audio_gen import generate_podcast

        fake_client = MagicMock()
        generate_podcast(script, fake_client)

    mock_host.assert_called_once()
    mock_learner.assert_called_once()
    mock_expert.assert_called_once()
    mock_merge.assert_called_once()


def test_merge_mp3_files_sorts_by_timestamp_and_merges_in_order(tmp_path, monkeypatch):
    """merge_mp3_files 本身只负责按文件名中的时间戳排序、依次合并并导出，
    真正的音频解码/编码逻辑属于 pydub + ffmpeg。这里把 audio_gen.AudioSegment
    整体替换为一个记录调用顺序的假实现，验证排序和合并/导出的编排逻辑，
    不依赖真实 ffmpeg 二进制或真实音频数据。"""
    monkeypatch.chdir(tmp_path)
    src_dir = tmp_path / "in_dir"
    src_dir.mkdir()
    (src_dir / "expert_2000000000.mp3").write_bytes(b"fake-expert")
    (src_dir / "host_1000000000.mp3").write_bytes(b"fake-host")
    (src_dir / "learner_1500000000.mp3").write_bytes(b"fake-learner")

    from funpaper.podcast import audio_gen

    class _FakeAudioSegment:
        def __init__(self, parts=None):
            self.parts = parts or []
            self.exported_to = None
            self.exported_format = None

        @classmethod
        def empty(cls):
            return cls([])

        @classmethod
        def from_mp3(cls, path):
            return cls([Path(path).name])

        def __add__(self, other):
            return _FakeAudioSegment(self.parts + other.parts)

        def export(self, output_file, format="mp3"):
            self.exported_to = output_file
            self.exported_format = format
            _FakeAudioSegment.last_exported = self

    with patch.object(audio_gen, "AudioSegment", _FakeAudioSegment):
        audio_gen.merge_mp3_files("in_dir", "out.mp3")

    merged = _FakeAudioSegment.last_exported
    assert merged.parts == [
        "host_1000000000.mp3",
        "learner_1500000000.mp3",
        "expert_2000000000.mp3",
    ]
    assert merged.exported_to == "out.mp3"
    assert merged.exported_format == "mp3"


# ---------------------------------------------------------------------------
# 5. 顶层编排函数 paper_to_podcast：mock 掉 LLM 和 TTS 相关的一切
# ---------------------------------------------------------------------------


def test_paper_to_podcast_orchestration_with_all_external_calls_mocked(tmp_path):
    """paper_to_podcast 需要真实的 DEEPSEEK_API_KEY / OPENAI_API_KEY 才能真正跑
    通（funai.llm.get_model 内部会读取密钥配置），这里把 get_model / ChatOpenAI /
    generate_script / generate_podcast 全部打桩，只验证编排逻辑本身可以正常
    跑完一遍而不抛异常、不触发任何真实网络调用。"""
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    with patch("funpaper.podcast.command.get_model") as mock_get_model, patch(
        "funpaper.podcast.command.ChatOpenAI"
    ) as mock_chat_openai, patch(
        "funpaper.podcast.command.generate_script", return_value="FAKE SCRIPT"
    ) as mock_generate_script, patch(
        "funpaper.podcast.command.generate_podcast"
    ) as mock_generate_podcast:
        mock_get_model.return_value = MagicMock(name="fake_deepseek_client")
        mock_chat_openai.return_value = MagicMock(name="fake_llm")

        from funpaper.podcast.command import paper_to_podcast

        paper_to_podcast(str(fake_pdf))

    mock_get_model.assert_called_once_with("deepseek")
    mock_chat_openai.assert_called_once()
    mock_generate_script.assert_called_once()
    mock_generate_podcast.assert_called_once_with(
        "FAKE SCRIPT", mock_get_model.return_value
    )


def test_initialize_discussion_chain_builds_working_rag_chain_offline(tmp_path):
    """initialize_discussion_chain 内部会构造 OpenAIEmbeddings() 并调用
    Chroma.from_documents(...) 做真实的 embedding API 调用。这里把两者替换为
    假实现（Chroma.from_documents 返回一个假 vectorstore，其 as_retriever()
    返回一个真正的 LangChain RunnableLambda，以保证 LCEL 的 `|` 组合正常工作），
    LLM 也换成 RunnableLambda，从而在完全离线、不依赖真实凭据的情况下验证
    返回的链可以正确 invoke 并产出字符串结果。"""
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    txt_file = tmp_path / "paper.txt"
    txt_file.write_text("hello world, this is fake paper content.", encoding="utf-8")

    fake_vectorstore = MagicMock()
    fake_vectorstore.as_retriever.return_value = RunnableLambda(
        lambda _query: [MagicMock(page_content="相关片段一")]
    )
    fake_llm = RunnableLambda(lambda _prompt_value: AIMessage(content="fake reply"))

    with patch("funpaper.podcast.script.OpenAIEmbeddings"), patch(
        "funpaper.podcast.script.Chroma"
    ) as mock_chroma:
        mock_chroma.from_documents.return_value = fake_vectorstore

        from funpaper.podcast.script import initialize_discussion_chain

        chain = initialize_discussion_chain(str(txt_file), fake_llm)
        result = chain.invoke(
            {"section_plan": "# Section 1", "previous_dialogue": "Host: hi"}
        )

    mock_chroma.from_documents.assert_called_once()
    assert result == "fake reply"


# ---------------------------------------------------------------------------
# 6. CLI 入口
# ---------------------------------------------------------------------------


def test_cli_entry_point_help_exits_cleanly():
    """[project.scripts] 里声明的 funpaper 命令行入口，--help 不应触发任何真实
    网络调用，也不应要求任何环境变量/凭据。"""
    funpaper_bin = Path(sys.executable).parent / "funpaper"
    assert funpaper_bin.exists(), f"未找到 CLI 可执行文件: {funpaper_bin}"

    result = subprocess.run(
        [str(funpaper_bin), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0
    assert "Usage" in result.stdout


def test_cli_podcast_subcommand_help_exits_cleanly():
    funpaper_bin = Path(sys.executable).parent / "funpaper"
    assert funpaper_bin.exists(), f"未找到 CLI 可执行文件: {funpaper_bin}"

    result = subprocess.run(
        [str(funpaper_bin), "podcast", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0
    assert "pdf_path" in result.stdout
