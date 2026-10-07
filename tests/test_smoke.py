"""烟雾测试（smoke tests）

funpaper 是一个"论文 PDF -> LangChain 生成播客脚本 -> TTS 合成语音"的命令行工具。
本测试套件的目标是在完全离线（不发起任何真实网络/LLM/TTS 调用）的前提下，验证：

1. 顶层包及各子模块可以正常 import；
2. PDF 解析相关的纯逻辑函数（parse_pdf / get_head / parse_script_plan）在真实样例
   PDF（仓库自带的 paper/2017/*.pdf）上工作正常；
3. 涉及真实 LLM（DeepSeek / OpenAI via langchain-openai）和 TTS（OpenAI audio.speech）
   调用的函数，使用 unittest.mock 打桩后可以被安全地调用一次，验证其编排逻辑
   （而不验证真实模型输出）；
4. CLI 入口 `funpaper` 的 `--help`、缺少/无效 `--pdf_path` 均能以预期退出码退出；
5. 边界与失败路径：PDF 缺少 "Conclusion"/"Introduction" 小节、脚本为空或不含
   任何角色标记、同一秒内重复调用导致的文件名/输出目录冲突。

不修复业务逻辑 bug（若审计中发现真实 bug，会在对应 commit 中单独说明并修复）；
如果某个函数无法在不改动源码的情况下被安全 mock 测试，则显式 skip 并说明原因。
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


# ---------------------------------------------------------------------------
# 4. 脚本生成编排：mock 掉 PDF、向量检索和 LLM 调用
# ---------------------------------------------------------------------------


def test_generate_script_orchestrates_multiple_sections(tmp_path, monkeypatch):
    """多小节大纲应按顺序传递上一段对话，并在最后统一润色。"""
    from funpaper.podcast.script import generate_script

    monkeypatch.chdir(tmp_path)
    plan_chain = MagicMock()
    plan_chain.invoke.return_value = ["# First", "# Second"]
    initial_chain = MagicMock()
    initial_chain.invoke.return_value = "Host: opening\n"
    enhance_chain = MagicMock()
    enhance_chain.invoke.return_value = "Host: polished"
    discuss_chain = MagicMock()
    discuss_chain.invoke.side_effect = ["Expert: first\n", "Learner: second\n"]
    chains = {
        "plan_script_chain": plan_chain,
        "initial_dialogue_chain": initial_chain,
        "enhance_chain": enhance_chain,
    }
    llm = MagicMock()

    def fake_parse_pdf(pdf_path, output_path):
        Path(output_path).write_text("paper body", encoding="utf-8")
        return output_path

    with patch("funpaper.podcast.script.parse_pdf", side_effect=fake_parse_pdf) as mock_parse_pdf, patch(
        "funpaper.podcast.script.get_head", return_value="paper head"
    ) as mock_get_head, patch(
        "funpaper.podcast.script.initialize_discussion_chain", return_value=discuss_chain
    ) as mock_initialize:
        result = generate_script("paper.pdf", chains, llm)

    assert result == "Host: polished"
    generated_text_path = mock_parse_pdf.call_args.args[1]
    assert generated_text_path.startswith("text_paper_")
    assert generated_text_path.endswith(".txt")
    plan_chain.invoke.assert_called_once_with({"paper": "paper body"})
    mock_get_head.assert_called_once_with("paper.pdf")
    initial_chain.invoke.assert_called_once_with({"paper_head": "paper head"})
    mock_initialize.assert_called_once_with(generated_text_path, llm)
    assert discuss_chain.invoke.call_args_list[0].args == (
        {"section_plan": "# First", "previous_dialogue": "Host: opening\n"},
    )
    assert discuss_chain.invoke.call_args_list[1].args == (
        {"section_plan": "# Second", "previous_dialogue": "Expert: first\n"},
    )
    enhance_chain.invoke.assert_called_once_with(
        {"draft_script": "Host: opening\nExpert: first\nLearner: second\n"}
    )


def test_generate_script_with_empty_plan_skips_section_generation(tmp_path, monkeypatch):
    """空大纲仍应生成并润色开场，但不能调用分段讨论链。"""
    from funpaper.podcast.script import generate_script

    monkeypatch.chdir(tmp_path)
    plan_chain = MagicMock()
    plan_chain.invoke.return_value = []
    initial_chain = MagicMock()
    initial_chain.invoke.return_value = "Host: opening"
    enhance_chain = MagicMock()
    enhance_chain.invoke.return_value = "Host: polished"
    discuss_chain = MagicMock()
    chains = {
        "plan_script_chain": plan_chain,
        "initial_dialogue_chain": initial_chain,
        "enhance_chain": enhance_chain,
    }
    llm = MagicMock()

    def fake_parse_pdf(pdf_path, output_path):
        Path(output_path).write_text("paper body", encoding="utf-8")
        return output_path

    with patch("funpaper.podcast.script.parse_pdf", side_effect=fake_parse_pdf), patch(
        "funpaper.podcast.script.get_head", return_value="paper head"
    ), patch(
        "funpaper.podcast.script.initialize_discussion_chain", return_value=discuss_chain
    ):
        result = generate_script("paper.pdf", chains, llm)

    assert result == "Host: polished"
    discuss_chain.invoke.assert_not_called()
    enhance_chain.invoke.assert_called_once_with({"draft_script": "Host: opening"})


def test_parse_pdf_without_conclusion_keeps_collecting_full_text(tmp_path):
    """边界场景：论文没有 "Conclusion" 小节时，`collecting` 标志不会被置为
    False，应收集全部页面文本，而不是静默产出空结果或抛异常。"""
    from funpaper.podcast.script import parse_pdf

    fake_pages = [MagicMock(), MagicMock()]
    fake_pages[0].extract_text.return_value = "Page one content, no magic word."
    fake_pages[1].extract_text.return_value = "Page two content, still nothing."

    output_path = tmp_path / "extracted.txt"
    with patch("funpaper.podcast.script.PdfReader") as mock_reader:
        mock_reader.return_value.pages = fake_pages
        result_path = parse_pdf("fake.pdf", str(output_path))

    assert result_path == str(output_path)
    content = output_path.read_text(encoding="utf-8")
    assert "Page one content" in content
    assert "Page two content" in content


def test_get_head_without_introduction_returns_full_text(tmp_path):
    """边界场景：论文没有 "Introduction" 小节时，应返回全部已收集文本，
    而不是空字符串或抛异常。"""
    from funpaper.podcast.script import get_head

    fake_pages = [MagicMock(), MagicMock()]
    fake_pages[0].extract_text.return_value = "Abstract without the magic word."
    fake_pages[1].extract_text.return_value = "More content, still no heading."

    with patch("funpaper.podcast.script.PdfReader") as mock_reader:
        mock_reader.return_value.pages = fake_pages
        head = get_head("fake.pdf")

    assert "Abstract without the magic word" in head
    assert "More content, still no heading" in head


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


class _FakeAudioSegment:
    """记录合并顺序的假 `AudioSegment`，不依赖真实 ffmpeg 二进制或音频数据。"""

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


def test_generate_podcast_with_empty_script_produces_no_segments(tmp_path, monkeypatch):
    """失败/边界路径：脚本为空字符串或不含任何 `Host:`/`Learner:`/`Expert:`
    标记时，正则匹配不到台词，三个角色的 TTS 函数都不应被调用，但仍应正常
    走到合并步骤（即便合并出的是空音频），而不是抛未处理异常。"""
    monkeypatch.chdir(tmp_path)

    with patch("funpaper.podcast.audio_gen.generate_host") as mock_host, patch(
        "funpaper.podcast.audio_gen.generate_expert"
    ) as mock_expert, patch(
        "funpaper.podcast.audio_gen.generate_learner"
    ) as mock_learner, patch(
        "funpaper.podcast.audio_gen.merge_mp3_files"
    ) as mock_merge:
        from funpaper.podcast.audio_gen import generate_podcast

        generate_podcast("", MagicMock())

    mock_host.assert_not_called()
    mock_learner.assert_not_called()
    mock_expert.assert_not_called()
    mock_merge.assert_called_once()


def test_merge_mp3_files_sorts_by_sequence_and_merges_in_order(tmp_path, monkeypatch):
    """merge_mp3_files 本身只负责按文件名中的序号排序、依次合并并导出。文件名
    格式为 `<角色>_<时间戳>_<序号>.mp3`（见 `_segment_filename`）；这里构造两个
    时间戳相同但序号不同的文件，验证排序以序号（而非时间戳或文件系统遍历顺序）
    为准——这正是修复「同一秒内同角色多段语音覆盖」问题时引入的排序键。"""
    monkeypatch.chdir(tmp_path)
    src_dir = tmp_path / "in_dir"
    src_dir.mkdir()
    (src_dir / "expert_1000000000_000002.mp3").write_bytes(b"fake-expert")
    (src_dir / "host_1000000000_000000.mp3").write_bytes(b"fake-host")
    (src_dir / "learner_1000000000_000001.mp3").write_bytes(b"fake-learner")

    from funpaper.podcast import audio_gen

    with patch.object(audio_gen, "AudioSegment", _FakeAudioSegment):
        audio_gen.merge_mp3_files("in_dir", "out.mp3")

    merged = _FakeAudioSegment.last_exported
    assert merged.parts == [
        "host_1000000000_000000.mp3",
        "learner_1000000000_000001.mp3",
        "expert_1000000000_000002.mp3",
    ]
    assert merged.exported_to == "out.mp3"
    assert merged.exported_format == "mp3"


def test_generate_host_twice_within_same_second_uses_distinct_filenames(
    tmp_path, monkeypatch
):
    """回归测试：修复前 `generate_host` 用秒级时间戳命名文件，同一秒内连续调用
    两次会生成同名文件，第二段台词覆盖第一段，最终合并音频丢词。本测试在不
    打桩时间的情况下连续调用两次（单元测试耗时远小于 1 秒，天然触发该场景），
    断言写入的两个文件名互不相同。"""
    from funpaper.podcast.audio_gen import generate_host

    monkeypatch.chdir(tmp_path)
    fake_client, fake_response = _make_fake_tts_client()

    generate_host("line one", fake_client, "out_dir")
    generate_host("line two", fake_client, "out_dir")

    calls = fake_response.stream_to_file.call_args_list
    assert len(calls) == 2
    filenames = [call.args[0] for call in calls]
    assert filenames[0] != filenames[1], "同一秒内两段台词的文件名不应相同"


def test_generate_podcast_twice_within_same_second_uses_distinct_output_dirs(
    tmp_path, monkeypatch
):
    """回归测试：修复前 `generate_podcast` 的输出目录名只精确到秒，同一秒内
    连续调用两次会 `os.mkdir` 到同名目录而抛 `FileExistsError`。"""
    monkeypatch.chdir(tmp_path)
    script = "Host: hi\n"

    with (
        patch("funpaper.podcast.audio_gen.generate_host"),
        patch("funpaper.podcast.audio_gen.merge_mp3_files"),
    ):
        from funpaper.podcast.audio_gen import generate_podcast

        fake_client = MagicMock()
        generate_podcast(script, fake_client)
        generate_podcast(script, fake_client)  # 不应抛 FileExistsError

    podcast_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(podcast_dirs) == 2


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


def test_cli_podcast_missing_pdf_path_fails_with_nonzero_exit(tmp_path):
    """失败路径：`--pdf_path` 指向不存在的文件时，Click 的
    `click.Path(exists=True)` 应在参数解析阶段直接报错退出，不应把 `None`
    或不存在的路径透传给下游的 `PdfReader`。"""
    funpaper_bin = Path(sys.executable).parent / "funpaper"
    assert funpaper_bin.exists(), f"未找到 CLI 可执行文件: {funpaper_bin}"

    missing_path = tmp_path / "does-not-exist.pdf"
    result = subprocess.run(
        [str(funpaper_bin), "podcast", "--pdf_path", str(missing_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode != 0
    assert (
        "does-not-exist.pdf" in result.stderr or "does-not-exist.pdf" in result.stdout
    )


def test_cli_podcast_requires_pdf_path_option():
    """失败路径：缺少必填的 `--pdf_path` 选项时应非零退出并给出清晰提示。"""
    funpaper_bin = Path(sys.executable).parent / "funpaper"
    assert funpaper_bin.exists(), f"未找到 CLI 可执行文件: {funpaper_bin}"

    result = subprocess.run(
        [str(funpaper_bin), "podcast"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode != 0
    assert "pdf_path" in result.stderr or "pdf_path" in result.stdout
