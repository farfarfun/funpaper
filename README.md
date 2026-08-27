# funpaper

论文转播客工具：输入一篇论文 PDF，用 LLM（默认 DeepSeek）先生成「主持人 / 学习者 / 专家」三人访谈式播客脚本，再用 OpenAI TTS 把脚本合成语音并拼接成一个完整的 mp3 播客文件。

## 安装

```bash
pip install funpaper
```

需要配置 `DEEPSEEK_API_KEY`（脚本生成，经 [funai](https://github.com/farfarfun/funai) 调用）和 `OPENAI_API_KEY`（`tts-1` 语音合成）等环境变量。

## 命令行用法

```bash
funpaper podcast --pdf_path /path/to/paper.pdf
```

对应 `funpaper.podcast.command:funpaper`（`pyproject.toml` 中的 `[project.scripts]`），内部依次执行：

1. `funpaper.podcast.script.generate_script`：用 `PyPDF2` 抽取 PDF 正文，按 `plan_prompt` 生成播客大纲，再按大纲逐段生成对话，最后用 `enhance_prompt` 润色成最终脚本（LangChain + `Chroma` 向量检索辅助生成每一段的上下文）。
2. `funpaper.podcast.audio_gen.generate_podcast`：解析脚本中 `Host:` / `Learner:` / `Expert:` 三种角色的台词，分别用 OpenAI TTS（`tts-1` 模型，`alloy` / `fable` / `nova` 三种音色）合成语音片段，保存到 `podcast_<时间戳>/` 目录，再用 `pydub` 按时间顺序合并成 `podcast_<时间戳>.mp3`。

## 作为库使用

```python
from funpaper.podcast.command import paper_to_podcast

paper_to_podcast("/path/to/paper.pdf")
```
