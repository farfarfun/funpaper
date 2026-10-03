# Changelog

## 1.0.9

### 新增

- `dependency-groups.dev` 补充 `funbuild>=1.6.66`：构建、版本递增与发布统一走组织标准的 `funbuild` 流程，不手写发布脚本。
- `tests/` 补充边界与失败路径用例：PDF 缺少 "Conclusion"/"Introduction" 小节、脚本为空或不含角色标记、CLI 缺少/指向不存在的 `--pdf_path`、同一秒内重复调用导致的文件名与输出目录冲突。

### 修复

- README 中 PDF 抽取库名称由 `PyPDF2` 更正为实际使用的 `pypdf`。
- README 安装步骤补充 `pydub` 依赖的系统级 `ffmpeg` 安装前置条件（Linux/macOS/Windows）。
- 修复 `generate_host`/`generate_expert`/`generate_learner` 用秒级时间戳命名语音片段文件，同一角色在同一秒内生成多段台词时后一段会覆盖前一段、导致最终合并音频丢词的问题：改为时间戳 + 单调递增序号组合命名，并同步调整 `merge_mp3_files` 的排序键。
- 修复 `generate_podcast` 的输出目录名只精确到秒、同一秒内重复调用会 `os.mkdir` 冲突抛 `FileExistsError` 的问题：目录名追加短随机后缀。
- GitHub 仓库 description 与实际三人（主持人/学习者/专家）播客脚本实现不一致，已更正为据实描述。

### 变更

- 日志迁移到 `farlog`，移除 `funutil` 依赖。
- `print()` 诊断/进度输出迁移到 `farlog` logger。
- 补全公开函数的类型标注与中文 docstring。
- README 补充组织统一页脚；`pyproject.toml` 补充 `license = "MIT"`。

### 废弃

- 无

> 1.0.9 之前的版本未维护 CHANGELOG，历史变更请参考 `git log`。
