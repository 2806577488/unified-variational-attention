# Unified Variational Attention (UVA)

中文认知对话原型：精度分词器 + 词态印记库 + 预期自由能（EFE）驱动的 `CognitiveDialogueAgent`。

## 功能概览

- **对话**：`example_run.py` 交互 / 批量训练；`dialogue_web_ui.py` Gradio 调参台
- **核心包**：`uva_model/`（`tokenizer`、`dialogue`、`word_imprints`、`model` 等）
- **审计与修复说明**：`docs/AUDIT_FINDINGS_AND_REMEDIATION.md`

## 环境

- Python 3.10+
- Windows / Linux 均可

```bash
cd unified-variational-attention
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -e ".[dev]"         # 或: pip install -r requirements-dev.txt && pip install -e .
pip install -r requirements-web.txt   # 仅需 Web 界面时
```

可选 Cython 加速分词热路径：

```bash
pip install Cython
python setup.py build_ext --inplace
```

## 快速开始

### 测试

```bash
pytest tests/ -q
```

### 交互对话（需先有分词器 JSON）

```bash
python train_tokenizer_from_chunks.py   # 从语料训练分词器，见脚本 --help
python example_run.py --mode dialogue --tokenizer-model tokenizer_zh_from_chunks_v2.json
```

仓库**不包含**大型 `tokenizer_*.json`（体积过大）。请本地训练或使用 Release 附件。

### Web 调参台

```bash
python dialogue_web_ui.py
# 浏览器打开终端显示的地址（默认 http://127.0.0.1:7860）
```

## 目录结构

```
uva_model/           # 库代码
tests/               # 单元测试与审计回归
example_run.py       # CLI：对话 / 训练 / 语料
dialogue_web_ui.py   # Gradio Web UI
train_tokenizer_from_chunks.py
docs/                # 审计与修复文档
```

## 语料与 QQ 导出

结构化 QQ 聊天记录（chunked JSONL + manifest）可通过上层 `corpus_tools` / `prepare_corpus.py` 准备（若你另有 monorepo）。本仓库仅保留 UVA 核心。

## License

未指定许可证时默认保留所有权利；如需开源请自行添加 `LICENSE` 文件。
