# Unified Variational Attention (UVA)

中文认知对话原型：精度分词器 + 词态印记库 + 预期自由能（EFE）驱动的 `CognitiveDialogueAgent`。

## 功能概览

| 入口 | 说明 |
|------|------|
| `example_run.py` | 命令行：交互对话、批量训练、语料导入 |
| `dialogue_web_ui.py` | Gradio Web 调参台（单轮对话 / 内心一拍 / 偏好反馈） |
| `docs/AUDIT_FINDINGS_AND_REMEDIATION.md` | 性能审计结论与已落地修复记录 |

## 核心包 `uva_model`

Python 包名：`uva_model`。对外主要类型见 `uva_model/__init__.py`。

| 模块 | 文件 | 职责 |
|------|------|------|
| 分词与资源 | `tokenizer.py` | `PrecisionTokenizer`：精度驱动分词、惊奇 trace、R/m 自调节 |
| 对话与 EFE | `dialogue.py` | `CognitiveDialogueAgent`：倾听 → 意图 → 候选稿 → 预期自由能择优；`turn()` 含 C 方案自动反馈 |
| 自动反馈 | `auto_feedback.py` | 无标注行为信号（接纳 / 回避 / 持续）→ `apply_dialogue_feedback` |
| 词态印记 | `word_imprints.py` | `WordStateMemory`：语境向量印记、联想、激活前沿 |
| 变分注意基座 | `model.py` | `UnifiedVariationalAttentionModel`：自由能、精度、容量约束（算术课等示例） |
| 语料 IO | `corpus_jsonl.py` | JSONL 流式读入与 episode 切分 |
| 检查点 | `checkpoint_json.py` | 对话/分词器 JSON 读写（含 gzip） |
| 算术示例 | `arithmetic.py` | 预测编码算术课（与对话主路径独立） |
| 课程与评测 | `curriculum.py` | 训练课程与评测钩子 |
| 可选加速 | `_tokenizer_accel.pyx` | Cython 分词热路径（`setup.py build_ext` 编译） |

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

## 仓库目录

```
unified-variational-attention/
├── uva_model/              # 核心库（见上表）
├── tests/                  # 单元测试与审计回归
├── docs/                   # 设计 / 审计文档
├── example_run.py          # 主 CLI
├── dialogue_web_ui.py      # Web UI
├── train_tokenizer_from_chunks.py
├── compact_uva_checkpoint.py
├── setup.py                # Cython 扩展构建
└── pyproject.toml
```

## 语料与 QQ 导出

结构化 QQ 聊天记录（chunked JSONL + manifest）可通过上层 `corpus_tools` / `prepare_corpus.py` 准备（若你另有 monorepo）。本仓库仅保留 UVA 核心。

## License

未指定许可证时默认保留所有权利；如需开源请自行添加 `LICENSE` 文件。
