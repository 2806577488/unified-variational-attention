from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

from uva_model import PrecisionTokenizer


def iter_jsonl_texts(chunks_dir: Path) -> Iterator[str]:
    for jsonl_path in sorted(chunks_dir.glob("*.jsonl")):
        with jsonl_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = obj.get("content", {})
                if not isinstance(content, dict):
                    continue
                text = content.get("text", "")
                if not isinstance(text, str):
                    continue
                cleaned = text.strip()
                if cleaned:
                    yield cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 chunks 目录下的 JSONL 语料逐行训练分词模型")
    parser.add_argument(
        "--chunks-dir",
        type=str,
        required=True,
        help="chunks 目录路径（内部应包含 chunk_*.jsonl）",
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default="tokenizer_from_chunks.json",
        help="输出分词模型 JSON 路径（训练正常结束或 Ctrl+C 中断时都会写入）",
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default="",
        help="已有模型路径；提供后将基于该模型继续增量训练",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=0,
        help="最多读取多少行，0 表示不限制",
    )
    args = parser.parse_args()

    chunks_dir = Path(args.chunks_dir)
    if not chunks_dir.exists() or not chunks_dir.is_dir():
        raise ValueError(f"无效目录: {chunks_dir}")

    if args.load_model:
        tokenizer = PrecisionTokenizer.load_model(args.load_model)
        print(f"已加载模型: {args.load_model}")
    else:
        tokenizer = PrecisionTokenizer()

    def line_stream() -> Iterator[str]:
        count = 0
        for text in iter_jsonl_texts(chunks_dir):
            if args.max_lines > 0 and count >= args.max_lines:
                break
            count += 1
            if count % 5000 == 0:
                print(f"已读取 {count} 行...")
            yield text

    seen = 0
    interrupted = False
    try:
        if args.load_model:
            seen = tokenizer.partial_fit_stream(line_stream())
        else:
            seen = tokenizer.fit_stream(line_stream())
    except KeyboardInterrupt:
        interrupted = True
        print("\n收到中断，仍将当前分词器写入 --save-model。")
    finally:
        tokenizer.save_model(args.save_model)
        print(f"分词器已保存到: {args.save_model}")

    if interrupted:
        raise SystemExit(130)

    print("训练完成。")
    print(f"训练模式: {'增量训练' if args.load_model else '从零训练'}")
    print(f"训练行数: {seen}")


if __name__ == "__main__":
    main()
