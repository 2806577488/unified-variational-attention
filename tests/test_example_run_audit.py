"""
对 example_run.py 的专项审计：纯函数边界、IO 健壮性、合并指标、安全与死代码路径。

不修改 example_run.py 本体；若断言暴露产品缺陷，应以修复 example_run 为准。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import example_run as er
from uva_model.dialogue import CognitiveDialogueAgent
from uva_model.tokenizer import PrecisionTokenizer


class EffectiveDialogueTrainTicksTests(unittest.TestCase):
    def test_slow_final_first_episode_uses_normal_ticks(self) -> None:
        bt, pt = er.effective_dialogue_train_ticks(
            global_one_based_episode_index=1,
            schedule_total_episodes=10,
            slow_final_episodes=3,
            between_turn_ticks=0,
            post_episode_ticks=1,
            between_turn_ticks_slow=5,
            post_episode_ticks_slow=9,
        )
        self.assertEqual((bt, pt), (0, 1))

    def test_slow_final_boundary_episode_uses_slow_ticks(self) -> None:
        # total=10, sf=3 -> start_slow_at=8
        bt, pt = er.effective_dialogue_train_ticks(
            global_one_based_episode_index=8,
            schedule_total_episodes=10,
            slow_final_episodes=3,
            between_turn_ticks=0,
            post_episode_ticks=1,
            between_turn_ticks_slow=5,
            post_episode_ticks_slow=9,
        )
        self.assertEqual((bt, pt), (5, 9))


class ResolveDialogueModelPathTests(unittest.TestCase):
    def test_fresh_returns_empty(self) -> None:
        self.assertEqual(
            er.resolve_dialogue_model_load_path("a.json", "x.dialogue.json", dialogue_fresh=True),
            "",
        )

    def test_explicit_path_wins(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".dialogue.json", delete=False) as fh:
            p = fh.name
        try:
            self.assertEqual(
                er.resolve_dialogue_model_load_path("a.json", p, dialogue_fresh=False),
                p,
            )
        finally:
            Path(p).unlink(missing_ok=True)


class LoadDialogueTrainingJsonlTests(unittest.TestCase):
    def test_invalid_json_raises(self) -> None:
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".jsonl",
            encoding="utf-8",
            delete=False,
        ) as fh:
            fh.write("not json\n")
            path = fh.name
        try:
            with self.assertRaises(json.JSONDecodeError):
                er.load_dialogue_training_jsonl(path)
        finally:
            Path(path).unlink(missing_ok=True)


class CleanChunkedDialogueTextTests(unittest.TestCase):
    def test_short_alnum_passes_through(self) -> None:
        """含字母数字的短串在 _clean_chunked_dialogue_text 中应保留（与 clean_qq 链式一致）。"""
        self.assertEqual(er._clean_chunked_dialogue_text("ab12"), "ab12")

    def test_dead_branch_after_alnum_gate(self) -> None:
        """
        源码 283 已要求 any(isalnum)；285「len<=4 且全非 alnum」在逻辑上不可达，属死代码 smell。
        此处用行为确认：含 alnum 的四字串不会被 285 误杀。
        """
        self.assertEqual(er._clean_chunked_dialogue_text("a1b2"), "a1b2")


class ChunkedRootStreamingMergedMetricsTests(unittest.TestCase):
    def test_new_imprint_tokens_should_match_total_vocab_growth_across_chunks(self) -> None:
        """
        run_dialogue_training_chunked_root_streaming 应对各 chunk 的 new_imprint 增量累加，
        与词库净增一致。
        """
        tok = PrecisionTokenizer()
        tok.fit(["你好 世界", "测试 语料", "alpha beta"])
        agent = CognitiveDialogueAgent(tok, learn_tokenizer_from_user=False)
        before = int(agent._word_memory.stats()["total_tokens"])

        def fake_jobs(*_a, **_k):
            return [
                ("dummy_a.jsonl", 4, 1, 180),
                ("dummy_b.jsonl", 4, 1, 180),
            ]

        def fake_iter(chunk_jobs, *, workers=1, prefetch_chunks=0):
            yield [["第一轮测试句子"]]
            yield [["第二轮不同内容"]]

        with patch.object(er, "_collect_chunked_root_jobs", fake_jobs):
            with patch.object(er, "_iter_chunk_job_results", fake_iter):
                metrics = er.run_dialogue_training_chunked_root_streaming(
                    agent,
                    "/unused/root",
                    epochs=1,
                    output_fn=lambda *a, **k: None,
                    training_fast_path=True,
                    skip_internal_efe=True,
                    progress_every=0,
                    live_progress=False,
                    workers=1,
                )
        after = int(agent._word_memory.stats()["total_tokens"])
        truth_new = after - before
        self.assertEqual(
            metrics["new_imprint_tokens"],
            truth_new,
            msg="chunked_root_streaming 应对各 chunk 的 new_imprint 增量汇总，与词库净增一致",
        )


class CollectChunkedRootJobsPathTests(unittest.TestCase):
    def test_parent_relative_path_outside_manifest_is_skipped(self) -> None:
        """manifest 中 relativePath 含 .. 逃出目录时被拒绝，不加入 chunk jobs。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sub = root / "export"
            sub.mkdir()
            outside = root / "evil.jsonl"
            outside.write_text(
                '{"type":"text","system":false,"recalled":false,"content":{"text":"x"},"timestamp":1}\n',
                encoding="utf-8",
            )
            (sub / "manifest.json").write_text(
                json.dumps(
                    {
                        "metadata": {"format": "chunked-jsonl"},
                        "chunked": {
                            "chunks": [{"relativePath": "../evil.jsonl"}],
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            jobs = er._collect_chunked_root_jobs(
                str(root),
                episode_size=4,
                min_episode_turns=1,
                gap_seconds=180,
            )
            self.assertEqual(len(jobs), 0)

    def test_valid_chunk_relative_path_is_collected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sub = root / "export"
            sub.mkdir()
            chunk = sub / "chunk_0.jsonl"
            chunk.write_text(
                '{"type":"text","system":false,"recalled":false,"content":{"text":"ab12"},"timestamp":1}\n',
                encoding="utf-8",
            )
            (sub / "manifest.json").write_text(
                json.dumps(
                    {
                        "metadata": {"format": "chunked-jsonl"},
                        "chunked": {"chunks": [{"relativePath": "chunk_0.jsonl"}]},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            jobs = er._collect_chunked_root_jobs(
                str(root),
                episode_size=4,
                min_episode_turns=1,
                gap_seconds=180,
            )
            self.assertEqual(len(jobs), 1)
            self.assertTrue(Path(jobs[0][0]).is_file())


if __name__ == "__main__":
    unittest.main()
