import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from example_run import (
    build_dialogue_episodes_from_lines,
    clean_qq_dialogue_line,
    effective_dialogue_train_ticks,
    is_qq_dialogue_boundary_line,
    load_dialogue_training_chunked_root,
    load_dialogue_training_chunked_jsonl,
    load_dialogue_training_corpus_jsonl,
    load_dialogue_training_haid_jsonl,
    load_dialogue_training_jsonl,
    main,
    run_dialogue_training_chunked_root_streaming,
    run_dialogue_training,
)
from uva_model.dialogue import CognitiveDialogueAgent
from uva_model.tokenizer import PrecisionTokenizer
from uva_model.word_imprints import WordStateImprint


class DialogueTrainingTests(unittest.TestCase):
    def test_train_fast_internal_surprise_hist_cap_zero_means_no_truncation(self) -> None:
        tokenizer = PrecisionTokenizer()
        agent = CognitiveDialogueAgent(
            tokenizer,
            train_fast_internal_surprise_hist_cap=0,
        )
        self.assertIsNone(agent.train_fast_internal_surprise_hist_cap)
        agent_neg = CognitiveDialogueAgent(
            tokenizer,
            train_fast_internal_surprise_hist_cap=-3,
        )
        self.assertIsNone(agent_neg.train_fast_internal_surprise_hist_cap)
        agent_k = CognitiveDialogueAgent(
            tokenizer,
            train_fast_internal_surprise_hist_cap=8,
        )
        self.assertEqual(agent_k.train_fast_internal_surprise_hist_cap, 8)

    def test_effective_dialogue_train_ticks_slow_final_tail(self) -> None:
        bt_s, pt_s = 9, 8
        bt_f, pt_f = 1, 2
        self.assertEqual(
            effective_dialogue_train_ticks(
                global_one_based_episode_index=7,
                schedule_total_episodes=10,
                slow_final_episodes=3,
                between_turn_ticks=bt_f,
                post_episode_ticks=pt_f,
                between_turn_ticks_slow=bt_s,
                post_episode_ticks_slow=pt_s,
            ),
            (bt_f, pt_f),
        )
        self.assertEqual(
            effective_dialogue_train_ticks(
                global_one_based_episode_index=8,
                schedule_total_episodes=10,
                slow_final_episodes=3,
                between_turn_ticks=bt_f,
                post_episode_ticks=pt_f,
                between_turn_ticks_slow=bt_s,
                post_episode_ticks_slow=pt_s,
            ),
            (bt_s, pt_s),
        )

    def test_run_dialogue_training_slow_final_episodes_adds_post_ticks(self) -> None:
        eps = [[f"t{i}"] for i in range(5)]
        tok_b = PrecisionTokenizer()
        tok_b.fit(["你好 世界", "今天 天气"])
        agent_b = CognitiveDialogueAgent(tok_b, learn_tokenizer_from_user=True)
        baseline = run_dialogue_training(
            agent_b,
            eps,
            epochs=1,
            between_turn_ticks=0,
            post_episode_ticks=0,
            shuffle=False,
            training_fast_path=True,
            skip_internal_efe=True,
            slow_final_episodes=0,
        )
        tok_s = PrecisionTokenizer()
        tok_s.fit(["你好 世界", "今天 天气"])
        agent_s = CognitiveDialogueAgent(tok_s, learn_tokenizer_from_user=True)
        slowed = run_dialogue_training(
            agent_s,
            eps,
            epochs=1,
            between_turn_ticks=0,
            post_episode_ticks=0,
            shuffle=False,
            training_fast_path=True,
            skip_internal_efe=True,
            slow_final_episodes=2,
            between_turn_ticks_slow=0,
            post_episode_ticks_slow=3,
        )
        self.assertEqual(baseline["internal_ticks"], 0)
        self.assertEqual(slowed["internal_ticks"], 6)

    def test_clean_qq_dialogue_line_filters_obvious_noise(self) -> None:
        self.assertEqual(clean_qq_dialogue_line("   "), "")
        self.assertEqual(clean_qq_dialogue_line("https://example.com"), "")
        self.assertEqual(clean_qq_dialogue_line("!!!!!"), "")
        self.assertEqual(clean_qq_dialogue_line("123456"), "")
        self.assertEqual(clean_qq_dialogue_line("a" * 400), "")
        self.assertEqual(clean_qq_dialogue_line("你撤回了一条消息"), "")
        self.assertEqual(clean_qq_dialogue_line("文件已过期"), "")
        self.assertEqual(clean_qq_dialogue_line("群主跑路了  "), "群主跑路了")

    def test_build_dialogue_episodes_from_lines_cleans_and_chunks(self) -> None:
        lines = [
            "https://example.com",
            "群主跑路了",
            "真的假的",
            "!!!!!",
            "今晚考试",
            "我还在玩游戏",
            "123456",
        ]
        episodes = build_dialogue_episodes_from_lines(lines, episode_size=2)
        self.assertEqual(
            episodes,
            [["群主跑路了", "真的假的"], ["今晚考试", "我还在玩游戏"]],
        )

    def test_build_dialogue_episodes_respects_blank_line_boundaries(self) -> None:
        lines = [
            "群主跑路了",
            "真的假的",
            "",
            "今晚考试",
            "我还在玩游戏",
        ]
        episodes = build_dialogue_episodes_from_lines(lines, episode_size=8)
        self.assertEqual(
            episodes,
            [["群主跑路了", "真的假的"], ["今晚考试", "我还在玩游戏"]],
        )

    def test_build_dialogue_episodes_respects_metadata_boundaries(self) -> None:
        lines = [
            "2024-05-06 13:22:11",
            "群主跑路了",
            "真的假的",
            "2024/05/06 13:30",
            "今晚考试",
            "我还在玩游戏",
        ]
        episodes = build_dialogue_episodes_from_lines(lines, episode_size=8)
        self.assertEqual(
            episodes,
            [["群主跑路了", "真的假的"], ["今晚考试", "我还在玩游戏"]],
        )

    def test_nickname_timestamp_line_is_boundary(self) -> None:
        self.assertTrue(is_qq_dialogue_boundary_line("张三 2024-05-06 13:22:11"))
        self.assertTrue(is_qq_dialogue_boundary_line("Alice 2024/05/06 09:03"))

    def test_build_dialogue_episodes_skips_system_lines(self) -> None:
        lines = [
            "张三 2024-05-06 13:22:11",
            "群主跑路了",
            "你撤回了一条消息",
            "真的假的",
            "李四 2024-05-06 13:25:10",
            "文件已过期",
            "今晚考试",
            "我还在玩游戏",
        ]
        episodes = build_dialogue_episodes_from_lines(lines, episode_size=8)
        self.assertEqual(
            episodes,
            [["群主跑路了", "真的假的"], ["今晚考试", "我还在玩游戏"]],
        )

    def test_load_dialogue_training_corpus_jsonl_windows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "corpus.jsonl"
            lines = [
                {"text": "甲", "meta": {"src": "wiki"}},
                {"text": "乙", "meta": {}},
                {"text": "丙", "meta": {}},
                {"text": "丁", "meta": {}},
                {"text": "戊", "meta": {}},
            ]
            p.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in lines) + "\n",
                encoding="utf-8",
            )
            eps = load_dialogue_training_corpus_jsonl(
                [str(p)],
                episode_size=2,
                min_episode_turns=1,
            )
            self.assertEqual(eps, [["甲", "乙"], ["丙", "丁"], ["戊"]])
            eps_alias = load_dialogue_training_haid_jsonl(
                [str(p)],
                episode_size=2,
                min_episode_turns=1,
            )
            self.assertEqual(eps_alias, eps)

    def test_run_dialogue_training_episode_iter_factory_streams(self) -> None:
        tok = PrecisionTokenizer()
        tok.fit(["你好"])
        agent = CognitiveDialogueAgent(tok, learn_tokenizer_from_user=False)

        def factory():
            yield ["一条"]
            yield ["二条", "三条"]

        metrics = run_dialogue_training(
            agent,
            episode_iter_factory=factory,
            epochs=1,
            shuffle=False,
            training_fast_path=True,
            skip_internal_efe=True,
        )
        self.assertEqual(metrics["episodes"], 2)
        self.assertEqual(metrics["turns"], 3)

    def test_run_dialogue_training_episode_iter_factory_rejects_shuffle(self) -> None:
        tok = PrecisionTokenizer()
        tok.fit(["你好"])
        agent = CognitiveDialogueAgent(tok, learn_tokenizer_from_user=False)

        def factory():
            yield ["x"]

        with self.assertRaises(ValueError):
            run_dialogue_training(
                agent,
                episode_iter_factory=factory,
                epochs=1,
                shuffle=True,
                training_fast_path=True,
                skip_internal_efe=True,
            )

    def test_main_dialogue_train_mode_accepts_corpus_jsonl_format(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            train_path = Path(td) / "corpus.jsonl"
            tok_path = Path(td) / "tok.json"
            dlg_path = Path(td) / "tok.dialogue.json"
            train_path.write_text(
                "\n".join(
                    json.dumps({"text": f"句{i}", "meta": {"i": i}}, ensure_ascii=False)
                    for i in range(4)
                )
                + "\n",
                encoding="utf-8",
            )
            tok = PrecisionTokenizer()
            tok.fit(["你好 世界", "今天 天气"])
            tok.save_model(str(tok_path))
            argv = [
                "example_run.py",
                "--mode",
                "dialogue-train",
                "--tokenizer-model",
                str(tok_path),
                "--dialogue-train-format",
                "corpus_jsonl",
                "--dialogue-train-file",
                str(train_path),
                "--dialogue-train-episode-size",
                "2",
                "--dialogue-train-min-turns",
                "1",
                "--save-tokenizer",
                str(tok_path),
                "--save-dialogue",
                str(dlg_path),
                "--dialogue-train-epochs",
                "1",
            ]
            with patch("sys.argv", argv), redirect_stdout(io.StringIO()):
                main()
            self.assertTrue(dlg_path.is_file())

    def test_load_dialogue_training_jsonl_reads_episode_turns(self) -> None:
        rows = [
            {"theme": "greeting", "turns": ["你好", "你在吗"]},
            {"theme": "gossip", "turns": ["群主跑路了", "真的假的"]},
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dialogue.jsonl"
            path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                encoding="utf-8",
            )
            episodes = load_dialogue_training_jsonl(str(path))
        self.assertEqual(episodes, [["你好", "你在吗"], ["群主跑路了", "真的假的"]])

    def test_load_dialogue_training_chunked_jsonl_filters_noise_and_keeps_text_reply(self) -> None:
        rows = [
            {
                "time": "2026-04-06T13:52:39.000Z",
                "timestamp": 1775483559000,
                "type": "text",
                "system": False,
                "recalled": False,
                "content": {"text": "群主跑路了"},
            },
            {
                "time": "2026-04-06T13:52:41.000Z",
                "timestamp": 1775483561000,
                "type": "text",
                "system": False,
                "recalled": False,
                "content": {"text": "😂😂😂😂"},
            },
            {
                "time": "2026-04-06T13:52:42.000Z",
                "timestamp": 1775483562000,
                "type": "reply",
                "system": False,
                "recalled": False,
                "content": {"text": "[回复消息]@牢大 真的假的"},
            },
            {
                "time": "2026-04-06T13:52:50.000Z",
                "timestamp": 1775483570000,
                "type": "text",
                "system": False,
                "recalled": False,
                "content": {"text": "[图片:abc.jpg]"},
            },
            {
                "time": "2026-04-06T13:55:03.000Z",
                "timestamp": 1775483703000,
                "type": "system",
                "system": False,
                "recalled": True,
                "content": {"text": "用户 撤回了一条消息"},
            },
            {
                "time": "2026-04-06T13:57:11.000Z",
                "timestamp": 1775483831000,
                "type": "text",
                "system": False,
                "recalled": False,
                "content": {"text": "你来给我搬，再给我200块"},
            },
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "chunk_0001.jsonl"
            path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                encoding="utf-8",
            )
            episodes = load_dialogue_training_chunked_jsonl(
                str(path),
                episode_size=8,
                min_episode_turns=1,
                gap_seconds=60,
            )
        self.assertEqual(
            episodes,
            [["群主跑路了", "真的假的"], ["你来给我搬，再给我200块"]],
        )

    def test_main_dialogue_train_mode_accepts_chunked_jsonl(self) -> None:
        rows = [
            {
                "time": "2026-04-06T13:52:39.000Z",
                "timestamp": 1775483559000,
                "type": "text",
                "system": False,
                "recalled": False,
                "content": {"text": "群主跑路了"},
            },
            {
                "time": "2026-04-06T13:52:41.000Z",
                "timestamp": 1775483561000,
                "type": "reply",
                "system": False,
                "recalled": False,
                "content": {"text": "[回复消息]@牢大 真的假的"},
            },
        ]
        with tempfile.TemporaryDirectory() as td:
            chunk_path = Path(td) / "chunk_0001.jsonl"
            tok_path = Path(td) / "tok.json"
            dlg_path = Path(td) / "tok.dialogue.json"
            chunk_path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                encoding="utf-8",
            )
            tok = PrecisionTokenizer()
            tok.fit(["你好 世界", "今天 天气"])
            tok.save_model(str(tok_path))
            argv = [
                "example_run.py",
                "--mode",
                "dialogue-train",
                "--tokenizer-model",
                str(tok_path),
                "--dialogue-train-chunked-jsonl",
                str(chunk_path),
                "--save-tokenizer",
                str(tok_path),
                "--save-dialogue",
                str(dlg_path),
                "--dialogue-train-epochs",
                "1",
            ]
            with patch("sys.argv", argv), redirect_stdout(io.StringIO()):
                main()
            self.assertTrue(tok_path.is_file())
            self.assertTrue(dlg_path.is_file())

    def test_load_dialogue_training_chunked_root_collects_multiple_groups(self) -> None:
        rows_a = [
            {
                "time": "2026-04-06T13:52:39.000Z",
                "timestamp": 1775483559000,
                "type": "text",
                "system": False,
                "recalled": False,
                "content": {"text": "群主跑路了"},
            }
        ]
        rows_b = [
            {
                "time": "2026-04-06T14:52:39.000Z",
                "timestamp": 1775487159000,
                "type": "text",
                "system": False,
                "recalled": False,
                "content": {"text": "今晚考试"},
            }
        ]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for idx, rows in enumerate((rows_a, rows_b), start=1):
                grp = root / f"group_demo_{idx}_chunked_jsonl"
                chunks = grp / "chunks"
                chunks.mkdir(parents=True)
                (grp / "manifest.json").write_text(
                    json.dumps(
                        {
                            "metadata": {"format": "chunked-jsonl"},
                            "chunked": {
                                "chunksDir": "chunks",
                                "chunks": [
                                    {"relativePath": "chunks/chunk_0001.jsonl"}
                                ],
                            },
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                (chunks / "chunk_0001.jsonl").write_text(
                    "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                    encoding="utf-8",
                )
            episodes = load_dialogue_training_chunked_root(str(root), min_episode_turns=1)
        self.assertEqual(episodes, [["群主跑路了"], ["今晚考试"]])

    def test_load_dialogue_training_chunked_root_with_workers_preserves_order(self) -> None:
        rows = [
            [{"timestamp": 1, "type": "text", "system": False, "recalled": False, "content": {"text": "甲"}}],
            [{"timestamp": 2, "type": "text", "system": False, "recalled": False, "content": {"text": "乙"}}],
            [{"timestamp": 3, "type": "text", "system": False, "recalled": False, "content": {"text": "丙"}}],
        ]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for idx, chunk_rows in enumerate(rows, start=1):
                grp = root / f"group_demo_{idx}_chunked_jsonl"
                chunks = grp / "chunks"
                chunks.mkdir(parents=True)
                (grp / "manifest.json").write_text(
                    json.dumps(
                        {
                            "metadata": {"format": "chunked-jsonl"},
                            "chunked": {
                                "chunksDir": "chunks",
                                "chunks": [{"relativePath": "chunks/chunk_0001.jsonl"}],
                            },
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                (chunks / "chunk_0001.jsonl").write_text(
                    "\n".join(json.dumps(r, ensure_ascii=False) for r in chunk_rows),
                    encoding="utf-8",
                )
            episodes = load_dialogue_training_chunked_root(
                str(root),
                min_episode_turns=1,
                workers=3,
            )
        self.assertEqual(episodes, [["甲"], ["乙"], ["丙"]])

    def test_main_dialogue_train_mode_accepts_chunked_root(self) -> None:
        rows = [
            {
                "time": "2026-04-06T13:52:39.000Z",
                "timestamp": 1775483559000,
                "type": "text",
                "system": False,
                "recalled": False,
                "content": {"text": "群主跑路了"},
            },
            {
                "time": "2026-04-06T13:52:41.000Z",
                "timestamp": 1775483561000,
                "type": "reply",
                "system": False,
                "recalled": False,
                "content": {"text": "[回复消息]@牢大 真的假的"},
            },
        ]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "exportsmessage"
            grp = root / "group_demo_chunked_jsonl"
            chunks = grp / "chunks"
            chunks.mkdir(parents=True)
            (grp / "manifest.json").write_text(
                json.dumps(
                    {
                        "metadata": {"format": "chunked-jsonl"},
                        "chunked": {
                            "chunksDir": "chunks",
                            "chunks": [{"relativePath": "chunks/chunk_0001.jsonl"}],
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (chunks / "chunk_0001.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                encoding="utf-8",
            )
            tok_path = Path(td) / "tok.json"
            dlg_path = Path(td) / "tok.dialogue.json"
            tok = PrecisionTokenizer()
            tok.fit(["你好 世界", "今天 天气"])
            tok.save_model(str(tok_path))
            argv = [
                "example_run.py",
                "--mode",
                "dialogue-train",
                "--tokenizer-model",
                str(tok_path),
                "--dialogue-train-chunked-root",
                str(root),
                "--save-tokenizer",
                str(tok_path),
                "--save-dialogue",
                str(dlg_path),
                "--dialogue-train-epochs",
                "1",
            ]
            with patch("sys.argv", argv), redirect_stdout(io.StringIO()):
                main()
            self.assertTrue(tok_path.is_file())
            self.assertTrue(dlg_path.is_file())

    def test_run_dialogue_training_updates_imprints_and_tokenizer(self) -> None:
        tok = PrecisionTokenizer()
        tok.fit(["你好 世界", "今天 天气"])
        agent = CognitiveDialogueAgent(tok, learn_tokenizer_from_user=True)
        before_total_chars = tok.total_chars
        metrics = run_dialogue_training(
            agent,
            [["群主跑路了", "真的假的"], ["今晚考试", "我还在玩游戏"]],
            epochs=2,
            between_turn_ticks=1,
            post_episode_ticks=2,
            shuffle=False,
        )
        self.assertGreater(metrics["episodes"], 0)
        self.assertGreater(metrics["turns"], 0)
        self.assertGreater(metrics["internal_ticks"], 0)
        self.assertGreater(metrics["imprint_tokens"], 0)
        self.assertGreater(tok.total_chars, before_total_chars)
        self.assertGreater(len(agent._word_memory.tokens()), 0)  # noqa: SLF001

    def test_run_dialogue_training_reports_learning_progress(self) -> None:
        tok = PrecisionTokenizer()
        tok.fit(["你好 世界", "今天 天气"])
        agent = CognitiveDialogueAgent(tok, learn_tokenizer_from_user=True)
        buf = io.StringIO()
        metrics = run_dialogue_training(
            agent,
            [["群主跑路了", "真的假的"], ["今晚考试", "我还在玩游戏"]],
            epochs=1,
            shuffle=False,
            progress_every=1,
            device_label="cpu",
            output_fn=lambda *args, **kwargs: print(*args, file=buf, **kwargs),
        )
        out = buf.getvalue()
        self.assertGreater(metrics["episodes"], 0)
        self.assertIn("设备=cpu", out)
        self.assertIn("新增印记词=", out)
        self.assertIn("分支=", out)
        self.assertIn("合并印记=", out)
        self.assertIn("活跃词=", out)
        self.assertIn("休眠词=", out)
        self.assertIn("关联激活词=", out)
        self.assertIn("冷词补扫数=", out)
        self.assertIn("关联探针=", out)

    def test_run_dialogue_training_reports_compaction_and_dormancy_metrics(self) -> None:
        tok = PrecisionTokenizer()
        tok.fit(["你好 世界", "今天 天气"])
        agent = CognitiveDialogueAgent(tok, learn_tokenizer_from_user=True)
        metrics = run_dialogue_training(
            agent,
            [["哈哈", "哈哈"], ["哈哈", "哈哈"]],
            epochs=1,
            shuffle=False,
            between_turn_ticks=1,
            post_episode_ticks=1,
            training_fast_path=True,
            skip_internal_efe=True,
        )
        self.assertIn("merged_imprints", metrics)
        self.assertGreaterEqual(metrics["merged_imprints"], 0)
        self.assertIn("active_tokens", metrics)
        self.assertIn("dormant_tokens", metrics)
        self.assertIn("associated_activated_tokens", metrics)
        self.assertIn("cold_probe_tokens", metrics)
        self.assertIn("associative_probes", metrics)

    def test_run_dialogue_training_batch_tokenizer_updates_keeps_learning_effect(self) -> None:
        tok = PrecisionTokenizer()
        tok.fit(["你好 世界", "今天 天气"])
        agent = CognitiveDialogueAgent(tok, learn_tokenizer_from_user=True)
        before_chars = tok.total_chars
        metrics = run_dialogue_training(
            agent,
            [["群主跑路了", "真的假的"], ["今晚考试", "我还在玩游戏"]],
            epochs=1,
            shuffle=False,
            batch_tokenizer_updates=True,
        )
        self.assertGreater(metrics["turns"], 0)
        self.assertGreater(tok.total_chars, before_chars)

    def test_agent_train_step_updates_imprints_without_full_reply_generation(self) -> None:
        tok = PrecisionTokenizer()
        tok.fit(["你好 世界", "今天 天气"])
        agent = CognitiveDialogueAgent(tok, learn_tokenizer_from_user=False)
        result = agent.train_step("群主跑路了")
        self.assertEqual(result["output_branch"], "train_fast")
        self.assertGreater(result["epsilon_social_in"], 0.0)
        self.assertGreater(len(agent._word_memory.tokens()), 0)  # noqa: SLF001
        self.assertGreater(len(agent.recent_attention_words), 0)

    def test_internal_tick_train_fast_many_skips_cold_scan_tokens(self) -> None:
        """训练快路径内心 tick 不应触发 cold_scan_tokens（十万词下曾周期性全库枚举）。"""
        tok = PrecisionTokenizer()
        tok.fit(["你好 世界", "今天 天气"])
        agent = CognitiveDialogueAgent(tok, learn_tokenizer_from_user=False)
        imp = WordStateImprint(1.0, 1.0, 0.0, 0.1, 0.1)
        for w in ("种子词", "一跳邻"):
            agent._word_memory.record(w, imp)  # noqa: SLF001
        agent._word_memory.record("种子词", WordStateImprint(1.0, 1.0, 0.0, 0.1, 0.1, "", "一跳邻"))  # noqa: SLF001
        agent._record_recent_attention(["种子词"], "种子词")  # noqa: SLF001
        calls = 0
        orig = agent._word_memory.cold_scan_tokens  # noqa: SLF001

        def wrapped(*args: object, **kwargs: object):
            nonlocal calls
            calls += 1
            return orig(*args, **kwargs)

        agent._word_memory.cold_scan_tokens = wrapped  # type: ignore[method-assign]  # noqa: SLF001
        agent.internal_tick_train_fast_many(3)
        self.assertEqual(calls, 0)

    def test_tokenizer_batch_ingest_matches_single_line_updates(self) -> None:
        lines = ["群主跑路了", "真的假的", "今晚考试", "我还在玩游戏"]
        tok_a = PrecisionTokenizer()
        tok_b = PrecisionTokenizer()
        tok_a.fit(["你好 世界", "今天 天气"])
        tok_b.fit(["你好 世界", "今天 天气"])
        for line in lines:
            tok_a.ingest_interaction_line(line)
        tok_b.ingest_interaction_lines(lines)
        self.assertEqual(tok_a.total_chars, tok_b.total_chars)
        self.assertEqual(tok_a.unigram, tok_b.unigram)
        self.assertEqual(tok_a.bigram, tok_b.bigram)
        self.assertEqual(tok_a.follow_counts, tok_b.follow_counts)

    def test_run_dialogue_training_fast_path_uses_train_step_batch(self) -> None:
        class CountingBatchAgent(CognitiveDialogueAgent):
            def __init__(self, tokenizer: PrecisionTokenizer) -> None:
                super().__init__(tokenizer, learn_tokenizer_from_user=False)
                self.batch_calls: list[list[str]] = []

            def train_step_batch(self, user_texts: list[str]) -> list[dict[str, object]]:
                self.batch_calls.append(list(user_texts))
                return super().train_step_batch(user_texts)

        tok = PrecisionTokenizer()
        tok.fit(["你好 世界", "今天 天气"])
        agent = CountingBatchAgent(tok)
        metrics = run_dialogue_training(
            agent,
            [["群主跑路了", "真的假的", "今晚考试"]],
            epochs=1,
            shuffle=False,
            training_fast_path=True,
            skip_internal_efe=True,
        )
        self.assertEqual(metrics["turns"], 3)
        self.assertTrue(agent.batch_calls)
        self.assertTrue(any(len(batch) > 1 for batch in agent.batch_calls))

    def test_main_dialogue_train_mode_writes_models(self) -> None:
        rows = [
            {"theme": "gossip", "turns": ["群主跑路了", "真的假的"]},
            {"theme": "study", "turns": ["今晚考试", "我还在玩游戏"]},
        ]
        with tempfile.TemporaryDirectory() as td:
            train_path = Path(td) / "dialogue.jsonl"
            tok_path = Path(td) / "tok.json"
            dlg_path = Path(td) / "tok.dialogue.json"
            train_path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                encoding="utf-8",
            )
            tok = PrecisionTokenizer()
            tok.fit(["你好 世界", "今天 天气"])
            tok.save_model(str(tok_path))
            argv = [
                "example_run.py",
                "--mode",
                "dialogue-train",
                "--tokenizer-model",
                str(tok_path),
                "--dialogue-train-file",
                str(train_path),
                "--save-tokenizer",
                str(tok_path),
                "--save-dialogue",
                str(dlg_path),
                "--dialogue-train-epochs",
                "1",
            ]
            with patch("sys.argv", argv), redirect_stdout(io.StringIO()):
                main()
            self.assertTrue(tok_path.is_file())
            self.assertTrue(dlg_path.is_file())

    def test_main_dialogue_train_mode_reports_device_and_progress(self) -> None:
        rows = [
            {"theme": "gossip", "turns": ["哈哈", "哈哈"]},
            {"theme": "study", "turns": ["哈哈", "哈哈"]},
        ]
        with tempfile.TemporaryDirectory() as td:
            train_path = Path(td) / "dialogue.jsonl"
            tok_path = Path(td) / "tok.json"
            dlg_path = Path(td) / "tok.dialogue.json"
            train_path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                encoding="utf-8",
            )
            tok = PrecisionTokenizer()
            tok.fit(["你好 世界", "今天 天气"])
            tok.save_model(str(tok_path))
            argv = [
                "example_run.py",
                "--mode",
                "dialogue-train",
                "--tokenizer-model",
                str(tok_path),
                "--dialogue-train-file",
                str(train_path),
                "--dialogue-train-device",
                "cpu",
                "--dialogue-train-progress-every",
                "1",
                "--save-tokenizer",
                str(tok_path),
                "--save-dialogue",
                str(dlg_path),
                "--dialogue-train-epochs",
                "1",
            ]
            stdout = io.StringIO()
            with patch("sys.argv", argv), redirect_stdout(stdout):
                main()
            out = stdout.getvalue()
            self.assertIn("训练设备: cpu", out)
            self.assertIn("新增印记词=", out)
            self.assertIn("训练路径=train_fast", out)
            self.assertIn("内部EFE=skip", out)
            self.assertIn("合并印记=", out)
            self.assertIn("活跃词=", out)
            self.assertIn("休眠词=", out)
            self.assertIn("关联激活词=", out)
            self.assertIn("冷词补扫数=", out)
            self.assertIn("关联探针=", out)
            self.assertTrue(tok_path.is_file())
            self.assertTrue(dlg_path.is_file())

    def test_run_dialogue_training_chunked_root_streaming_updates_state(self) -> None:
        rows_a = [
            {"timestamp": 1, "type": "text", "system": False, "recalled": False, "content": {"text": "群主跑路了"}},
            {"timestamp": 2, "type": "reply", "system": False, "recalled": False, "content": {"text": "[回复消息]@牢大 真的假的"}},
        ]
        rows_b = [
            {"timestamp": 3, "type": "text", "system": False, "recalled": False, "content": {"text": "今晚考试"}},
            {"timestamp": 4, "type": "text", "system": False, "recalled": False, "content": {"text": "我还在玩游戏"}},
        ]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "exportsmessage"
            for idx, rows in enumerate((rows_a, rows_b), start=1):
                grp = root / f"group_{idx}_chunked_jsonl"
                chunks = grp / "chunks"
                chunks.mkdir(parents=True)
                (grp / "manifest.json").write_text(
                    json.dumps(
                        {
                            "metadata": {"format": "chunked-jsonl"},
                            "chunked": {
                                "chunksDir": "chunks",
                                "chunks": [{"relativePath": "chunks/chunk_0001.jsonl"}],
                            },
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                (chunks / "chunk_0001.jsonl").write_text(
                    "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                    encoding="utf-8",
                )
            tok = PrecisionTokenizer()
            tok.fit(["你好 世界", "今天 天气"])
            agent = CognitiveDialogueAgent(tok, learn_tokenizer_from_user=True)
            metrics = run_dialogue_training_chunked_root_streaming(
                agent,
                str(root),
                epochs=1,
                shuffle=False,
                workers=2,
                prefetch_chunks=1,
                batch_tokenizer_updates=True,
                training_fast_path=True,
            )
        self.assertEqual(metrics["episodes"], 2)
        self.assertGreater(metrics["turns"], 0)
        self.assertGreater(metrics["imprint_tokens"], 0)

    def test_chunked_root_streaming_slow_final_preflights_without_max_episodes(self) -> None:
        rows_a = [
            {"timestamp": 1, "type": "text", "system": False, "recalled": False, "content": {"text": "群主跑路了"}},
            {"timestamp": 2, "type": "reply", "system": False, "recalled": False, "content": {"text": "[回复消息]@牢大 真的假的"}},
        ]
        rows_b = [
            {"timestamp": 3, "type": "text", "system": False, "recalled": False, "content": {"text": "今晚考试"}},
            {"timestamp": 4, "type": "text", "system": False, "recalled": False, "content": {"text": "我还在玩游戏"}},
        ]
        pieces: list[str] = []

        def capture_fn(*args: object, **kwargs: object) -> None:
            pieces.extend(str(a) for a in args)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "exportsmessage"
            for idx, rows in enumerate((rows_a, rows_b), start=1):
                grp = root / f"group_{idx}_chunked_jsonl"
                chunks = grp / "chunks"
                chunks.mkdir(parents=True)
                (grp / "manifest.json").write_text(
                    json.dumps(
                        {
                            "metadata": {"format": "chunked-jsonl"},
                            "chunked": {
                                "chunksDir": "chunks",
                                "chunks": [{"relativePath": "chunks/chunk_0001.jsonl"}],
                            },
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                (chunks / "chunk_0001.jsonl").write_text(
                    "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                    encoding="utf-8",
                )

            tok_b = PrecisionTokenizer()
            tok_b.fit(["你好 世界", "今天 天气"])
            agent_b = CognitiveDialogueAgent(tok_b, learn_tokenizer_from_user=True)
            baseline = run_dialogue_training_chunked_root_streaming(
                agent_b,
                str(root),
                epochs=1,
                shuffle=False,
                workers=1,
                prefetch_chunks=1,
                training_fast_path=True,
                skip_internal_efe=True,
                between_turn_ticks=0,
                post_episode_ticks=0,
                slow_final_episodes=0,
            )

            tok_s = PrecisionTokenizer()
            tok_s.fit(["你好 世界", "今天 天气"])
            agent_s = CognitiveDialogueAgent(tok_s, learn_tokenizer_from_user=True)
            slowed = run_dialogue_training_chunked_root_streaming(
                agent_s,
                str(root),
                epochs=1,
                shuffle=False,
                workers=1,
                prefetch_chunks=1,
                training_fast_path=True,
                skip_internal_efe=True,
                between_turn_ticks=0,
                post_episode_ticks=0,
                slow_final_episodes=1,
                between_turn_ticks_slow=0,
                post_episode_ticks_slow=7,
                output_fn=capture_fn,
            )

        self.assertEqual(baseline["internal_ticks"], 0)
        self.assertEqual(slowed["internal_ticks"], 7)
        self.assertIn("预扫描", "".join(pieces))

    def test_run_dialogue_training_fast_path_skips_internal_efe_emits(self) -> None:
        tok = PrecisionTokenizer()
        tok.fit(["你好 世界", "今天 天气"])
        agent = CognitiveDialogueAgent(tok, learn_tokenizer_from_user=True)
        metrics = run_dialogue_training(
            agent,
            [["群主跑路了", "真的假的"], ["今晚考试", "我还在玩游戏"]],
            epochs=1,
            between_turn_ticks=1,
            post_episode_ticks=2,
            shuffle=False,
            training_fast_path=True,
            skip_internal_efe=True,
        )
        self.assertGreater(metrics["internal_ticks"], 0)
        self.assertEqual(metrics["internal_emits"], 0)

    def test_main_dialogue_train_mode_accepts_raw_qq_text(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            raw_path = Path(td) / "qq.txt"
            tok_path = Path(td) / "tok.json"
            dlg_path = Path(td) / "tok.dialogue.json"
            raw_path.write_text(
                "\n".join(
                    [
                        "https://example.com",
                        "群主跑路了",
                        "真的假的",
                        "今晚考试",
                        "我还在玩游戏",
                    ]
                ),
                encoding="utf-8",
            )
            tok = PrecisionTokenizer()
            tok.fit(["你好 世界", "今天 天气"])
            tok.save_model(str(tok_path))
            argv = [
                "example_run.py",
                "--mode",
                "dialogue-train",
                "--tokenizer-model",
                str(tok_path),
                "--dialogue-train-raw-file",
                str(raw_path),
                "--dialogue-train-episode-size",
                "2",
                "--save-tokenizer",
                str(tok_path),
                "--save-dialogue",
                str(dlg_path),
                "--dialogue-train-epochs",
                "1",
            ]
            with patch("sys.argv", argv), redirect_stdout(io.StringIO()):
                main()
            self.assertTrue(tok_path.is_file())
            self.assertTrue(dlg_path.is_file())


if __name__ == "__main__":
    unittest.main()
