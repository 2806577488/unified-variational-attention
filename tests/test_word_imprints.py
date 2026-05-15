import json
import tempfile
import unittest
from pathlib import Path

from uva_model.word_imprints import WordStateImprint, WordStateMemory
from uva_model.word_imprints import torch


class WordImprintsTests(unittest.TestCase):
    def _assert_active_dormant_consistent(self, mem: WordStateMemory) -> None:
        brute_active = 0
        brute_dormant = 0
        for tok in mem._store:
            meta = mem._ensure_meta(tok)
            if meta.is_dormant:
                brute_dormant += 1
            else:
                brute_active += 1
        st = mem.stats()
        self.assertEqual(st["total_tokens"], len(mem._store))
        self.assertEqual(st["active_tokens"], brute_active)
        self.assertEqual(st["dormant_tokens"], brute_dormant)

    def test_incremental_vocab_stats_match_store_meta(self) -> None:
        tiny = WordStateImprint(1, 1, 0, 0.1, 0.1)
        mem = WordStateMemory(capacity=10)
        self._assert_active_dormant_consistent(mem)
        mem.record("甲", tiny)
        mem.record("乙", tiny)
        self._assert_active_dormant_consistent(mem)
        mem.note_trigger_success("甲")
        mem.soft_freeze_cold_tokens(max_active_tokens=1)
        self._assert_active_dormant_consistent(mem)
        mem.note_access("乙")
        self._assert_active_dormant_consistent(mem)

    def test_progress_new_vocab_snapshot_counts_first_imprint_only(self) -> None:
        tiny = WordStateImprint(1, 1, 0, 0.1, 0.1)
        mem = WordStateMemory(capacity=10)
        mem.progress_new_vocab_snapshot()
        mem.record("新甲", tiny)
        mem.record("新乙", tiny)
        mem.record("新甲", tiny)
        n, samples = mem.progress_new_vocab_snapshot()
        self.assertEqual(n, 2)
        self.assertEqual(samples, ["新甲", "新乙"])
        mem.record("新甲", tiny)
        n2, samples2 = mem.progress_new_vocab_snapshot()
        self.assertEqual(n2, 0)
        self.assertEqual(samples2, [])

    def test_empty_memory_zero_surprise(self) -> None:
        mem = WordStateMemory(capacity=10)
        cur = WordStateImprint(1.0, 0.5, 0.1, 0.6, 0.3)
        self.assertEqual(mem.semantic_surprise_for_token("你好", cur), 0.0)

    def test_surprise_after_record(self) -> None:
        mem = WordStateMemory(capacity=100)
        old = WordStateImprint(F_ema=2.0, R=0.9, m=0.0, u_curiosity=0.2, u_task=0.1)
        mem.record("游戏", old)
        cur = WordStateImprint(F_ema=8.0, R=0.1, m=0.8, u_curiosity=1.0, u_task=0.9)
        s = mem.semantic_surprise_for_token("游戏", cur)
        self.assertGreater(s, 0.5)

    def test_semantic_surprise_hist_scan_limit_tail_only(self) -> None:
        """末尾截断时若最优印记仅在序列前端，惊奇应高于全量最小。"""
        mem = WordStateMemory(capacity=200)
        near_cur = WordStateImprint(
            F_ema=2.0, R=0.5, m=0.1, u_curiosity=0.2, u_task=0.2
        )
        far = WordStateImprint(F_ema=9.0, R=0.1, m=0.9, u_curiosity=1.0, u_task=1.0)
        cur = WordStateImprint(
            F_ema=2.05, R=0.5, m=0.1, u_curiosity=0.2, u_task=0.2
        )
        mem.record("tok", near_cur)
        for _ in range(40):
            mem.record("tok", far)
        full = mem.semantic_surprise_for_token("tok", cur)
        tail1 = mem.semantic_surprise_for_token("tok", cur, hist_scan_limit=1)
        self.assertLess(full, tail1)

    def test_tokens_returns_recorded_surface_tokens(self) -> None:
        mem = WordStateMemory(capacity=10)
        mem.record("游戏", WordStateImprint(1, 1, 0, 0.1, 0.1))
        mem.record("跑路", WordStateImprint(1, 1, 0, 0.1, 0.1))
        self.assertEqual(mem.tokens(), ["游戏", "跑路"])

    def test_merge_record_does_not_invalidate_torch_cache_version(self) -> None:
        mem = WordStateMemory(capacity=10)
        imp = WordStateImprint(
            1.0,
            1.0,
            0.0,
            0.2,
            0.2,
            context_before="a",
            context_after="b",
        )
        mem.record("k", imp)
        v0 = mem._cache_version
        mem.record("k", imp)
        self.assertEqual(mem._cache_version, v0)
        mem.record("new", WordStateImprint(2.0, 1.0, 0.0, 0.1, 0.1))
        self.assertGreater(mem._cache_version, v0)

    def test_top_surprises_returns_highest_global_mismatches(self) -> None:
        mem = WordStateMemory(capacity=10)
        mem.record("稳", WordStateImprint(F_ema=1.0, R=0.6, m=0.1, u_curiosity=0.1, u_task=0.1))
        mem.record("裂", WordStateImprint(F_ema=1.0, R=0.6, m=0.1, u_curiosity=0.1, u_task=0.1))
        current = WordStateImprint(F_ema=9.0, R=0.1, m=0.9, u_curiosity=1.0, u_task=0.9)
        top = mem.top_surprises(current, k=1)
        self.assertEqual(len(top), 1)
        self.assertIn(top[0][1], {"稳", "裂"})
        self.assertGreater(top[0][0], 0.0)

    def test_top_surprises_respects_exclude_and_empty_memory(self) -> None:
        empty = WordStateMemory(capacity=10)
        current = WordStateImprint(1, 1, 0, 0.1, 0.1)
        self.assertEqual(empty.top_surprises(current, k=3), [])

        mem = WordStateMemory(capacity=10)
        mem.record("游戏", WordStateImprint(1, 1, 0, 0.1, 0.1))
        mem.record("考试", WordStateImprint(1, 1, 0, 0.1, 0.1))
        top = mem.top_surprises(current, k=3, exclude={"游戏"})
        self.assertEqual([tok for _surp, tok in top], ["考试"])

    def test_hot_tokens_prioritize_trigger_success_then_recent_access(self) -> None:
        mem = WordStateMemory(capacity=10)
        for tok in ("冷", "热", "旧"):
            mem.record(tok, WordStateImprint(1, 1, 0, 0.1, 0.1))
        mem.note_access("旧")
        mem.note_access("热")
        mem.note_trigger_success("热")
        mem.note_access("冷")
        self.assertEqual(mem.hot_tokens(limit=3), ["热", "冷", "旧"])

    def test_cold_scan_tokens_rotates_shards(self) -> None:
        mem = WordStateMemory(capacity=10)
        for tok in ("甲", "乙", "丙", "丁"):
            mem.record(tok, WordStateImprint(1, 1, 0, 0.1, 0.1))
        part0 = set(mem.cold_scan_tokens(limit=10, shard_count=2))
        part1 = set(mem.cold_scan_tokens(limit=10, shard_count=2))
        self.assertNotEqual(part0, part1)
        self.assertEqual(part0 | part1, {"甲", "乙", "丙", "丁"})

    def test_top_surprises_lazy_combines_recent_hot_and_cold_tiers(self) -> None:
        mem = WordStateMemory(capacity=10)
        mem.record("近", WordStateImprint(1, 1, 0, 0.1, 0.1))
        mem.record("热", WordStateImprint(5, 0.1, 0.8, 1.0, 1.0))
        mem.record("冷1", WordStateImprint(4, 0.2, 0.7, 0.8, 0.8))
        mem.record("冷2", WordStateImprint(3, 0.3, 0.6, 0.7, 0.7))
        mem.note_access("热")
        mem.note_trigger_success("热")
        current = WordStateImprint(9, 0.1, 0.9, 1.0, 1.0)
        out = mem.top_surprises_lazy(
            current,
            k=8,
            recent_tokens=["近"],
            pending_tokens={"挂起"},
            hot_budget=1,
            cold_budget=1,
            shard_count=8,
        )
        got = [tok for _surp, tok in out]
        self.assertIn("近", got)
        self.assertIn("热", got)
        self.assertLessEqual(len(got), 3)

    def test_record_merges_near_identical_imprints_but_preserves_association_counts(self) -> None:
        mem = WordStateMemory(capacity=10)
        mem.record("跑路", WordStateImprint(1, 1, 0, 0.1, 0.1, "群", "封禁"))
        mem.record("跑路", WordStateImprint(1, 1, 0, 0.1, 0.1, "群", "封禁"))
        self.assertEqual(len(mem._store.get("跑路", [])), 1)  # noqa: SLF001
        assoc = mem.associated_tokens("跑路", k=3)
        self.assertIn(("群", 2), assoc)
        self.assertIn(("封禁", 2), assoc)

    def test_soft_freeze_cold_tokens_excludes_them_from_active_scan(self) -> None:
        mem = WordStateMemory(capacity=10)
        for tok in ("热1", "热2", "冷1", "冷2"):
            mem.record(tok, WordStateImprint(1, 1, 0, 0.1, 0.1))
        mem.note_trigger_success("热1")
        mem.note_access("热2")
        frozen = set(mem.soft_freeze_cold_tokens(max_active_tokens=2))
        self.assertEqual(frozen, {"冷1", "冷2"})
        self.assertEqual(set(mem.hot_tokens(limit=10)), {"热1", "热2"})
        self.assertEqual(set(mem.cold_scan_tokens(limit=10, shard_count=1)), {"热1", "热2"})

    def test_note_access_reactivates_soft_frozen_token(self) -> None:
        mem = WordStateMemory(capacity=10)
        for tok in ("热1", "热2", "冷1", "冷2"):
            mem.record(tok, WordStateImprint(1, 1, 0, 0.1, 0.1))
        mem.note_trigger_success("热1")
        mem.note_access("热2")
        mem.soft_freeze_cold_tokens(max_active_tokens=2)
        mem.note_access("冷1")
        self.assertIn("冷1", mem.hot_tokens(limit=10))

    def test_top_surprises_lazy_applies_soft_freeze_and_reactivates_recent_token(self) -> None:
        mem = WordStateMemory(capacity=10)
        mem.record("热", WordStateImprint(1, 1, 0, 0.1, 0.1))
        mem.record("冷1", WordStateImprint(8, 0.1, 0.9, 1.0, 1.0))
        mem.record("冷2", WordStateImprint(7, 0.1, 0.8, 0.9, 0.9))
        mem.note_trigger_success("热")
        current = WordStateImprint(9, 0.1, 0.9, 1.0, 1.0)
        first = mem.top_surprises_lazy(
            current,
            k=4,
            hot_budget=1,
            cold_budget=4,
            shard_count=1,
            max_active_tokens=1,
        )
        self.assertEqual([tok for _surp, tok in first], ["热"])
        second = mem.top_surprises_lazy(
            current,
            k=4,
            recent_tokens=["冷1"],
            hot_budget=1,
            cold_budget=4,
            shard_count=1,
            max_active_tokens=1,
        )
        self.assertIn("冷1", [tok for _surp, tok in second])

    def test_activation_frontier_includes_one_hop_associates(self) -> None:
        mem = WordStateMemory(capacity=10)
        mem.record("种子", WordStateImprint(1, 1, 0, 0.1, 0.1, "", "一跳"))
        mem.record("种子", WordStateImprint(1, 1, 0, 0.1, 0.1, "", "一跳"))
        mem.record("一跳", WordStateImprint(1, 1, 0, 0.1, 0.1))
        out = mem.activation_frontier_tokens(
            recent_tokens=["种子"],
            pending_tokens=set(),
            limit=8,
            one_hop_budget=2,
            two_hop_budget=0,
            cold_budget=0,
            shard_count=1,
        )
        self.assertEqual(out[:2], ["种子", "一跳"])

    def test_activation_frontier_limits_two_hop_and_keeps_small_cold_shard(self) -> None:
        mem = WordStateMemory(capacity=10)
        mem.record("种子", WordStateImprint(1, 1, 0, 0.1, 0.1, "", "一跳甲"))
        mem.record("种子", WordStateImprint(1, 1, 0, 0.1, 0.1, "", "一跳乙"))
        mem.record("冷词", WordStateImprint(1, 1, 0, 0.1, 0.1))
        mem.record("一跳甲", WordStateImprint(1, 1, 0, 0.1, 0.1, "", "二跳甲"))
        mem.record("一跳乙", WordStateImprint(1, 1, 0, 0.1, 0.1, "", "二跳乙"))
        mem.record("二跳甲", WordStateImprint(1, 1, 0, 0.1, 0.1))
        mem.record("二跳乙", WordStateImprint(1, 1, 0, 0.1, 0.1))
        out = mem.activation_frontier_tokens(
            recent_tokens=["种子"],
            pending_tokens=set(),
            limit=10,
            one_hop_budget=2,
            two_hop_budget=1,
            two_hop_per_token=1,
            cold_budget=1,
            shard_count=1,
        )
        self.assertIn("种子", out)
        self.assertIn("一跳甲", out)
        self.assertIn("一跳乙", out)
        self.assertEqual(len({"二跳甲", "二跳乙"} & set(out)), 1)
        self.assertIn("冷词", out)

    def test_save_load_roundtrip_preserves_soft_freeze_and_occurrence_counts(self) -> None:
        mem = WordStateMemory(capacity=5)
        mem.record("x", WordStateImprint(1, 2, 0, 0.5, 0.5, "a", "b"))
        mem.record("x", WordStateImprint(1, 2, 0, 0.5, 0.5, "a", "b"))
        mem.record("y", WordStateImprint(1, 2, 0, 0.5, 0.5, "a", "b"))
        mem.note_trigger_success("x")
        mem.soft_freeze_cold_tokens(max_active_tokens=1)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "im.json"
            mem.save(str(p))
            mem2 = WordStateMemory.load(str(p))
            self.assertEqual(len(mem2._store.get("x", [])), 1)  # noqa: SLF001
            self.assertIn(("a", 2), mem2.associated_tokens("x", k=3))
            self.assertEqual(set(mem2.cold_scan_tokens(limit=10, shard_count=1)), {"x"})

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA torch")
    def test_top_surprises_torch_matches_cpu(self) -> None:
        mem = WordStateMemory(capacity=10)
        mem.record("游戏", WordStateImprint(1.0, 0.8, 0.1, 0.2, 0.1))
        mem.record("游戏", WordStateImprint(1.2, 0.7, 0.1, 0.2, 0.1))
        mem.record("考试", WordStateImprint(2.0, 0.2, 0.6, 0.9, 0.9))
        mem.record("睡觉", WordStateImprint(0.9, 0.9, 0.0, 0.1, 0.1))
        current = WordStateImprint(1.8, 0.3, 0.7, 0.8, 0.8)

        mem.set_compute_device("cpu")
        top_cpu = mem.top_surprises(current, k=3)
        mem.set_compute_device("cuda")
        top_gpu = mem.top_surprises(current, k=3)

        self.assertEqual([tok for _surp, tok in top_cpu], [tok for _surp, tok in top_gpu])
        for (surp_cpu, _), (surp_gpu, _) in zip(top_cpu, top_gpu, strict=True):
            self.assertAlmostEqual(surp_cpu, surp_gpu, places=5)

    def test_associated_tokens_counts_context_neighbors(self) -> None:
        mem = WordStateMemory(capacity=10)
        mem.record("跑路", WordStateImprint(1, 1, 0, 0.1, 0.1, "群", "封禁"))
        mem.record("跑路", WordStateImprint(1, 1, 0, 0.1, 0.1, "群", "bbs"))
        mem.record("跑路", WordStateImprint(1, 1, 0, 0.1, 0.1, "跑路", ""))
        assoc = mem.associated_tokens("跑路", k=3)
        self.assertEqual(assoc[0], ("群", 2))
        self.assertIn(("封禁", 1), assoc)
        self.assertIn(("bbs", 1), assoc)
        self.assertNotIn(("跑路", 1), assoc)

    def test_associated_tokens_uses_cache_and_record_invalidates_it(self) -> None:
        mem = WordStateMemory(capacity=10)
        mem.record("跑路", WordStateImprint(1, 1, 0, 0.1, 0.1, "群", "封禁"))
        first = mem.associated_tokens("跑路", k=3)
        self.assertGreaterEqual(mem._associated_tokens_cache_hits, 0)  # noqa: SLF001
        before_hits = mem._associated_tokens_cache_hits  # noqa: SLF001
        second = mem.associated_tokens("跑路", k=3)
        self.assertEqual(first, second)
        self.assertGreater(mem._associated_tokens_cache_hits, before_hits)  # noqa: SLF001
        mem.record("跑路", WordStateImprint(1, 1, 0, 0.1, 0.1, "群", "新邻居"))
        third = mem.associated_tokens("跑路", k=5)
        self.assertIn(("新邻居", 1), third)

    def test_structural_caches_lru_bounded_and_remain_correct(self) -> None:
        """长跑训练下关联/一跳缓存必须有界；驱逐后重算结果仍正确。"""
        mem = WordStateMemory(capacity=10)
        mem._structural_cache_max = 2  # noqa: SLF001
        tiny_a = WordStateImprint(1, 1, 0, 0.1, 0.1, "邻甲", "邻乙")
        tiny_x = WordStateImprint(1, 1, 0, 0.1, 0.1, "邻丙", "邻丁")
        tiny_p = WordStateImprint(1, 1, 0, 0.1, 0.1, "邻戊", "邻己")
        mem.record("词a", tiny_a)
        mem.record("词x", tiny_x)
        mem.record("词p", tiny_p)
        mem.associated_tokens("词a", k=3)
        mem.associated_tokens("词x", k=3)
        mem.associated_tokens("词p", k=3)
        self.assertLessEqual(len(mem._associated_tokens_cache), 2)  # noqa: SLF001
        ca = mem.associated_tokens("词a", k=3)
        self.assertTrue(ca)
        mem._frontier_one_hop_neighbors("词a")
        mem._frontier_one_hop_neighbors("词x")
        mem._frontier_one_hop_neighbors("词p")
        self.assertLessEqual(len(mem._frontier_one_hop_cache), 2)  # noqa: SLF001

    def test_activation_frontier_uses_one_hop_cache_and_record_invalidates_it(self) -> None:
        mem = WordStateMemory(capacity=10)
        mem.record("种子", WordStateImprint(1, 1, 0, 0.1, 0.1, "", "一跳"))
        mem.record("一跳", WordStateImprint(1, 1, 0, 0.1, 0.1))
        first = mem.activation_frontier_tokens(
            recent_tokens=["种子"],
            pending_tokens=set(),
            limit=8,
            one_hop_budget=2,
            two_hop_budget=0,
            cold_budget=0,
            shard_count=1,
        )
        before_hits = mem._frontier_one_hop_cache_hits  # noqa: SLF001
        second = mem.activation_frontier_tokens(
            recent_tokens=["种子"],
            pending_tokens=set(),
            limit=8,
            one_hop_budget=2,
            two_hop_budget=0,
            cold_budget=0,
            shard_count=1,
        )
        self.assertEqual(first, second)
        self.assertGreater(mem._frontier_one_hop_cache_hits, before_hits)  # noqa: SLF001
        mem.record("种子", WordStateImprint(1, 1, 0, 0.1, 0.1, "", "新一跳"))
        mem.record("新一跳", WordStateImprint(1, 1, 0, 0.1, 0.1))
        third = mem.activation_frontier_tokens(
            recent_tokens=["种子"],
            pending_tokens=set(),
            limit=8,
            one_hop_budget=4,
            two_hop_budget=0,
            cold_budget=0,
            shard_count=1,
        )
        self.assertIn("新一跳", third)

    def test_activation_frontier_trims_active_tokens_once_per_batch(self) -> None:
        mem = WordStateMemory(capacity=10)
        tokens = [f"词{i}" for i in range(12)]
        for tok in tokens:
            mem.record(tok, WordStateImprint(1, 1, 0, 0.1, 0.1))
        mem.soft_freeze_cold_tokens(max_active_tokens=4)

        trim_calls = 0
        original_trim = mem._trim_active_tokens  # noqa: SLF001

        def counted_trim(max_active_tokens: int, *, protected: set[str] | None = None) -> None:
            nonlocal trim_calls
            trim_calls += 1
            original_trim(max_active_tokens, protected=protected)

        mem._trim_active_tokens = counted_trim  # type: ignore[method-assign]  # noqa: SLF001

        out = mem.activation_frontier_tokens(
            recent_tokens=tokens,
            pending_tokens=set(),
            limit=len(tokens),
            one_hop_budget=0,
            two_hop_budget=0,
            cold_budget=0,
            shard_count=1,
            max_active_tokens=4,
        )

        self.assertEqual(out, tokens)
        self.assertLessEqual(trim_calls, 1)

    def test_save_load_roundtrip(self) -> None:
        mem = WordStateMemory(capacity=5)
        mem.record(
            "x",
            WordStateImprint(1, 2, 0, 0.5, 0.5, "a", "b"),
        )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "im.json"
            mem.save(str(p))
            mem2 = WordStateMemory.load(str(p))
            self.assertGreater(len(mem2._store.get("x", [])), 0)  # noqa: SLF001
            d = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(d.get("format"), "word_state_memory_v1")


if __name__ == "__main__":
    unittest.main()
