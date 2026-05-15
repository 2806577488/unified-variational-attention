"""
审计用测试：不改业务实现，仅断言/记录已知的性能热点与重复计算路径，
供代码整理与后续优化对照（见同目录主代码审查说明）。
"""

from __future__ import annotations

import inspect
import unittest
from unittest.mock import patch

from uva_model.model import UnifiedVariationalAttentionModel
from uva_model.word_imprints import WordStateImprint, WordStateMemory


class ModelDuplicateComputationTests(unittest.TestCase):
    """UVA 标量模型：同一调用路径内对感官误差的重复遍历。"""

    def setUp(self) -> None:
        self.model = UnifiedVariationalAttentionModel(
            observation=[1.2, -0.4],
            mu=[0.3, -0.1],
            pi=[1.5, 1.2],
            pi_min=0.5,
            c_max=2.0,
            alpha=1.2,
            beta=0.7,
            gamma_phi=0.9,
            lambda_c=1.1,
            sigma0=0.3,
            tau_pi=0.8,
            dt=0.01,
            task_input=[0.2, 0.6],
            delta_s=[0.8, 0.3],
        )

    def test_free_energy_calls_sensory_error_once(self) -> None:
        """free_energy 复用同一次 eps_o 传入 effective_drive，避免重复 sensory_error。"""
        with patch.object(
            self.model,
            "sensory_error",
            wraps=self.model.sensory_error,
        ) as spy:
            _ = self.model.free_energy()
            self.assertEqual(spy.call_count, 1)

    def test_gradients_calls_sensory_error_once(self) -> None:
        """gradients 与 free_energy 相同：effective_drive 复用已算 eps_o。"""
        with patch.object(
            self.model,
            "sensory_error",
            wraps=self.model.sensory_error,
        ) as spy:
            _ = self.model.gradients()
            self.assertEqual(spy.call_count, 1)


class WordStateMemoryHotspotTests(unittest.TestCase):
    """印记库：分配与算法复杂度热点（与 cold_scan 避免全量 keys 复制的注释对照）。"""

    def test_tokens_returns_fresh_list_each_time(self) -> None:
        """tokens() = list(_store.keys())，大词表下每次全量复制键。"""
        mem = WordStateMemory(capacity=10)
        mem.record("a", WordStateImprint(1, 1, 0, 0.1, 0.1))
        a, b = mem.tokens(), mem.tokens()
        self.assertEqual(a, b)
        self.assertIsNot(a, b)

    def test_frontier_one_hop_neighbors_returns_cached_tuple_on_hit(self) -> None:
        """缓存命中直接返回 tuple，避免 list(cached) 拷贝。"""
        mem = WordStateMemory(capacity=5)
        mem.record(
            "seed",
            WordStateImprint(1, 1, 0, 0.1, 0.1, context_before="", context_after="x"),
        )
        mem.record(
            "x",
            WordStateImprint(1, 1, 0, 0.1, 0.1, context_before="seed", context_after=""),
        )
        n1 = mem._frontier_one_hop_neighbors("seed")
        n2 = mem._frontier_one_hop_neighbors("seed")
        self.assertEqual(n1, n2)
        self.assertIs(n1, n2)

    def test_top_surprises_full_scan_iterates_store_without_tokens_copy(self) -> None:
        """全库 top_surprises 直接遍历 _store，不调用 tokens() 全键复制。"""
        mem = WordStateMemory(capacity=10)
        for i in range(30):
            mem.record(f"t{i}", WordStateImprint(float(i), 1, 0, 0.1, 0.1))
        cur = WordStateImprint(0.0, 1.0, 0.0, 0.1, 0.1)
        calls: list[int] = []

        def counting_tokens() -> list[str]:
            calls.append(1)
            return list(mem._store.keys())

        with patch.object(mem, "tokens", side_effect=counting_tokens):
            out = mem.top_surprises(cur, k=3, exclude=set())
        self.assertGreaterEqual(len(out), 1)
        self.assertEqual(calls, [])

    def test_trim_active_tokens_source_uses_pop_from_end(self) -> None:
        """_trim_active_tokens 对 reverse 排序的 removable 使用 pop()，避免 pop(0) O(n²)。"""
        src = inspect.getsource(WordStateMemory._trim_active_tokens)
        self.assertIn("pop()", src)
        self.assertNotIn("pop(0)", src)

    def test_top_surprises_lazy_throttles_soft_freeze(self) -> None:
        """lazy 路径对 soft_freeze 节流，非每次 max_active_tokens 都全库排序。"""
        src = inspect.getsource(WordStateMemory.top_surprises_lazy)
        self.assertIn("_lazy_soft_freeze", src)

    def test_record_merge_skips_global_torch_cache_invalidate(self) -> None:
        """合并命中仅失效关联缓存；新增/替换印记仍 _invalidate_cache。"""
        src_record = inspect.getsource(WordStateMemory.record)
        merge_block = src_record.split("self._can_merge_imprint")[1].split("return")[0]
        self.assertNotIn("_invalidate_cache()", merge_block)
        self.assertIn("_invalidate_cache()", src_record)


if __name__ == "__main__":
    unittest.main()
