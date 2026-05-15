"""交互对话仅保留空闲内心路径：不再有阻塞 stdin、内心仅随用户输入的开关。"""
from __future__ import annotations

import unittest

import example_run


class DialogueInternalIdleModeTests(unittest.TestCase):
    def test_blocking_dialogue_loop_removed(self) -> None:
        self.assertFalse(
            hasattr(example_run, "_run_interactive_dialogue_blocking"),
            "已移除同步阻塞对话循环；内心独白不应用 CLI「0」切换为仅随用户发言。",
        )

    def test_min_idle_interval_is_positive(self) -> None:
        self.assertGreater(example_run._MIN_DIALOGUE_INTERNAL_IDLE_SEC, 0.0)


if __name__ == "__main__":
    unittest.main()
