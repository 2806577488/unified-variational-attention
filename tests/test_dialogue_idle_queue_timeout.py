"""``_dialogue_idle_queue_timeout``：回应后冷却与空闲 [内心] 门槛。"""
from __future__ import annotations

import time
import unittest

from example_run import _MIN_DIALOGUE_INTERNAL_IDLE_SEC, _dialogue_idle_queue_timeout


class DialogueIdleQueueTimeoutTests(unittest.TestCase):
    def test_zero_gate_allows_speak_with_idle_timeout(self) -> None:
        t, ok = _dialogue_idle_queue_timeout(
            internal_idle_sec=1.5,
            inner_speak_not_before=0.0,
        )
        self.assertTrue(ok)
        self.assertEqual(t, max(_MIN_DIALOGUE_INTERNAL_IDLE_SEC, 1.5))

    def test_far_future_gate_blocks_speak(self) -> None:
        far = time.monotonic() + 1e6
        t, ok = _dialogue_idle_queue_timeout(
            internal_idle_sec=2.0,
            inner_speak_not_before=far,
        )
        self.assertFalse(ok)
        self.assertGreaterEqual(t, _MIN_DIALOGUE_INTERNAL_IDLE_SEC)
        self.assertLessEqual(t, 2.0)


if __name__ == "__main__":
    unittest.main()
