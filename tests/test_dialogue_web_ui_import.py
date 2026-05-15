"""dialogue_web_ui 可导入且能构建 Blocks（需已安装 gradio）。"""
from __future__ import annotations

import unittest


class DialogueWebUiImportTests(unittest.TestCase):
    def test_build_ui_import(self) -> None:
        try:
            import gradio as gr  # noqa: F401
        except ImportError:
            self.skipTest("未安装 gradio，跳过：pip install -r requirements-web.txt")
        from dialogue_web_ui import build_ui

        demo = build_ui()
        self.assertTrue(hasattr(demo, "launch"))


if __name__ == "__main__":
    unittest.main()
