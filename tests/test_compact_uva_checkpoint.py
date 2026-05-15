"""compact_uva_checkpoint.py：紧凑重写 JSON，且不覆盖同路径（默认）。"""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from uva_model.dialogue import DIALOGUE_MODEL_FORMAT
from uva_model.tokenizer import PrecisionTokenizer
from uva_model.word_imprints import WORD_STATE_MEMORY_FORMAT, WordStateMemory


def _load_compact_script():
    script_path = Path(__file__).resolve().parent.parent / "compact_uva_checkpoint.py"
    spec = importlib.util.spec_from_file_location("compact_uva_checkpoint", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CompactUvaCheckpointTests(unittest.TestCase):
    def test_compact_reduces_pretty_print_size(self) -> None:
        script = Path(__file__).resolve().parent.parent / "compact_uva_checkpoint.py"
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src = td_path / "a.json"
            dst = td_path / "a.small.json"
            obj = {"sigma_slot_1": 0.5, "nested": {"x": [1, 2, 3]}}
            src.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            r = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--no-stream",
                    "--tokenizer-in",
                    str(src),
                    "--tokenizer-out",
                    str(dst),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(r.returncode, 0, msg=r.stderr + r.stdout)
            back = json.loads(dst.read_text(encoding="utf-8"))
            self.assertEqual(back, obj)
            self.assertLess(dst.stat().st_size, src.stat().st_size)

    def test_same_path_rejected_without_in_place(self) -> None:
        script = Path(__file__).resolve().parent.parent / "compact_uva_checkpoint.py"
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src = td_path / "same.json"
            src.write_text('{"a":1}', encoding="utf-8")
            r = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--no-stream",
                    "--tokenizer-in",
                    str(src),
                    "--tokenizer-out",
                    str(src),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(r.returncode, 3)

    def test_in_place_allowed(self) -> None:
        script = Path(__file__).resolve().parent.parent / "compact_uva_checkpoint.py"
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src = td_path / "same.json"
            src.write_text(json.dumps({"k": "v"}, indent=2), encoding="utf-8")
            before = src.read_bytes()
            r = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--no-stream",
                    "--tokenizer-in",
                    str(src),
                    "--tokenizer-out",
                    str(src),
                    "--in-place",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(r.returncode, 0, msg=r.stderr + r.stdout)
            after = src.read_bytes()
            self.assertLess(len(after), len(before))

    def test_optimize_dialogue_word_imprints_loads_same_stats(self) -> None:
        mod = _load_compact_script()
        wi: dict = {
            "format": WORD_STATE_MEMORY_FORMAT,
            "capacity": 100,
            "tokens": {
                "甲": [
                    {
                        "F_ema": 1.0,
                        "R": 0.5,
                        "m": 0.0,
                        "u_curiosity": 0.1,
                        "u_task": 0.0,
                        "context_before": "",
                        "context_after": "",
                        "occurrence_count": 1,
                    }
                ]
            },
            "token_meta": {
                "甲": {
                    "first_seen_order": 0,
                    "last_access_seq": 0,
                    "last_trigger_seq": 0,
                    "trigger_success_count": 0,
                    "imprint_count": 1,
                    "is_dormant": False,
                }
            },
            "seq": 0,
            "cold_scan_cursor": 0,
            "merged_imprints_total": 0,
        }
        dlg = {
            "format": DIALOGUE_MODEL_FORMAT,
            "sigma": {"sigma_slot_1": 0.5, "sigma_slot_2": 0.5, "sigma_slot_3": 0.5, "sigma_slot_4": 0.5},
            "word_imprints": wi,
            "preference_state": {
                "format": "preference_state_v1",
                "branch_bias": {"noop": 0.0},
                "token_value": {},
                "association_value": {},
            },
        }
        m0 = WordStateMemory.from_dict(json.loads(json.dumps(wi)))
        opt = mod.optimize_payload("dialogue", dlg, optimize=True)
        m1 = WordStateMemory.from_dict(json.loads(json.dumps(opt["word_imprints"])))
        self.assertEqual(m0.stats(), m1.stats())

    def test_optimize_tokenizer_strips_default_alpha(self) -> None:
        mod = _load_compact_script()
        raw = {
            "alpha": 1.1,
            "unigram": {"x": 3},
            "bigram": {},
            "follow_counts": {},
            "total_chars": 0,
            "fitted": True,
            "R": 1.0,
            "m": 0.0,
            "F_ema": 1.2,
        }
        t0 = PrecisionTokenizer.from_dict(json.loads(json.dumps(raw)))
        opt = mod.optimize_payload("tokenizer", raw, optimize=True)
        self.assertNotIn("alpha", opt)
        t1 = PrecisionTokenizer.from_dict(json.loads(json.dumps(opt)))
        self.assertEqual(t0.unigram, t1.unigram)
        self.assertEqual(t0.R, t1.R)


if __name__ == "__main__":
    unittest.main()
