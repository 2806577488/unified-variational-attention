import json
import tempfile
import unittest
from pathlib import Path

try:
    import ijson  # noqa: F401

    from uva_model.compact_json_stream import stream_available, stream_compact_checkpoint
except ImportError:
    stream_available = lambda: False  # type: ignore[assignment,misc]
    stream_compact_checkpoint = None  # type: ignore[assignment,misc]


@unittest.skipUnless(stream_available(), "需要 pip install ijson")
class CompactJsonStreamTests(unittest.TestCase):
    def test_stream_matches_json_dumps_compact(self) -> None:
        obj = {
            "a": 1,
            "b": {"x": [1, 2], "y": "z"},
            "c": None,
        }
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src = td_path / "in.json"
            dst = td_path / "out.json"
            src.write_text(json.dumps(obj, indent=2), encoding="utf-8")
            stream_compact_checkpoint(src, dst, tokenizer_strip=None, optimize_dialogue=False)
            got = json.loads(dst.read_text(encoding="utf-8"))
            self.assertEqual(got, obj)

    def test_stream_dialogue_optimize_imprint(self) -> None:
        dlg = {
            "format": "cognitive_dialogue_agent_v1",
            "sigma": {"sigma_slot_1": 0.5},
            "word_imprints": {
                "format": "word_state_memory_v1",
                "capacity": 100,
                "tokens": {
                    "T": [
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
                    "T": {
                        "first_seen_order": 0,
                        "last_access_seq": 0,
                        "imprint_count": 1,
                        "is_dormant": False,
                    }
                },
                "seq": 0,
            },
            "preference_state": {
                "format": "preference_state_v1",
                "branch_bias": {"noop": 0.0, "keep": 0.5},
                "token_value": {},
                "association_value": {"a": {"b": 0.0, "c": 1.0}}},
        }
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src = td_path / "d.json"
            dst = td_path / "d.out.json"
            src.write_text(json.dumps(dlg, indent=2), encoding="utf-8")
            stream_compact_checkpoint(
                src, dst, tokenizer_strip=None, optimize_dialogue=True
            )
            got = json.loads(dst.read_text(encoding="utf-8"))
            wi = got["word_imprints"]
            self.assertNotIn("seq", wi)
            self.assertNotIn("context_before", wi["tokens"]["T"][0])
            self.assertNotIn("last_access_seq", wi["token_meta"]["T"])
            self.assertNotIn("noop", got["preference_state"]["branch_bias"])
            self.assertNotIn("b", got["preference_state"]["association_value"]["a"])


if __name__ == "__main__":
    unittest.main()
