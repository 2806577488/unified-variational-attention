import tempfile
import unittest
from pathlib import Path

from uva_model.checkpoint_json import read_json_document, write_json_document
from uva_model.tokenizer import PrecisionTokenizer


class CheckpointJsonTests(unittest.TestCase):
    def test_gzip_tokenizer_roundtrip(self) -> None:
        t0 = PrecisionTokenizer()
        t0.fit(["你好 世界", "测试 文本"])
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.json.gz"
            t0.save_model(str(p))
            t1 = PrecisionTokenizer.load_model(str(p))
        self.assertEqual(t0.unigram, t1.unigram)
        self.assertEqual(t0.bigram, t1.bigram)
        self.assertEqual(t0.follow_counts, t1.follow_counts)

    def test_plain_json_still_readable_as_bytes(self) -> None:
        obj = {"a": 1, "b": [2, 3]}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "x.json"
            write_json_document(path, obj, compact=True, compression=None)
            self.assertEqual(read_json_document(path), obj)

    def test_legacy_dict_ngrams_load(self) -> None:
        raw = {
            "alpha": 1.1,
            "unigram": {"^": 2, "a": 1},
            "bigram": {"^": {"a": 1}},
            "follow_counts": {},
            "total_chars": 0,
            "fitted": True,
            "R": 1.0,
            "m": 0.0,
            "F_ema": 1.2,
        }
        t = PrecisionTokenizer.from_dict(raw)
        self.assertEqual(t.unigram["^"], 2)
        self.assertEqual(t.bigram["^"]["a"], 1)

    def test_compact_v1_equivalent_to_dict_layout(self) -> None:
        t = PrecisionTokenizer()
        t.fit(["abcd", "bcde", "汉字 混合"])
        t1 = PrecisionTokenizer.from_dict(t.to_dict(ngram_layout="compact_v1"))
        t2 = PrecisionTokenizer.from_dict(t.to_dict(ngram_layout="dict"))
        self.assertEqual(t1.unigram, t2.unigram)
        self.assertEqual(t1.bigram, t2.bigram)
        self.assertEqual(t1.follow_counts, t2.follow_counts)


class ZstdCheckpointTests(unittest.TestCase):
    def test_zstd_roundtrip_if_installed(self) -> None:
        try:
            import zstandard  # noqa: F401
        except ImportError:
            self.skipTest("zstandard 未安装")
        t0 = PrecisionTokenizer()
        t0.fit(["zstd", "round", "trip"])
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.json.zst"
            t0.save_model(str(p))
            t1 = PrecisionTokenizer.load_model(str(p))
        self.assertEqual(t0.unigram, t1.unigram)


if __name__ == "__main__":
    unittest.main()
