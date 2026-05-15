import gzip
import json
import tempfile
import unittest
from pathlib import Path

from uva_model.corpus_jsonl import (
    iter_corpus_jsonl_training_episodes,
    load_corpus_jsonl_episodes_list,
)


class CorpusJsonlTests(unittest.TestCase):
    def test_iter_windows_and_gzip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.jsonl"
            p.write_text(
                "\n".join(
                    json.dumps({"text": str(i), "meta": {}}, ensure_ascii=False)
                    for i in range(5)
                )
                + "\n",
                encoding="utf-8",
            )
            eps = list(
                iter_corpus_jsonl_training_episodes([p], episode_size=2, min_episode_turns=1)
            )
            self.assertEqual(eps, [["0", "1"], ["2", "3"], ["4"]])

            gz = Path(td) / "b.jsonl.gz"
            with gzip.open(gz, "wt", encoding="utf-8") as fh:
                fh.write('{"text":"z","meta":{}}\n')
            self.assertEqual(
                load_corpus_jsonl_episodes_list([gz], episode_size=1, min_episode_turns=1),
                [["z"]],
            )


if __name__ == "__main__":
    unittest.main()
