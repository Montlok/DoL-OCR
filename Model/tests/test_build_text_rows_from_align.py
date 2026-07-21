from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from Model.config import BOS_ID, EOS_ID, IGNORE_INDEX
from scripts.build_text_rows_from_align import main


class BuildTextRowsFromAlignTest(unittest.TestCase):
    def test_bundle_qa_uses_public_load_and_decode_apis(self) -> None:
        decoded: list[list[int]] = []

        class FakeTokenizerBundle:
            @classmethod
            def from_dir(cls, path: str):
                self.assertEqual(path, str(bundle_dir))
                tokenizer = SimpleNamespace(
                    decode=lambda ids: decoded.append(list(ids)) or "decoded text"
                )
                return SimpleNamespace(tokenizer=tokenizer)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            align = root / "align.jsonl"
            output = root / "packed"
            bundle_dir = root / "bundle"
            bundle_dir.mkdir()
            align.write_text(
                json.dumps(
                    {
                        "input_ids": [900, 400, EOS_ID],
                        "labels": [IGNORE_INDEX, 400, EOS_ID],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with (
                mock.patch(
                    "Tokenizer.unified.TokenizerBundle", FakeTokenizerBundle
                ),
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                rc = main(
                    [
                        "--align",
                        str(align),
                        "--output",
                        str(output),
                        "--seq-len",
                        "3",
                        "--bundle",
                        str(bundle_dir),
                    ]
                )

            self.assertEqual(rc, 0)
            self.assertEqual(decoded, [[400]])
            self.assertIn("decoded text", stdout.getvalue())
            row = json.loads((output / "text_0000.jsonl").read_text())
            self.assertEqual(row["input_ids"], [BOS_ID, 400, EOS_ID])


if __name__ == "__main__":
    unittest.main()
