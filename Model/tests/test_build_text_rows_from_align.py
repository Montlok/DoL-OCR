from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from Model.config import BOS_ID, EOS_ID, IGNORE_INDEX
from Tokenizer.pretraining.data_contract import (
    PRETRAINING_PRODUCER_OCR_ALIGN_TEXT,
    pretraining_producer_algorithm_contract,
)
from scripts.build_text_rows_from_align import main


class BuildTextRowsFromAlignTest(unittest.TestCase):
    def _run_builder(
        self,
        root: Path,
        source_rows: list[dict],
        *,
        seq_len: int,
        track_table: tuple[int, ...],
    ) -> tuple[int, str, Path]:
        decoded: list[list[int]] = []
        align = root / "align.jsonl"
        align.write_text(
            "".join(json.dumps(row) + "\n" for row in source_rows),
            encoding="utf-8",
        )
        align_receipt = root / "ocr_data_contract.json"
        align_receipt.write_text("{}", encoding="utf-8")
        output = root / "packed"
        bundle_dir = root / "bundle"
        bundle_dir.mkdir()

        class FakeTokenizerBundle:
            @classmethod
            def from_dir(cls, path: str):
                self.assertEqual(path, str(bundle_dir))
                tokenizer = SimpleNamespace(
                    decode=lambda ids: decoded.append(list(ids)) or "decoded text"
                )
                return SimpleNamespace(tokenizer=tokenizer)

        with (
            ExitStack() as stack,
            contextlib.redirect_stdout(io.StringIO()) as stdout,
        ):
            stack.enter_context(
                mock.patch(
                    "scripts.build_text_rows_from_align.TokenizerBundle",
                    FakeTokenizerBundle,
                )
            )
            stack.enter_context(
                mock.patch(
                    "scripts.build_text_rows_from_align."
                    "native_tokenization_contract",
                    return_value={"target_encoding": "native"},
                )
            )
            validate_source = stack.enter_context(
                mock.patch(
                    "scripts.build_text_rows_from_align."
                    "load_and_validate_ocr_alignment_data_contract",
                    return_value={"contract_canonical_sha256": "a" * 64},
                )
            )
            stack.enter_context(
                mock.patch(
                    "scripts.build_text_rows_from_align."
                    "tokenizer_morphology_track_table",
                    return_value=track_table,
                )
            )
            stack.enter_context(
                mock.patch(
                    "scripts.build_text_rows_from_align."
                    "tokenizer_bundle_contract",
                    return_value={"files_canonical_sha256": "b" * 64},
                )
            )
            stack.enter_context(
                mock.patch(
                    "scripts.build_text_rows_from_align."
                    "tokenizer_algorithm_contract",
                    return_value={"files_canonical_sha256": "c" * 64},
                )
            )
            rc = main(
                [
                    "--align",
                    str(align),
                    "--align-receipt",
                    str(align_receipt),
                    "--output",
                    str(output),
                    "--seq-len",
                    str(seq_len),
                    "--bundle",
                    str(bundle_dir),
                ]
            )
        validate_source.assert_called_once_with(
            str(align_receipt),
            str(align),
            {"target_encoding": "native"},
        )
        self.assertTrue(decoded)
        return rc, stdout.getvalue(), output

    def test_bundle_qa_uses_public_load_and_decode_apis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            table = [0] * 1000
            table[400] = 1
            rc, stdout, output = self._run_builder(
                Path(tmp),
                [
                    {
                        "input_ids": [900, 400, EOS_ID],
                        "labels": [IGNORE_INDEX, 400, EOS_ID],
                    }
                ],
                seq_len=3,
                track_table=tuple(table),
            )

            row = json.loads((output / "text_0000.jsonl").read_text())
            receipt = json.loads(
                (output / "pretraining_data_receipt.json").read_text()
            )
        self.assertEqual(rc, 0)
        self.assertIn("decoded text", stdout)
        self.assertEqual(row["input_ids"], [BOS_ID, 400, EOS_ID])
        self.assertEqual(row["word_pos"], [0, 0, 0])
        self.assertEqual(row["morph_depth"], [0, 0, 0])
        self.assertEqual(receipt["data_file_count"], 1)
        self.assertEqual(
            receipt["tokenizer_bundle"]["files_canonical_sha256"],
            "b" * 64,
        )
        self.assertEqual(
            receipt["producer_kind"],
            PRETRAINING_PRODUCER_OCR_ALIGN_TEXT,
        )
        self.assertEqual(
            receipt["producer_algorithm"],
            pretraining_producer_algorithm_contract(
                PRETRAINING_PRODUCER_OCR_ALIGN_TEXT
            ),
        )

    def test_long_word_depth_and_each_document_bos_reset_are_persisted(self) -> None:
        word = list(range(400, 409))
        table = [0] * 1000
        for token_id in word:
            table[token_id] = 1
        source_rows = [
            {
                "input_ids": [900, *word, EOS_ID],
                "labels": [IGNORE_INDEX, *word, EOS_ID],
            },
            {
                "input_ids": [901, *word, EOS_ID],
                "labels": [IGNORE_INDEX, *word, EOS_ID],
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            rc, _stdout, output = self._run_builder(
                Path(tmp),
                source_rows,
                seq_len=2 * (1 + len(word) + 1),
                track_table=tuple(table),
            )
            row = json.loads((output / "text_0000.jsonl").read_text())

        document_len = 1 + len(word) + 1
        self.assertEqual(rc, 0)
        self.assertEqual(
            row["morph_depth"][:document_len],
            [0, *range(len(word)), 0],
        )
        self.assertEqual(
            row["morph_depth"][document_len:],
            [0, *range(len(word)), 0],
        )
        self.assertEqual(row["word_pos"][1:1 + len(word)], [0] * len(word))
        second_word_start = document_len + 1
        self.assertEqual(
            row["word_pos"][second_word_start:second_word_start + len(word)],
            [1] * len(word),
        )

    def test_source_receipt_failure_prevents_output_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            align = root / "align.jsonl"
            align.write_text("{}\n", encoding="utf-8")
            source_receipt = root / "ocr_data_contract.json"
            source_receipt.write_text("{}", encoding="utf-8")
            bundle_dir = root / "bundle"
            bundle_dir.mkdir()
            output = root / "packed"
            fake_bundle = SimpleNamespace(tokenizer=object())
            with (
                mock.patch(
                    "scripts.build_text_rows_from_align.TokenizerBundle.from_dir",
                    return_value=fake_bundle,
                ),
                mock.patch(
                    "scripts.build_text_rows_from_align."
                    "tokenizer_bundle_contract",
                    return_value={"files_canonical_sha256": "b" * 64},
                ),
                mock.patch(
                    "scripts.build_text_rows_from_align."
                    "tokenizer_algorithm_contract",
                    return_value={"files_canonical_sha256": "c" * 64},
                ),
                mock.patch(
                    "scripts.build_text_rows_from_align."
                    "native_tokenization_contract",
                    return_value={"target_encoding": "native"},
                ),
                mock.patch(
                    "scripts.build_text_rows_from_align."
                    "load_and_validate_ocr_alignment_data_contract",
                    side_effect=ValueError("shard bytes differ"),
                ),
                contextlib.redirect_stderr(io.StringIO()) as stderr,
            ):
                rc = main(
                    [
                        "--align",
                        str(align),
                        "--align-receipt",
                        str(source_receipt),
                        "--output",
                        str(output),
                        "--bundle",
                        str(bundle_dir),
                    ]
                )
            receipt_exists = (
                output / "pretraining_data_receipt.json"
            ).exists()
        self.assertEqual(rc, 2)
        self.assertIn("shard bytes differ", stderr.getvalue())
        self.assertFalse(receipt_exists)


if __name__ == "__main__":
    unittest.main()
