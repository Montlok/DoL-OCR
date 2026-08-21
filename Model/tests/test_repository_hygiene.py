from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from scripts.check_repository_hygiene import (
    MAX_TRACKED_BYTES,
    SCRIPT_ALLOWLIST,
    TOKENIZER_TOOL_ALLOWLIST,
    check_paths,
)


class RepositoryHygieneTest(unittest.TestCase):
    def _materialize(self, root: Path, path: Path, content: str = "") -> None:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(content, encoding="utf-8")

    def test_clean_source_file_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = Path("Model/module.py")
            self._materialize(root, path, "value = 1\n")
            self.assertEqual(check_paths(root, [path]), [])

    def test_approved_entry_points_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [
                Path("scripts/train_rdt.py"),
                Path("scripts/train_ocr_anyres_grpo.py"),
                Path("Tokenizer/tools/build_pretraining_data.py"),
            ]
            for path in paths:
                self._materialize(root, path)
            self.assertEqual(check_paths(root, paths), [])

    def test_allowlists_match_expected_production_inventory(self) -> None:
        self.assertIn("scripts/train_rdt.py", SCRIPT_ALLOWLIST)
        self.assertIn("scripts/train_ocr_anyres_grpo.py", SCRIPT_ALLOWLIST)
        self.assertIn(
            "scripts/eval_ocr_anyres_locked.py",
            SCRIPT_ALLOWLIST,
        )
        self.assertNotIn("scripts/train_grpo.py", SCRIPT_ALLOWLIST)
        self.assertNotIn("scripts/build_ocr_rl_manifests.py", SCRIPT_ALLOWLIST)
        self.assertIn(
            "Tokenizer/tools/build_pretraining_data.py",
            TOKENIZER_TOOL_ALLOWLIST,
        )
        self.assertNotIn(
            "Tokenizer/tools/normalize_mongolian.py",
            TOKENIZER_TOOL_ALLOWLIST,
        )

    def test_unapproved_script_and_tokenizer_tool_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [
                Path("scripts/scrape_corpus.py"),
                Path("Tokenizer/tools/download_dataset.py"),
            ]
            for path in paths:
                self._materialize(root, path)
            violations = check_paths(root, paths)
            self.assertTrue(any("unapproved scripts" in item for item in violations))
            self.assertTrue(any("unapproved tokenizer" in item for item in violations))

    def test_dataset_and_media_extensions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [
                Path("fixture/rows.jsonl"),
                Path("fixture/labels.csv"),
                Path("fixture/splits.tsv"),
                Path("fixture/page.png"),
                Path("fixture/scan.pdf"),
                Path("fixture/audio.wav"),
            ]
            for path in paths:
                self._materialize(root, path)
            violations = check_paths(root, paths)
            rejected = [item for item in violations if "dataset, or media" in item]
            self.assertEqual(len(rejected), len(paths))

    def test_model_artifact_and_archive_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [
                Path("release/model.safetensors"),
                Path("release/optimizer.pt"),
                Path("release/export.tar.zst"),
            ]
            for path in paths:
                self._materialize(root, path)
            violations = check_paths(root, paths)
            rejected = [item for item in violations if "artifact, dataset" in item]
            self.assertEqual(len(rejected), len(paths))

    def test_retired_and_local_data_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [
                Path("Encoding Mapping/README.md"),
                Path("Tokenizer/data/README.md"),
                Path("datasets/README.md"),
                Path("private/DO NOT GIT IT/README.md"),
            ]
            for path in paths:
                self._materialize(root, path)
            violations = check_paths(root, paths)
            self.assertTrue(any("retired path" in item for item in violations))
            self.assertTrue(any("local-only path marker" in item for item in violations))

    def test_placeholder_and_local_system_file_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [Path("unused/.gitkeep"), Path(".DS_Store")]
            for path in paths:
                self._materialize(root, path)
            violations = check_paths(root, paths)
            self.assertTrue(any("placeholder" in item for item in violations))
            self.assertTrue(any("local-system" in item for item in violations))

    @unittest.skipIf(os.name == "nt", "symlink creation is not reliable on Windows")
    def test_symbolic_link_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = Path("target.py")
            link = Path("Model/link.py")
            self._materialize(root, target)
            (root / link).parent.mkdir(parents=True)
            (root / link).symlink_to(root / target)
            violations = check_paths(root, [link])
            self.assertTrue(any("symbolic link" in item for item in violations))

    def test_oversized_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = Path("Model/large_source.py")
            (root / path).parent.mkdir(parents=True)
            with (root / path).open("wb") as stream:
                stream.truncate(MAX_TRACKED_BYTES + 1)
            violations = check_paths(root, [path])
            self.assertTrue(any("exceeds" in item for item in violations))


if __name__ == "__main__":
    unittest.main()
