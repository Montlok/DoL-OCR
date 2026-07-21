from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.check_repository_hygiene import MAX_TRACKED_BYTES, check_paths


class RepositoryHygieneTest(unittest.TestCase):
    def test_clean_source_file_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = Path("src/module.py")
            (root / path).parent.mkdir(parents=True)
            (root / path).write_text("value = 1\n", encoding="utf-8")
            self.assertEqual(check_paths(root, [path]), [])

    def test_artifact_and_placeholder_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = Path("weights/model.safetensors")
            placeholder = Path("unused/.gitkeep")
            for path in (artifact, placeholder):
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                (root / path).touch()
            violations = check_paths(root, [artifact, placeholder])
            self.assertTrue(any("artifact or corpus" in item for item in violations))
            self.assertTrue(any("placeholder" in item for item in violations))

    def test_oversized_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = Path("large-source.txt")
            with (root / path).open("wb") as stream:
                stream.truncate(MAX_TRACKED_BYTES + 1)
            violations = check_paths(root, [path])
            self.assertTrue(any("exceeds" in item for item in violations))


if __name__ == "__main__":
    unittest.main()
