# -*- coding: utf-8 -*-

import json
import os
import tempfile
import unittest

from Tokenizer.morphbpe import MorphBPETrainer
from Tokenizer.unified.bundle import TokenizerBundle


class TokenizerBundleTest(unittest.TestCase):
    def _train_tiny_morphbpe(self, tmp: str) -> str:
        trainer = MorphBPETrainer(vocab_size=200, min_pair_freq=1)
        tokenizer = trainer.train(["ᠮᠣᠩᠭᠣᠯ ᠪᠢᠴᠢᠭ", "ᠮᠣᠩᠭᠣᠯ text"])
        path = os.path.join(tmp, "tiny_morphbpe.json")
        tokenizer.save(path)
        return path

    def test_bundle_save_load_encode_and_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            morphbpe_path = self._train_tiny_morphbpe(tmp)
            bundle = TokenizerBundle.from_files(morphbpe_path)
            out_dir = os.path.join(tmp, "bundle")
            bundle.save_dir(out_dir)

            self.assertTrue(os.path.exists(os.path.join(out_dir, "config.json")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "morphbpe.json")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "general.json")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "vocab.json")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "manifest.json")))
            with open(os.path.join(out_dir, "manifest.json"), "r", encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertIn("vocab.json", manifest["files"])

            loaded = TokenizerBundle.from_dir(out_dir)
            self.assertEqual(loaded.validate(), [])

            encoded = loaded.encode_with_spans("ᠮᠣᠩᠭᠣᠯ 文字 test", add_bos=True, add_eos=True)
            self.assertEqual(len(encoded.input_ids), len(encoded.tokens))
            self.assertGreater(len(encoded.input_ids), 4)
            self.assertEqual(encoded.tokens[0].token, "<bos>")
            self.assertEqual(encoded.tokens[-1].token, "<eos>")

            mm = loaded.encode_multimodal(
                "文字 <image> test",
                images=["img"],
                image_sizes=[(14, 14)],
            )
            self.assertEqual(len(mm.image_token_spans), 1)
            start, end = mm.image_token_spans[0]
            self.assertEqual([tok.token for tok in mm.tokens[start:end]], [
                "<image_start>",
                "<image_patch>",
                "<image_end>",
            ])
            self.assertEqual(len(mm.attention_mask), len(mm.input_ids))

            with open(os.path.join(out_dir, "vocab.json"), "a", encoding="utf-8") as f:
                f.write("\n")
            issues = loaded.validate()
            self.assertTrue(
                any("manifest hash mismatch for vocab.json" in issue for issue in issues)
            )

    def test_from_dir_loads_legacy_v1_config(self):
        # Pre-v2 bundles stored zh_source/en_source/use_smoke_hf in config.json.
        # Those keys must be dropped (with a warning), not raise TypeError.
        import json
        import warnings

        with tempfile.TemporaryDirectory() as tmp:
            morphbpe_path = self._train_tiny_morphbpe(tmp)
            bundle = TokenizerBundle.from_files(morphbpe_path)
            out_dir = os.path.join(tmp, "bundle")
            bundle.save_dir(out_dir)

            config_path = os.path.join(out_dir, "config.json")
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg.update(
                {
                    "zh_source": "qwen",
                    "en_source": "gpt2",
                    "use_smoke_hf": True,
                }
            )
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f)
            os.remove(os.path.join(out_dir, "manifest.json"))

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loaded = TokenizerBundle.from_dir(out_dir)
            self.assertEqual(loaded.validate(), [])
            self.assertFalse(hasattr(loaded.config, "zh_source"))
            self.assertTrue(
                any("legacy" in str(w.message).lower() for w in caught),
                "expected a warning about legacy config keys",
            )

    def test_from_dir_rejects_unknown_config_key(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            morphbpe_path = self._train_tiny_morphbpe(tmp)
            bundle = TokenizerBundle.from_files(morphbpe_path)
            out_dir = os.path.join(tmp, "bundle")
            bundle.save_dir(out_dir)

            config_path = os.path.join(out_dir, "config.json")
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["totally_unknown_key"] = 1
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f)

            with self.assertRaises(TypeError):
                TokenizerBundle.from_dir(out_dir)


if __name__ == "__main__":
    unittest.main()
