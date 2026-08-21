# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from Model.posttrain.release_contract import (
    MONTLOK_DOL_1_2_OCR_REPOSITORY_ID,
    MONTLOK_DOL_1_2_OCR_REVISION,
    TRUSTED_REVIEWED_OCR_RELEASE_LOCKS,
    VISUAL_SOURCE_CONTRACT_ALIGNMENT_V3,
    admit_visual_ocr_source,
    validate_reviewed_streaming_v2_ocr_release,
)
from Model.ocr.visual_input_contract import (
    DOL_OCR_ANYRES_V2,
    DOL_OCR_LINE_LETTERBOX_224_V1,
)
from Model.ocr.tokenization import (
    native_tokenization_contract,
    tokenizer_vocab_sha256,
)
from Tokenizer.morphbpe import MorphBPETrainer
from Tokenizer.unified.bundle import TokenizerBundle
from Tokenizer.unified.contract import tokenizer_bundle_contract


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {"size": len(payload), "sha256": _sha(payload)}


def _canonical_sha(value: object) -> str:
    return _sha(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


class _ReleaseFixture:
    repository_id = "example/ocr"
    revision = "1" * 40

    def __init__(
        self,
        base: Path,
        mutate_metadata=None,
        *,
        declare_visual_input_contract: bool = True,
    ):
        self.root = base / "release"
        self.root.mkdir()
        self.tokenizer_root = self.root / "tokenizer"
        (self.root / "model.pt").write_bytes(b"not-deserialized-model")
        trainer = MorphBPETrainer(vocab_size=200, min_pair_freq=1)
        morphbpe = trainer.train(["ᠮᠣᠩᠭᠣᠯ ᠪᠢᠴᠢᠭ", "ᠮᠣᠩᠭᠣᠯ text"])
        morph_path = base / "tiny-morphbpe.json"
        morphbpe.save(str(morph_path))
        TokenizerBundle.from_files(str(morph_path)).save_dir(
            str(self.tokenizer_root)
        )
        self.bundle = TokenizerBundle.from_dir(str(self.tokenizer_root))
        manifest = json.loads(
            (self.tokenizer_root / "manifest.json").read_text(encoding="utf-8")
        )
        self.manifest_canonical_sha = _canonical_sha(manifest)
        self.bundle_contract = tokenizer_bundle_contract(self.tokenizer_root)
        self.token_id_map_sha = tokenizer_vocab_sha256(self.bundle.tokenizer)
        self.runtime_contract = native_tokenization_contract(
            self.bundle.tokenizer, self.tokenizer_root
        )
        self.vocab_extent = max(self.bundle.tokenizer.vocab.values()) + 1
        self.corpus_manifest = {
            "version": 1,
            "seed": 42,
            "mix_schedule": ["wds", "wds", "hanshi"],
            "wds": {
                "shards": [{"id": 0, "name": "shard-00000.tar", "size": 100}]
            },
            "hanshi": {
                "meta_name": "meta.jsonl",
                "meta_size": 100,
                "pages_name": "pages",
            },
            "excluded_shard_ids": [],
            "split": {
                "train": [0, 10],
                "val": [10, 20],
                "test": [20, None],
                "group_key": "src_doc",
            },
        }
        self.cursor = {
            "version": 1,
            "mix_position": 3,
            "counts": {"total": 3, "wds": 2, "hanshi": 1},
            "exhausted": {"wds": False, "hanshi": False},
            "wds": {"shard_position": 0, "sample_position": 2},
            "hanshi": {"line_number": 1, "byte_offset": 12},
        }
        metadata = {
            "phase": "vlm_align",
            "freeze_rdt": True,
            "frozen_vision": False,
            "final": True,
            "stop_reason": "loss_plateau",
            "ocr_position_contract": "boundary_v1",
            "ocr_position_contract_version": 1,
            "ocr_target_encoding": "native",
            "ocr_tokenization_contract_version": 2,
            "rdt_config": {"vocab_size": self.vocab_extent},
            "streaming": {
                "corpus_complete": False,
                "corpus_cursor": self.cursor,
                "corpus_manifest": self.corpus_manifest,
                "corpus_manifest_sha256": _canonical_sha(self.corpus_manifest),
                "ocr_target_encoding": "native",
                "ocr_tokenization_contract_version": 2,
                "tokenizer_manifest_sha256": self.manifest_canonical_sha,
            },
        }
        if declare_visual_input_contract:
            metadata.update(
                ocr_visual_input_contract=DOL_OCR_LINE_LETTERBOX_224_V1,
                ocr_visual_input_contract_version=1,
            )
        if mutate_metadata is not None:
            mutate_metadata(metadata)
        torch.save({"step": 7, "metadata": metadata}, self.root / "meta.pt")
        self._write_manifests_and_lock()

    def _write_manifests_and_lock(self) -> None:
        top_names = ["model.pt", "meta.pt"]
        tokenizer_names = [
            "config.json",
            "general.json",
            "manifest.json",
            "morphbpe.json",
            "vocab.json",
        ]
        backup = {
            "schema_version": 1,
            "release_name": "fixture-release",
            "parameter_count": 123,
            "files": {
                name: _file_record(self.root / name) for name in top_names
            },
            "tokenizer": {
                "path": "tokenizer",
                "files": {
                    name: _file_record(self.tokenizer_root / name)
                    for name in tokenizer_names
                },
            },
        }
        backup_path = self.root / "BACKUP_MANIFEST.json"
        backup_path.write_text(json.dumps(backup, indent=2) + "\n", encoding="utf-8")
        rows = []
        for name in top_names:
            rows.append(f"{backup['files'][name]['sha256']}  {name}")
        for name in tokenizer_names:
            rows.append(
                f"{backup['tokenizer']['files'][name]['sha256']}  tokenizer/{name}"
            )
        sums_path = self.root / "SHA256SUMS"
        sums_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        manifest_path = self.tokenizer_root / "manifest.json"
        self.lock_payload = {
            "schema_version": 1,
            "kind": "reviewed_streaming_v2_ocr_release",
            "source": {
                "repository_id": self.repository_id,
                "revision": self.revision,
            },
            "release": {
                "name": "fixture-release",
                "parameter_count": 123,
                "backup_manifest_path": "BACKUP_MANIFEST.json",
                "backup_manifest_raw_sha256": _sha(backup_path.read_bytes()),
                "sha256sums_path": "SHA256SUMS",
                "sha256sums_raw_sha256": _sha(sums_path.read_bytes()),
                "model_path": "model.pt",
                "model_raw_sha256": backup["files"]["model.pt"]["sha256"],
                "metadata_path": "meta.pt",
                "metadata_raw_sha256": backup["files"]["meta.pt"]["sha256"],
            },
            "tokenizer": {
                "root": "tokenizer",
                "manifest_path": "tokenizer/manifest.json",
                "manifest_raw_sha256": _sha(manifest_path.read_bytes()),
                "manifest_canonical_sha256": self.manifest_canonical_sha,
                "bundle_files_canonical_sha256": self.bundle_contract[
                    "files_canonical_sha256"
                ],
                "vocab_path": "tokenizer/vocab.json",
                "vocab_file_raw_sha256": backup["tokenizer"]["files"]["vocab.json"][
                    "sha256"
                ],
                "token_id_map_sha256": self.token_id_map_sha,
            },
            "contracts": {
                "phase": "vlm_align",
                "freeze_rdt": True,
                "frozen_vision": False,
                "final": True,
                "stop_reason": "loss_plateau",
                "ocr_position_contract": "boundary_v1",
                "ocr_position_contract_version": 1,
                "ocr_target_encoding": "native",
                "ocr_tokenization_contract_version": 2,
                "checkpoint_step": 7,
                "tokenizer_id_extent": self.vocab_extent,
                "model_vocab_capacity": self.vocab_extent,
            },
            "streaming": {
                "corpus_manifest_canonical_sha256": _canonical_sha(
                    self.corpus_manifest
                ),
                "tokenizer_manifest_canonical_sha256": (
                    self.manifest_canonical_sha
                ),
                "cursor_canonical_sha256": _canonical_sha(self.cursor),
                "samples_exposed": 3,
                "corpus_complete": False,
            },
        }
        self.lock_path = self.root.parent / "reviewed-lock.json"
        self.write_lock()

    def write_lock(self) -> None:
        self.lock_path.write_text(
            json.dumps(self.lock_payload, indent=2) + "\n", encoding="utf-8"
        )

    def trust_patch(self):
        return mock.patch.dict(
            TRUSTED_REVIEWED_OCR_RELEASE_LOCKS,
            {(self.repository_id, self.revision): _sha(self.lock_path.read_bytes())},
        )


class ReviewedOCRReleaseContractTests(unittest.TestCase):
    def _validate(self, fixture: _ReleaseFixture, **kwargs):
        with fixture.trust_patch():
            return validate_reviewed_streaming_v2_ocr_release(
                fixture.root,
                fixture.lock_path,
                runtime_native_tokenization_contract=fixture.runtime_contract,
                **kwargs,
            )

    def test_accepts_verified_release_and_keeps_hash_domains_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _ReleaseFixture(Path(tmp))
            real_load = torch.load
            with mock.patch(
                "Model.posttrain.release_contract.torch.load",
                wraps=real_load,
            ) as load:
                result = self._validate(
                    fixture,
                    expected_repository_id=fixture.repository_id,
                    expected_revision=fixture.revision,
                )
            load.assert_called_once()
            self.assertTrue(load.call_args.kwargs["weights_only"])
            self.assertEqual(result["metadata"]["ocr_position_contract"], "boundary_v1")
            lineage = result["lineage"]
            hash_domains = {
                lineage["tokenizer_manifest_raw_sha256"],
                lineage["tokenizer_manifest_canonical_sha256"],
                lineage["tokenizer_bundle_files_canonical_sha256"],
                lineage["tokenizer_token_id_map_sha256"],
            }
            self.assertEqual(len(hash_domains), 4)
            self.assertEqual(lineage["visual_samples_exposed"], 3)
            self.assertEqual(
                lineage["ocr_visual_input_contract"],
                DOL_OCR_LINE_LETTERBOX_224_V1,
            )
            self.assertEqual(lineage["ocr_visual_input_contract_version"], 1)

    def test_byte_tamper_fails_before_any_torch_deserialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _ReleaseFixture(Path(tmp))
            (fixture.tokenizer_root / "general.json").write_bytes(b"tampered")
            with mock.patch(
                "Model.posttrain.release_contract.torch.load"
            ) as load, self.assertRaisesRegex(ValueError, "size mismatch"):
                self._validate(fixture)
            load.assert_not_called()

    def test_rejects_path_traversal_and_duplicate_checksum_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _ReleaseFixture(Path(tmp))
            fixture.lock_payload["release"]["model_path"] = "../model.pt"
            fixture.write_lock()
            with self.assertRaisesRegex(ValueError, "normalized relative path"):
                self._validate(fixture)

        with tempfile.TemporaryDirectory() as tmp:
            fixture = _ReleaseFixture(Path(tmp))
            sums_path = fixture.root / "SHA256SUMS"
            first_line = sums_path.read_text(encoding="utf-8").splitlines()[0]
            sums_path.write_text(
                sums_path.read_text(encoding="utf-8") + first_line + "\n",
                encoding="utf-8",
            )
            fixture.lock_payload["release"]["sha256sums_raw_sha256"] = _sha(
                sums_path.read_bytes()
            )
            fixture.write_lock()
            with mock.patch(
                "Model.posttrain.release_contract.torch.load"
            ) as load, self.assertRaisesRegex(ValueError, "duplicate SHA256SUMS path"):
                self._validate(fixture)
            load.assert_not_called()

    def test_requires_external_lock_and_honors_source_pins(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _ReleaseFixture(Path(tmp))
            internal_lock = fixture.root / "lock.json"
            internal_lock.write_bytes(fixture.lock_path.read_bytes())
            with self.assertRaisesRegex(ValueError, "external"):
                validate_reviewed_streaming_v2_ocr_release(
                    fixture.root,
                    internal_lock,
                    runtime_native_tokenization_contract=fixture.runtime_contract,
                )
            with self.assertRaisesRegex(ValueError, "repository mismatch"):
                self._validate(
                    fixture,
                    expected_repository_id="wrong/repository",
                )
            with self.assertRaisesRegex(ValueError, "revision mismatch"):
                self._validate(
                    fixture,
                    expected_revision="2" * 40,
                )

    def test_rejects_unanchored_and_symlink_locks(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _ReleaseFixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, "trust registry"):
                validate_reviewed_streaming_v2_ocr_release(
                    fixture.root,
                    fixture.lock_path,
                    runtime_native_tokenization_contract=fixture.runtime_contract,
                )
            symlink = Path(tmp) / "symlink-lock.json"
            symlink.symlink_to(fixture.lock_path)
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                validate_reviewed_streaming_v2_ocr_release(
                    fixture.root,
                    symlink,
                    runtime_native_tokenization_contract=fixture.runtime_contract,
                )

    def test_runtime_native_v3_mapping_must_match_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _ReleaseFixture(Path(tmp))
            runtime = dict(fixture.runtime_contract)
            runtime["tokenizer_vocab_sha256"] = "0" * 64
            with fixture.trust_patch(), self.assertRaisesRegex(
                ValueError, "token-id map"
            ):
                validate_reviewed_streaming_v2_ocr_release(
                    fixture.root,
                    fixture.lock_path,
                    runtime_native_tokenization_contract=runtime,
                )

    def test_production_lock_uses_independent_golden_hash_domains(self):
        repo_root = Path(__file__).resolve().parents[2]
        lock_path = (
            repo_root
            / "Model"
            / "posttrain"
            / "release_locks"
            / "dol_1_2_ocr.json"
        )
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        self.assertEqual(
            _sha(lock_path.read_bytes()),
            "01a291e921dab65d89a1bfd07570c8089a2427a7add1e28d7db30dced46a6044",
        )
        tokenizer = payload["tokenizer"]
        self.assertEqual(tokenizer["manifest_raw_sha256"], "dbee445fc5573b70306ff84cdec182b798111384c9c2a463abe14574cebac737")
        self.assertEqual(tokenizer["manifest_canonical_sha256"], "8fa21f2cac49dc6595f3b746460b36b79db77c5602fd67788ba57d52c52c5f6f")
        self.assertEqual(tokenizer["bundle_files_canonical_sha256"], "c6d2c3ba18f5e358deaf2c0d9b7ef503efa61dad270d669bbc2ee622a48a4361")
        self.assertEqual(tokenizer["token_id_map_sha256"], "e0338bb7b9c09cd9c74a2434dd8f37a3943c1f124228ebb44d6fc881bd047053")

    def test_exact_official_release_identity_binds_legacy_metadata_to_line_v1(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(
                _ReleaseFixture,
                "repository_id",
                MONTLOK_DOL_1_2_OCR_REPOSITORY_ID,
            ),
            mock.patch.object(
                _ReleaseFixture,
                "revision",
                MONTLOK_DOL_1_2_OCR_REVISION,
            ),
        ):
            fixture = _ReleaseFixture(
                Path(tmp),
                declare_visual_input_contract=False,
            )
            result = self._validate(
                fixture,
                expected_repository_id=MONTLOK_DOL_1_2_OCR_REPOSITORY_ID,
                expected_revision=MONTLOK_DOL_1_2_OCR_REVISION,
            )
            self.assertNotIn("ocr_visual_input_contract", result["metadata"])
            self.assertEqual(
                result["lineage"]["ocr_visual_input_contract"],
                DOL_OCR_LINE_LETTERBOX_224_V1,
            )
            self.assertEqual(
                result["lineage"]["ocr_visual_input_contract_version"], 1
            )

    def test_official_line_v1_release_rejects_anyres_declaration(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(
                _ReleaseFixture,
                "repository_id",
                MONTLOK_DOL_1_2_OCR_REPOSITORY_ID,
            ),
            mock.patch.object(
                _ReleaseFixture,
                "revision",
                MONTLOK_DOL_1_2_OCR_REVISION,
            ),
        ):
            fixture = _ReleaseFixture(
                Path(tmp),
                mutate_metadata=lambda meta: meta.update(
                    ocr_visual_input_contract=DOL_OCR_ANYRES_V2,
                    ocr_visual_input_contract_version=2,
                ),
            )
            with self.assertRaisesRegex(
                ValueError, "OCR visual input contract mismatch"
            ):
                self._validate(fixture)

    def test_dispatcher_has_no_contract_fallback_and_checks_full_lineage(self):
        metadata = {
            "phase": "vlm_align",
            "ocr_position_contract": "boundary_v1",
            "ocr_position_contract_version": 1,
            "ocr_visual_input_contract": DOL_OCR_LINE_LETTERBOX_224_V1,
            "ocr_visual_input_contract_version": 1,
        }
        runtime = {"target_encoding": "native", "tokenization_contract_version": 3}
        with mock.patch(
            "Model.posttrain.checkpointing.validate_visual_ocr_source_contract",
            return_value={"source_checkpoint_model_sha256": "a" * 64},
        ):
            admitted = admit_visual_ocr_source(
                VISUAL_SOURCE_CONTRACT_ALIGNMENT_V3,
                checkpoint_dir="fixture",
                metadata=metadata,
                runtime_native_tokenization_contract=runtime,
                tokenizer_vocab_extent=1,
            )
            self.assertEqual(
                admitted["lineage"]["source_contract_kind"], "alignment-v3"
            )
            self.assertEqual(
                admitted["lineage"]["ocr_position_contract"], "boundary_v1"
            )
            self.assertEqual(
                admitted["lineage"]["ocr_visual_input_contract"],
                DOL_OCR_LINE_LETTERBOX_224_V1,
            )
            with self.assertRaisesRegex(
                ValueError, "OCR visual input contract mismatch"
            ):
                admit_visual_ocr_source(
                    VISUAL_SOURCE_CONTRACT_ALIGNMENT_V3,
                    checkpoint_dir="fixture",
                    metadata=metadata,
                    runtime_native_tokenization_contract=runtime,
                    tokenizer_vocab_extent=1,
                    requested_visual_input_contract=DOL_OCR_ANYRES_V2,
                )
            with self.assertRaisesRegex(ValueError, "expected lineage"):
                admit_visual_ocr_source(
                    VISUAL_SOURCE_CONTRACT_ALIGNMENT_V3,
                    checkpoint_dir="fixture",
                    metadata=metadata,
                    runtime_native_tokenization_contract=runtime,
                    tokenizer_vocab_extent=1,
                    expected_lineage={"different": True},
                )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            admit_visual_ocr_source(
                "unknown",
                checkpoint_dir="fixture",
                metadata=metadata,
                runtime_native_tokenization_contract=runtime,
                tokenizer_vocab_extent=1,
            )

    def test_rejects_each_required_training_semantic_after_valid_hashing(self):
        mutations = {
            "position": lambda meta: meta.update(
                ocr_position_contract="legacy_sequential_v0"
            ),
            "position-version": lambda meta: meta.update(
                ocr_position_contract_version=2
            ),
            "encoding": lambda meta: meta.update(ocr_target_encoding="bytes"),
            "tokenization-version": lambda meta: meta.update(
                ocr_tokenization_contract_version=3
            ),
            "not-final": lambda meta: meta.update(final=False),
            "stop-reason": lambda meta: meta.update(stop_reason="max_steps"),
            "streaming-encoding": lambda meta: meta["streaming"].update(
                ocr_target_encoding="bytes"
            ),
            "streaming-token-version": lambda meta: meta["streaming"].update(
                ocr_tokenization_contract_version=3
            ),
            "streaming-manifest": lambda meta: meta["streaming"][
                "corpus_manifest"
            ].update(seed=99),
            "streaming-cursor": lambda meta: meta["streaming"][
                "corpus_cursor"
            ].update(mix_position=4),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                fixture = _ReleaseFixture(Path(tmp), mutate_metadata=mutate)
                with self.assertRaises(ValueError):
                    self._validate(fixture)

    def test_duplicate_json_keys_are_rejected_before_torch_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _ReleaseFixture(Path(tmp))
            fixture.lock_path.write_text(
                '{"schema_version":1,"schema_version":1}\n', encoding="utf-8"
            )
            with mock.patch(
                "Model.posttrain.release_contract.torch.load"
            ) as load, self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                validate_reviewed_streaming_v2_ocr_release(
                    fixture.root,
                    fixture.lock_path,
                    runtime_native_tokenization_contract=fixture.runtime_contract,
                )
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
