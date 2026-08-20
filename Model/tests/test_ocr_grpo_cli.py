# -*- coding: utf-8 -*-

"""CLI contract tests for OCR RL validation and locked-golden evaluation."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from Model.config import OMVTConfig, RDTConfig
from Model.model import RDTForCausalLM
from Model.ocr.alignment_contract import (
    OCR_ALIGNMENT_DATA_SCHEMA_VERSION,
    OCR_IMAGE_BINDING_MODE,
)
from Model.ocr.position_contract import (
    BOUNDARY_V1,
    OCR_POSITION_CONTRACT_METADATA_VERSION,
)
from Model.ocr.visual_input_contract import DOL_OCR_LINE_LETTERBOX_224_V1
from Model.ocr.tokenization import (
    OCR_NATIVE_TARGET_ENCODING,
    OCR_TOKENIZATION_CONTRACT_VERSION,
    canonical_json_sha256,
    canonicalize_native_ocr_text,
    native_tokenization_contract,
)
from Model.omvt import OMVTInjector
from Model.posttrain.checkpointing import OCR_GRPO_CONTRACT_VERSION
from Model.posttrain.ocr_manifests import (
    golden_identity_semantic_sha256,
    load_golden_identity_manifest,
)


def _rdt() -> RDTConfig:
    return RDTConfig(
        d_model=32,
        n_heads=4,
        head_dim=8,
        kv_lora_rank=8,
        rope_head_dim=4,
        nope_head_dim=4,
        ffn_hidden=64,
        ffn_multiple=32,
        n_prelude=1,
        n_coda=1,
        mamba_per_block=1,
        attn_per_block=1,
        recurrent_steps=1,
        mamba_d_state=8,
        mamba_expand=2,
        mamba_headdim=16,
        use_official_mamba=False,
        max_seq_len=24,
    )


def _omvt() -> OMVTConfig:
    return OMVTConfig(
        image_size=16,
        vertical_patch=(8, 4),
        horizontal_patch=(4, 8),
        square_patch=(4, 4),
        layout_patch=(16, 16),
        d_vision=32,
        vision_n_heads=4,
        vision_ffn_hidden=64,
        compress_to=2,
        compressor_layers=1,
        compressor_heads=4,
        n_vertical_layers=1,
        n_horizontal_layers=1,
        n_local_attn_layers=1,
        n_layout_layers=1,
    )


def _checkpoint(
    root: Path,
    metadata: dict,
    *,
    step_number: int = 0,
) -> Path:
    rdt_cfg, omvt_cfg = _rdt(), _omvt()
    model = RDTForCausalLM(rdt_cfg)
    model.vision._omvt_cfg = omvt_cfg
    model.vision.omvt = OMVTInjector(rdt_cfg, omvt_cfg)
    step = root / f"step_{step_number:08d}"
    step.mkdir(parents=True)
    torch.save(model.state_dict(), step / "model.pt")
    payload = {
        "rdt_config": asdict(rdt_cfg),
        "omvt_config": asdict(omvt_cfg),
        **metadata,
    }
    torch.save(
        {"step": step_number, "metadata": payload},
        step / "meta.pt",
    )
    (step / "COMPLETE").write_text(
        f"step={step_number}\n",
        encoding="ascii",
    )
    return step


def _receipt_checkpoint(
    root: Path,
    metadata: dict,
    *,
    step_number: int,
) -> Path:
    """Small checkpoint fixture for receipt validation without model rebuilds."""

    step = root / f"step_{step_number:08d}"
    step.mkdir(parents=True)
    (step / "model.pt").write_bytes(f"model-step-{step_number}".encode())
    torch.save(
        {"step": step_number, "metadata": metadata},
        step / "meta.pt",
    )
    (step / "COMPLETE").write_text(
        f"step={step_number}\n",
        encoding="ascii",
    )
    return step


def _image_and_row(root: Path, name: str, split: str) -> dict:
    from PIL import Image

    image = root / f"{name}.png"
    color = tuple(hashlib.sha256(name.encode("utf-8")).digest()[:3])
    Image.new("RGB", (8, 12), color).save(image)
    return {
        "id": name,
        "group_id": f"group-{name}",
        "split": split,
        "image": image.name,
        "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "reference": "a",
        "domain": "photo",
    }


class _FakeTokenizer:
    vocab = {"a": 300, "<unused>": 65535}
    id_to_token = {300: "a", 65535: "<unused>"}
    unk_id = 1
    general_global_to_local = {}
    mn_local_to_global = {0: 300}

    class general:
        @staticmethod
        def decode(_ids):
            return ""

    @staticmethod
    def decode(ids):
        return "".join("a" for token in ids if token == 300)

    @staticmethod
    def encode_plain_text(text):
        if text != "a":
            raise ValueError(f"test tokenizer only supports 'a', got {text!r}")
        return [300]

    @staticmethod
    def encode_with_spans(
        text,
        add_bos=False,
        add_eos=False,
        interpret_special_tokens=True,
    ):
        from Tokenizer.unified.encoded import DualTrackResult, EncodedToken

        del add_bos, add_eos, interpret_special_tokens
        if text != "a":
            raise ValueError(f"test tokenizer only supports 'a', got {text!r}")
        return DualTrackResult(
            input_ids=[300],
            tokens=[EncodedToken(300, "a", "general", 0, 1)],
            spans=[],
        )


class _FakeBundle:
    tokenizer = _FakeTokenizer()

    class config:
        morphbpe_file = "morphbpe.json"
        general_file = "general.json"

    @staticmethod
    def validate():
        return []

    @staticmethod
    def encode(_text, add_bos=False, add_eos=False):
        del add_bos, add_eos
        return [300]


def _tokenizer_contract(root: Path) -> tuple[Path, dict]:
    tokenizer_dir = root / "tokenizer"
    tokenizer_dir.mkdir()
    for name in (
        "config.json",
        "morphbpe.json",
        "general.json",
        "vocab.json",
        "manifest.json",
    ):
        (tokenizer_dir / name).write_text(
            json.dumps({"schema_version": 1, "name": name}) + "\n",
            encoding="utf-8",
        )
    with patch(
        "Tokenizer.unified.contract.TokenizerBundle.from_dir",
        return_value=_FakeBundle(),
    ):
        contract = native_tokenization_contract(
            _FakeTokenizer(),
            tokenizer_dir,
        )
    return tokenizer_dir, contract


def _identity_row(labeled_row: dict) -> dict:
    return {
        "schema_version": 1,
        "id": labeled_row["id"],
        "group_id": labeled_row["group_id"],
        "split": "golden",
        "image_sha256": labeled_row["sha256"],
    }


def _public_dataset_contract(
    root: Path,
    *,
    train: Path,
    validation: Path,
    identity: Path,
    token_contract: dict,
    locked_receipt_sha256: str = "f" * 64,
) -> Path:
    split_lock = root / "split_lock.json"
    split_lock.write_text(
        json.dumps({"schema_version": 1}) + "\n",
        encoding="utf-8",
    )
    identity_rows = load_golden_identity_manifest(identity)
    payload = {
        "schema_version": 2,
        "image_root": ".",
        "images_materialized": True,
        "ocr_tokenization_contract": token_contract,
        "locked_golden_receipt_sha256": locked_receipt_sha256,
        "split_lock_sha256": hashlib.sha256(
            split_lock.read_bytes()
        ).hexdigest(),
        "manifests": {
            "rl_train_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
            "rl_val_sha256": hashlib.sha256(
                validation.read_bytes()
            ).hexdigest(),
            "golden_identity_sha256": hashlib.sha256(
                identity.read_bytes()
            ).hexdigest(),
            "golden_identity_semantic_sha256": (
                golden_identity_semantic_sha256(identity_rows)
            ),
        },
        "reference_symbol_support": {
            "rl_train": {},
            "rl_val": {},
        },
    }
    path = root / "dataset_contract.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _locked_build_receipt(
    root: Path,
    *,
    golden: Path,
    identity: Path,
) -> Path:
    identity_rows = load_golden_identity_manifest(identity)
    payload = {
        "schema_version": 2,
        "kind": "ocr_locked_golden_build_receipt",
        "locked_golden_sha256": hashlib.sha256(
            golden.read_bytes()
        ).hexdigest(),
        "golden_identity_sha256": hashlib.sha256(
            identity.read_bytes()
        ).hexdigest(),
        "golden_identity_semantic_sha256": (
            golden_identity_semantic_sha256(identity_rows)
        ),
    }
    path = root / "build_receipt.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _visual_metadata(token_contract: dict) -> dict:
    entries = [
        {
            "name": "align.jsonl",
            "sha256": "f" * 64,
            "size_bytes": 1,
        }
    ]
    return {
        "phase": "vlm_align",
        "freeze_rdt": True,
        "frozen_vision": False,
        "ocr_target_encoding": OCR_NATIVE_TARGET_ENCODING,
        "ocr_tokenization_contract_version": (
            OCR_TOKENIZATION_CONTRACT_VERSION
        ),
        "ocr_position_contract": BOUNDARY_V1,
        "ocr_position_contract_version": OCR_POSITION_CONTRACT_METADATA_VERSION,
        "ocr_visual_input_contract": DOL_OCR_LINE_LETTERBOX_224_V1,
        "ocr_visual_input_contract_version": 1,
        "final": True,
        "stop_reason": "loss_plateau",
        "source_rdt_tokenizer_bundle": token_contract["tokenizer_bundle"],
        "source_rdt_tokenizer_algorithm": token_contract[
            "pretraining_tokenizer_algorithm"
        ],
        "ocr_data_contract": {
            "schema_version": OCR_ALIGNMENT_DATA_SCHEMA_VERSION,
            "kind": "pretokenized_ocr_alignment",
            "data_layout": "single_jsonl",
            "data_file_count": 1,
            "data_files": entries,
            "data_sha256": canonical_json_sha256(entries),
            "image_binding": OCR_IMAGE_BINDING_MODE,
            "image_manifest_sha256": "8" * 64,
            "image_reference_count": 1,
            "image_total_size_bytes": 1,
            "shard_completion": {
                "mode": "direct_builder_v1",
                "sentinel_count": 0,
                "sentinels": [],
                "sentinel_manifest_sha256": canonical_json_sha256([]),
            },
            "ocr_tokenization_contract": token_contract,
        },
    }


class OCRGRPOCLITest(unittest.TestCase):
    def test_one_shot_claim_and_stub_are_durable_before_golden_hash(self):
        from scripts import eval_ocr_grpo

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            golden = root / "golden.jsonl"
            golden.write_text('{"reference":"secret"}\n', encoding="utf-8")
            digest = hashlib.sha256(golden.read_bytes()).hexdigest()
            marker = root / "ledger" / "receipt.json"
            destination = root / "golden_report.json"
            temporary = root / ".golden_report.json.tmp"
            temporary.touch()
            claim_payload = {
                "schema_version": 2,
                "kind": "ocr_locked_golden_claimed",
                "report_path": str(destination),
            }

            real_fsync_directory = eval_ocr_grpo._fsync_directory
            with patch.object(
                eval_ocr_grpo,
                "_fsync_directory",
                wraps=real_fsync_directory,
            ) as fsync_directory:

                def verify_after_claim(receipt, *, golden_manifest):
                    self.assertTrue(marker.is_file())
                    self.assertTrue(fsync_directory.called)
                    stub = json.loads(destination.read_text(encoding="utf-8"))
                    self.assertEqual(
                        stub["status"],
                        "claimed_evaluation_incomplete",
                    )
                    self.assertEqual(Path(golden_manifest), golden)
                    self.assertEqual(receipt["locked_golden_sha256"], digest)
                    return digest

                with patch.object(
                    eval_ocr_grpo,
                    "verify_locked_golden_manifest",
                    side_effect=verify_after_claim,
                ):
                    actual = eval_ocr_grpo._claim_and_verify_locked_golden(
                        consumption_marker=marker,
                        destination=destination,
                        temporary=temporary,
                        claim_payload=claim_payload,
                        golden_receipt={"locked_golden_sha256": digest},
                        golden_manifest=golden,
                    )

            self.assertEqual(actual, digest)
            self.assertFalse(temporary.exists())

    def test_post_claim_golden_hash_failure_keeps_consumed_incomplete_stub(self):
        from scripts import eval_ocr_grpo

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            golden = root / "golden.jsonl"
            golden.write_text('{"reference":"secret"}\n', encoding="utf-8")
            marker = root / "ledger" / "receipt.json"
            destination = root / "golden_report.json"
            temporary = root / ".golden_report.json.tmp"
            temporary.touch()
            claim_payload = {
                "schema_version": 2,
                "kind": "ocr_locked_golden_claimed",
                "report_path": str(destination),
            }

            with patch.object(
                eval_ocr_grpo,
                "verify_locked_golden_manifest",
                side_effect=RuntimeError("injected golden hash failure"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "injected golden hash failure",
                ):
                    eval_ocr_grpo._claim_and_verify_locked_golden(
                        consumption_marker=marker,
                        destination=destination,
                        temporary=temporary,
                        claim_payload=claim_payload,
                        golden_receipt={"locked_golden_sha256": "a" * 64},
                        golden_manifest=golden,
                    )

            self.assertTrue(marker.is_file())
            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8"))["status"],
                "claimed_evaluation_incomplete",
            )
            self.assertFalse(temporary.exists())

    def test_global_golden_ledger_rejects_symlink_and_non_directory(self):
        from scripts.eval_ocr_grpo import _global_golden_consumption_marker

        with tempfile.TemporaryDirectory() as tmp:
            locked_root = Path(tmp)
            ledger = locked_root / "golden_evaluation_ledger"
            ledger.write_text("not a directory", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a directory"):
                _global_golden_consumption_marker(locked_root, "a" * 64)

            ledger.unlink()
            real_directory = locked_root / "other"
            real_directory.mkdir()
            ledger.symlink_to(real_directory, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                _global_golden_consumption_marker(locked_root, "a" * 64)

    def test_terminal_receipt_binds_relocatable_final_best_state(self):
        from Model.posttrain.ocr_selection import (
            build_selection_receipt,
            load_and_validate_selection_receipt,
            write_selection_receipt,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = _receipt_checkpoint(
                root / "best",
                {},
                step_number=3,
            )
            data_contract = {"dataset": "immutable"}

            def terminal_metadata(**health_overrides):
                health = {
                    "stop_reason": "validation_plateau",
                    # A moved/archived run may retain the old absolute mount.
                    "best_checkpoint": (
                        "/old/mount/run/best/step_00000003"
                    ),
                    "best_val_step": 3,
                    "best_val_eligible": True,
                    **health_overrides,
                }
                return {
                    "phase": "grpo",
                    "task": "ocr",
                    "final": True,
                    "stop_reason": "validation_plateau",
                    "health_state": health,
                    "data_contract": data_contract,
                }

            terminal = _receipt_checkpoint(
                root,
                terminal_metadata(),
                step_number=10,
            )
            payload = build_selection_receipt(
                selected_checkpoint=selected,
                selected_step=3,
                terminal_checkpoint=terminal,
                terminal_step=10,
                stop_reason="validation_plateau",
                data_contract=data_contract,
            )
            receipt = selected.parent / "SELECTION_FINALIZED.json"
            write_selection_receipt(receipt, payload)
            loaded = load_and_validate_selection_receipt(
                receipt,
                selected_checkpoint=selected,
                data_contract=data_contract,
            )
            self.assertEqual(loaded["selected_step"], 3)

            failures = (
                (
                    {"best_checkpoint": "/old/run/best/step_00000002"},
                    "best_checkpoint differs",
                ),
                ({"best_val_step": 2}, "best_val_step differs"),
                ({"best_val_eligible": False}, "not eligible"),
            )
            for overrides, expected_error in failures:
                with self.subTest(overrides=overrides):
                    torch.save(
                        {
                            "step": 10,
                            "metadata": terminal_metadata(**overrides),
                        },
                        terminal / "meta.pt",
                    )
                    with self.assertRaisesRegex(ValueError, expected_error):
                        load_and_validate_selection_receipt(
                            receipt,
                            selected_checkpoint=selected,
                            data_contract=data_contract,
                        )
            torch.save(
                {"step": 10, "metadata": terminal_metadata()},
                terminal / "meta.pt",
            )
            noncanonical_selected = _receipt_checkpoint(
                root / "archive",
                {},
                step_number=3,
            )
            with self.assertRaisesRegex(
                ValueError,
                "terminal sibling best/<step>",
            ):
                build_selection_receipt(
                    selected_checkpoint=noncanonical_selected,
                    selected_step=3,
                    terminal_checkpoint=terminal,
                    terminal_step=10,
                    stop_reason="validation_plateau",
                    data_contract=data_contract,
                )

    def test_training_contract_failure_precedes_model_and_distributed_init(self):
        from scripts import train_grpo

        class _ProbeTokenizer:
            vocab = {"probe": 65535}
            last = ""

            def decode(self, _ids):
                return self.last

        class _ProbeBundle:
            tokenizer = _ProbeTokenizer()

            @staticmethod
            def validate():
                return []

        def encode_probe(text):
            _ProbeBundle.tokenizer.last = canonicalize_native_ocr_text(text)
            return [300]

        encode_probe.stats = {"byte_fallback": 0}
        token_contract = {
            "target_encoding": OCR_NATIVE_TARGET_ENCODING,
            "tokenization_contract_version": (
                OCR_TOKENIZATION_CONTRACT_VERSION
            ),
            "tokenizer_manifest_canonical_sha256": "manifest",
            "tokenizer_vocab_sha256": "vocab",
            "tokenizer_bundle": {
                "schema_version": 1,
                "files": [
                    {
                        "role": "vocab",
                        "name": "vocab.json",
                        "size_bytes": 1,
                        "sha256": "a" * 64,
                    }
                ],
                "files_canonical_sha256": "b" * 64,
            },
            "native_route": "dual-track-pretraining-special-aware",
            "mongolian_general_fallback": "forbidden",
            "reference_canonicalization": "native-pretraining-v3",
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = _checkpoint(root / "checkpoint", {})
            identity = root / "golden_identity.jsonl"
            identity.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "id": "golden",
                        "group_id": "group-golden",
                        "split": "golden",
                        "image_sha256": "a" * 64,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            train.touch()
            validation.touch()
            output = root / "must-not-exist"
            with (
                patch(
                    "Tokenizer.unified.bundle.TokenizerBundle.from_dir",
                    return_value=_ProbeBundle(),
                ),
                patch.object(
                    train_grpo,
                    "native_tokenization_contract",
                    return_value=token_contract,
                ),
                patch.object(
                    train_grpo,
                    "make_ocr_target_encoder",
                    return_value=encode_probe,
                ),
                patch.object(train_grpo, "init_distributed") as init_dist,
                patch.object(train_grpo, "RDTForCausalLM") as model_ctor,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "OCR visual input contract metadata",
                ):
                    train_grpo.main(
                        [
                            "--task",
                            "ocr",
                            "--tokenizer",
                            str(root / "tokenizer"),
                            "--data",
                            str(train),
                            "--validation-manifest",
                            str(validation),
                            "--golden-identity-manifest",
                            str(identity),
                            "--init-checkpoint",
                            str(checkpoint),
                            "--output",
                            str(output),
                        ]
                    )
            init_dist.assert_not_called()
            model_ctor.assert_not_called()
            self.assertFalse(output.exists())

    def test_fresh_training_reconstructs_only_the_admitted_source_identity(self):
        from scripts import train_grpo

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _checkpoint(root / "admitted", {})
            admitted_metadata = {"phase": "vlm_align"}
            model_sha256 = "a" * 64
            metadata_sha256 = "b" * 64
            lineage = {
                "source_checkpoint": str(source),
                "source_checkpoint_model_sha256": model_sha256,
                "source_checkpoint_metadata_sha256": metadata_sha256,
            }
            restored = SimpleNamespace(
                checkpoint_dir=source,
                metadata=admitted_metadata,
                model_sha256=model_sha256,
                metadata_sha256=metadata_sha256,
            )
            args = SimpleNamespace(
                resume="",
                init_checkpoint=str(root / "raw-cli-alias-must-not-be-used"),
                task="ocr",
                smoke=False,
            )
            fallback = _rdt()
            with patch.object(
                train_grpo,
                "reconstruct_policy_from_checkpoint",
                return_value=restored,
            ) as reconstruct:
                actual = train_grpo._reconstruct_requested_policy(
                    args,
                    fallback,
                    lineage,
                    admitted_metadata,
                )

            self.assertIs(actual, restored)
            reconstruct.assert_called_once_with(
                source,
                fallback_rdt_config=fallback,
                require_vision=True,
                metadata_override=admitted_metadata,
                expected_metadata_sha256=metadata_sha256,
                expected_model_sha256=model_sha256,
            )

            missing_metadata_hash = dict(lineage)
            missing_metadata_hash.pop("source_checkpoint_metadata_sha256")
            with patch.object(
                train_grpo,
                "reconstruct_policy_from_checkpoint",
            ) as reconstruct:
                with self.assertRaisesRegex(
                    ValueError,
                    "source_checkpoint_metadata_sha256",
                ):
                    train_grpo._reconstruct_requested_policy(
                        args,
                        fallback,
                        missing_metadata_hash,
                        admitted_metadata,
                    )
            reconstruct.assert_not_called()

    def test_fresh_source_reconstruction_failure_does_not_create_output(self):
        from scripts import train_grpo

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "must-not-exist"
            with (
                patch.object(train_grpo, "_validate_args", return_value=0),
                patch.object(
                    train_grpo,
                    "_preflight_artifacts",
                    return_value=(None, None, {}, {}, {}, {}),
                ),
                patch.object(
                    train_grpo,
                    "init_distributed",
                    return_value=(0, 1, 0),
                ) as init_distributed,
                patch.object(
                    train_grpo,
                    "_reconstruct_requested_policy",
                    side_effect=ValueError("injected admitted source failure"),
                ),
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "injected admitted source failure",
                ):
                    train_grpo.main(
                        [
                            "--task",
                            "ocr",
                            "--init-checkpoint",
                            "unused-cli-alias",
                            "--ocr-position-contract",
                            BOUNDARY_V1,
                            "--output",
                            str(output),
                        ]
                    )
            init_distributed.assert_not_called()
            self.assertFalse(output.exists())

    def test_manifest_validator_reports_lossless_completion_budget(self):
        from scripts.validate_ocr_rl_manifests import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenizer_dir, token_contract = _tokenizer_contract(root)
            checkpoint = _checkpoint(
                root / "checkpoint",
                _visual_metadata(token_contract),
            )
            paths = {}
            for name, split in (("train", "rl_train"), ("validation", "rl_val")):
                manifest = root / f"{name}.jsonl"
                manifest.write_text(
                    json.dumps(_image_and_row(root, name, split)) + "\n",
                    encoding="utf-8",
                )
                paths[name] = manifest
            golden_labeled = _image_and_row(root, "golden", "golden")
            golden_identity = root / "golden_identity.jsonl"
            golden_identity.write_text(
                json.dumps(_identity_row(golden_labeled)) + "\n",
                encoding="utf-8",
            )
            dataset_contract = _public_dataset_contract(
                root,
                train=paths["train"],
                validation=paths["validation"],
                identity=golden_identity,
                token_contract=token_contract,
            )
            output = io.StringIO()
            with (
                patch(
                    "Tokenizer.unified.bundle.TokenizerBundle.from_dir",
                    return_value=_FakeBundle(),
                ),
                contextlib.redirect_stdout(output),
            ):
                rc = main(
                    [
                        "--checkpoint", str(checkpoint),
                        "--tokenizer", str(tokenizer_dir),
                        "--train", str(paths["train"]),
                        "--validation", str(paths["validation"]),
                        "--golden-identity", str(golden_identity),
                        "--dataset-contract", str(dataset_contract),
                        "--image-root", str(root),
                    ]
                )
            self.assertEqual(rc, 0)
            report = json.loads(output.getvalue())
            self.assertEqual(report["recommended_max_new_tokens"], 2)
            self.assertEqual(report["splits"]["golden_identity"]["samples"], 1)
            self.assertTrue(report["splits"]["golden_identity"]["locked"])
            self.assertNotIn(
                "completion_tokens_including_eos",
                report["splits"]["golden_identity"],
            )

    def test_manifest_source_failure_precedes_model_reconstruction(self):
        from scripts import validate_ocr_rl_manifests

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenizer_dir, _ = _tokenizer_contract(root)
            checkpoint = _checkpoint(root / "checkpoint", {})
            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            golden_identity = root / "golden_identity.jsonl"
            for path in (train, validation, golden_identity):
                path.touch()
            with (
                patch(
                    "Tokenizer.unified.bundle.TokenizerBundle.from_dir",
                    return_value=_FakeBundle(),
                ),
                patch.object(
                    validate_ocr_rl_manifests,
                    "reconstruct_policy_from_checkpoint",
                ) as reconstruct,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "OCR visual input contract metadata",
                ):
                    validate_ocr_rl_manifests.main(
                        [
                            "--checkpoint", str(checkpoint),
                            "--tokenizer", str(tokenizer_dir),
                            "--train", str(train),
                            "--validation", str(validation),
                            "--golden-identity", str(golden_identity),
                        ]
                    )
            reconstruct.assert_not_called()

    def test_streaming_v2_validator_reuses_admitted_metadata_and_hashes(self):
        from scripts import validate_ocr_rl_manifests
        from Model.posttrain.release_contract import (
            VISUAL_SOURCE_CONTRACT_STREAMING_V2_RELEASE,
        )

        class _ReconstructionReached(Exception):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenizer_dir = root / "tokenizer"
            tokenizer_dir.mkdir()
            checkpoint = _checkpoint(root / "checkpoint", {})
            admitted_metadata = {"phase": "vlm_align"}
            model_sha256 = "c" * 64
            metadata_sha256 = "d" * 64
            lineage = {
                "source_checkpoint": str(checkpoint),
                "source_checkpoint_model_sha256": model_sha256,
                "source_checkpoint_metadata_sha256": metadata_sha256,
            }

            def assert_reconstruction(path, **kwargs):
                self.assertEqual(Path(path), checkpoint)
                self.assertTrue(kwargs["require_vision"])
                self.assertIs(kwargs["metadata_override"], admitted_metadata)
                self.assertEqual(
                    kwargs["expected_metadata_sha256"],
                    metadata_sha256,
                )
                self.assertEqual(
                    kwargs["expected_model_sha256"],
                    model_sha256,
                )
                raise _ReconstructionReached

            with (
                patch(
                    "Tokenizer.unified.bundle.TokenizerBundle.from_dir",
                    return_value=_FakeBundle(),
                ),
                patch.object(
                    validate_ocr_rl_manifests,
                    "native_tokenization_contract",
                    return_value={},
                ),
                patch.object(
                    validate_ocr_rl_manifests,
                    "load_verified_policy_metadata",
                ) as preload_metadata,
                patch.object(
                    validate_ocr_rl_manifests,
                    "admit_visual_ocr_source",
                    return_value={
                        "metadata": admitted_metadata,
                        "lineage": lineage,
                    },
                ) as admit,
                patch.object(
                    validate_ocr_rl_manifests,
                    "reconstruct_policy_from_checkpoint",
                    side_effect=assert_reconstruction,
                ),
            ):
                with self.assertRaises(_ReconstructionReached):
                    validate_ocr_rl_manifests.main(
                        [
                            "--checkpoint",
                            str(checkpoint),
                            "--visual-source-contract",
                            VISUAL_SOURCE_CONTRACT_STREAMING_V2_RELEASE,
                            "--tokenizer",
                            str(tokenizer_dir),
                            "--train",
                            str(root / "train.jsonl"),
                            "--validation",
                            str(root / "validation.jsonl"),
                            "--golden-identity",
                            str(root / "golden_identity.jsonl"),
                        ]
                    )

            preload_metadata.assert_not_called()
            self.assertIsNone(admit.call_args.kwargs["metadata"])

    def test_locked_golden_cli_accepts_only_registered_selected_artifacts(self):
        from scripts.eval_ocr_grpo import _vocab_sha256, main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_a = root / "run_a"
            golden = root / "golden.jsonl"
            golden.write_text(
                json.dumps(_image_and_row(root, "golden", "golden")) + "\n",
                encoding="utf-8",
            )
            labeled_row = json.loads(golden.read_text(encoding="utf-8"))
            golden_identity = root / "golden_identity.jsonl"
            golden_identity.write_text(
                json.dumps(_identity_row(labeled_row)) + "\n",
                encoding="utf-8",
            )
            identity_rows = load_golden_identity_manifest(golden_identity)
            tokenizer_dir, token_contract = _tokenizer_contract(root)
            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            train.write_text(
                json.dumps(_image_and_row(root, "train", "rl_train")) + "\n",
                encoding="utf-8",
            )
            validation.write_text(
                json.dumps(_image_and_row(root, "validation", "rl_val"))
                + "\n",
                encoding="utf-8",
            )
            golden_receipt = _locked_build_receipt(
                root,
                golden=golden,
                identity=golden_identity,
            )
            public_contract = _public_dataset_contract(
                root,
                train=train,
                validation=validation,
                identity=golden_identity,
                token_contract=token_contract,
                locked_receipt_sha256=hashlib.sha256(
                    golden_receipt.read_bytes()
                ).hexdigest(),
            )
            checkpoint_root = run_a / "best"
            source_dir = _checkpoint(
                root / "visual-source",
                _visual_metadata(token_contract),
            )
            from Model.posttrain.release_contract import (
                VISUAL_SOURCE_CONTRACT_ALIGNMENT_V3,
                admit_visual_ocr_source,
            )
            from Model.training.checkpoint import load_checkpoint_metadata

            source_admission = admit_visual_ocr_source(
                VISUAL_SOURCE_CONTRACT_ALIGNMENT_V3,
                checkpoint_dir=source_dir,
                metadata=load_checkpoint_metadata(source_dir),
                runtime_native_tokenization_contract=token_contract,
                tokenizer_vocab_extent=65536,
            )
            source_lineage = source_admission["lineage"]
            source_checkpoint = str(source_dir)
            source_model_sha256 = source_lineage[
                "source_checkpoint_model_sha256"
            ]
            reference_metadata = {
                "phase": "grpo_reference",
                "contract_version": OCR_GRPO_CONTRACT_VERSION,
                "source_checkpoint": source_checkpoint,
                "source_checkpoint_model_sha256": source_model_sha256,
                "ocr_tokenization_contract": token_contract,
                "ocr_position_contract": BOUNDARY_V1,
                "ocr_position_contract_version": (
                    OCR_POSITION_CONTRACT_METADATA_VERSION
                ),
                "immutable": True,
            }
            reference = _checkpoint(
                run_a / "reference",
                reference_metadata,
            )
            immutable_data_contract = {
                "train_manifest_sha256": hashlib.sha256(
                    train.read_bytes()
                ).hexdigest(),
                "validation_manifest_sha256": hashlib.sha256(
                    validation.read_bytes()
                ).hexdigest(),
                "golden_identity_manifest_sha256": hashlib.sha256(
                    golden_identity.read_bytes()
                ).hexdigest(),
                "golden_identity_semantic_sha256": (
                    golden_identity_semantic_sha256(identity_rows)
                ),
                "tokenizer_vocab_sha256": _vocab_sha256(_FakeTokenizer()),
                "ocr_tokenization_contract": token_contract,
                "ocr_position_contract": BOUNDARY_V1,
                "ocr_position_contract_version": (
                    OCR_POSITION_CONTRACT_METADATA_VERSION
                ),
                "visual_source": source_lineage,
                "reference_model_sha256": hashlib.sha256(
                    (reference / "model.pt").read_bytes()
                ).hexdigest(),
                "image_root": str(root),
                "public_dataset_contract_sha256": hashlib.sha256(
                    public_contract.read_bytes()
                ).hexdigest(),
            }
            metadata = {
                "phase": "grpo",
                "contract_version": OCR_GRPO_CONTRACT_VERSION,
                "task": "ocr",
                "ocr_position_contract": BOUNDARY_V1,
                "ocr_position_contract_version": (
                    OCR_POSITION_CONTRACT_METADATA_VERSION
                ),
                "source_checkpoint": source_checkpoint,
                "reference_checkpoint": str(reference),
                "golden_identity_manifest": str(golden_identity),
                "validation_manifest": str(validation),
                "public_dataset_contract": str(public_contract),
                "training_config": {"train_data": str(train)},
                "golden_split": "golden",
                "grpo_config": {"max_new_tokens": 2, "recurrent_steps": 1},
                "health_state": {
                    "best_checkpoint": (
                        "/old/mount/run/best/step_00000001"
                    ),
                    "best_val_grapheme_cer": 0.2,
                    "best_val_step": 1,
                    "best_val_eligible": True,
                    "stop_reason": "validation_plateau",
                },
                "data_contract": immutable_data_contract,
                "final": True,
                "stop_reason": "validation_plateau",
            }
            checkpoint = _checkpoint(
                checkpoint_root,
                metadata,
                step_number=1,
            )
            terminal = _checkpoint(
                run_a,
                metadata,
                step_number=2,
            )
            from Model.posttrain.ocr_selection import (
                build_selection_receipt,
                write_selection_receipt,
            )

            selection_payload = build_selection_receipt(
                selected_checkpoint=checkpoint,
                selected_step=1,
                terminal_checkpoint=terminal,
                terminal_step=2,
                stop_reason="validation_plateau",
                data_contract=immutable_data_contract,
            )
            selection_receipt = checkpoint.parent / "SELECTION_FINALIZED.json"
            write_selection_receipt(selection_receipt, selection_payload)
            real = {
                "grapheme_cer": 0.1,
                "norm_cer": 0.1,
                "raw_cer": 0.1,
                "wer": 0.1,
                "line_exact": 0.9,
                "normalization_backend": "python",
                "eos_rate": 1.0,
                "invalid_output_rate": 0.0,
                "samples": 1.0,
            }
            blank = {**real, "grapheme_cer": 0.8}
            reference_metrics = {**real, "grapheme_cer": 0.3}
            rendered = io.StringIO()
            with (
                patch(
                    "Tokenizer.unified.bundle.TokenizerBundle.from_dir",
                    return_value=_FakeBundle(),
                ),
                patch(
                    "scripts.eval_ocr_grpo.evaluate_ocr_manifest",
                    side_effect=[real, blank, reference_metrics],
                ),
                contextlib.redirect_stdout(rendered),
            ):
                source_model = source_dir / "model.pt"
                original_source_model = source_model.read_bytes()
                source_model.write_bytes(original_source_model + b"tampered")
                rejected_report = root / "must-not-consume-golden.json"
                with self.assertRaisesRegex(ValueError, "expected lineage"):
                    main(
                        [
                            "--checkpoint", str(checkpoint),
                            "--tokenizer", str(tokenizer_dir),
                            "--golden-manifest", str(golden),
                            "--golden-identity-manifest", str(golden_identity),
                            "--dataset-contract", str(public_contract),
                            "--golden-receipt", str(golden_receipt),
                            "--selection-receipt", str(selection_receipt),
                            "--out", str(rejected_report),
                            "--device", "cpu",
                        ]
                    )
                self.assertFalse(rejected_report.exists())
                self.assertFalse(
                    (root / "golden_evaluation_ledger").exists()
                )
                source_model.write_bytes(original_source_model)

                # The selected-model receipt is validated before reconstruction.
                # Replacing model.pt in that narrow window must still be caught
                # by the expected hash consumed by the atomic loader, without
                # burning the one-shot golden evaluation.
                from Model.posttrain.checkpointing import (
                    reconstruct_policy_from_checkpoint as real_reconstruct,
                )

                selected_model_path = checkpoint / "model.pt"
                original_selected_model = selected_model_path.read_bytes()
                selected_swapped = False

                def swap_selected_before_load(path, **kwargs):
                    nonlocal selected_swapped
                    if (
                        not selected_swapped
                        and Path(path).resolve() == checkpoint.resolve()
                    ):
                        selected_swapped = True
                        selected_model_path.write_bytes(
                            original_selected_model + b"tampered-after-receipt"
                        )
                    return real_reconstruct(path, **kwargs)

                selected_tamper_report = root / "tampered_selected_report.json"
                try:
                    with patch(
                        "scripts.eval_ocr_grpo.reconstruct_policy_from_checkpoint",
                        side_effect=swap_selected_before_load,
                    ):
                        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                            main(
                                [
                                    "--checkpoint", str(checkpoint),
                                    "--tokenizer", str(tokenizer_dir),
                                    "--golden-manifest", str(golden),
                                    "--golden-identity-manifest", str(golden_identity),
                                    "--dataset-contract", str(public_contract),
                                    "--golden-receipt", str(golden_receipt),
                                    "--selection-receipt", str(selection_receipt),
                                    "--out", str(selected_tamper_report),
                                    "--device", "cpu",
                                ]
                            )
                finally:
                    selected_model_path.write_bytes(original_selected_model)
                self.assertFalse(selected_tamper_report.exists())
                self.assertFalse(
                    (root / "golden_evaluation_ledger").exists()
                )

                # Apply the same post-metadata swap to the immutable reference.
                # The selected model may load successfully, but reference hash
                # failure must still happen before any claim/report is created.
                reference_model_path = reference / "model.pt"
                original_reference_model = reference_model_path.read_bytes()
                reference_swapped = False

                def swap_reference_before_load(path, **kwargs):
                    nonlocal reference_swapped
                    if (
                        not reference_swapped
                        and Path(path).resolve() == reference.resolve()
                    ):
                        reference_swapped = True
                        reference_model_path.write_bytes(
                            original_reference_model + b"tampered-after-metadata"
                        )
                    return real_reconstruct(path, **kwargs)

                reference_tamper_report = root / "tampered_reference_report.json"
                try:
                    with patch(
                        "scripts.eval_ocr_grpo.reconstruct_policy_from_checkpoint",
                        side_effect=swap_reference_before_load,
                    ):
                        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                            main(
                                [
                                    "--checkpoint", str(checkpoint),
                                    "--tokenizer", str(tokenizer_dir),
                                    "--golden-manifest", str(golden),
                                    "--golden-identity-manifest", str(golden_identity),
                                    "--dataset-contract", str(public_contract),
                                    "--golden-receipt", str(golden_receipt),
                                    "--selection-receipt", str(selection_receipt),
                                    "--out", str(reference_tamper_report),
                                    "--device", "cpu",
                                ]
                            )
                finally:
                    reference_model_path.write_bytes(original_reference_model)
                self.assertFalse(reference_tamper_report.exists())
                self.assertFalse(
                    (root / "golden_evaluation_ledger").exists()
                )

                rc = main(
                    [
                        "--checkpoint", str(checkpoint),
                        "--tokenizer", str(tokenizer_dir),
                        "--golden-manifest", str(golden),
                        "--golden-identity-manifest", str(golden_identity),
                        "--dataset-contract", str(public_contract),
                        "--golden-receipt", str(golden_receipt),
                        "--selection-receipt", str(selection_receipt),
                        "--out", str(root / "golden_report.json"),
                        "--device", "cpu",
                    ]
                )
                global_marker = (
                    root
                    / "golden_evaluation_ledger"
                    / (
                        hashlib.sha256(golden_receipt.read_bytes()).hexdigest()
                        + ".json"
                    )
                )
                self.assertTrue(global_marker.is_file())
                claim = json.loads(global_marker.read_text(encoding="utf-8"))
                self.assertEqual(
                    claim["selected_model_sha256"],
                    selection_payload["selected_model_sha256"],
                )
                run_b = root / "run_b"
                reference_b = _checkpoint(
                    run_b / "reference",
                    reference_metadata,
                )
                data_contract_b = {
                    **immutable_data_contract,
                    "reference_model_sha256": hashlib.sha256(
                        (reference_b / "model.pt").read_bytes()
                    ).hexdigest(),
                }
                checkpoint_b_path = run_b / "best" / "step_00000001"
                metadata_b = {
                    **metadata,
                    "reference_checkpoint": str(reference_b),
                    "data_contract": data_contract_b,
                    "health_state": {
                        **metadata["health_state"],
                        "best_checkpoint": str(checkpoint_b_path),
                    },
                }
                checkpoint_b = _checkpoint(
                    run_b / "best",
                    metadata_b,
                    step_number=1,
                )
                terminal_b = _checkpoint(
                    run_b,
                    metadata_b,
                    step_number=2,
                )
                selection_payload_b = build_selection_receipt(
                    selected_checkpoint=checkpoint_b,
                    selected_step=1,
                    terminal_checkpoint=terminal_b,
                    terminal_step=2,
                    stop_reason="validation_plateau",
                    data_contract=data_contract_b,
                )
                selection_receipt_b = (
                    checkpoint_b.parent / "SELECTION_FINALIZED.json"
                )
                write_selection_receipt(
                    selection_receipt_b,
                    selection_payload_b,
                )
                with self.assertRaisesRegex(
                    FileExistsError,
                    "already consumed",
                ):
                    main(
                        [
                            "--checkpoint", str(checkpoint_b),
                            "--tokenizer", str(tokenizer_dir),
                            "--golden-manifest", str(golden),
                            "--golden-identity-manifest", str(golden_identity),
                            "--dataset-contract", str(public_contract),
                            "--golden-receipt", str(golden_receipt),
                            "--selection-receipt", str(selection_receipt_b),
                            "--out", str(root / "second_report.json"),
                            "--device", "cpu",
                        ]
                    )
                saved_meta = torch.load(
                    checkpoint / "meta.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                saved_meta["metadata"]["health_state"]["best_checkpoint"] = (
                    "/old/mount/run/best/step_00000002"
                )
                torch.save(saved_meta, checkpoint / "meta.pt")
                with self.assertRaisesRegex(
                    ValueError,
                    "non-selected checkpoint",
                ):
                    main(
                        [
                            "--checkpoint", str(checkpoint),
                            "--tokenizer", str(tokenizer_dir),
                            "--golden-manifest", str(golden),
                            "--golden-identity-manifest", str(golden_identity),
                            "--dataset-contract", str(public_contract),
                            "--golden-receipt", str(golden_receipt),
                            "--selection-receipt", str(selection_receipt),
                            "--out", str(root / "third_report.json"),
                            "--device", "cpu",
                        ]
                    )
            self.assertEqual(rc, 0)
            report = json.loads(rendered.getvalue())
            self.assertEqual(
                report["selected_policy_model_sha256"],
                selection_payload["selected_model_sha256"],
            )
            self.assertEqual(
                report["immutable_reference_model_sha256"],
                immutable_data_contract["reference_model_sha256"],
            )
            self.assertAlmostEqual(
                report["reference_grapheme_cer_improvement"], 0.2
            )
            self.assertTrue(report["gates"]["reference_improvement_pass"])


if __name__ == "__main__":
    unittest.main()
