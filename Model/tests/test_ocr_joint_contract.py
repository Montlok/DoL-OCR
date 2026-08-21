# -*- coding: utf-8 -*-

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from Model.config import OMVTConfig, RDTConfig
from Model.model import RDTForCausalLM
from Model.ocr.anyres_preprocess_contract import build_anyres_preprocess_contract
from Model.ocr.position_contract import BOUNDARY_V1
from Model.ocr.tokenization import canonical_json_sha256
from Model.omvt import OMVTInjector
from Model.posttrain.checkpointing import ReconstructedPolicy
from Model.posttrain.ocr_joint_contract import (
    admit_and_prepare_anyres_policy,
    admit_joint_policy_for_grpo,
    promote_visual_policy_to_joint,
    resume_anyres_policy,
)
from Model.posttrain.release_contract import (
    MONTLOK_DOL_1_2_OCR_REPOSITORY_ID,
    MONTLOK_DOL_1_2_OCR_REVISION,
    STREAMING_V2_RELEASE_SOURCE_CONTRACT_KIND,
    VISUAL_SOURCE_CONTRACT_STREAMING_V2_RELEASE,
)


def _rdt_cfg() -> RDTConfig:
    return RDTConfig(
        d_model=8,
        n_heads=1,
        head_dim=8,
        kv_lora_rank=4,
        rope_head_dim=4,
        nope_head_dim=4,
        ffn_hidden=16,
        ffn_multiple=8,
        n_prelude=1,
        n_coda=0,
        mamba_per_block=0,
        attn_per_block=1,
        recurrent_steps=1,
        mamba_d_state=4,
        mamba_expand=2,
        mamba_headdim=8,
        use_official_mamba=False,
        max_seq_len=16,
        bidirectional=False,
    )


def _omvt_cfg() -> OMVTConfig:
    return OMVTConfig(
        image_size=16,
        vertical_patch=(8, 4),
        horizontal_patch=(4, 8),
        square_patch=(4, 4),
        layout_patch=(16, 16),
        d_vision=8,
        n_vertical_layers=1,
        n_horizontal_layers=1,
        n_local_attn_layers=1,
        n_layout_layers=1,
        vision_n_heads=2,
        vision_ffn_hidden=16,
        compress_to=256,
        compressor_layers=1,
        compressor_heads=1,
    )


def _tokenizer_contract() -> dict:
    return {
        "target_encoding": "native",
        "tokenization_contract_version": 3,
        "tokenizer_bundle": {"fixture": True},
        "tokenizer_vocab_sha256": "b" * 64,
    }


def _lineage(path: str = "/fixture/source") -> dict:
    return {
        "source_contract_kind": STREAMING_V2_RELEASE_SOURCE_CONTRACT_KIND,
        "source_checkpoint": path,
        "source_checkpoint_model_sha256": "a" * 64,
        "source_checkpoint_metadata_sha256": "c" * 64,
        "source_repository_id": MONTLOK_DOL_1_2_OCR_REPOSITORY_ID,
        "source_revision": MONTLOK_DOL_1_2_OCR_REVISION,
        "ocr_visual_input_contract": "dol_ocr_line_letterbox_224_v1",
        "ocr_visual_input_contract_version": 1,
        "ocr_position_contract": "boundary_v1",
        "ocr_position_contract_version": 1,
    }


def _source_policy() -> tuple[ReconstructedPolicy, dict]:
    rdt = _rdt_cfg()
    omvt = _omvt_cfg()
    model = RDTForCausalLM(rdt)
    model.vision._omvt_cfg = omvt
    model.vision.omvt = OMVTInjector(rdt, omvt)
    metadata = {
        "rdt_config": asdict(rdt),
        "omvt_config": asdict(omvt),
    }
    return (
        ReconstructedPolicy(
            model=model,
            rdt_config=rdt,
            omvt_config=omvt,
            metadata=metadata,
            checkpoint_dir=Path("/fixture/source"),
            model_sha256="a" * 64,
            metadata_sha256="c" * 64,
        ),
        metadata,
    )


def _preprocess(omvt: OMVTConfig) -> dict:
    return build_anyres_preprocess_contract(
        omvt,
        1_000_000,
        1024,
        4,
        16,
        256,
        4,
        520,
        256,
    )


def _prepare(stage: str = "visual", *, lineage: dict | None = None):
    source, metadata = _source_policy()
    source.model.eval()
    admitted = {
        "metadata": metadata,
        "lineage": _lineage() if lineage is None else lineage,
    }
    with (
        mock.patch(
            "Model.posttrain.ocr_joint_contract.admit_visual_ocr_source",
            return_value=admitted,
        ) as admit,
        mock.patch(
            "Model.posttrain.ocr_joint_contract.reconstruct_policy_from_checkpoint",
            return_value=source,
        ) as reconstruct,
    ):
        prepared = admit_and_prepare_anyres_policy(
            "/fixture/source",
            VISUAL_SOURCE_CONTRACT_STREAMING_V2_RELEASE,
            _tokenizer_contract(),
            65536,
            _preprocess(source.omvt_config),
            stage,
            520,
            {
                "max_detail_tokens": 256,
                "source_tokens_per_detail_token": 4,
            },
            {"memory_dim": 8, "n_heads": 1, "dropout": 0.0},
        )
    self_call = admit.call_args
    if self_call.kwargs["requested_visual_input_contract"] != (
        "dol_ocr_line_letterbox_224_v1"
    ):
        raise AssertionError("fresh admission did not request the v1 source")
    if reconstruct.call_args.kwargs["expected_model_sha256"] != "a" * 64:
        raise AssertionError("fresh reconstruction did not bind model bytes")
    return source, prepared


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _embedded_contract(kind: str) -> dict[str, str]:
    base = {"kind": kind}
    return {**base, "contract_sha256": canonical_json_sha256(base)}


def _text_partition_contract(
    contracts: dict[str, dict[str, str]],
) -> dict[str, object]:
    split_sha256 = {
        split: contract["contract_sha256"]
        for split, contract in contracts.items()
    }
    base: dict[str, object] = {
        "contract_version": 1,
        "kind": "dol_text_ce_replay_partition",
        "public_splits": [
            "train",
            "sft_validation",
            "kl_selection",
            "formal_monitor",
        ],
        "manifest_file_sha256": "1" * 64,
        "all_rows_canonical_sha256": "2" * 64,
        "split_contract_sha256": split_sha256,
        "split_selected_rows_canonical_sha256": {
            split: str(index) * 64
            for index, split in enumerate(split_sha256, start=3)
        },
        "split_document_sha256_canonical_sha256": {
            split: str(index) * 64
            for index, split in enumerate(split_sha256, start=4)
        },
        "split_text_sha256_canonical_sha256": {
            split: str(index) * 64
            for index, split in enumerate(split_sha256, start=5)
        },
        "split_token_stats": {
            split: {"samples": 1}
            for split in split_sha256
        },
        "shared_contract": {"position_contract": BOUNDARY_V1},
        "leakage_contract": {
            "document_id_cross_split": "forbidden",
            "exact_text_sha256_cross_split": "forbidden",
            "ngram_sha256_cross_split": "forbidden",
            "ngram_codepoints": 13,
            "global_cross_split_leakage_validated": True,
        },
    }
    return {**base, "contract_sha256": canonical_json_sha256(base)}


class OCRJointContractTests(unittest.TestCase):
    def test_fresh_v1_to_v2_is_detail_off_bit_exact(self) -> None:
        torch.manual_seed(17)
        source, _metadata = _source_policy()
        source.model.eval()
        input_ids = torch.tensor([[2, 300, 301, 3]], dtype=torch.long)
        with torch.no_grad():
            before = source.model(
                input_ids,
                position_contract=BOUNDARY_V1,
            )["logits"].clone()

        admitted = {"metadata": source.metadata, "lineage": _lineage()}
        with (
            mock.patch(
                "Model.posttrain.ocr_joint_contract.admit_visual_ocr_source",
                return_value=admitted,
            ),
            mock.patch(
                "Model.posttrain.ocr_joint_contract.reconstruct_policy_from_checkpoint",
                return_value=source,
            ),
        ):
            prepared = admit_and_prepare_anyres_policy(
                "/fixture/source",
                VISUAL_SOURCE_CONTRACT_STREAMING_V2_RELEASE,
                _tokenizer_contract(),
                65536,
                _preprocess(source.omvt_config),
                "visual",
                520,
                {
                    "max_detail_tokens": 256,
                    "source_tokens_per_detail_token": 4,
                },
                {"memory_dim": 8, "n_heads": 1, "dropout": 0.0},
            )
        prepared.policy.eval()
        with torch.no_grad():
            after = prepared.policy(
                input_ids,
                position_contract=BOUNDARY_V1,
            )["logits"]
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(prepared.rdt_config.max_seq_len, 520)
        self.assertEqual(
            prepared.metadata_template["ocr_visual_input_contract"],
            "dol_ocr_anyres_v2",
        )
        self.assertTrue(
            prepared.metadata_template["native_migration_receipt"]["payload"][
                "initialized_from_legacy"
            ]
        )

    def test_metadata_roundtrips_through_strict_v2_reconstruction(self) -> None:
        _source, prepared = _prepare("visual")
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "derived"
            checkpoint.mkdir()
            torch.save(prepared.policy.state_dict(), checkpoint / "model.pt")
            torch.save(
                {"metadata": prepared.metadata_template},
                checkpoint / "meta.pt",
            )
            resumed = resume_anyres_policy(
                checkpoint,
                prepared.run_contract,
                _sha(checkpoint / "model.pt"),
                _sha(checkpoint / "meta.pt"),
            )
        self.assertIsNotNone(resumed.policy.vision.native_detail_tower)
        self.assertIsNotNone(resumed.policy.vision_cross_attention)
        self.assertEqual(resumed.run_contract, prepared.run_contract)
        self.assertEqual(
            set(resumed.policy.state_dict()), set(prepared.policy.state_dict())
        )
        for name, expected in prepared.policy.state_dict().items():
            self.assertTrue(torch.equal(resumed.policy.state_dict()[name], expected), name)

    def test_only_hash_bound_eligible_best_can_promote_to_joint(self) -> None:
        _source, prepared = _prepare("visual")
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp).resolve() / "best"
            checkpoint.mkdir()
            validation = {"eligibility": {"eligible": True}}
            baseline_validation = {
                "real": {
                    "buckets": {
                        bucket: {"raw_grapheme_cer": 1.0}
                        for bucket in (
                            "print",
                            "handwritten_good",
                            "handwritten_medium",
                            "handwritten_poor",
                        )
                    }
                },
                "text_replay": {"token_nll": 1.0},
            }
            frozen_contract = {
                "language_sha256": "c" * 64,
                "legacy_omvt_sha256": "d" * 64,
            }
            runtime_source = {"canonical_sha256": "e" * 64}
            runtime_environment = {"device": "test"}
            metadata = {
                **prepared.metadata_template,
                "training_stage": "visual",
                "final": False,
                "last_validation": validation,
                "baseline_validation": baseline_validation,
                "frozen_parameter_contract": frozen_contract,
                "runtime_source_receipt": runtime_source,
                "runtime_environment": runtime_environment,
            }
            torch.save(prepared.policy.state_dict(), checkpoint / "model.pt")
            torch.save({"metadata": metadata}, checkpoint / "meta.pt")
            identity = {
                "path": str(checkpoint),
                "model_sha256": _sha(checkpoint / "model.pt"),
                "metadata_sha256": _sha(checkpoint / "meta.pt"),
            }
            result = {
                "schema_version": 1,
                "kind": "dol_ocr_anyres_visual_stage_result_v1",
                "run_contract_sha256": prepared.run_contract[
                    "contract_canonical_sha256"
                ],
                "final_checkpoint": dict(identity),
                "best_eligible_checkpoint": dict(identity),
                "baseline_validation_sha256": canonical_json_sha256(
                    baseline_validation
                ),
                "final_validation_sha256": "b" * 64,
                "best_validation_sha256": canonical_json_sha256(validation),
                "last_eligibility": {"eligible": True},
                "promotion_allowed": True,
                "completed_cycles": 2,
                "stop_reason": "validation_plateau",
                "frozen_parameter_contract": frozen_contract,
                "runtime_source_receipt": runtime_source,
                "runtime_environment": runtime_environment,
            }
            result["canonical_sha256"] = canonical_json_sha256(result)
            with mock.patch(
                "Model.posttrain.ocr_joint_contract.joint_eval_eligibility",
                return_value=validation["eligibility"],
            ):
                promoted = promote_visual_policy_to_joint(
                    checkpoint,
                    identity["model_sha256"],
                    identity["metadata_sha256"],
                    result,
                )
            self.assertEqual(
                promoted.run_contract["metadata_contract"]["stage"], "joint"
            )
            self.assertEqual(
                promoted.run_contract["metadata_contract"][
                    "parent_anyres_checkpoint"
                ]["path"],
                str(checkpoint),
            )
            self.assertTrue(promoted.policy.embed.weight.requires_grad)
            self.assertFalse(
                any(
                    parameter.requires_grad
                    for parameter in promoted.policy.vision.encoder.parameters()
                )
            )

            tampered = deepcopy(result)
            tampered["promotion_allowed"] = False
            with self.assertRaises(ValueError):
                promote_visual_policy_to_joint(
                    checkpoint,
                    identity["model_sha256"],
                    identity["metadata_sha256"],
                    tampered,
                )

            joint_checkpoint = Path(tmp).resolve() / "joint-best"
            joint_checkpoint.mkdir()
            sft_image_sha256 = _embedded_contract("validation")[
                "contract_sha256"
            ]
            sft_text_sha256 = _embedded_contract("text-sft-validation")[
                "contract_sha256"
            ]
            joint_validation = {
                "eligibility": {"eligible": True},
                "image_dataset_contract_sha256": sft_image_sha256,
                "text_replay": {
                    "dataset_contract_sha256": sft_text_sha256,
                    "token_nll": 1.0,
                },
            }
            joint_baseline = {
                "deployment_weighted_cer": 1.0,
                "image_dataset_contract_sha256": sft_image_sha256,
                "text_replay": {
                    "dataset_contract_sha256": sft_text_sha256,
                    "token_nll": 1.0,
                },
            }
            historical_validation = {
                "historical": True,
                "image_dataset_contract_sha256": sft_image_sha256,
                "text_replay": {
                    "dataset_contract_sha256": sft_text_sha256,
                    "token_nll": 1.0,
                },
            }
            optimizer_contract = {
                "groups": [],
                "canonical_sha256": "f" * 64,
            }
            sampler_state = {"draw_counter": 80}
            text_cursor = {"canonical_sha256": "9" * 64}
            joint_runtime_source = {"canonical_sha256": "8" * 64}
            joint_runtime_environment = {"device": "joint-test"}
            text_contracts = {
                "train": _embedded_contract("text-train"),
                "sft_validation": _embedded_contract("text-sft-validation"),
                "kl_selection": _embedded_contract("text-kl-selection"),
                "formal_monitor": _embedded_contract("text-formal-monitor"),
            }
            text_partition_contract = _text_partition_contract(text_contracts)
            joint_metadata = {
                **promoted.metadata_template,
                "training_stage": "joint",
                "final": False,
                "last_validation": joint_validation,
                "joint_runtime_baseline": joint_baseline,
                "historical_visual_best_validation": historical_validation,
                "optimizer_contract": optimizer_contract,
                "quota_sampler_state": sampler_state,
                "text_replay_cursor": text_cursor,
                "dataset_admission_report": {"kind": "dataset-report"},
                "train_dataset_contract": _embedded_contract("train"),
                "validation_dataset_contract": _embedded_contract("validation"),
                "sft_validation_dataset_contract": _embedded_contract(
                    "validation"
                ),
                "kl_selection_dataset_contract": _embedded_contract(
                    "kl-selection"
                ),
                "formal_monitor_dataset_contract": _embedded_contract(
                    "formal-monitor"
                ),
                "text_replay_train_contract": text_contracts["train"],
                "text_replay_sft_validation_contract": text_contracts[
                    "sft_validation"
                ],
                "text_replay_kl_selection_contract": text_contracts[
                    "kl_selection"
                ],
                "text_replay_formal_monitor_contract": text_contracts[
                    "formal_monitor"
                ],
                "text_replay_validation_contract": text_contracts[
                    "sft_validation"
                ],
                "text_replay_partition_contract": text_partition_contract,
                "runtime_source_receipt": joint_runtime_source,
                "runtime_environment": joint_runtime_environment,
            }
            torch.save(promoted.policy.state_dict(), joint_checkpoint / "model.pt")
            torch.save({"metadata": joint_metadata}, joint_checkpoint / "meta.pt")
            joint_identity = {
                "path": str(joint_checkpoint),
                "model_sha256": _sha(joint_checkpoint / "model.pt"),
                "metadata_sha256": _sha(joint_checkpoint / "meta.pt"),
            }
            joint_result = {
                "schema_version": 1,
                "kind": "dol_ocr_anyres_joint_stage_result_v1",
                "run_contract_sha256": promoted.run_contract[
                    "contract_canonical_sha256"
                ],
                "parent_anyres_checkpoint": promoted.metadata_template[
                    "parent_anyres_checkpoint"
                ],
                "final_checkpoint": dict(joint_identity),
                "best_eligible_checkpoint": dict(joint_identity),
                "baseline_validation_sha256": canonical_json_sha256(
                    joint_baseline
                ),
                "historical_visual_validation_sha256": canonical_json_sha256(
                    historical_validation
                ),
                "final_validation_sha256": canonical_json_sha256(
                    joint_validation
                ),
                "best_validation_sha256": canonical_json_sha256(
                    joint_validation
                ),
                "last_eligibility": {"eligible": True},
                "grpo_promotion_allowed": True,
                "completed_cycles": 2,
                "stop_reason": "validation_plateau",
                "optimizer_contract_sha256": optimizer_contract[
                    "canonical_sha256"
                ],
                "quota_sampler_state_sha256": canonical_json_sha256(
                    sampler_state
                ),
                "text_replay_cursor_sha256": text_cursor[
                    "canonical_sha256"
                ],
                "runtime_source_receipt": joint_runtime_source,
                "runtime_environment": joint_runtime_environment,
            }
            joint_result["canonical_sha256"] = canonical_json_sha256(
                joint_result
            )
            with mock.patch(
                "Model.posttrain.ocr_joint_contract.dual_baseline_joint_eligibility",
                return_value=joint_validation["eligibility"],
            ):
                grpo = admit_joint_policy_for_grpo(
                    joint_checkpoint,
                    joint_identity["model_sha256"],
                    joint_identity["metadata_sha256"],
                    joint_result,
                    reward_contract_sha256="e" * 64,
                    kl_coef=0.04,
                )
            self.assertIsNotNone(grpo.reference)
            self.assertIsNot(grpo.policy, grpo.reference)
            self.assertFalse(grpo.reference.reverse_loss_enabled)
            self.assertTrue(
                all(
                    not parameter.requires_grad
                    for parameter in grpo.reference.parameters()
                )
            )
            self.assertEqual(
                grpo.admission.policy_checkpoint_sha256,
                joint_identity["model_sha256"],
            )
            self.assertEqual(
                grpo.admission.sft_validation_dataset_contract_sha256,
                joint_metadata["sft_validation_dataset_contract"][
                    "contract_sha256"
                ],
            )
            self.assertEqual(
                grpo.admission.kl_selection_dataset_contract_sha256,
                joint_metadata["kl_selection_dataset_contract"][
                    "contract_sha256"
                ],
            )
            self.assertEqual(
                grpo.admission.formal_monitor_dataset_contract_sha256,
                joint_metadata["formal_monitor_dataset_contract"][
                    "contract_sha256"
                ],
            )
            self.assertEqual(
                grpo.admission.text_replay_train_contract_sha256,
                text_contracts["train"]["contract_sha256"],
            )
            self.assertEqual(
                grpo.admission.text_replay_sft_validation_contract_sha256,
                text_contracts["sft_validation"]["contract_sha256"],
            )
            self.assertEqual(
                grpo.admission.text_replay_kl_selection_contract_sha256,
                text_contracts["kl_selection"]["contract_sha256"],
            )
            self.assertEqual(
                grpo.admission.text_replay_formal_monitor_contract_sha256,
                text_contracts["formal_monitor"]["contract_sha256"],
            )
            with mock.patch(
                "Model.posttrain.ocr_joint_contract.dual_baseline_joint_eligibility",
                return_value=joint_validation["eligibility"],
            ):
                zero_kl_grpo = admit_joint_policy_for_grpo(
                    joint_checkpoint,
                    joint_identity["model_sha256"],
                    joint_identity["metadata_sha256"],
                    joint_result,
                    reward_contract_sha256="e" * 64,
                    kl_coef=0.0,
                )
            self.assertIsNone(zero_kl_grpo.reference)
            self.assertEqual(
                zero_kl_grpo.admission.reference_checkpoint_sha256,
                joint_identity["model_sha256"],
            )
            self.assertEqual(
                zero_kl_grpo.admission.reference_metadata_sha256,
                joint_identity["metadata_sha256"],
            )
            rejected_joint = deepcopy(joint_result)
            rejected_joint["grpo_promotion_allowed"] = False
            rejected_joint["best_eligible_checkpoint"] = None
            rejected_joint["best_validation_sha256"] = None
            rejected_joint.pop("canonical_sha256")
            rejected_joint["canonical_sha256"] = canonical_json_sha256(
                rejected_joint
            )
            with self.assertRaisesRegex(ValueError, "does not allow GRPO"):
                admit_joint_policy_for_grpo(
                    joint_checkpoint,
                    joint_identity["model_sha256"],
                    joint_identity["metadata_sha256"],
                    rejected_joint,
                    reward_contract_sha256="e" * 64,
                    kl_coef=0.04,
                )

            text_alias_drift_metadata = deepcopy(joint_metadata)
            text_alias_drift_metadata["text_replay_validation_contract"] = (
                _embedded_contract("not-text-sft-validation")
            )
            torch.save(
                {"metadata": text_alias_drift_metadata},
                joint_checkpoint / "meta.pt",
            )
            text_alias_drift_identity = {
                **joint_identity,
                "metadata_sha256": _sha(joint_checkpoint / "meta.pt"),
            }
            text_alias_drift_result = deepcopy(joint_result)
            text_alias_drift_result["final_checkpoint"] = dict(
                text_alias_drift_identity
            )
            text_alias_drift_result["best_eligible_checkpoint"] = dict(
                text_alias_drift_identity
            )
            text_alias_drift_result.pop("canonical_sha256")
            text_alias_drift_result["canonical_sha256"] = canonical_json_sha256(
                text_alias_drift_result
            )
            with (
                mock.patch(
                    "Model.posttrain.ocr_joint_contract.dual_baseline_joint_eligibility",
                    return_value=joint_validation["eligibility"],
                ),
                self.assertRaisesRegex(
                    ValueError,
                    "text_replay_validation_contract is only",
                ),
            ):
                admit_joint_policy_for_grpo(
                    joint_checkpoint,
                    text_alias_drift_identity["model_sha256"],
                    text_alias_drift_identity["metadata_sha256"],
                    text_alias_drift_result,
                    reward_contract_sha256="e" * 64,
                    kl_coef=0.04,
                )

            alias_drift_metadata = deepcopy(joint_metadata)
            alias_drift_metadata["validation_dataset_contract"] = (
                _embedded_contract("not-sft-validation")
            )
            torch.save(
                {"metadata": alias_drift_metadata},
                joint_checkpoint / "meta.pt",
            )
            alias_drift_identity = {
                **joint_identity,
                "metadata_sha256": _sha(joint_checkpoint / "meta.pt"),
            }
            alias_drift_result = deepcopy(joint_result)
            alias_drift_result["final_checkpoint"] = dict(
                alias_drift_identity
            )
            alias_drift_result["best_eligible_checkpoint"] = dict(
                alias_drift_identity
            )
            alias_drift_result.pop("canonical_sha256")
            alias_drift_result["canonical_sha256"] = canonical_json_sha256(
                alias_drift_result
            )
            with (
                mock.patch(
                    "Model.posttrain.ocr_joint_contract.dual_baseline_joint_eligibility",
                    return_value=joint_validation["eligibility"],
                ),
                self.assertRaisesRegex(ValueError, "only an SFT validation alias"),
            ):
                admit_joint_policy_for_grpo(
                    joint_checkpoint,
                    alias_drift_identity["model_sha256"],
                    alias_drift_identity["metadata_sha256"],
                    alias_drift_result,
                    reward_contract_sha256="e" * 64,
                    kl_coef=0.04,
                )

    def test_source_preprocess_and_stage_drift_fail_closed(self) -> None:
        bad_lineage = _lineage()
        bad_lineage["source_repository_id"] = "other/repository"
        with self.assertRaisesRegex(ValueError, "source_repository_id"):
            _prepare(lineage=bad_lineage)

        source, metadata = _source_policy()
        preprocess = _preprocess(source.omvt_config)
        preprocess["budgets"]["window"]["max_windows_per_asset"] += 1
        with (
            mock.patch(
                "Model.posttrain.ocr_joint_contract.admit_visual_ocr_source",
                return_value={"metadata": metadata, "lineage": _lineage()},
            ),
            self.assertRaises(ValueError),
        ):
            admit_and_prepare_anyres_policy(
                "/fixture/source",
                VISUAL_SOURCE_CONTRACT_STREAMING_V2_RELEASE,
                _tokenizer_contract(),
                65536,
                preprocess,
                "visual",
                520,
                {
                    "max_detail_tokens": 256,
                    "source_tokens_per_detail_token": 4,
                },
                {"memory_dim": 8, "n_heads": 1, "dropout": 0.0},
            )
        with self.assertRaisesRegex(ValueError, "stage"):
            _prepare("hybrid")


if __name__ == "__main__":
    unittest.main()
