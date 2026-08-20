# -*- coding: utf-8 -*-

"""One-shot locked-golden evaluation for a selected OCR GRPO checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Model.config import (  # noqa: E402
    BOS_ID,
    IMAGE_END_ID,
    IMAGE_PATCH_ID,
    IMAGE_START_ID,
)
from Model.ocr.tokenization import (  # noqa: E402
    canonicalize_native_ocr_text,
    native_tokenization_contract,
    tokenizer_vocab_sha256,
)
from Model.ocr.position_contract import (  # noqa: E402
    OCR_POSITION_CONTRACT_METADATA_VERSION,
    resolve_checkpoint_ocr_position_contract,
)
from Model.posttrain.checkpointing import (  # noqa: E402
    OCR_GRPO_CONTRACT_VERSION,
    load_verified_policy_metadata,
    reconstruct_policy_from_checkpoint,
)
from Model.posttrain.release_contract import (  # noqa: E402
    STREAMING_V2_RELEASE_SOURCE_CONTRACT_KIND,
    VISUAL_SOURCE_CONTRACT_ALIGNMENT_V3,
    VISUAL_SOURCE_CONTRACT_STREAMING_V2_RELEASE,
    admit_visual_ocr_source,
)
from Model.posttrain.ocr_decode import decode_ocr_completion  # noqa: E402
from Model.posttrain.ocr_eval import evaluate_ocr_manifest  # noqa: E402
from Model.posttrain.ocr_manifests import (  # noqa: E402
    assert_labeled_golden_matches_identity,
    golden_identity_semantic_sha256,
    load_golden_identity_manifest,
    load_locked_golden_receipt_anchor,
    load_ocr_dataset_contract,
    verify_locked_golden_manifest,
)
from Model.posttrain.ocr_manifest_builder import (  # noqa: E402
    file_sha256 as _file_sha256,
)
from Model.posttrain.ocr_selection import (  # noqa: E402
    load_and_validate_selection_receipt,
)
from Model.posttrain.preference_data import OCRPromptDataset  # noqa: E402
from Model.training.checkpoint import (  # noqa: E402
    load_checkpoint_metadata,
    resolve_checkpoint_dir,
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GOLDEN_LEDGER_DIRNAME = "golden_evaluation_ledger"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the selected OCR GRPO policy on a locked golden manifest"
    )
    parser.add_argument("--checkpoint", required=True, help="usually OUTPUT/best/latest")
    parser.add_argument(
        "--visual-source-contract",
        choices=(
            VISUAL_SOURCE_CONTRACT_ALIGNMENT_V3,
            VISUAL_SOURCE_CONTRACT_STREAMING_V2_RELEASE,
        ),
        default=VISUAL_SOURCE_CONTRACT_ALIGNMENT_V3,
        help=(
            "visual source admission contract; streaming-v2 release admission "
            "is explicit and never used as a fallback"
        ),
    )
    parser.add_argument("--tokenizer", required=True, help="TokenizerBundle directory")
    parser.add_argument(
        "--golden-manifest",
        required=True,
        help="locked labeled golden JSONL (never recorded or opened by training)",
    )
    parser.add_argument(
        "--golden-identity-manifest",
        default="",
        help="defaults to the public identity path recorded by training",
    )
    parser.add_argument(
        "--dataset-contract",
        default="",
        help="defaults to the public dataset contract recorded by training",
    )
    parser.add_argument(
        "--golden-receipt",
        default="",
        help="defaults to build_receipt.json beside the locked golden manifest",
    )
    parser.add_argument(
        "--selection-receipt",
        default="",
        help="defaults to SELECTION_FINALIZED.json beside the selected step",
    )
    parser.add_argument("--image-root", default="",
                        help="defaults to the immutable training data contract")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"],
                        default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cer-backend", choices=["auto", "python", "rust"],
                        default="auto")
    parser.add_argument("--min-visual-cer-gap", type=float, default=0.0,
                        help="require blank CER - real CER to exceed this value")
    parser.add_argument(
        "--min-reference-cer-improvement",
        type=float,
        default=0.0,
        help="require immutable-reference CER - selected CER above this value",
    )
    parser.add_argument("--min-eos-rate", type=float, default=0.99)
    parser.add_argument(
        "--out",
        required=True,
        help="new JSON report path; final golden evaluation is one-shot",
    )
    return parser.parse_args(argv)


def _vocab_sha256(tokenizer) -> str:
    """Compatibility alias for older callers/tests."""

    return tokenizer_vocab_sha256(tokenizer)


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _claim_one_shot(path: Path, payload: dict) -> None:
    """Durably consume the sealed golden evaluation before labels are opened."""

    ledger = path.parent
    if ledger.is_symlink():
        raise ValueError(f"golden evaluation ledger must not be a symlink: {ledger}")
    created = False
    try:
        ledger.mkdir(mode=0o700)
        created = True
    except FileExistsError:
        pass
    if ledger.is_symlink() or not ledger.is_dir():
        raise ValueError(
            f"golden evaluation ledger is not a real directory: {ledger}"
        )
    if created:
        _fsync_directory(ledger.parent)

    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(ledger, directory_flags)
    except OSError as exc:
        raise ValueError(
            f"golden evaluation ledger is not an accessible real directory: {ledger}"
        ) from exc
    os.fchmod(directory_fd, 0o700)
    claim_fd: int | None = None
    try:
        claim_fd = os.open(
            path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=directory_fd,
        )
        with os.fdopen(claim_fd, "w", encoding="utf-8") as handle:
            claim_fd = None
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(directory_fd)
    except FileExistsError as exc:
        raise FileExistsError(
            "locked golden evaluation was already consumed: " f"{path}"
        ) from exc
    finally:
        if claim_fd is not None:
            os.close(claim_fd)
        os.close(directory_fd)


def _global_golden_consumption_marker(
    locked_root: str | Path,
    golden_receipt_sha256: str,
) -> Path:
    """Return the sole receipt-keyed ledger entry for one locked golden set."""

    if not _SHA256_RE.fullmatch(golden_receipt_sha256):
        raise ValueError("locked golden receipt SHA-256 is invalid")
    root = Path(locked_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"locked golden root is not a directory: {root}")
    ledger = root / _GOLDEN_LEDGER_DIRNAME
    if ledger.is_symlink():
        raise ValueError(f"golden evaluation ledger must not be a symlink: {ledger}")
    if ledger.exists() and not ledger.is_dir():
        raise ValueError(
            f"golden evaluation ledger is not a directory: {ledger}"
        )
    return ledger / f"{golden_receipt_sha256}.json"


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _claim_and_verify_locked_golden(
    *,
    consumption_marker: Path,
    destination: Path,
    temporary: Path,
    claim_payload: dict,
    golden_receipt: dict,
    golden_manifest: str | Path,
) -> str:
    """Publish the claim/stub durably, then and only then hash golden labels."""

    try:
        _claim_one_shot(consumption_marker, claim_payload)
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        **claim_payload,
                        "status": "claimed_evaluation_incomplete",
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return verify_locked_golden_manifest(
        golden_receipt,
        golden_manifest=golden_manifest,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.min_visual_cer_gap < 0:
        raise ValueError("--min-visual-cer-gap must be non-negative")
    if args.min_reference_cer_improvement < 0:
        raise ValueError("--min-reference-cer-improvement must be non-negative")
    if not 0.0 <= args.min_eos_rate <= 1.0:
        raise ValueError("--min-eos-rate must be in [0, 1]")
    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing locked-golden report: {destination}"
        )

    selected_checkpoint_dir = resolve_checkpoint_dir(args.checkpoint)
    metadata, selected_metadata_sha256 = load_verified_policy_metadata(
        selected_checkpoint_dir,
    )
    if not (selected_checkpoint_dir / "COMPLETE").is_file():
        raise ValueError("selected GRPO checkpoint has no COMPLETE marker")
    if metadata.get("phase") != "grpo" or metadata.get("task") != "ocr":
        raise ValueError("checkpoint is not an OCR GRPO policy")
    if metadata.get("contract_version") != OCR_GRPO_CONTRACT_VERSION:
        raise ValueError("checkpoint uses a different OCR GRPO contract")
    health = metadata.get("health_state")
    if not isinstance(health, dict):
        raise ValueError("checkpoint has no GRPO health/selection state")
    if health.get("best_val_eligible") is not True:
        raise ValueError(
            "selected checkpoint did not pass validation EOS/invalid-output gates; "
            "locked golden remains closed"
        )
    best_val_step = health.get("best_val_step")
    if type(best_val_step) is not int or best_val_step <= 0:
        raise ValueError(
            "no post-update checkpoint improved the validation selection; "
            "locked golden remains closed"
        )
    recorded_best = str(health.get("best_checkpoint", ""))
    if not recorded_best:
        raise ValueError("checkpoint metadata has no selected best checkpoint")
    recorded = Path(recorded_best)
    expected_checkpoint_name = f"step_{best_val_step:08d}"
    if (
        recorded.parent.name != "best"
        or recorded.name != expected_checkpoint_name
        or selected_checkpoint_dir.parent.name != "best"
        or selected_checkpoint_dir.name != expected_checkpoint_name
    ):
        raise ValueError(
            "refusing locked-golden evaluation of a non-selected checkpoint"
        )

    data_contract = metadata.get("data_contract")
    if not isinstance(data_contract, dict):
        raise ValueError("checkpoint has no immutable data contract")
    if (
        "ocr_position_contract" not in data_contract
        or "ocr_position_contract_version" not in data_contract
    ):
        raise ValueError(
            "checkpoint has no explicit OCR position contract; locked golden "
            "evaluation remains closed"
        )
    position_contract = resolve_checkpoint_ocr_position_contract(
        data_contract,
        None,
    )
    if resolve_checkpoint_ocr_position_contract(
        metadata,
        position_contract,
    ) != position_contract:
        raise ValueError("selected policy uses a different OCR position contract")

    from Tokenizer.multimodal import PILImageProcessor
    from Tokenizer.unified.bundle import TokenizerBundle

    bundle = TokenizerBundle.from_dir(args.tokenizer)
    bundle_issues = bundle.validate()
    if bundle_issues:
        raise ValueError(
            "invalid tokenizer bundle:\n  - " + "\n  - ".join(bundle_issues)
        )
    current_token_contract = native_tokenization_contract(
        bundle.tokenizer,
        args.tokenizer,
    )
    if data_contract.get("ocr_tokenization_contract") != current_token_contract:
        raise ValueError(
            "tokenizer/native contract differs from OCR GRPO training"
        )
    if _vocab_sha256(bundle.tokenizer) != data_contract.get(
        "tokenizer_vocab_sha256"
    ):
        raise ValueError("tokenizer vocabulary differs from OCR GRPO training")

    visual_source = data_contract.get("visual_source")
    if not isinstance(visual_source, dict):
        raise ValueError("checkpoint has no visual source lineage")
    expected_source_kind = (
        VISUAL_SOURCE_CONTRACT_ALIGNMENT_V3
        if args.visual_source_contract == VISUAL_SOURCE_CONTRACT_ALIGNMENT_V3
        else STREAMING_V2_RELEASE_SOURCE_CONTRACT_KIND
    )
    if visual_source.get("source_contract_kind") != expected_source_kind:
        raise ValueError(
            "--visual-source-contract differs from the selected policy lineage"
        )
    if visual_source.get("ocr_position_contract") != position_contract:
        raise ValueError(
            "visual source lineage uses a different OCR position contract"
        )
    if visual_source.get("ocr_position_contract_version") != (
        OCR_POSITION_CONTRACT_METADATA_VERSION
    ):
        raise ValueError(
            "visual source lineage uses an unsupported OCR position contract version"
        )
    source_checkpoint = visual_source.get("source_checkpoint")
    if not isinstance(source_checkpoint, str) or not source_checkpoint:
        raise ValueError("visual source lineage has no source checkpoint")
    source_dir = resolve_checkpoint_dir(source_checkpoint)
    source_metadata = (
        load_checkpoint_metadata(source_dir)
        if args.visual_source_contract == VISUAL_SOURCE_CONTRACT_ALIGNMENT_V3
        else None
    )
    admit_visual_ocr_source(
        args.visual_source_contract,
        checkpoint_dir=source_dir,
        metadata=source_metadata,
        runtime_native_tokenization_contract=current_token_contract,
        tokenizer_vocab_extent=(
            max(int(index) for index in bundle.tokenizer.vocab.values()) + 1
        ),
        expected_lineage=visual_source,
    )

    golden_identity_manifest = (
        args.golden_identity_manifest
        or str(metadata.get("golden_identity_manifest", ""))
    )
    if not golden_identity_manifest:
        raise ValueError("--golden-identity-manifest is required")
    expected_identity_file_sha = data_contract.get(
        "golden_identity_manifest_sha256"
    )
    if _file_sha256(golden_identity_manifest) != expected_identity_file_sha:
        raise ValueError(
            "golden identity manifest SHA-256 differs from OCR GRPO training"
        )
    identity_rows = load_golden_identity_manifest(
        golden_identity_manifest,
        required_split=str(metadata.get("golden_split", "golden")),
    )
    if golden_identity_semantic_sha256(identity_rows) != data_contract.get(
        "golden_identity_semantic_sha256"
    ):
        raise ValueError(
            "golden identity semantic hash differs from OCR GRPO training"
        )
    training_config = metadata.get("training_config")
    if not isinstance(training_config, dict):
        raise ValueError("checkpoint has no training_config")
    train_manifest = training_config.get("train_data")
    validation_manifest = metadata.get("validation_manifest")
    if not isinstance(train_manifest, str) or not train_manifest:
        raise ValueError("checkpoint has no train manifest path")
    if not isinstance(validation_manifest, str) or not validation_manifest:
        raise ValueError("checkpoint has no validation manifest path")
    public_dataset_contract_path = (
        args.dataset_contract
        or str(metadata.get("public_dataset_contract", ""))
    )
    if not public_dataset_contract_path:
        raise ValueError("--dataset-contract is required")
    public_dataset_contract = load_ocr_dataset_contract(
        public_dataset_contract_path,
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        golden_identity_manifest=golden_identity_manifest,
        tokenization_contract=current_token_contract,
    )
    if public_dataset_contract.get(
        "contract_file_sha256"
    ) != data_contract.get("public_dataset_contract_sha256"):
        raise ValueError(
            "public dataset contract SHA-256 differs from OCR GRPO training"
        )
    selection_receipt_path = Path(
        args.selection_receipt
        or selected_checkpoint_dir.parent / "SELECTION_FINALIZED.json"
    )
    selection_receipt = load_and_validate_selection_receipt(
        selection_receipt_path,
        selected_checkpoint=selected_checkpoint_dir,
        data_contract=data_contract,
    )

    reference_checkpoint = str(metadata.get("reference_checkpoint", ""))
    if not reference_checkpoint:
        raise ValueError("checkpoint has no immutable reference_checkpoint")
    expected_reference_sha256 = data_contract.get("reference_model_sha256")
    if not isinstance(expected_reference_sha256, str):
        raise ValueError("checkpoint data contract has no reference model SHA-256")
    # Validate every registered artifact before opening the locked labels for
    # inference. A broken/tampered reference must not consume the one-shot audit.
    reference_dir = resolve_checkpoint_dir(reference_checkpoint)
    reference_metadata, reference_metadata_sha256 = load_verified_policy_metadata(
        reference_dir,
    )
    if not (reference_dir / "COMPLETE").is_file():
        raise ValueError("immutable reference checkpoint has no COMPLETE marker")
    if reference_metadata.get("phase") != "grpo_reference":
        raise ValueError("registered reference is not an immutable GRPO reference")
    if reference_metadata.get("contract_version") != OCR_GRPO_CONTRACT_VERSION:
        raise ValueError("registered reference uses a different OCR GRPO contract")
    if reference_metadata.get("immutable") is not True:
        raise ValueError("registered reference is not marked immutable")
    if reference_metadata.get("source_checkpoint") != metadata.get("source_checkpoint"):
        raise ValueError("registered reference points to a different source policy")
    if reference_metadata.get(
        "ocr_tokenization_contract"
    ) != current_token_contract:
        raise ValueError(
            "registered reference uses a different OCR tokenization contract"
        )
    if resolve_checkpoint_ocr_position_contract(
        reference_metadata,
        position_contract,
    ) != position_contract:
        raise ValueError(
            "registered reference uses a different OCR position contract"
        )
    if reference_metadata.get(
        "source_checkpoint_model_sha256"
    ) != visual_source.get("source_checkpoint_model_sha256"):
        raise ValueError(
            "registered reference points to different visual source weights"
        )
    selected_rdt_config = metadata.get("rdt_config")
    if not isinstance(selected_rdt_config, dict):
        raise ValueError("selected policy is missing RDT config")
    if reference_metadata.get("rdt_config") != selected_rdt_config:
        raise ValueError("registered reference RDT config differs from selected policy")
    selected_omvt_config = metadata.get("omvt_config")
    if not isinstance(selected_omvt_config, dict):
        raise ValueError("selected OCR policy is missing OMVT config")
    if reference_metadata.get("omvt_config") != selected_omvt_config:
        raise ValueError("registered reference OMVT config differs from selected policy")
    golden_manifest = args.golden_manifest

    grpo_cfg = metadata.get("grpo_config")
    if not isinstance(grpo_cfg, dict):
        raise ValueError("checkpoint has no grpo_config")
    max_new_tokens = int(grpo_cfg["max_new_tokens"])
    recurrent_steps = grpo_cfg.get("recurrent_steps")
    if recurrent_steps is not None:
        recurrent_steps = int(recurrent_steps)
    device = _resolve_device(args.device)
    precision = (
        "bf16" if args.precision == "auto" and device.type == "cuda"
        else "fp32" if args.precision == "auto"
        else args.precision
    )
    if device.type != "cuda" and precision in {"bf16", "fp16"}:
        raise ValueError("bf16/fp16 golden evaluation requires CUDA")

    golden_receipt_path = Path(
        args.golden_receipt
        or Path(golden_manifest).parent / "build_receipt.json"
    )
    if not golden_receipt_path.is_file():
        raise ValueError(f"locked golden receipt is missing: {golden_receipt_path}")
    locked_image_root = Path(golden_manifest).parent.resolve()
    if args.image_root and Path(args.image_root).resolve() != locked_image_root:
        raise ValueError(
            "--image-root differs from the physically isolated golden root"
        )
    golden_receipt = load_locked_golden_receipt_anchor(
        golden_receipt_path,
        golden_manifest=golden_manifest,
        golden_identity_manifest=golden_identity_manifest,
        identity_rows=identity_rows,
        public_dataset_contract=public_dataset_contract_path,
    )
    golden_receipt_sha256 = str(golden_receipt["receipt_file_sha256"])
    consumption_marker = _global_golden_consumption_marker(
        locked_image_root,
        golden_receipt_sha256,
    )
    image_root = str(locked_image_root)

    # Only after every receipt, dataset, source, tokenizer, and position
    # contract has passed do we materialize either policy.  The metadata bytes
    # are pinned to the verified snapshots above, while the trusted selection
    # and data-contract hashes bind the exact model bytes consumed by torch.load.
    expected_selected_sha256 = selection_receipt.get("selected_model_sha256")
    if not isinstance(expected_selected_sha256, str):
        raise ValueError("selection receipt has no selected model SHA-256")
    restored = reconstruct_policy_from_checkpoint(
        selected_checkpoint_dir,
        require_vision=True,
        metadata_override=metadata,
        expected_metadata_sha256=selected_metadata_sha256,
        expected_model_sha256=expected_selected_sha256,
    )
    reference_restored = reconstruct_policy_from_checkpoint(
        reference_dir,
        require_vision=True,
        metadata_override=reference_metadata,
        expected_metadata_sha256=reference_metadata_sha256,
        expected_model_sha256=expected_reference_sha256,
    )
    loaded_selected_model_sha256 = restored.model_sha256
    loaded_reference_model_sha256 = reference_restored.model_sha256
    if loaded_selected_model_sha256 != expected_selected_sha256:
        raise ValueError("loaded selected model SHA-256 differs from selection")
    if loaded_reference_model_sha256 != expected_reference_sha256:
        raise ValueError("loaded reference model SHA-256 differs from training")
    omvt_cfg = restored.omvt_config
    if omvt_cfg is None or reference_restored.omvt_config is None:
        raise ValueError("selected/reference OCR policy is missing OMVT config")
    if asdict(reference_restored.rdt_config) != asdict(restored.rdt_config):
        raise ValueError("registered reference RDT config differs from selected policy")
    if asdict(reference_restored.omvt_config) != asdict(omvt_cfg):
        raise ValueError("registered reference OMVT config differs from selected policy")

    # Finish every non-label preflight before irreversibly claiming the
    # one-shot evaluation: report path, both model/device transfers, and image
    # processor construction. Runtime inference can still fail, but a typo,
    # permissions error, or incompatible checkpoint/device cannot burn golden.
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}"
    )
    with temporary.open("x", encoding="utf-8") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    try:
        selected_model = restored.model.to(device).eval()
        selected_model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        reference_model_preflight = reference_restored.model.to(device).eval()
        reference_model_preflight.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        selected_model = restored.model.to(device).eval()
        processor = PILImageProcessor(
            image_size=omvt_cfg.image_size,
            in_channels=omvt_cfg.in_channels,
        )
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    claim_payload = {
        "schema_version": 2,
        "kind": "ocr_locked_golden_claimed",
        "selected_model_sha256": loaded_selected_model_sha256,
        "selection_receipt_sha256": selection_receipt[
            "receipt_file_sha256"
        ],
        "golden_receipt_sha256": golden_receipt_sha256,
        "report_path": str(destination.absolute()),
    }
    actual_golden = _claim_and_verify_locked_golden(
        consumption_marker=consumption_marker,
        destination=destination,
        temporary=temporary,
        claim_payload=claim_payload,
        golden_receipt=golden_receipt,
        golden_manifest=golden_manifest,
    )

    dataset = OCRPromptDataset(
        golden_manifest,
        encode=lambda text: bundle.encode(text, add_bos=False, add_eos=False),
        n_image_tokens=omvt_cfg.compress_to,
        bos_id=BOS_ID,
        image_start_id=IMAGE_START_ID,
        image_patch_id=IMAGE_PATCH_ID,
        image_end_id=IMAGE_END_ID,
        canonicalize_reference=canonicalize_native_ocr_text,
        image_root=image_root,
        # The generation budget was fixed without consulting golden labels.
        # Evaluate long references as-is (and charge any truncation in CER)
        # instead of aborting after revealing their token lengths.
        max_prompt_len=restored.rdt_config.max_seq_len - max_new_tokens,
        max_completion_len=None,
        max_seq_len=restored.rdt_config.max_seq_len,
        inspect_reference_tokens=False,
        required_split=str(metadata.get("golden_split", "golden")),
        validate_images=True,
        verify_image_decode=False,
        require_sha256=True,
        require_group_id=True,
        require_domain=True,
        verify_sha256=False,
        allow_instruction=False,
    )
    assert_labeled_golden_matches_identity(dataset, identity_rows)
    decode = lambda ids: decode_ocr_completion(  # noqa: E731
        ids,
        bundle.tokenizer.decode,
        valid_token_ids=bundle.tokenizer.id_to_token,
        require_eos=True,
    )

    selected_checkpoint_dir = restored.checkpoint_dir
    selected_omvt_cfg = restored.omvt_config
    model = selected_model
    common = dict(
        policy=model,
        dataset=dataset,
        processor=processor,
        omvt_cfg=omvt_cfg,
        decode=decode,
        batch_size=args.batch_size,
        max_new_tokens=max_new_tokens,
        recurrent_steps=recurrent_steps,
        precision=precision,
        device=device,
        cer_backend=args.cer_backend,
        position_contract=position_contract,
    )
    real = evaluate_ocr_manifest(**common)
    blank = evaluate_ocr_manifest(**common, blank_visual=True)

    # Release the selected model before moving the already verified immutable
    # reference back to the GPU. The two policies share one locked-golden audit
    # but never need to coexist on the accelerator during scored inference.
    del common, model, restored
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if reference_restored.omvt_config is None or selected_omvt_cfg is None:
        raise ValueError("selected/reference OCR policy is missing OMVT config")
    reference_model = reference_restored.model.to(device).eval()
    reference_real = evaluate_ocr_manifest(
        policy=reference_model,
        dataset=dataset,
        processor=processor,
        omvt_cfg=selected_omvt_cfg,
        decode=decode,
        batch_size=args.batch_size,
        max_new_tokens=max_new_tokens,
        recurrent_steps=recurrent_steps,
        precision=precision,
        device=device,
        cer_backend=args.cer_backend,
        position_contract=position_contract,
    )
    visual_gap = blank["grapheme_cer"] - real["grapheme_cer"]
    reference_improvement = (
        reference_real["grapheme_cer"] - real["grapheme_cer"]
    )
    report = {
        "checkpoint": str(selected_checkpoint_dir),
        "selected_policy_model_sha256": loaded_selected_model_sha256,
        "immutable_reference_checkpoint": str(reference_restored.checkpoint_dir),
        "immutable_reference_model_sha256": loaded_reference_model_sha256,
        "golden_manifest_sha256": actual_golden,
        "golden_identity_manifest_sha256": expected_identity_file_sha,
        "golden_identity_semantic_sha256": (
            golden_identity_semantic_sha256(identity_rows)
        ),
        "public_dataset_contract_sha256": public_dataset_contract[
            "contract_file_sha256"
        ],
        "selection_receipt_sha256": selection_receipt[
            "receipt_file_sha256"
        ],
        "golden_build_receipt_sha256": golden_receipt[
            "receipt_file_sha256"
        ],
        "one_shot_consumption_marker": str(consumption_marker),
        "ocr_tokenization_contract": current_token_contract,
        "ocr_position_contract": position_contract,
        "ocr_position_contract_version": OCR_POSITION_CONTRACT_METADATA_VERSION,
        "best_val_grapheme_cer": health.get("best_val_grapheme_cer"),
        "best_val_step": health.get("best_val_step"),
        "golden": real,
        "blank_visual": blank,
        "immutable_reference": reference_real,
        "visual_grapheme_cer_gap": visual_gap,
        "reference_grapheme_cer_improvement": reference_improvement,
        "runtime": {
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else device.type
            ),
            "precision": precision,
            "cer_backend_requested": args.cer_backend,
            "normalization_backend_actual": real[
                "normalization_backend"
            ],
            "batch_size": args.batch_size,
        },
        "generation": {
            "greedy": True,
            "max_new_tokens": max_new_tokens,
            "recurrent_steps": recurrent_steps,
        },
        "thresholds": {
            "min_visual_cer_gap": args.min_visual_cer_gap,
            "min_reference_cer_improvement": args.min_reference_cer_improvement,
            "min_eos_rate": args.min_eos_rate,
        },
        "gates": {
            "visual_gap_pass": visual_gap > args.min_visual_cer_gap,
            "reference_improvement_pass": (
                reference_improvement > args.min_reference_cer_improvement
            ),
            "invalid_output_pass": real["invalid_output_rate"] == 0.0,
            "eos_rate_pass": real["eos_rate"] >= args.min_eos_rate,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return 0 if all(report["gates"].values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
