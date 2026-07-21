# -*- coding: utf-8 -*-

"""One-shot locked-golden evaluation for a selected OCR GRPO checkpoint."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
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
from Model.posttrain.checkpointing import (  # noqa: E402
    OCR_GRPO_CONTRACT_VERSION,
    reconstruct_policy_from_checkpoint,
)
from Model.posttrain.ocr_decode import decode_ocr_completion  # noqa: E402
from Model.posttrain.ocr_eval import evaluate_ocr_manifest  # noqa: E402
from Model.posttrain.preference_data import OCRPromptDataset  # noqa: E402
from Model.training.checkpoint import (  # noqa: E402
    load_checkpoint_metadata,
    resolve_checkpoint_dir,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the selected OCR GRPO policy on a locked golden manifest"
    )
    parser.add_argument("--checkpoint", required=True, help="usually OUTPUT/best/latest")
    parser.add_argument("--tokenizer", required=True, help="TokenizerBundle directory")
    parser.add_argument("--golden-manifest", default="",
                        help="defaults to path recorded by training metadata")
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
    parser.add_argument("--allow-nonbest", action="store_true")
    parser.add_argument("--out", default="", help="optional JSON report path")
    return parser.parse_args(argv)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vocab_sha256(tokenizer) -> str:
    payload = json.dumps(
        sorted((str(token), int(idx)) for token, idx in tokenizer.vocab.items()),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


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
    destination = Path(args.out) if args.out else None
    if destination is not None and destination.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing locked-golden report: {destination}"
        )

    restored = reconstruct_policy_from_checkpoint(
        args.checkpoint,
        require_vision=True,
    )
    if not (restored.checkpoint_dir / "COMPLETE").is_file():
        raise ValueError("selected GRPO checkpoint has no COMPLETE marker")
    metadata = restored.metadata
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
    if int(health.get("best_val_step", 0)) <= 0:
        raise ValueError(
            "no post-update checkpoint improved the validation selection; "
            "locked golden remains closed"
        )
    recorded_best = str(health.get("best_checkpoint", ""))
    if not args.allow_nonbest:
        if not recorded_best:
            raise ValueError("checkpoint metadata has no selected best checkpoint")
        try:
            recorded = Path(recorded_best).resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"recorded best checkpoint is missing: {recorded_best}"
            ) from exc
        if restored.checkpoint_dir.resolve() != recorded:
            raise ValueError(
                "refusing locked-golden evaluation of a non-selected checkpoint; "
                "pass OUTPUT/best/latest or explicitly use --allow-nonbest"
            )

    data_contract = metadata.get("data_contract")
    if not isinstance(data_contract, dict):
        raise ValueError("checkpoint has no immutable data contract")
    reference_checkpoint = str(metadata.get("reference_checkpoint", ""))
    if not reference_checkpoint:
        raise ValueError("checkpoint has no immutable reference_checkpoint")
    expected_reference_sha256 = data_contract.get("reference_model_sha256")
    if not isinstance(expected_reference_sha256, str):
        raise ValueError("checkpoint data contract has no reference model SHA-256")
    # Validate every registered artifact before opening the locked labels for
    # inference. A broken/tampered reference must not consume the one-shot audit.
    reference_dir = resolve_checkpoint_dir(reference_checkpoint)
    if not (reference_dir / "COMPLETE").is_file():
        raise ValueError("immutable reference checkpoint has no COMPLETE marker")
    reference_metadata = load_checkpoint_metadata(reference_dir)
    if reference_metadata.get("phase") != "grpo_reference":
        raise ValueError("registered reference is not an immutable GRPO reference")
    if reference_metadata.get("contract_version") != OCR_GRPO_CONTRACT_VERSION:
        raise ValueError("registered reference uses a different OCR GRPO contract")
    if reference_metadata.get("immutable") is not True:
        raise ValueError("registered reference is not marked immutable")
    if reference_metadata.get("source_checkpoint") != metadata.get("source_checkpoint"):
        raise ValueError("registered reference points to a different source policy")
    if reference_metadata.get("rdt_config") != asdict(restored.rdt_config):
        raise ValueError("registered reference RDT config differs from selected policy")
    if restored.omvt_config is None:
        raise ValueError("selected OCR policy is missing OMVT config")
    if reference_metadata.get("omvt_config") != asdict(restored.omvt_config):
        raise ValueError("registered reference OMVT config differs from selected policy")
    actual_reference_sha256 = _file_sha256(reference_dir / "model.pt")
    if actual_reference_sha256 != expected_reference_sha256:
        raise ValueError("immutable reference model SHA-256 differs from training")
    golden_manifest = args.golden_manifest or str(metadata.get("golden_manifest", ""))
    if not golden_manifest:
        raise ValueError("--golden-manifest is required (not present in metadata)")
    expected_golden = data_contract.get("golden_manifest_sha256")
    actual_golden = _file_sha256(golden_manifest)
    if actual_golden != expected_golden:
        raise ValueError(
            "golden manifest SHA-256 differs from the manifest registered at training"
        )

    from Tokenizer.multimodal import PILImageProcessor
    from Tokenizer.unified.bundle import TokenizerBundle

    bundle = TokenizerBundle.from_dir(args.tokenizer)
    bundle_issues = bundle.validate()
    if bundle_issues:
        raise ValueError("invalid tokenizer bundle:\n  - " + "\n  - ".join(bundle_issues))
    if _vocab_sha256(bundle.tokenizer) != data_contract.get("tokenizer_vocab_sha256"):
        raise ValueError("tokenizer vocabulary differs from OCR GRPO training")
    if _file_sha256(Path(args.tokenizer) / "manifest.json") != data_contract.get(
        "tokenizer_manifest_sha256"
    ):
        raise ValueError("tokenizer bundle manifest differs from OCR GRPO training")

    grpo_cfg = metadata.get("grpo_config")
    if not isinstance(grpo_cfg, dict):
        raise ValueError("checkpoint has no grpo_config")
    max_new_tokens = int(grpo_cfg["max_new_tokens"])
    recurrent_steps = grpo_cfg.get("recurrent_steps")
    if recurrent_steps is not None:
        recurrent_steps = int(recurrent_steps)
    omvt_cfg = restored.omvt_config
    assert omvt_cfg is not None
    image_root = args.image_root or str(data_contract.get("image_root", "")) or None
    dataset = OCRPromptDataset(
        golden_manifest,
        encode=lambda text: bundle.encode(text, add_bos=False, add_eos=False),
        n_image_tokens=omvt_cfg.compress_to,
        bos_id=BOS_ID,
        image_start_id=IMAGE_START_ID,
        image_patch_id=IMAGE_PATCH_ID,
        image_end_id=IMAGE_END_ID,
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
        verify_image_decode=True,
        require_sha256=True,
        require_group_id=True,
        require_domain=True,
        verify_sha256=True,
    )
    decode = lambda ids: decode_ocr_completion(  # noqa: E731
        ids,
        bundle.tokenizer.decode,
        valid_token_ids=bundle.tokenizer.id_to_token,
        require_eos=True,
    )

    device = _resolve_device(args.device)
    precision = (
        "bf16" if args.precision == "auto" and device.type == "cuda"
        else "fp32" if args.precision == "auto"
        else args.precision
    )
    if device.type != "cuda" and precision in {"bf16", "fp16"}:
        raise ValueError("bf16/fp16 golden evaluation requires CUDA")
    selected_checkpoint_dir = restored.checkpoint_dir
    selected_rdt_cfg = restored.rdt_config
    selected_omvt_cfg = restored.omvt_config
    model = restored.model.to(device).eval()
    processor = PILImageProcessor(
        image_size=omvt_cfg.image_size,
        in_channels=omvt_cfg.in_channels,
    )
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
    )
    real = evaluate_ocr_manifest(**common)
    blank = evaluate_ocr_manifest(**common, blank_visual=True)

    # Release the selected model before materializing the 1B immutable
    # reference. The two policies are evaluated in one locked-golden audit but
    # never need to coexist on GPU.
    del common, model, restored
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    reference_restored = reconstruct_policy_from_checkpoint(
        reference_dir,
        require_vision=True,
    )
    if not (reference_restored.checkpoint_dir / "COMPLETE").is_file():
        raise ValueError("immutable reference checkpoint has no COMPLETE marker")
    if reference_restored.metadata.get("phase") != "grpo_reference":
        raise ValueError("registered reference is not an immutable GRPO reference")
    if reference_restored.metadata.get("immutable") is not True:
        raise ValueError("registered reference is not marked immutable")
    if reference_restored.metadata.get("source_checkpoint") != metadata.get(
        "source_checkpoint"
    ):
        raise ValueError("registered reference points to a different source policy")
    if asdict(reference_restored.rdt_config) != asdict(selected_rdt_cfg):
        raise ValueError("registered reference RDT config differs from selected policy")
    if reference_restored.omvt_config is None or selected_omvt_cfg is None:
        raise ValueError("selected/reference OCR policy is missing OMVT config")
    if asdict(reference_restored.omvt_config) != asdict(selected_omvt_cfg):
        raise ValueError("registered reference OMVT config differs from selected policy")
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
    )
    visual_gap = blank["grapheme_cer"] - real["grapheme_cer"]
    reference_improvement = (
        reference_real["grapheme_cer"] - real["grapheme_cer"]
    )
    report = {
        "checkpoint": str(selected_checkpoint_dir),
        "immutable_reference_checkpoint": str(reference_restored.checkpoint_dir),
        "immutable_reference_model_sha256": actual_reference_sha256,
        "golden_manifest_sha256": actual_golden,
        "best_val_grapheme_cer": health.get("best_val_grapheme_cer"),
        "best_val_step": health.get("best_val_step"),
        "golden": real,
        "blank_visual": blank,
        "immutable_reference": reference_real,
        "visual_grapheme_cer_gap": visual_gap,
        "reference_grapheme_cer_improvement": reference_improvement,
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
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.tmp-{os.getpid()}"
        )
        rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    return 0 if all(report["gates"].values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
