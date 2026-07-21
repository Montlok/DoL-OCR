# -*- coding: utf-8 -*-

"""Strict image-conditioned OCR evaluation shared by GRPO and final audit."""

from __future__ import annotations

import contextlib
from collections.abc import Callable

import torch

from Model.config import EOS_ID, PAD_ID, OMVTConfig
from Model.ocr.metrics import ocr_report
from Model.omvt import collate_omvt_batch
from Model.posttrain.grpo import generate_sequences
from Model.posttrain.preference_data import OCRPromptDataset


def _precision_context(precision: str, device: torch.device):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def build_ocr_pixel_batch(
    rows: list[dict],
    processor,
    omvt_cfg: OMVTConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    images = processor([row["image"] for row in rows])
    return {
        key: value.to(device, non_blocking=True)
        for key, value in dict(collate_omvt_batch(images, omvt_cfg)).items()
    }


@torch.no_grad()
def evaluate_ocr_manifest(
    policy: torch.nn.Module,
    dataset: OCRPromptDataset,
    processor,
    omvt_cfg: OMVTConfig,
    decode: Callable[[torch.Tensor], str],
    *,
    batch_size: int,
    max_new_tokens: int,
    recurrent_steps: int | None,
    precision: str,
    device: torch.device,
    cer_backend: str,
    blank_visual: bool = False,
) -> dict[str, float]:
    """Greedy, full-manifest evaluation with no prompt padding.

    Rows are grouped by prompt length before batching. ``blank_visual`` zeros
    every visual tensor while retaining shared patch geometry, providing an
    image-ablation control with the exact same text prompt and decode budget.
    """

    if batch_size <= 0:
        raise ValueError("evaluation batch_size must be positive")
    by_length: dict[int, list[dict]] = {}
    for idx in range(len(dataset)):
        row = dataset[idx]
        by_length.setdefault(len(row["prompt_ids"]), []).append(row)

    predictions: list[str] = []
    references: list[str] = []
    eos_rows = 0
    invalid_rows = 0
    for prompt_len in sorted(by_length):
        rows_for_length = by_length[prompt_len]
        for start in range(0, len(rows_for_length), batch_size):
            rows = rows_for_length[start : start + batch_size]
            prompts = torch.tensor(
                [row["prompt_ids"] for row in rows],
                dtype=torch.long,
                device=device,
            )
            pixels = build_ocr_pixel_batch(rows, processor, omvt_cfg, device)
            if blank_visual:
                pixels = {
                    key: value if key.endswith("_bbox") else torch.zeros_like(value)
                    for key, value in pixels.items()
                }
            with _precision_context(precision, device):
                sequences = generate_sequences(
                    policy,
                    prompts,
                    max_new_tokens=max_new_tokens,
                    greedy=True,
                    eos_id=EOS_ID,
                    pad_id=PAD_ID,
                    recurrent_steps=recurrent_steps,
                    pixel_values=pixels,
                )
            tails = sequences[:, prompt_len:]
            for tail, row in zip(tails, rows):
                prediction = decode(tail)
                predictions.append(prediction)
                references.append(row["reference"])
                eos_rows += int(bool((tail == EOS_ID).any()))
                invalid_rows += int("\ufffd" in prediction)

    report = ocr_report(predictions, references, backend=cer_backend)
    denom = max(1, len(predictions))
    metrics = {
        "grapheme_cer": float(report.grapheme_cer),
        "norm_cer": float(report.norm_cer),
        "raw_cer": float(report.raw_cer),
        "wer": float(report.wer),
        "line_exact": float(report.line_exact),
        "eos_rate": eos_rows / denom,
        "invalid_output_rate": invalid_rows / denom,
        "samples": float(len(predictions)),
    }
    for bucket, values in (report.script_cer or {}).items():
        metrics[f"script_{bucket}_cer"] = float(values["cer"])
    return metrics


__all__ = ["build_ocr_pixel_batch", "evaluate_ocr_manifest"]
