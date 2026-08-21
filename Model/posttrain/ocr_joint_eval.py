# -*- coding: utf-8 -*-

"""Cache-free joint global/detail validation for admitted anyres OCR batches."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from Model.ocr.metrics import edit_distance, grapheme_clusters, ocr_report
from Model.ocr.position_contract import BOUNDARY_V1
from Model.posttrain.ocr_joint_forward import encode_anyres_visual_batch


DEPLOYMENT_BUCKET_WEIGHTS = {
    "print": 0.6,
    "handwritten_good": 0.1,
    "handwritten_medium": 0.2,
    "handwritten_poor": 0.1,
}


@torch.no_grad()
def evaluate_ocr_joint(
    model,
    val_batches: Iterable[Mapping[str, Any]],
    decode: Callable[[torch.Tensor], str],
    *,
    max_new_tokens: int,
    device: torch.device | str,
    expected_image_dataset_contract_sha256: str,
    text_replay_batches: Iterable[Mapping[str, Any]] | None = None,
    expected_text_replay_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Evaluate real, zero-visual, and same-bucket shuffled conditions."""

    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    if not callable(decode):
        raise TypeError("decode must be callable")
    device = torch.device(device)
    batches = list(val_batches)
    if not batches:
        raise ValueError("joint OCR validation batches must not be empty")
    image_contract = _uniform_dataset_contract(
        batches,
        expected=expected_image_dataset_contract_sha256,
        label="joint OCR image",
    )

    text_batches = (
        None if text_replay_batches is None else list(text_replay_batches)
    )
    if text_batches is None:
        if expected_text_replay_contract_sha256 is not None:
            raise ValueError(
                "expected text replay contract requires text replay batches"
            )
    elif expected_text_replay_contract_sha256 is None:
        raise ValueError(
            "text replay evaluation requires an expected dataset contract"
        )
    else:
        _uniform_dataset_contract(
            text_batches,
            expected=expected_text_replay_contract_sha256,
            label="text replay",
        )
    ignore_index = int(getattr(model.cfg, "ignore_index", -100))
    descriptors = [
        _describe_batch(batch, index, ignore_index=ignore_index)
        for index, batch in enumerate(batches)
    ]
    total_bucket_counts = Counter(
        bucket
        for descriptor in descriptors
        for bucket in descriptor["buckets"]
    )
    if set(total_bucket_counts) != set(DEPLOYMENT_BUCKET_WEIGHTS):
        raise ValueError("validation must contain all four deployment buckets")
    if any(total_bucket_counts[bucket] < 2 for bucket in DEPLOYMENT_BUCKET_WEIGHTS):
        raise ValueError("each deployment bucket needs at least two samples to shuffle")

    conditions = {
        "real": _new_condition_accumulator(),
        "blank": _new_condition_accumulator(),
        "shuffled": _new_condition_accumulator(),
    }
    first_token_nll_values: dict[str, list[float]] = {
        name: [] for name in conditions
    }
    was_training = bool(model.training)
    model.eval()
    try:
        for batch, descriptor in zip(batches, descriptors):
            visual = encode_anyres_visual_batch(model, batch, device=device)
            global_features = model.vision.encode_visual(
                visual["global_pixel_values"]
            )
            if not isinstance(global_features, torch.Tensor):
                raise TypeError("global visual encoder must return a tensor")
            batch_size = int(descriptor["prompts"].shape[0])
            if global_features.shape[0] != batch_size:
                raise ValueError("global visual features do not match text batch size")
            detail_memory = visual["detail_memory"]
            detail_cu = visual["detail_cu_seqlens"]
            _validate_detail_memory(detail_memory, detail_cu, batch_size)

            permutation = _same_bucket_cyclic_permutation(descriptor["buckets"])
            shuffled_global = global_features.index_select(
                0,
                permutation.to(global_features.device),
            )
            shuffled_detail, shuffled_cu = _permute_packed_detail(
                detail_memory,
                detail_cu,
                permutation,
            )
            condition_visuals = {
                "real": (global_features, detail_memory, detail_cu),
                "blank": (
                    torch.zeros_like(global_features),
                    torch.zeros_like(detail_memory),
                    detail_cu,
                ),
                "shuffled": (shuffled_global, shuffled_detail, shuffled_cu),
            }

            prompts = descriptor["prompts"].to(device, non_blocking=True)
            first_targets = descriptor["first_targets"].to(
                device,
                non_blocking=True,
            )
            for condition_name, (global_memory, detail, cu) in condition_visuals.items():
                sequences, eos_mask, hit_cap_mask = _cache_free_greedy_generate(
                    model,
                    prompts,
                    global_features=global_memory,
                    detail_memory=detail,
                    detail_cu_seqlens=cu,
                    max_new_tokens=max_new_tokens,
                )
                predictions, invalid_mask = _decode_sequences(
                    sequences,
                    prompt_length=prompts.shape[1],
                    eos_id=int(model.cfg.eos_id),
                    decode=decode,
                )
                accumulator = conditions[condition_name]
                accumulator["sample_ids"].extend(descriptor["sample_ids"])
                accumulator["predictions"].extend(predictions)
                accumulator["references"].extend(descriptor["references"])
                accumulator["buckets"].extend(descriptor["buckets"])
                accumulator["eos"].extend(eos_mask.detach().cpu().tolist())
                accumulator["hit_cap"].extend(hit_cap_mask.detach().cpu().tolist())
                accumulator["invalid"].extend(invalid_mask)

                first_token_nll_values[condition_name].extend(
                    _first_token_nll_values(
                        model,
                        prompts,
                        first_targets,
                        global_features=global_memory,
                        detail_memory=detail,
                        detail_cu_seqlens=cu,
                    )
                )

        reports = {
            name: _finalize_condition(accumulator)
            for name, accumulator in conditions.items()
        }
        real_buckets = reports["real"]["buckets"]
        deployment_weighted_cer = sum(
            DEPLOYMENT_BUCKET_WEIGHTS[bucket]
            * float(real_buckets[bucket]["raw_grapheme_cer"])
            for bucket in DEPLOYMENT_BUCKET_WEIGHTS
        )
        worst_bucket = max(
            DEPLOYMENT_BUCKET_WEIGHTS,
            key=lambda bucket: float(real_buckets[bucket]["raw_grapheme_cer"]),
        )
        nll = {
            name: sum(first_token_nll_values[name])
            / len(first_token_nll_values[name])
            for name in conditions
        }
        real_sample_cer = _sample_cer_values(conditions["real"])
        blank_sample_cer = _sample_cer_values(conditions["blank"])
        shuffled_sample_cer = _sample_cer_values(conditions["shuffled"])
        blank_cer_delta = [
            control - real
            for control, real in zip(blank_sample_cer, real_sample_cer, strict=True)
        ]
        shuffled_cer_delta = [
            control - real
            for control, real in zip(shuffled_sample_cer, real_sample_cer, strict=True)
        ]
        blank_nll_delta = [
            control - real
            for control, real in zip(
                first_token_nll_values["blank"],
                first_token_nll_values["real"],
                strict=True,
            )
        ]
        shuffled_nll_delta = [
            control - real
            for control, real in zip(
                first_token_nll_values["shuffled"],
                first_token_nll_values["real"],
                strict=True,
            )
        ]
        text_report = (
            None
            if text_batches is None
            else _evaluate_text_replay_nll(
                model,
                text_batches,
                device=device,
                expected_dataset_contract_sha256=(
                    expected_text_replay_contract_sha256
                ),
            )
        )
        result = {
            "schema_version": 1,
            "image_dataset_contract_sha256": image_contract,
            "decode_contract": {
                "greedy": True,
                "use_cache": False,
                "position_contract": BOUNDARY_V1,
                "visual": "preencoded_global_plus_packed_detail",
                "blank": "zero_encoded_global_and_detail",
                "shuffled": "same_bucket_cyclic_no_fixed_points",
                "max_new_tokens": max_new_tokens,
            },
            "real": reports["real"],
            "controls": {
                "blank": reports["blank"],
                "shuffled": reports["shuffled"],
            },
            "grounding": {
                "blank_cer_gap": (
                    reports["blank"]["overall"]["raw_grapheme_cer"]
                    - reports["real"]["overall"]["raw_grapheme_cer"]
                ),
                "shuffled_cer_gap": (
                    reports["shuffled"]["overall"]["raw_grapheme_cer"]
                    - reports["real"]["overall"]["raw_grapheme_cer"]
                ),
                "first_token_nll": nll,
                "blank_first_token_nll_gap": nll["blank"] - nll["real"],
                "shuffled_first_token_nll_gap": nll["shuffled"] - nll["real"],
                "paired_bootstrap_95ci": {
                    "blank_cer_gap": _bootstrap_mean_ci(blank_cer_delta),
                    "shuffled_cer_gap": _bootstrap_mean_ci(shuffled_cer_delta),
                    "blank_first_token_nll_gap": _bootstrap_mean_ci(blank_nll_delta),
                    "shuffled_first_token_nll_gap": _bootstrap_mean_ci(
                        shuffled_nll_delta
                    ),
                },
            },
            "deployment_weighted_cer": deployment_weighted_cer,
            "deployment_weights": dict(DEPLOYMENT_BUCKET_WEIGHTS),
            "worst_bucket": {
                "name": worst_bucket,
                "raw_grapheme_cer": real_buckets[worst_bucket][
                    "raw_grapheme_cer"
                ],
            },
            "text_replay": text_report,
            "selection_records": _selection_records(conditions["real"]),
        }
        return result
    finally:
        model.train(was_training)


def joint_eval_eligibility(
    report: Mapping[str, Any],
    *,
    baseline_bucket_cer: Mapping[str, float],
    baseline_text_token_nll: float,
    min_relative_cer_improvement: float = 0.0,
) -> dict[str, Any]:
    """Apply deployment gates without reading any baseline or golden files."""

    if set(baseline_bucket_cer) != set(DEPLOYMENT_BUCKET_WEIGHTS):
        raise ValueError("baseline_bucket_cer must contain exactly four buckets")
    if not math.isfinite(float(baseline_text_token_nll)) or baseline_text_token_nll <= 0:
        raise ValueError("baseline_text_token_nll must be finite and positive")
    if (
        not math.isfinite(float(min_relative_cer_improvement))
        or not 0.0 <= float(min_relative_cer_improvement) < 1.0
    ):
        raise ValueError("min_relative_cer_improvement must be in [0,1)")

    reasons: list[str] = []
    overall = report["real"]["overall"]
    if float(overall["eos_rate"]) < 0.995:
        reasons.append("eos_rate_below_0.995")
    if int(overall["invalid_count"]) != 0:
        reasons.append("invalid_output_nonzero")
    if float(overall["hit_cap_rate"]) > 0.005:
        reasons.append("hit_cap_rate_above_0.005")

    grounding = report["grounding"]
    for name in (
        "blank_cer_gap",
        "shuffled_cer_gap",
        "blank_first_token_nll_gap",
        "shuffled_first_token_nll_gap",
    ):
        if not math.isfinite(float(grounding[name])) or float(grounding[name]) <= 0:
            reasons.append(f"{name}_not_positive")
    confidence = grounding.get("paired_bootstrap_95ci")
    if not isinstance(confidence, Mapping):
        reasons.append("grounding_confidence_missing")
    else:
        for name in (
            "blank_cer_gap",
            "shuffled_cer_gap",
            "blank_first_token_nll_gap",
            "shuffled_first_token_nll_gap",
        ):
            interval = confidence.get(name)
            if (
                not isinstance(interval, Mapping)
                or not math.isfinite(float(interval.get("low", float("nan"))))
                or float(interval["low"]) <= 0
            ):
                reasons.append(f"{name}_ci_lower_not_positive")

    bucket_checks: dict[str, bool] = {}
    baseline_weighted_cer = 0.0
    for bucket in DEPLOYMENT_BUCKET_WEIGHTS:
        baseline = float(baseline_bucket_cer[bucket])
        if not math.isfinite(baseline) or baseline < 0:
            raise ValueError(f"baseline CER for {bucket} must be finite and non-negative")
        current = float(report["real"]["buckets"][bucket]["raw_grapheme_cer"])
        baseline_weighted_cer += DEPLOYMENT_BUCKET_WEIGHTS[bucket] * baseline
        bucket_checks[bucket] = current <= baseline
        if not bucket_checks[bucket]:
            reasons.append(f"bucket_regression:{bucket}")
    current_weighted_cer = float(report["deployment_weighted_cer"])
    required_weighted_cer = baseline_weighted_cer * (
        1.0 - float(min_relative_cer_improvement)
    )
    if current_weighted_cer > required_weighted_cer:
        reasons.append("deployment_cer_improvement_below_required")

    text_report = report.get("text_replay")
    if not isinstance(text_report, Mapping):
        text_drift = None
        reasons.append("text_replay_nll_missing")
    else:
        current_text_nll = float(text_report["token_nll"])
        text_drift = current_text_nll / float(baseline_text_token_nll) - 1.0
        if not math.isfinite(text_drift) or text_drift > 0.02:
            reasons.append("text_token_nll_drift_above_2pct")

    return {
        "eligible": not reasons,
        "reasons": reasons,
        "checks": {
            "eos_rate": float(overall["eos_rate"]),
            "invalid_count": int(overall["invalid_count"]),
            "hit_cap_rate": float(overall["hit_cap_rate"]),
            "grounding_gaps": {
                name: float(grounding[name])
                for name in (
                    "blank_cer_gap",
                    "shuffled_cer_gap",
                    "blank_first_token_nll_gap",
                    "shuffled_first_token_nll_gap",
                )
            },
            "bucket_non_regression": bucket_checks,
            "deployment_cer": {
                "baseline": baseline_weighted_cer,
                "current": current_weighted_cer,
                "required_max": required_weighted_cer,
                "min_relative_improvement": float(
                    min_relative_cer_improvement
                ),
            },
            "text_token_nll_relative_drift": text_drift,
        },
    }


def dual_baseline_joint_eligibility(
    report: Mapping[str, Any],
    *,
    runtime_baseline: Mapping[str, Any],
    historical_visual_baseline: Mapping[str, Any],
) -> dict[str, Any]:
    """Require non-regression against both runtime and historical baselines."""

    def inputs(baseline: Mapping[str, Any]) -> tuple[dict[str, float], float]:
        return (
            {
                bucket: float(
                    baseline["real"]["buckets"][bucket]["raw_grapheme_cer"]
                )
                for bucket in DEPLOYMENT_BUCKET_WEIGHTS
            },
            float(baseline["text_replay"]["token_nll"]),
        )

    runtime_buckets, runtime_text = inputs(runtime_baseline)
    historical_buckets, historical_text = inputs(historical_visual_baseline)
    result = joint_eval_eligibility(
        report,
        baseline_bucket_cer=runtime_buckets,
        baseline_text_token_nll=runtime_text,
        min_relative_cer_improvement=0.0,
    )
    historical = joint_eval_eligibility(
        report,
        baseline_bucket_cer=historical_buckets,
        baseline_text_token_nll=historical_text,
        min_relative_cer_improvement=0.0,
    )
    result["historical_visual_best_gate"] = historical
    if historical["eligible"] is not True:
        result["eligible"] = False
        result["reasons"] = [
            *result["reasons"],
            *(
                f"historical_visual_best:{reason}"
                for reason in historical["reasons"]
            ),
        ]
    return result


def _describe_batch(
    batch: Mapping[str, Any],
    index: int,
    *,
    ignore_index: int,
) -> dict[str, Any]:
    if not isinstance(batch, Mapping):
        raise TypeError(f"val_batches[{index}] must be a mapping")
    if batch.get("position_contract") != BOUNDARY_V1:
        raise ValueError(f"val_batches[{index}] must use boundary_v1")
    if any(name in batch for name in ("word_pos", "morph_depth", "token_offsets")):
        raise ValueError("boundary_v1 validation batches must not materialize positions")
    input_ids = batch.get("input_ids")
    attention_mask = batch.get("attention_mask")
    labels = batch.get("labels")
    if not all(isinstance(value, torch.Tensor) for value in (input_ids, attention_mask, labels)):
        raise TypeError("joint OCR input_ids, attention_mask, and labels must be tensors")
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape or labels.shape != input_ids.shape:
        raise ValueError("joint OCR text tensors must have aligned [B, T] shapes")
    batch_size = int(input_ids.shape[0])
    if batch_size <= 0:
        raise ValueError("joint OCR validation batch must not be empty")

    supervised = labels != ignore_index
    first_positions: list[int] = []
    first_targets: list[int] = []
    for row_index in range(batch_size):
        positions = torch.nonzero(supervised[row_index], as_tuple=False).flatten()
        if positions.numel() == 0:
            raise ValueError(f"val_batches[{index}] row {row_index} has no target")
        first = int(positions[0].item())
        first_positions.append(first)
        first_targets.append(int(labels[row_index, first].item()))
    if len(set(first_positions)) != 1:
        raise ValueError("joint OCR batch prompt lengths must be uniform")
    prompt_length = first_positions[0]
    prompts = input_ids[:, :prompt_length]
    if not bool((attention_mask[:, :prompt_length] == 1).all()):
        raise ValueError("joint OCR prompts must not contain padding")

    metadata = batch.get("sample_metadata")
    buckets = batch.get("quota_buckets")
    if not isinstance(metadata, Sequence) or isinstance(metadata, (str, bytes)):
        raise TypeError("sample_metadata must be a sequence")
    if not isinstance(buckets, Sequence) or isinstance(buckets, (str, bytes)):
        raise TypeError("quota_buckets must be a sequence")
    if len(metadata) != batch_size or len(buckets) != batch_size:
        raise ValueError("sample metadata and quota buckets must match batch size")
    references: list[str] = []
    sample_ids: list[str] = []
    parsed_buckets: list[str] = []
    for row_index, (sample, bucket) in enumerate(zip(metadata, buckets)):
        if not isinstance(sample, Mapping):
            raise TypeError(f"sample_metadata[{row_index}] must be a mapping")
        reference = sample.get("reference_model")
        if not isinstance(reference, Mapping) or not isinstance(reference.get("text"), str):
            raise ValueError(f"sample_metadata[{row_index}] has no model reference")
        if bucket not in DEPLOYMENT_BUCKET_WEIGHTS:
            raise ValueError(f"sample_metadata[{row_index}] has invalid quota bucket")
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"sample_metadata[{row_index}] has no sample_id")
        references.append(str(reference["text"]))
        sample_ids.append(sample_id)
        parsed_buckets.append(str(bucket))
    counts = Counter(parsed_buckets)
    if any(count == 1 for count in counts.values()):
        raise ValueError("same-bucket shuffle is not evaluable with a singleton bucket")
    return {
        "prompts": prompts,
        "first_targets": torch.tensor(first_targets, dtype=torch.long),
        "references": references,
        "buckets": parsed_buckets,
        "sample_ids": sample_ids,
    }


def _same_bucket_cyclic_permutation(buckets: Sequence[str]) -> torch.Tensor:
    permutation = list(range(len(buckets)))
    for bucket in DEPLOYMENT_BUCKET_WEIGHTS:
        indices = [index for index, value in enumerate(buckets) if value == bucket]
        if len(indices) == 1:
            raise ValueError(f"bucket {bucket!r} cannot be shuffled with one sample")
        for position, target_index in enumerate(indices):
            permutation[target_index] = indices[(position + 1) % len(indices)]
    if any(source == target for target, source in enumerate(permutation)):
        raise RuntimeError("same-bucket cyclic shuffle produced a fixed point")
    return torch.tensor(permutation, dtype=torch.long)


def _validate_detail_memory(memory: object, cu: object, batch_size: int) -> None:
    if not isinstance(memory, torch.Tensor) or memory.ndim != 2:
        raise TypeError("detail_memory must be a [sumM, D] tensor")
    if not isinstance(cu, torch.Tensor) or cu.ndim != 1 or cu.numel() != batch_size + 1:
        raise TypeError("detail_cu_seqlens must have shape [B + 1]")
    if cu.device.type != "cpu":
        raise ValueError("detail_cu_seqlens must remain on CPU")
    boundaries = [int(value) for value in cu.tolist()]
    if boundaries[0] != 0 or boundaries[-1] != memory.shape[0]:
        raise ValueError("detail_cu_seqlens boundaries do not match detail_memory")
    if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("every validation sample must have non-empty detail memory")


def _permute_packed_detail(
    memory: torch.Tensor,
    cu: torch.Tensor,
    permutation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cu.device.type != "cpu":
        raise ValueError("detail_cu_seqlens must remain on CPU")
    boundaries = [int(value) for value in cu.tolist()]
    segments: list[torch.Tensor] = []
    cumulative = [0]
    for source_index in permutation.tolist():
        segment = memory[boundaries[source_index]:boundaries[source_index + 1]]
        segments.append(segment)
        cumulative.append(cumulative[-1] + int(segment.shape[0]))
    return torch.cat(segments, dim=0), torch.tensor(
        cumulative,
        dtype=cu.dtype,
        device="cpu",
    )


def _cache_free_greedy_generate(
    model,
    prompts: torch.Tensor,
    *,
    global_features: torch.Tensor,
    detail_memory: torch.Tensor,
    detail_cu_seqlens: torch.Tensor,
    max_new_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence = prompts
    attention = torch.ones_like(prompts, dtype=torch.long)
    batch_size = prompts.shape[0]
    eos_id = int(model.cfg.eos_id)
    pad_id = int(model.cfg.pad_id)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=prompts.device)
    eos_seen = torch.zeros_like(finished)
    for _ in range(max_new_tokens):
        output = model(
            input_ids=sequence,
            attention_mask=attention,
            visual_features=global_features,
            detail_memory=detail_memory,
            detail_cu_seqlens=detail_cu_seqlens,
            position_contract=BOUNDARY_V1,
            return_logits=True,
        )
        logits = output.get("logits") if isinstance(output, Mapping) else None
        if not isinstance(logits, torch.Tensor) or logits.shape[:2] != sequence.shape:
            raise RuntimeError("joint OCR model returned invalid generation logits")
        next_token = logits[:, -1, :].float().argmax(dim=-1)
        next_token = torch.where(
            finished,
            torch.full_like(next_token, pad_id),
            next_token,
        )
        sequence = torch.cat((sequence, next_token.unsqueeze(1)), dim=1)
        attention = torch.cat(
            (attention, (~finished).to(dtype=torch.long).unsqueeze(1)),
            dim=1,
        )
        newly_finished = (~finished) & (next_token == eos_id)
        eos_seen |= newly_finished
        finished |= newly_finished
        if bool(finished.all()):
            break
    return sequence, eos_seen, ~eos_seen


def _decode_sequences(
    sequences: torch.Tensor,
    *,
    prompt_length: int,
    eos_id: int,
    decode: Callable[[torch.Tensor], str],
) -> tuple[list[str], list[bool]]:
    predictions: list[str] = []
    invalid: list[bool] = []
    for row in sequences[:, prompt_length:].detach().cpu():
        eos_positions = torch.nonzero(row == eos_id, as_tuple=False).flatten()
        content = row[: int(eos_positions[0].item())] if eos_positions.numel() else row
        try:
            prediction = decode(content)
            if not isinstance(prediction, str):
                raise TypeError("decode must return str")
            is_invalid = _invalid_text(prediction)
        except Exception:
            prediction = ""
            is_invalid = True
        predictions.append(prediction)
        invalid.append(is_invalid)
    return predictions, invalid


def _invalid_text(text: str) -> bool:
    return any(
        character in {"\x00", "\ufffd"} or 0xD800 <= ord(character) <= 0xDFFF
        for character in text
    )


def _first_token_nll_values(
    model,
    prompts: torch.Tensor,
    targets: torch.Tensor,
    *,
    global_features: torch.Tensor,
    detail_memory: torch.Tensor,
    detail_cu_seqlens: torch.Tensor,
) -> list[float]:
    output = model(
        input_ids=prompts,
        attention_mask=torch.ones_like(prompts, dtype=torch.long),
        visual_features=global_features,
        detail_memory=detail_memory,
        detail_cu_seqlens=detail_cu_seqlens,
        position_contract=BOUNDARY_V1,
        return_logits=True,
    )
    logits = output.get("logits") if isinstance(output, Mapping) else None
    if not isinstance(logits, torch.Tensor):
        raise RuntimeError("joint OCR model returned no first-token logits")
    return (
        F.cross_entropy(
            logits[:, -1, :].float(),
            targets,
            reduction="none",
        )
        .detach()
        .cpu()
        .tolist()
    )


def _sample_cer_values(accumulator: Mapping[str, list[Any]]) -> list[float]:
    return [
        float(ocr_report([prediction], [reference], backend="python").grapheme_cer)
        for prediction, reference in zip(
            accumulator["predictions"],
            accumulator["references"],
            strict=True,
        )
    ]


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int = 42,
    resamples: int = 2000,
) -> dict[str, float | int]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    if tensor.numel() < 2:
        raise ValueError("paired bootstrap requires at least two samples")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("paired bootstrap values must be finite")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(
        tensor.numel(),
        (resamples, tensor.numel()),
        generator=generator,
    )
    means = tensor[indices].mean(dim=1).sort().values
    low_index = max(0, int(0.025 * resamples) - 1)
    high_index = min(resamples - 1, int(0.975 * resamples))
    return {
        "mean": float(tensor.mean()),
        "low": float(means[low_index]),
        "high": float(means[high_index]),
        "samples": int(tensor.numel()),
        "resamples": resamples,
        "seed": seed,
    }


def _new_condition_accumulator() -> dict[str, list[Any]]:
    return {
        "sample_ids": [],
        "predictions": [],
        "references": [],
        "buckets": [],
        "eos": [],
        "hit_cap": [],
        "invalid": [],
    }


def _selection_records(
    accumulator: Mapping[str, list[Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sample_id, bucket, prediction, reference in zip(
        accumulator["sample_ids"],
        accumulator["buckets"],
        accumulator["predictions"],
        accumulator["references"],
        strict=True,
    ):
        if sample_id in seen:
            raise ValueError(f"duplicate validation sample_id {sample_id!r}")
        seen.add(sample_id)
        pred_graphemes = grapheme_clusters(prediction)
        ref_graphemes = grapheme_clusters(reference)
        if not ref_graphemes:
            raise ValueError(f"validation sample {sample_id!r} has empty reference")
        records.append(
            {
                "sample_id": sample_id,
                "bucket": bucket,
                "grapheme_edits": edit_distance(pred_graphemes, ref_graphemes),
                "reference_graphemes": len(ref_graphemes),
            }
        )
    return sorted(records, key=lambda row: row["sample_id"])


def _finalize_condition(accumulator: Mapping[str, list[Any]]) -> dict[str, Any]:
    overall = _metric_block(
        accumulator["predictions"],
        accumulator["references"],
        accumulator["eos"],
        accumulator["hit_cap"],
        accumulator["invalid"],
    )
    buckets: dict[str, Any] = {}
    for bucket in DEPLOYMENT_BUCKET_WEIGHTS:
        indices = [
            index
            for index, value in enumerate(accumulator["buckets"])
            if value == bucket
        ]
        if not indices:
            raise RuntimeError(f"condition has no samples for bucket {bucket!r}")
        buckets[bucket] = _metric_block(
            [accumulator["predictions"][index] for index in indices],
            [accumulator["references"][index] for index in indices],
            [accumulator["eos"][index] for index in indices],
            [accumulator["hit_cap"][index] for index in indices],
            [accumulator["invalid"][index] for index in indices],
        )
    return {"overall": overall, "buckets": buckets}


def _metric_block(
    predictions: Sequence[str],
    references: Sequence[str],
    eos: Sequence[bool],
    hit_cap: Sequence[bool],
    invalid: Sequence[bool],
) -> dict[str, Any]:
    report = ocr_report(predictions, references, backend="python")
    count = len(predictions)
    return {
        "samples": count,
        "raw_grapheme_cer": float(report.grapheme_cer),
        "raw_line_exact": float(report.raw_line_exact),
        "eos_count": int(sum(bool(value) for value in eos)),
        "eos_rate": sum(bool(value) for value in eos) / count,
        "hit_cap_count": int(sum(bool(value) for value in hit_cap)),
        "hit_cap_rate": sum(bool(value) for value in hit_cap) / count,
        "invalid_count": int(sum(bool(value) for value in invalid)),
        "invalid_rate": sum(bool(value) for value in invalid) / count,
        "symbol_metrics": report.symbol_metrics,
    }


def _evaluate_text_replay_nll(
    model,
    batches: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    expected_dataset_contract_sha256: str,
) -> dict[str, Any]:
    if not batches:
        raise ValueError("text replay evaluation batches must not be empty")
    dataset_contract = _uniform_dataset_contract(
        batches,
        expected=expected_dataset_contract_sha256,
        label="text replay",
    )
    weighted_loss = 0.0
    token_count = 0
    for index, batch in enumerate(batches):
        if batch.get("position_contract") != BOUNDARY_V1:
            raise ValueError(f"text replay batch {index} must use boundary_v1")
        if any(name in batch for name in ("word_pos", "morph_depth", "token_offsets")):
            raise ValueError("text replay evaluation must not materialize positions")
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        labels = batch.get("labels")
        if not all(isinstance(value, torch.Tensor) for value in (input_ids, attention_mask, labels)):
            raise TypeError("text replay tensors are missing")
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_contract=BOUNDARY_V1,
            return_logits=False,
            loss_chunk_size=4096,
        )
        parts = output.get("loss_parts") if isinstance(output, Mapping) else None
        if not isinstance(parts, Mapping) or "forward" not in parts:
            raise RuntimeError("text replay model output has no forward NLL")
        ignore_index = int(getattr(model.cfg, "ignore_index", -100))
        valid = int((labels[:, 1:] != ignore_index).sum().item())
        if valid <= 0:
            raise ValueError("text replay batch has no causal target tokens")
        forward_nll = float(parts["forward"])
        if not math.isfinite(forward_nll) or forward_nll < 0:
            raise RuntimeError("text replay forward NLL is invalid")
        weighted_loss += forward_nll * valid
        token_count += valid
    return {
        "dataset_contract_sha256": dataset_contract,
        "token_nll": weighted_loss / token_count,
        "target_tokens": token_count,
        "batches": len(batches),
        "position_contract": BOUNDARY_V1,
    }


def _require_sha256(value: object, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{where} must be lowercase SHA-256")
    return value


def _uniform_dataset_contract(
    batches: Sequence[Mapping[str, Any]],
    *,
    expected: object,
    label: str,
) -> str:
    expected_contract = _require_sha256(expected, f"expected {label} dataset contract")
    contracts = {
        _require_sha256(
            batch.get("dataset_contract_sha256"),
            f"{label} batch {index} dataset contract",
        )
        for index, batch in enumerate(batches)
    }
    if len(contracts) != 1:
        raise ValueError(f"{label} validation batches mix dataset contracts")
    actual = next(iter(contracts))
    if actual != expected_contract:
        raise ValueError(f"{label} dataset contract differs from expected")
    return actual


__all__ = [
    "DEPLOYMENT_BUCKET_WEIGHTS",
    "dual_baseline_joint_eligibility",
    "evaluate_ocr_joint",
    "joint_eval_eligibility",
]
