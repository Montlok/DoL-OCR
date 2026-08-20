# -*- coding: utf-8 -*-

"""Anyres global/detail GRPO core with unique-image encoding per stage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import torch
import torch.nn.functional as F

from Model.ocr.position_contract import BOUNDARY_V1
from Model.posttrain.grpo import (
    GRPOConfig,
    _completion_action_mask,
    _generate_sequences_with_behavior,
    _unwrap_model,
    grpo_loss,
    group_relative_advantages,
)
from Model.posttrain.logprobs import token_logprobs_with_mask
from Model.posttrain.ocr_joint_forward import encode_anyres_visual_batch
from Model.posttrain.ocr_anyres_reward import AnyresOCRRewardAdapter


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class AnyresGRPOAdmission:
    policy_checkpoint_sha256: str
    policy_metadata_sha256: str
    reference_checkpoint_sha256: str | None
    reference_metadata_sha256: str | None
    joint_stage_result_sha256: str
    tokenizer_contract_sha256: str
    visual_contract_sha256: str
    preprocess_contract_sha256: str
    native_migration_receipt_sha256: str
    trainability_contract_sha256: str
    trainable_parameter_names_sha256: str
    reward_contract_sha256: str
    dataset_admission_report_sha256: str
    train_dataset_contract_sha256: str
    sft_validation_dataset_contract_sha256: str
    kl_selection_dataset_contract_sha256: str
    formal_monitor_dataset_contract_sha256: str
    text_replay_train_contract_sha256: str
    text_replay_sft_validation_contract_sha256: str
    text_replay_kl_selection_contract_sha256: str
    text_replay_formal_monitor_contract_sha256: str
    runtime_source_receipt_sha256: str
    recommended_max_new_tokens: int
    minimum_reward_spread: float = 0.005
    position_contract: str = BOUNDARY_V1

    def __post_init__(self) -> None:
        for name in (
            "policy_checkpoint_sha256",
            "policy_metadata_sha256",
            "joint_stage_result_sha256",
            "tokenizer_contract_sha256",
            "visual_contract_sha256",
            "preprocess_contract_sha256",
            "native_migration_receipt_sha256",
            "trainability_contract_sha256",
            "trainable_parameter_names_sha256",
            "reward_contract_sha256",
            "dataset_admission_report_sha256",
            "train_dataset_contract_sha256",
            "sft_validation_dataset_contract_sha256",
            "kl_selection_dataset_contract_sha256",
            "formal_monitor_dataset_contract_sha256",
            "text_replay_train_contract_sha256",
            "text_replay_sft_validation_contract_sha256",
            "text_replay_kl_selection_contract_sha256",
            "text_replay_formal_monitor_contract_sha256",
            "runtime_source_receipt_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if self.reference_checkpoint_sha256 is not None:
            _require_sha256(
                self.reference_checkpoint_sha256,
                "reference_checkpoint_sha256",
            )
        if self.reference_metadata_sha256 is not None:
            _require_sha256(
                self.reference_metadata_sha256,
                "reference_metadata_sha256",
            )
        if (self.reference_checkpoint_sha256 is None) != (
            self.reference_metadata_sha256 is None
        ):
            raise ValueError("reference model/metadata hashes must be paired")
        image_contract_sha256s = {
            self.train_dataset_contract_sha256,
            self.sft_validation_dataset_contract_sha256,
            self.kl_selection_dataset_contract_sha256,
            self.formal_monitor_dataset_contract_sha256,
        }
        if len(image_contract_sha256s) != 4:
            raise ValueError(
                "image train, SFT validation, KL selection, and formal monitor "
                "contracts must be distinct"
            )
        text_contract_sha256s = {
            self.text_replay_train_contract_sha256,
            self.text_replay_sft_validation_contract_sha256,
            self.text_replay_kl_selection_contract_sha256,
            self.text_replay_formal_monitor_contract_sha256,
        }
        if len(text_contract_sha256s) != 4:
            raise ValueError(
                "text train, SFT validation, KL selection, and formal monitor "
                "dataset contracts must be distinct"
            )
        if (
            isinstance(self.recommended_max_new_tokens, bool)
            or not isinstance(self.recommended_max_new_tokens, int)
            or self.recommended_max_new_tokens <= 0
        ):
            raise ValueError("recommended_max_new_tokens must be positive")
        if self.position_contract != BOUNDARY_V1:
            raise ValueError("anyres GRPO admission must bind boundary_v1")
        if (
            isinstance(self.minimum_reward_spread, bool)
            or not isinstance(self.minimum_reward_spread, (int, float))
            or not math.isfinite(float(self.minimum_reward_spread))
            or float(self.minimum_reward_spread) <= 0
        ):
            raise ValueError("minimum_reward_spread must be finite and positive")

    @property
    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "dol_ocr_anyres_grpo_admission_v1",
            "policy_checkpoint_sha256": self.policy_checkpoint_sha256,
            "policy_metadata_sha256": self.policy_metadata_sha256,
            "reference_checkpoint_sha256": self.reference_checkpoint_sha256,
            "reference_metadata_sha256": self.reference_metadata_sha256,
            "joint_stage_result_sha256": self.joint_stage_result_sha256,
            "tokenizer_contract_sha256": self.tokenizer_contract_sha256,
            "visual_contract_sha256": self.visual_contract_sha256,
            "preprocess_contract_sha256": self.preprocess_contract_sha256,
            "native_migration_receipt_sha256": (
                self.native_migration_receipt_sha256
            ),
            "trainability_contract_sha256": self.trainability_contract_sha256,
            "trainable_parameter_names_sha256": (
                self.trainable_parameter_names_sha256
            ),
            "reward_contract_sha256": self.reward_contract_sha256,
            "dataset_admission_report_sha256": (
                self.dataset_admission_report_sha256
            ),
            "train_dataset_contract_sha256": self.train_dataset_contract_sha256,
            "sft_validation_dataset_contract_sha256": (
                self.sft_validation_dataset_contract_sha256
            ),
            "kl_selection_dataset_contract_sha256": (
                self.kl_selection_dataset_contract_sha256
            ),
            "formal_monitor_dataset_contract_sha256": (
                self.formal_monitor_dataset_contract_sha256
            ),
            "text_replay_train_contract_sha256": (
                self.text_replay_train_contract_sha256
            ),
            "text_replay_sft_validation_contract_sha256": (
                self.text_replay_sft_validation_contract_sha256
            ),
            "text_replay_kl_selection_contract_sha256": (
                self.text_replay_kl_selection_contract_sha256
            ),
            "text_replay_formal_monitor_contract_sha256": (
                self.text_replay_formal_monitor_contract_sha256
            ),
            "runtime_source_receipt_sha256": (
                self.runtime_source_receipt_sha256
            ),
            "recommended_max_new_tokens": self.recommended_max_new_tokens,
            "minimum_reward_spread": float(self.minimum_reward_spread),
            "position_contract": self.position_contract,
        }

    @property
    def canonical_sha256(self) -> str:
        return _canonical_sha256(self.canonical_payload)


def repeat_packed_detail_prompt_major(
    memory: torch.Tensor,
    cu_seqlens: torch.Tensor,
    repeats: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat each ragged prompt segment contiguously before the next prompt."""

    if memory.ndim != 2:
        raise ValueError("detail memory must have shape [sumM, D]")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("detail cu_seqlens must have shape [P + 1]")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("detail cu_seqlens must use an integer dtype")
    if cu_seqlens.device.type != "cpu":
        raise ValueError("detail cu_seqlens must remain on CPU")
    if type(repeats) is not int or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    boundaries = [int(value) for value in cu_seqlens.tolist()]
    if boundaries[0] != 0 or boundaries[-1] != memory.shape[0]:
        raise ValueError("detail cu_seqlens do not bound detail memory")
    if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("every prompt must have non-empty detail memory")
    segments: list[torch.Tensor] = []
    repeated_boundaries = [0]
    for prompt_index in range(len(boundaries) - 1):
        segment = memory[boundaries[prompt_index]:boundaries[prompt_index + 1]]
        for _ in range(repeats):
            segments.append(segment)
            repeated_boundaries.append(
                repeated_boundaries[-1] + int(segment.shape[0])
            )
    return torch.cat(segments, dim=0), torch.tensor(
        repeated_boundaries,
        dtype=cu_seqlens.dtype,
        device="cpu",
    )


def anyres_grpo_compute_loss(
    policy,
    reference,
    collated_batch: Mapping[str, Any],
    reward_adapter: AnyresOCRRewardAdapter,
    admission: AnyresGRPOAdmission,
    cfg: GRPOConfig,
    device: torch.device | str,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Roll out and score one prompt-major anyres GRPO batch."""

    if not isinstance(cfg, GRPOConfig):
        raise TypeError("cfg must be a GRPOConfig")
    if cfg.group_size <= 1 or cfg.max_new_tokens <= 0:
        raise ValueError("anyres GRPO requires group_size>=2 and max_new_tokens>0")
    if cfg.clip_eps is not None:
        raise ValueError("single-update anyres OCR GRPO requires clip_eps=None")
    if cfg.advantage_mode != "centered":
        raise ValueError("anyres OCR GRPO requires centered advantages")
    if cfg.max_behavior_log_ratio is None:
        raise ValueError(
            "anyres OCR GRPO requires an explicit behavior-log-ratio gate"
        )
    if not isinstance(reward_adapter, AnyresOCRRewardAdapter):
        raise TypeError("reward_adapter must be AnyresOCRRewardAdapter")
    if not isinstance(admission, AnyresGRPOAdmission):
        raise TypeError("admission must be AnyresGRPOAdmission")
    if cfg.max_new_tokens != admission.recommended_max_new_tokens:
        raise ValueError("GRPO max_new_tokens differs from admitted recommendation")
    if float(cfg.min_reward_spread) != float(admission.minimum_reward_spread):
        raise ValueError("GRPO min_reward_spread differs from admission")
    if (
        cfg.tool_result_open_ids is not None
        or cfg.tool_result_close_ids is not None
    ):
        raise ValueError("anyres OCR GRPO forbids tool-result masking")
    if admission.reward_contract_sha256 != reward_adapter.contract[
        "canonical_sha256"
    ]:
        raise ValueError("reward adapter contract differs from GRPO admission")
    if reward_adapter.contract.get("construction") != (
        "reviewed_native_tokenizer_bundle_v1"
    ):
        raise ValueError("GRPO reward adapter is not from a reviewed bundle")
    if admission.tokenizer_contract_sha256 != reward_adapter.contract[
        "tokenizer_contract_sha256"
    ]:
        raise ValueError("reward tokenizer contract differs from GRPO admission")
    if cfg.kl_coef > 0 and reference is None:
        raise ValueError("positive kl_coef requires a frozen reference model")
    if cfg.kl_coef > 0 and admission.reference_checkpoint_sha256 is None:
        raise ValueError("positive KL requires an admitted reference checkpoint")
    device = torch.device(device)
    policy = _require_unwrapped_single_model(policy, role="policy")
    actual_trainable_names_sha256 = _canonical_sha256(
        sorted(
            name
            for name, parameter in policy.named_parameters()
            if parameter.requires_grad
        )
    )
    if actual_trainable_names_sha256 != (
        admission.trainable_parameter_names_sha256
    ):
        raise ValueError("policy trainable parameter names differ from admission")
    if cfg.kl_coef > 0:
        reference = _require_unwrapped_single_model(
            reference,
            role="reference",
        )
        if reference is policy:
            raise ValueError("policy and frozen reference must be distinct models")
    if collated_batch.get("dataset_contract_sha256") != (
        admission.train_dataset_contract_sha256
    ):
        raise ValueError("GRPO batch differs from admitted train dataset contract")
    batch = _describe_batch(policy, collated_batch, device=device)
    if batch["preprocess_contracts"] != {admission.preprocess_contract_sha256}:
        raise ValueError("GRPO batch preprocess contract differs from admission")
    prompt_count = int(batch["prompts"].shape[0])
    group_size = cfg.group_size
    prompt_length = int(batch["prompts"].shape[1])
    policy_max_seq_len = int(policy.cfg.max_seq_len)
    if prompt_length + cfg.max_new_tokens > policy_max_seq_len:
        raise ValueError(
            "anyres GRPO prompt plus rollout exceeds policy max_seq_len: "
            f"{prompt_length} + {cfg.max_new_tokens} > {policy_max_seq_len}"
        )
    eos_id = int(policy.cfg.eos_id)
    pad_id = int(policy.cfg.pad_id)
    if eos_id != reward_adapter.eos_id:
        raise ValueError("model EOS differs from reward adapter")

    policy_was_training = bool(policy.training)
    use_reference = bool(cfg.kl_coef > 0)
    reference_was_training = (
        bool(reference.training) if use_reference else False
    )
    if use_reference:
        if any(parameter.requires_grad for parameter in reference.parameters()):
            raise ValueError("anyres GRPO reference parameters must be frozen")
        _require_matching_model_contract(policy, reference, cfg)
    policy.eval()
    if use_reference:
        reference.eval()
    try:
        with torch.no_grad():
            rollout_unique = _encode_unique(policy, collated_batch, device)
            rollout_global, rollout_detail, rollout_cu = _repeat_visuals(
                rollout_unique,
                group_size,
            )
            grouped_prompts = batch["prompts"].repeat_interleave(
                group_size,
                dim=0,
            )
            rollout_adapter = _PreencodedAnyresRollout(
                policy,
                rollout_global,
                rollout_detail,
                rollout_cu,
            )
            sequences, behavior_logp = _generate_sequences_with_behavior(
                rollout_adapter,
                grouped_prompts,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                eos_id=eos_id,
                pad_id=pad_id,
                recurrent_steps=cfg.recurrent_steps,
                position_contract=BOUNDARY_V1,
            )

        objective_mask = _completion_action_mask(
            sequences,
            prompt_length,
            eos_id,
        )
        score_attention = torch.ones_like(sequences, dtype=torch.long)
        score_attention[:, prompt_length:] = objective_mask[
            :, prompt_length:
        ].long()
        token_tails = sequences[:, prompt_length:]
        eos_mask = token_tails == eos_id
        repeated_references = [
            reference_text
            for reference_text in batch["references"]
            for _ in range(group_size)
        ]
        with torch.no_grad():
            reward_output = reward_adapter.score_group(
                repeated_references,
                token_tails.detach(),
                eos_mask.detach(),
            )
        responses = reward_output["responses"]
        rewards = torch.as_tensor(
            reward_output["group_rewards"],
            dtype=torch.float32,
            device=sequences.device,
        ).detach().reshape(-1)
        if rewards.numel() != prompt_count * group_size:
            raise ValueError(
                "reward adapter must return one reward per sampled response"
            )
        if not bool(torch.isfinite(rewards).all()):
            raise ValueError("reward adapter returned a non-finite reward")
        advantages = group_relative_advantages(
            rewards,
            group_size,
            mode="centered",
            min_reward_spread=cfg.min_reward_spread,
        ).detach()

        policy_unique = _encode_unique(policy, collated_batch, device)
        policy_global, policy_detail, policy_cu = _repeat_visuals(
            policy_unique,
            group_size,
        )
        policy_logp, shifted_mask = token_logprobs_with_mask(
            policy,
            sequences,
            objective_mask,
            attention_mask=score_attention,
            recurrent_steps=cfg.recurrent_steps,
            completion_start=prompt_length - 1,
            position_contract=BOUNDARY_V1,
            visual_features=policy_global,
            detail_memory=policy_detail,
            detail_cu_seqlens=policy_cu,
        )

        old_logp = behavior_logp[:, 1:]
        ref_logp = None
        if use_reference:
            with torch.no_grad():
                reference_unique = _encode_unique(
                    reference,
                    collated_batch,
                    device,
                )
                reference_global, reference_detail, reference_cu = _repeat_visuals(
                    reference_unique,
                    group_size,
                )
                ref_logp, _ = token_logprobs_with_mask(
                    reference,
                    sequences,
                    objective_mask,
                    attention_mask=score_attention,
                    recurrent_steps=cfg.recurrent_steps,
                    completion_start=prompt_length - 1,
                    position_contract=BOUNDARY_V1,
                    visual_features=reference_global,
                    detail_memory=reference_detail,
                    detail_cu_seqlens=reference_cu,
                )

        loss, loss_metrics = grpo_loss(
            policy_logp,
            old_logp,
            advantages,
            shifted_mask,
            ref_token_logp=ref_logp,
            cfg=cfg,
            enforce_behavior_gate=True,
        )
        reward_groups = rewards.view(prompt_count, group_size)
        spread = reward_groups.max(dim=1).values - reward_groups.min(dim=1).values
        extra_values = torch.stack(
            (
                rewards.mean(),
                spread.mean(),
                (spread > cfg.min_reward_spread).float().mean(),
                eos_mask.any(dim=1).float().mean(),
            )
        ).detach().cpu().tolist()
        invalid_rate = sum(
            bool(sample["diagnostics"]["invalid"])
            for sample in reward_output["samples"]
        ) / len(reward_output["samples"])
        exact_rate = sum(
            bool(sample["diagnostics"]["exact"])
            for sample in reward_output["samples"]
        ) / len(reward_output["samples"])
        metrics = dict(loss_metrics)
        metrics.update(
            {
                "prompts": float(prompt_count),
                "group_size": float(group_size),
                "reward_mean": extra_values[0],
                "reward_spread_mean": extra_values[1],
                "active_group_frac": extra_values[2],
                "eos_rate": extra_values[3],
                "reward_invalid_rate": invalid_rate,
                "reward_exact_rate": exact_rate,
                "unique_response_frac": _unique_response_fraction(
                    responses,
                    group_size,
                ),
                "visual_unique_prompts_per_stage": float(prompt_count),
            }
        )
        return loss, metrics
    finally:
        policy.train(policy_was_training)
        if use_reference:
            reference.train(reference_was_training)


def _describe_batch(policy, batch: Mapping[str, Any], *, device: torch.device) -> dict[str, Any]:
    if not isinstance(batch, Mapping):
        raise TypeError("collated_batch must be a mapping")
    if batch.get("position_contract") != BOUNDARY_V1:
        raise ValueError("anyres GRPO batch must use boundary_v1")
    if any(key in batch for key in ("word_pos", "morph_depth", "token_offsets")):
        raise ValueError("boundary_v1 anyres GRPO must not materialize positions")
    input_ids = batch.get("input_ids")
    attention_mask = batch.get("attention_mask")
    labels = batch.get("labels")
    if not all(isinstance(value, torch.Tensor) for value in (input_ids, attention_mask, labels)):
        raise TypeError("collated text fields must be tensors")
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape or labels.shape != input_ids.shape:
        raise ValueError("collated text tensors must have aligned [P, T] shapes")
    ignore_index = int(getattr(policy.cfg, "ignore_index", -100))
    first_positions: list[int] = []
    for row in labels:
        positions = torch.nonzero(row != ignore_index, as_tuple=False).flatten()
        if positions.numel() == 0:
            raise ValueError("every anyres GRPO row must have a supervised reference")
        first_positions.append(int(positions[0].item()))
    if len(set(first_positions)) != 1:
        raise ValueError("anyres GRPO prompt lengths must be uniform")
    prompt_length = first_positions[0]
    if prompt_length <= 0:
        raise ValueError("anyres GRPO prompt must contain at least one token")
    prompts = input_ids[:, :prompt_length].to(device, non_blocking=True)
    if not bool((attention_mask[:, :prompt_length] == 1).all()):
        raise ValueError("anyres GRPO prompts must not contain padding")
    metadata = batch.get("sample_metadata")
    if not isinstance(metadata, Sequence) or isinstance(metadata, (str, bytes)):
        raise TypeError("sample_metadata must be a sequence")
    if len(metadata) != input_ids.shape[0]:
        raise ValueError("sample_metadata length differs from prompt count")
    references: list[str] = []
    preprocess_contracts: set[str] = set()
    for index, sample in enumerate(metadata):
        if not isinstance(sample, Mapping):
            raise TypeError(f"sample_metadata[{index}] must be a mapping")
        reference = sample.get("reference_model")
        if not isinstance(reference, Mapping) or not isinstance(reference.get("text"), str):
            raise ValueError(f"sample_metadata[{index}] has no reference_model.text")
        references.append(str(reference["text"]))
        preprocess = sample.get("preprocess_contract_sha256")
        if not isinstance(preprocess, str):
            raise ValueError(
                f"sample_metadata[{index}] has no preprocess contract"
            )
        preprocess_contracts.add(preprocess)
    return {
        "prompts": prompts,
        "references": references,
        "preprocess_contracts": preprocess_contracts,
    }


def _require_unwrapped_single_model(model, *, role: str):
    if model is None:
        raise ValueError(f"{role} model is required")
    core = _unwrap_model(model)
    if core is not model:
        raise ValueError(
            "anyres GRPO currently supports only an unwrapped single-process "
            f"{role} model"
        )
    return core


def _require_matching_model_contract(policy, reference, cfg: GRPOConfig) -> None:
    for field in (
        "vocab_size",
        "eos_id",
        "pad_id",
        "ignore_index",
        "max_seq_len",
        "d_model",
    ):
        policy_value = getattr(policy.cfg, field, None)
        reference_value = getattr(reference.cfg, field, None)
        if policy_value != reference_value:
            raise ValueError(
                f"policy/reference {field} differs: "
                f"{policy_value!r} != {reference_value!r}"
            )
    if cfg.recurrent_steps is None and (
        getattr(policy.cfg, "recurrent_steps", None)
        != getattr(reference.cfg, "recurrent_steps", None)
    ):
        raise ValueError(
            "policy/reference default recurrent_steps differ without an "
            "explicit GRPO override"
        )


def _encode_unique(model, batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    visual = encode_anyres_visual_batch(model, batch, device=device)
    global_features = model.vision.encode_visual(visual["global_pixel_values"])
    detail_memory = visual["detail_memory"]
    detail_cu = visual["detail_cu_seqlens"]
    if not all(
        isinstance(value, torch.Tensor)
        for value in (global_features, detail_memory, detail_cu)
    ):
        raise TypeError("anyres visual encoders must return tensors")
    prompt_count = int(batch["input_ids"].shape[0])
    if global_features.shape[0] != prompt_count or detail_cu.numel() != prompt_count + 1:
        raise ValueError("unique anyres visual outputs differ from prompt count")
    return {
        "global_features": global_features,
        "detail_memory": detail_memory,
        "detail_cu_seqlens": detail_cu,
    }


def _repeat_visuals(
    visual: Mapping[str, torch.Tensor],
    repeats: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global_features = visual["global_features"].repeat_interleave(repeats, dim=0)
    detail_memory, detail_cu = repeat_packed_detail_prompt_major(
        visual["detail_memory"],
        visual["detail_cu_seqlens"],
        repeats,
    )
    return global_features, detail_memory, detail_cu


class _PreencodedAnyresRollout:
    def __init__(
        self,
        model,
        global_features: torch.Tensor,
        detail_memory: torch.Tensor,
        detail_cu_seqlens: torch.Tensor,
    ) -> None:
        self.model = model
        self.global_features = global_features
        self.detail_memory = detail_memory
        self.detail_cu_seqlens = detail_cu_seqlens

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float | None,
        eos_id: int,
        pad_id: int,
        recurrent_steps: int | None,
        position_contract: str,
        on_sample,
        **kwargs,
    ) -> torch.Tensor:
        if kwargs:
            raise ValueError(f"unsupported anyres rollout arguments: {sorted(kwargs)}")
        if temperature != 1.0 or top_p is not None:
            raise ValueError("anyres behavior capture requires raw policy sampling")
        if position_contract != BOUNDARY_V1:
            raise ValueError("anyres rollout requires boundary_v1")
        sequence = input_ids
        attention = torch.ones_like(sequence, dtype=torch.long)
        finished = torch.zeros(
            sequence.shape[0],
            dtype=torch.bool,
            device=sequence.device,
        )
        for step in range(max_new_tokens):
            output = self.model(
                input_ids=sequence,
                attention_mask=attention,
                steps=recurrent_steps,
                return_logits=True,
                visual_features=self.global_features,
                detail_memory=self.detail_memory,
                detail_cu_seqlens=self.detail_cu_seqlens,
                position_contract=BOUNDARY_V1,
            )
            logits = output["logits"][:, -1, :].float()
            logp = F.log_softmax(logits, dim=-1)
            sampled = torch.multinomial(logp.exp(), num_samples=1).squeeze(-1)
            sampled_logp = logp.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
            sampled = torch.where(
                finished,
                torch.full_like(sampled, pad_id),
                sampled,
            )
            sampled_logp = torch.where(
                finished,
                torch.zeros_like(sampled_logp),
                sampled_logp,
            )
            on_sample(step, sampled, sampled_logp)
            sequence = torch.cat((sequence, sampled.unsqueeze(1)), dim=1)
            attention = torch.cat(
                (attention, (~finished).long().unsqueeze(1)),
                dim=1,
            )
            finished |= sampled == eos_id
            if bool(finished.all()):
                break
        return sequence


def _unique_response_fraction(responses: Sequence[str], group_size: int) -> float:
    if len(responses) % group_size != 0:
        raise ValueError("responses are not group-contiguous")
    groups = [
        responses[start:start + group_size]
        for start in range(0, len(responses), group_size)
    ]
    return sum(len(set(group)) / group_size for group in groups) / len(groups)


__all__ = [
    "AnyresGRPOAdmission",
    "anyres_grpo_compute_loss",
    "repeat_packed_detail_prompt_major",
]
