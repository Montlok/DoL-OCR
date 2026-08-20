# -*- coding: utf-8 -*-

"""Explicit, auditable OMVT-v1 to native-detail parameter migration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

import torch
import torch.nn as nn

from Model.omvt.patcher import PATCH_KINDS


MIGRATION_CONTRACT = "omvt_v1_to_native_detail_v1"


@dataclass(frozen=True)
class NativeMigrationMapping:
    target: str
    action: str
    shape: tuple[int, ...]
    dtype: str
    source: str | None = None

    def canonical_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "target": self.target,
            "action": self.action,
            "shape": list(self.shape),
            "dtype": self.dtype,
        }
        if self.source is not None:
            payload["source"] = self.source
        return payload


@dataclass(frozen=True)
class NativeMigrationReceipt:
    initialized_from_legacy: bool
    source_class: str | None
    target_class: str
    source_schema_sha256: str | None
    target_schema_sha256: str
    max_detail_tokens_per_sample: int
    source_tokens_per_detail_token: int
    mappings: tuple[NativeMigrationMapping, ...]
    contract: str = MIGRATION_CONTRACT

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "initialized_from_legacy": self.initialized_from_legacy,
            "source_class": self.source_class,
            "target_class": self.target_class,
            "source_schema_sha256": self.source_schema_sha256,
            "target_schema_sha256": self.target_schema_sha256,
            "max_detail_tokens_per_sample": self.max_detail_tokens_per_sample,
            "source_tokens_per_detail_token": self.source_tokens_per_detail_token,
            "mappings": [mapping.canonical_payload() for mapping in self.mappings],
        }

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def canonical_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


def _qualified_class_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _state_schema_sha256(state: Mapping[str, torch.Tensor]) -> str:
    payload = [
        {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
        for name, tensor in sorted(state.items())
    ]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class _MigrationBuilder:
    def __init__(
        self,
        *,
        source: nn.Module | None,
        target: nn.Module,
    ) -> None:
        self.source_module = source
        self.target_module = target
        self.source = (
            dict(source.state_dict(keep_vars=True)) if source is not None else {}
        )
        self.target = dict(target.state_dict(keep_vars=True))
        self.used_source: set[str] = set()
        self.used_target: set[str] = set()
        self.mappings: list[NativeMigrationMapping] = []

    def _target_tensor(self, name: str) -> torch.Tensor:
        if name not in self.target:
            raise ValueError(f"native migration target tensor is missing: {name}")
        if name in self.used_target:
            raise ValueError(f"native migration target was initialized twice: {name}")
        return self.target[name]

    def copy(self, source_name: str, target_name: str) -> None:
        if source_name not in self.source:
            raise ValueError(f"legacy migration source tensor is missing: {source_name}")
        if source_name in self.used_source:
            raise ValueError(f"legacy migration source was consumed twice: {source_name}")
        source = self.source[source_name]
        target = self._target_tensor(target_name)
        if tuple(source.shape) != tuple(target.shape):
            raise ValueError(
                "native migration shape mismatch: "
                f"{source_name}{tuple(source.shape)} -> "
                f"{target_name}{tuple(target.shape)}"
            )
        if source.dtype != target.dtype:
            raise ValueError(
                "native migration dtype mismatch: "
                f"{source_name}={source.dtype} -> {target_name}={target.dtype}"
            )
        if source.device != target.device:
            raise ValueError(
                "native migration device mismatch: "
                f"{source_name}={source.device} -> {target_name}={target.device}"
            )
        with torch.no_grad():
            target.copy_(source)
        if not torch.equal(target, source):
            raise RuntimeError(
                f"native migration copy verification failed: {source_name} -> {target_name}"
            )
        self.used_source.add(source_name)
        self.used_target.add(target_name)
        self.mappings.append(
            NativeMigrationMapping(
                source=source_name,
                target=target_name,
                action="copy_exact",
                shape=tuple(target.shape),
                dtype=str(target.dtype),
            )
        )

    def fill(self, target_name: str, *, value: float, action: str) -> None:
        target = self._target_tensor(target_name)
        with torch.no_grad():
            target.fill_(value)
        expected = torch.full_like(target, value)
        if not torch.equal(target, expected):
            raise RuntimeError(f"native migration initialization failed: {target_name}")
        self.used_target.add(target_name)
        self.mappings.append(
            NativeMigrationMapping(
                target=target_name,
                action=action,
                shape=tuple(target.shape),
                dtype=str(target.dtype),
            )
        )

    def copy_prefix_repeat(self, source_name: str, target_name: str) -> None:
        if source_name not in self.source:
            raise ValueError(f"legacy migration source tensor is missing: {source_name}")
        if source_name in self.used_source:
            raise ValueError(f"legacy migration source was consumed twice: {source_name}")
        source = self.source[source_name]
        target = self._target_tensor(target_name)
        if (
            source.ndim == 0
            or target.ndim != source.ndim
            or tuple(target.shape[1:]) != tuple(source.shape[1:])
            or target.shape[0] < source.shape[0]
        ):
            raise ValueError(
                f"native migration prefix shape mismatch: {source_name}"
                f"{tuple(source.shape)} -> {target_name}{tuple(target.shape)}"
            )
        if source.dtype != target.dtype or source.device != target.device:
            raise ValueError("native migration prefix dtype/device mismatch")
        repeats = (target.shape[0] + source.shape[0] - 1) // source.shape[0]
        expected = source.repeat((repeats,) + (1,) * (source.ndim - 1))[
            : target.shape[0]
        ]
        with torch.no_grad():
            target.copy_(expected)
        if not torch.equal(target, expected):
            raise RuntimeError(
                f"native migration prefix verification failed: {source_name} -> {target_name}"
            )
        self.used_source.add(source_name)
        self.used_target.add(target_name)
        self.mappings.append(
            NativeMigrationMapping(
                source=source_name,
                target=target_name,
                action="copy_prefix_repeat_extension",
                shape=tuple(target.shape),
                dtype=str(target.dtype),
            )
        )

    def mark_fresh(self, target_name: str) -> None:
        target = self._target_tensor(target_name)
        self.used_target.add(target_name)
        self.mappings.append(
            NativeMigrationMapping(
                target=target_name,
                action="fresh_initialization",
                shape=tuple(target.shape),
                dtype=str(target.dtype),
            )
        )

    def finish(self, *, require_all_source: bool) -> tuple[NativeMigrationMapping, ...]:
        missing_targets = sorted(set(self.target) - self.used_target)
        if missing_targets:
            raise ValueError(
                "native migration left target tensors unclassified: "
                + ", ".join(missing_targets)
            )
        if require_all_source:
            missing_sources = sorted(set(self.source) - self.used_source)
            if missing_sources:
                raise ValueError(
                    "native migration left legacy tensors unmapped: "
                    + ", ".join(missing_sources)
                )
        return tuple(self.mappings)


_DIRECTIONAL_LAYER_STATE = (
    "ssm.A_log",
    "ssm.norm.weight",
    "ssm.norm.bias",
    "ssm.in_proj.weight",
    "ssm.in_proj.bias",
    "ssm.out_proj.weight",
    "ssm.out_proj.bias",
    "ffn.0.weight",
    "ffn.0.bias",
    "ffn.1.weight",
    "ffn.1.bias",
    "ffn.4.weight",
    "ffn.4.bias",
)

_SQUARE_LAYER_STATE = (
    "norm.weight",
    "norm.bias",
    "qkv.weight",
    "qkv.bias",
    "out.weight",
    "out.bias",
    "ffn.0.weight",
    "ffn.0.bias",
    "ffn.1.weight",
    "ffn.1.bias",
    "ffn.4.weight",
    "ffn.4.bias",
)

_LAYOUT_DIRECT_STATE = (
    "bbox_proj.weight",
    "bbox_proj.bias",
    "norm.weight",
    "norm.bias",
    "mixer.0.weight",
    "mixer.0.bias",
)

_COMPRESSOR_CROSS_RENAMES = (
    ("q_proj.weight", "query_proj.weight"),
    ("q_proj.bias", "query_proj.bias"),
    ("kv_proj.weight", "key_value_proj.weight"),
    ("kv_proj.bias", "key_value_proj.bias"),
    ("out.weight", "out.weight"),
    ("out.bias", "out.bias"),
    ("norm_q.weight", "norm_query.weight"),
    ("norm_q.bias", "norm_query.bias"),
    ("norm_kv.weight", "norm_context.weight"),
    ("norm_kv.bias", "norm_context.bias"),
)

_COMPRESSOR_FFN_STATE = (
    "ffn.0.weight",
    "ffn.0.bias",
    "ffn.1.weight",
    "ffn.1.bias",
    "ffn.3.weight",
    "ffn.3.bias",
)

_CONFIG_COMPATIBILITY_FIELDS = (
    "in_channels",
    "vertical_patch",
    "horizontal_patch",
    "square_patch",
    "layout_patch",
    "d_vision",
    "n_vertical_layers",
    "n_horizontal_layers",
    "n_local_attn_layers",
    "n_layout_layers",
    "vision_n_heads",
    "vision_ffn_hidden",
    "vision_dropout",
    "router_min_route_prob",
    "router_temperature",
    "compress_to",
    "compressor_layers",
    "compressor_heads",
)


def _assert_config_compatibility(legacy_tower: nn.Module, native_tower: nn.Module) -> None:
    if not hasattr(legacy_tower, "cfg") or not hasattr(native_tower, "cfg"):
        raise TypeError("legacy and native towers must expose their OMVT config")
    for field in _CONFIG_COMPATIBILITY_FIELDS:
        source_value = getattr(legacy_tower.cfg, field)
        target_value = getattr(native_tower.cfg, field)
        if source_value != target_value:
            raise ValueError(
                f"OMVT config field {field!r} is incompatible: "
                f"legacy={source_value!r}, native={target_value!r}"
            )
    source_latents = legacy_tower.compressor.latents
    target_latents = native_tower.compressor.latents
    if (
        source_latents.ndim != target_latents.ndim
        or tuple(source_latents.shape[1:]) != tuple(target_latents.shape[1:])
        or target_latents.shape[0] < source_latents.shape[0]
    ):
        raise ValueError(
            "native max_detail_tokens must be at least legacy compress_to "
            "with the same latent width"
        )


def _initialize_new_native_parameters(builder: _MigrationBuilder, native_tower: nn.Module) -> None:
    for kind in PATCH_KINDS:
        prefix = f"encoders.{kind}"
        builder.fill(
            f"{prefix}.bbox_embed.weight",
            value=0.0,
            action="zero_new_geometry",
        )
        builder.fill(
            f"{prefix}.bbox_embed.bias",
            value=0.0,
            action="zero_new_geometry",
        )
        builder.fill(
            f"{prefix}.validity_embed.weight",
            value=0.0,
            action="zero_new_validity",
        )
        builder.fill(
            f"{prefix}.validity_embed.bias",
            value=0.0,
            action="zero_new_validity",
        )
        builder.fill(
            f"{prefix}.input_norm.weight",
            value=1.0,
            action="identity_new_layer_norm_affine",
        )
        builder.fill(
            f"{prefix}.input_norm.bias",
            value=0.0,
            action="identity_new_layer_norm_affine",
        )
    for layer_index in range(len(native_tower.encoders["layout"].layers)):
        prefix = f"encoders.layout.layers.{layer_index}.context_proj"
        builder.fill(
            f"{prefix}.weight",
            value=0.0,
            action="zero_new_sample_context",
        )
        builder.fill(
            f"{prefix}.bias",
            value=0.0,
            action="zero_new_sample_context",
        )


def _copy_legacy_state(builder: _MigrationBuilder, legacy_tower: nn.Module, native_tower: nn.Module) -> None:
    builder.copy("router.bias", "router.bias")
    builder.copy("router.log_temperature", "router.log_temperature")

    for kind in PATCH_KINDS:
        builder.copy(
            f"encoders.{kind}.embed.weight",
            f"encoders.{kind}.patch_embed.weight",
        )
        builder.copy(
            f"encoders.{kind}.embed.bias",
            f"encoders.{kind}.patch_embed.bias",
        )
        source_layers = legacy_tower.encoders[kind].layers
        target_layers = native_tower.encoders[kind].layers
        if len(source_layers) != len(target_layers):
            raise ValueError(
                f"{kind} mixer layer count mismatch: "
                f"legacy={len(source_layers)}, native={len(target_layers)}"
            )
        for layer_index in range(len(source_layers)):
            source_prefix = f"encoders.{kind}.layers.{layer_index}"
            target_prefix = f"encoders.{kind}.layers.{layer_index}"
            if kind in {"vertical", "horizontal"}:
                for suffix in _DIRECTIONAL_LAYER_STATE:
                    builder.copy(
                        f"{source_prefix}.{suffix}",
                        f"{target_prefix}.{suffix}",
                    )
            elif kind == "square":
                for suffix in _SQUARE_LAYER_STATE:
                    builder.copy(
                        f"{source_prefix}.{suffix}",
                        f"{target_prefix}.{suffix}",
                    )
            else:
                for suffix in _LAYOUT_DIRECT_STATE:
                    builder.copy(
                        f"{source_prefix}.{suffix}",
                        f"{target_prefix}.{suffix}",
                    )
                builder.copy(
                    f"{source_prefix}.mixer.2.weight",
                    f"{target_prefix}.mixer.3.weight",
                )
                builder.copy(
                    f"{source_prefix}.mixer.2.bias",
                    f"{target_prefix}.mixer.3.bias",
                )

    builder.copy("fuse_norm.weight", "fuse_norm.weight")
    builder.copy("fuse_norm.bias", "fuse_norm.bias")
    if tuple(legacy_tower.compressor.latents.shape) == tuple(
        native_tower.compressor.latents.shape
    ):
        builder.copy("compressor.latents", "compressor.latents")
    else:
        builder.copy_prefix_repeat("compressor.latents", "compressor.latents")
    source_blocks = legacy_tower.compressor.blocks
    target_blocks = native_tower.compressor.blocks
    if len(source_blocks) != len(target_blocks):
        raise ValueError(
            "compressor layer count mismatch: "
            f"legacy={len(source_blocks)}, native={len(target_blocks)}"
        )
    for block_index in range(len(source_blocks)):
        source_prefix = f"compressor.blocks.{block_index}"
        target_prefix = f"compressor.blocks.{block_index}"
        for source_suffix, target_suffix in _COMPRESSOR_CROSS_RENAMES:
            builder.copy(
                f"{source_prefix}.cross.{source_suffix}",
                f"{target_prefix}.cross.{target_suffix}",
            )
        for suffix in _COMPRESSOR_FFN_STATE:
            builder.copy(
                f"{source_prefix}.{suffix}",
                f"{target_prefix}.{suffix}",
            )
    builder.copy("compressor.final_norm.weight", "compressor.final_norm.weight")
    builder.copy("compressor.final_norm.bias", "compressor.final_norm.bias")


def migrate_legacy_omvt_to_native_detail(
    legacy_tower: nn.Module,
    native_tower: nn.Module,
) -> NativeMigrationReceipt:
    """Copy every compatible tensor and classify every new native tensor."""

    _assert_config_compatibility(legacy_tower, native_tower)
    builder = _MigrationBuilder(source=legacy_tower, target=native_tower)
    _copy_legacy_state(builder, legacy_tower, native_tower)
    _initialize_new_native_parameters(builder, native_tower)
    mappings = builder.finish(require_all_source=True)
    return NativeMigrationReceipt(
        initialized_from_legacy=True,
        source_class=_qualified_class_name(legacy_tower),
        target_class=_qualified_class_name(native_tower),
        source_schema_sha256=_state_schema_sha256(builder.source),
        target_schema_sha256=_state_schema_sha256(builder.target),
        max_detail_tokens_per_sample=native_tower.compressor.max_detail_tokens_per_sample,
        source_tokens_per_detail_token=native_tower.compressor.source_tokens_per_detail_token,
        mappings=mappings,
    )


def initialize_fresh_native_detail(
    native_tower: nn.Module,
) -> NativeMigrationReceipt:
    """Classify fresh weights while still neutralizing all new geometry paths."""

    builder = _MigrationBuilder(source=None, target=native_tower)
    _initialize_new_native_parameters(builder, native_tower)
    for target_name in sorted(set(builder.target) - builder.used_target):
        builder.mark_fresh(target_name)
    mappings = builder.finish(require_all_source=False)
    return NativeMigrationReceipt(
        initialized_from_legacy=False,
        source_class=None,
        target_class=_qualified_class_name(native_tower),
        source_schema_sha256=None,
        target_schema_sha256=_state_schema_sha256(builder.target),
        max_detail_tokens_per_sample=native_tower.compressor.max_detail_tokens_per_sample,
        source_tokens_per_detail_token=native_tower.compressor.source_tokens_per_detail_token,
        mappings=mappings,
    )


__all__ = [
    "MIGRATION_CONTRACT",
    "NativeMigrationMapping",
    "NativeMigrationReceipt",
    "initialize_fresh_native_detail",
    "migrate_legacy_omvt_to_native_detail",
]
