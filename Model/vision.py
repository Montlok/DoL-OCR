# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn


class MLPVisionEncoder(nn.Module):
    """Lightweight MLP+LN patch encoder.

    Kept as a fallback for smoke tests and small-data experiments; production
    VLM training should use ``Model.omvt.OMVTVisionTower`` instead.
    """

    def __init__(self, cfg, patch_pixels: int = 14 * 14 * 3):
        super().__init__()

        if patch_pixels <= 0:
            raise ValueError("patch_pixels must be positive")

        self.cfg = cfg
        self.patch_pixels = patch_pixels
        self.d_model = cfg.d_model

        self.patch_embed = nn.Linear(patch_pixels, cfg.d_model)

        self.encoder = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.GELU(),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )

        self.to_llm = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim not in {2, 3}:
            raise ValueError("pixel_values must have shape [N, P] or [B, N, P]")

        if pixel_values.shape[-1] != self.patch_pixels:
            raise ValueError(
                f"expected patch_pixels={self.patch_pixels}, got {pixel_values.shape[-1]}"
            )

        x = self.patch_embed(pixel_values)
        x = x + self.encoder(x)
        return self.to_llm(x)


# Back-compat alias; legacy code expecting VisionEncoder still works.
VisionEncoder = MLPVisionEncoder


def inject_visual_features(
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    visual_features: torch.Tensor,
    image_patch_id: int,
) -> torch.Tensor:
    if inputs_embeds.ndim != 3:
        raise ValueError("inputs_embeds must have shape [B, L, D]")

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B, L]")

    if input_ids.shape != inputs_embeds.shape[:2]:
        raise ValueError("input_ids shape must match inputs_embeds[:2]")

    if visual_features.ndim not in {2, 3}:
        raise ValueError("visual_features must have shape [N, D] or [B, N, D]")

    bsz, _seq_len, dim = inputs_embeds.shape
    mask = input_ids == image_patch_id
    n_patches = int(mask.sum().item())

    if n_patches == 0:
        return inputs_embeds

    out = inputs_embeds.clone()

    if visual_features.ndim == 2:
        if bsz != 1:
            raise ValueError(
                "unbatched visual_features are only supported when batch size is 1"
            )

        if visual_features.shape != (n_patches, dim):
            raise ValueError(
                f"expected visual_features {(n_patches, dim)}, got {tuple(visual_features.shape)}"
            )

        out[mask] = visual_features.to(device=out.device, dtype=out.dtype)
        return out

    if visual_features.shape[0] != bsz or visual_features.shape[-1] != dim:
        raise ValueError(
            f"expected visual_features [B, N, {dim}], got {tuple(visual_features.shape)}"
        )

    counts = mask.sum(dim=1).tolist()

    for batch_idx, count in enumerate(counts):
        if visual_features.shape[1] != count:
            raise ValueError(
                f"batch {batch_idx}: visual feature count must equal image_patch "
                f"count (got {visual_features.shape[1]} vs {count})"
            )

        if count > 0:
            out[batch_idx, mask[batch_idx]] = visual_features[
                batch_idx,
                :count,
            ].to(device=out.device, dtype=out.dtype)

    return out


class VisionInjector(nn.Module):
    """Dispatcher between MLP fallback and OMVT vision tower.

    When ``pixel_values`` is a ``Tensor``, the lightweight :class:`MLPVisionEncoder`
    is used (legacy / smoke path). When it is a ``Mapping`` (the multi-scale
    batch produced by :class:`Model.omvt.MultiScalePatcher`), the OMVT tower
    + Perceiver compressor are dispatched instead.

    The OMVT tower is constructed lazily on the first dict input so smoke runs
    that never see vision input pay no parameter cost.
    """

    def __init__(
        self,
        cfg,
        patch_pixels: int = 14 * 14 * 3,
        omvt_cfg=None,
    ):
        super().__init__()

        self.cfg = cfg
        self.patch_pixels = patch_pixels
        self.encoder = MLPVisionEncoder(cfg, patch_pixels)

        self._omvt_cfg = omvt_cfg
        self.omvt = None  # type: ignore[assignment]
        self.native_detail_tower = None  # type: ignore[assignment]
        self.native_detail_migration_receipt = None

    def _ensure_omvt(self) -> None:
        if self.omvt is not None:
            return
        if self._omvt_cfg is None:
            from Model.config import OMVTConfig

            self._omvt_cfg = OMVTConfig()

        from Model.omvt import OMVTInjector

        injector = OMVTInjector(self.cfg, self._omvt_cfg)

        # Lazy submodules must be migrated to the parent's current device
        # and dtype: assigning after `RDTForCausalLM.to('cuda')` does not
        # auto-migrate, so the first GPU batch would crash with a device
        # mismatch. Infer the target from an existing parameter; fall back
        # to MLP encoder's first parameter, then CPU/float32 if the module
        # is genuinely parameterless (shouldn't happen in practice).
        target_param = next(self.parameters(), None)
        if target_param is not None:
            injector = injector.to(device=target_param.device, dtype=target_param.dtype)

        self.omvt = injector

    def install_native_detail_tower(
        self,
        omvt_cfg,
        max_detail_tokens: int,
        ratio: int,
        initialize_from_legacy: bool = True,
    ):
        """Explicitly install the opt-in native-detail tower.

        The default v1 module tree remains unchanged until this method is
        called.  Weight migration is exact and receipt-backed; no partial or
        non-strict state-dict loading is used.
        """

        if self.native_detail_tower is not None:
            raise RuntimeError("native detail tower is already installed")
        if type(max_detail_tokens) is not int or max_detail_tokens <= 0:
            raise ValueError("max_detail_tokens must be a positive integer")
        if type(ratio) is not int or ratio <= 0:
            raise ValueError("ratio must be a positive integer")
        if type(initialize_from_legacy) is not bool:
            raise ValueError("initialize_from_legacy must be bool")
        if omvt_cfg is None:
            raise ValueError("omvt_cfg is required for native detail installation")
        if initialize_from_legacy and self.omvt is None:
            raise RuntimeError(
                "legacy OMVT must already be installed before native migration"
            )

        from Model.omvt.native_migration import (
            initialize_fresh_native_detail,
            migrate_legacy_omvt_to_native_detail,
        )
        from Model.omvt.native_tower import NativeOMVTDetailTower

        native_tower = NativeOMVTDetailTower(
            omvt_cfg,
            max_detail_tokens_per_sample=max_detail_tokens,
            source_tokens_per_detail_token=ratio,
        )
        if initialize_from_legacy:
            legacy_tower = getattr(self.omvt, "tower", None)
            if legacy_tower is None:
                raise TypeError("installed legacy OMVT does not expose its tower")
            target_parameter = next(legacy_tower.parameters(), None)
        else:
            target_parameter = next(self.parameters(), None)
        if target_parameter is not None:
            native_tower = native_tower.to(
                device=target_parameter.device,
                dtype=target_parameter.dtype,
            )

        if initialize_from_legacy:
            receipt = migrate_legacy_omvt_to_native_detail(
                legacy_tower,
                native_tower,
            )
        else:
            receipt = initialize_fresh_native_detail(native_tower)

        # Register only after every shape/layer/tensor check has succeeded, so
        # a failed migration never leaves a partially initialized live module.
        self.native_detail_tower = native_tower
        self.native_detail_migration_receipt = receipt
        return receipt

    def encode_visual(
        self,
        pixel_values: torch.Tensor | Mapping[str, torch.Tensor],
        *,
        pixel_repeats: int = 1,
    ) -> torch.Tensor:
        """Encode/project pixels once, before injecting them into token slots."""

        if type(pixel_repeats) is not int or pixel_repeats <= 0:
            raise ValueError("pixel_repeats must be a positive integer")
        if isinstance(pixel_values, Mapping):
            self._ensure_omvt()
            return self.omvt(pixel_values, pixel_repeats=pixel_repeats)
        if pixel_repeats != 1:
            raise ValueError(
                "pixel_repeats is only supported for OMVT mapping inputs"
            )
        return self.encoder(pixel_values)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
        visual_features: torch.Tensor | None = None,
        pixel_repeats: int = 1,
    ) -> torch.Tensor:
        if type(pixel_repeats) is not int or pixel_repeats <= 0:
            raise ValueError("pixel_repeats must be a positive integer")
        if pixel_values is not None and visual_features is not None:
            raise ValueError("pass pixel_values or visual_features, not both")
        if visual_features is not None and pixel_repeats != 1:
            raise ValueError(
                "pixel_repeats must be 1 for precomputed visual_features"
            )
        if pixel_values is None and visual_features is None:
            if pixel_repeats != 1:
                raise ValueError("pixel_repeats requires pixel_values")
            return inputs_embeds
        if visual_features is None:
            assert pixel_values is not None
            visual_features = self.encode_visual(
                pixel_values,
                pixel_repeats=pixel_repeats,
            )
        return inject_visual_features(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            visual_features=visual_features,
            image_patch_id=self.cfg.image_patch_id,
        )


def _check() -> None:
    from Model.config import tiny_config

    torch.manual_seed(0)

    cfg = tiny_config()
    patch_pixels = 14 * 14 * 3

    input_ids = torch.tensor(
        [
            [
                cfg.bos_id,
                300,
                301,
                cfg.image_start_id,
                cfg.image_patch_id,
                cfg.image_patch_id,
                cfg.image_patch_id,
                cfg.image_patch_id,
                cfg.image_end_id,
                302,
                303,
                cfg.eos_id,
            ]
        ]
    )

    inputs_embeds = torch.randn(1, input_ids.shape[1], cfg.d_model)
    pixel_values = torch.randn(4, patch_pixels)

    injector = VisionInjector(cfg, patch_pixels)
    out = injector(inputs_embeds, input_ids, pixel_values)

    mask = input_ids == cfg.image_patch_id
    changed = (out != inputs_embeds).any(dim=-1)

    print("VisionInjector")
    print(f"  shape: {tuple(inputs_embeds.shape)} -> {tuple(out.shape)}")
    print(f"  patch_count: {int(mask.sum().item())}")
    print(f"  changed: {changed[0].nonzero().squeeze(-1).tolist()}")
    print(f"  expected: {mask[0].nonzero().squeeze(-1).tolist()}")
    print(f"  exact_match: {torch.equal(changed, mask)}")

    out_text = injector(inputs_embeds, input_ids, pixel_values=None)
    print(f"  text_only_equal: {torch.equal(out_text, inputs_embeds)}")

    pv = torch.randn(4, patch_pixels, requires_grad=True)
    out = injector(inputs_embeds, input_ids, pv)
    out.sum().backward()

    print(f"  grad_norm: {pv.grad.norm().item():.6f}")

    batched_ids = torch.tensor(
        [
            [cfg.bos_id, cfg.image_patch_id, cfg.image_patch_id, cfg.eos_id],
            [cfg.bos_id, 301, cfg.image_patch_id, cfg.eos_id],
        ]
    )
    batched_embeds = torch.randn(2, 4, cfg.d_model)
    batched_pixels = torch.randn(2, 2, patch_pixels)

    batched_out = injector(batched_embeds, batched_ids, batched_pixels)
    print(f"  batched_shape: {tuple(batched_out.shape)}")


if __name__ == "__main__":
    _check()
