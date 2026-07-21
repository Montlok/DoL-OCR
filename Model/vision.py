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

    def encode_visual(
        self,
        pixel_values: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode/project pixels once, before injecting them into token slots."""

        if isinstance(pixel_values, Mapping):
            self._ensure_omvt()
            return self.omvt(pixel_values)
        return self.encoder(pixel_values)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
        visual_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pixel_values is not None and visual_features is not None:
            raise ValueError("pass pixel_values or visual_features, not both")
        if pixel_values is None and visual_features is None:
            return inputs_embeds
        if visual_features is None:
            assert pixel_values is not None
            visual_features = self.encode_visual(pixel_values)
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
