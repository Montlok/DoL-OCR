# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from Model.blocks import StandardBlock
from Model.config import RDTConfig
from Model.layers.rmsnorm import RMSNorm
from Model.recurrent import RecurrentCore
from Model.segmented import SegmentedCore
from Model.two_stage import TwoStageCore
from Model.vision import VisionInjector


class RDTForCausalLM(nn.Module):
    def __init__(self, cfg: RDTConfig, patch_pixels: int = 14 * 14 * 3):
        super().__init__()

        self.cfg = cfg
        self.patch_pixels = patch_pixels

        self.embed = nn.Embedding(
            cfg.vocab_size,
            cfg.d_model,
            padding_idx=cfg.pad_id,
        )

        self.vision = VisionInjector(cfg, patch_pixels)

        self.prelude = nn.ModuleList(
            StandardBlock(cfg, layer_idx=i) for i in range(cfg.n_prelude)
        )

        if cfg.core_type == "two_stage":
            self.recurrent = TwoStageCore(cfg)
        elif cfg.core_type == "segmented":
            self.recurrent = SegmentedCore(cfg)
        else:
            self.recurrent = RecurrentCore(cfg)

        self.coda = nn.ModuleList(
            StandardBlock(cfg, layer_idx=cfg.n_prelude + i) for i in range(cfg.n_coda)
        )

        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.rmsnorm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.bidirectional = cfg.bidirectional

        if self.bidirectional:
            self.reverse_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        else:
            self.reverse_head = None

        # Runtime toggle: alignment phases (SFT/DPO/GRPO) disable the
        # bidirectional reverse-LM auxiliary loss so it does not pollute the
        # supervised/preference gradient. Pretraining keeps it enabled.
        self.reverse_loss_enabled = True

        self.apply(self._init_weights)
        self._scale_residual_projections()

        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight
            if self.reverse_head is not None:
                self.reverse_head.weight = self.embed.weight

    def _forward_decode(
        self,
        input_ids: torch.Tensor,
        word_pos: torch.Tensor,
        morph_depth: torch.Tensor,
        cache,
        pixel_values: torch.Tensor | None = None,
        steps: int | None = None,
    ) -> torch.Tensor:
        """Incremental forward over the new tokens ``input_ids`` (``[B, m]``).

        ``word_pos`` / ``morph_depth`` are the *absolute* per-position values of
        the new tokens (computed by the caller from the full running sequence).
        Reuses ``cache`` so the result is bit-exact with a full
        :meth:`forward` over the whole prefix. Returns logits ``[B, m, vocab]``;
        callers typically read the last position. Generation is pad-free, so
        ``attn_mask`` is omitted throughout.
        """

        if self.cfg.core_type not in {"two_stage", "segmented"}:
            raise NotImplementedError(
                "incremental KV/state cache is implemented for core_type="
                "'two_stage' / 'segmented' only"
            )

        bsz, seq_len = input_ids.shape
        pos_offset = cache.seq_len

        h = self.embed(input_ids)
        if pixel_values is not None:
            h = self.vision(h, input_ids, pixel_values)

        for i, block in enumerate(self.prelude):
            h = block(
                h,
                word_pos=word_pos,
                morph_depth=morph_depth,
                attn_mask=None,
                causal=True,
                cache=cache.mla_cache(f"prelude.{i}"),
                pos_offset=pos_offset,
            )

        h, _rec_info = self.recurrent(
            h,
            word_pos=word_pos,
            morph_depth=morph_depth,
            attn_mask=None,
            causal=True,
            cache=cache,
            pos_offset=pos_offset,
            steps=steps,
        )

        for i, block in enumerate(self.coda):
            h = block(
                h,
                word_pos=word_pos,
                morph_depth=morph_depth,
                attn_mask=None,
                causal=True,
                cache=cache.mla_cache(f"coda.{i}"),
                pos_offset=pos_offset,
            )

        h = self.final_norm(h)
        logits = self.lm_head(h)
        cache.seq_len = pos_offset + seq_len
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        word_pos: torch.Tensor | None = None,
        morph_depth: torch.Tensor | None = None,
        steps: int | None = None,
        bptt_window: int | None = None,
        return_logits: bool = True,
        loss_chunk_size: int | None = None,
    ) -> dict[str, torch.Tensor | dict | None]:
        self._check_inputs(input_ids, attention_mask, labels)
        if loss_chunk_size is None:
            loss_chunk_size = self.cfg.loss_chunk_size
        elif loss_chunk_size <= 0:
            raise ValueError("loss_chunk_size must be positive")

        bsz, seq_len = input_ids.shape

        if attention_mask is None:
            attention_mask = (input_ids != self.cfg.pad_id).long()

        if word_pos is None or morph_depth is None:
            word_pos, morph_depth = self._default_morph_info(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        h = self.embed(input_ids)

        if pixel_values is not None:
            h = self.vision(h, input_ids, pixel_values)

        for block in self.prelude:
            h = self._maybe_ckpt(
                block,
                h,
                word_pos=word_pos,
                morph_depth=morph_depth,
                attn_mask=attention_mask,
                causal=True,
            )

        e0 = h

        h, rec_info = self.recurrent(
            e0,
            word_pos=word_pos,
            morph_depth=morph_depth,
            attn_mask=attention_mask,
            causal=True,
            steps=steps,
            bptt_window=bptt_window,
        )

        for block in self.coda:
            h = self._maybe_ckpt(
                block,
                h,
                word_pos=word_pos,
                morph_depth=morph_depth,
                attn_mask=attention_mask,
                causal=True,
            )

        h = self.final_norm(h)
        logits = None

        loss = None
        loss_parts: dict[str, float] = {}

        if labels is not None:
            if not return_logits and loss_chunk_size is not None:
                loss, loss_parts = self._losses_chunked(
                    h,
                    labels,
                    rec_info,
                    loss_chunk_size,
                )
            else:
                logits = self.lm_head(h)
                loss, loss_parts = self._losses(h, logits, labels, rec_info)
        elif return_logits:
            logits = self.lm_head(h)

        return {
            "loss": loss,
            "logits": logits if return_logits else None,
            "loss_parts": loss_parts,
            "rec_info": rec_info,
        }

    def _losses(
        self,
        h: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        rec_info: dict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        forward = self._causal_loss(logits, labels)
        loss = forward
        parts = {"forward": float(forward.detach())}

        if self.reverse_head is not None and self.reverse_loss_enabled:
            rev_logits = self.reverse_head(h)
            reverse = self._reverse_loss(rev_logits, labels)
            loss = loss + self.cfg.reverse_loss_weight * reverse
            parts["reverse"] = float(reverse.detach())

        ponder = rec_info.get("ponder_cost")
        if self.cfg.use_act and isinstance(ponder, torch.Tensor):
            loss = loss + self.cfg.act_ponder_cost * ponder
            parts["ponder"] = float(ponder.detach())

        return loss, parts

    def _losses_chunked(
        self,
        h: torch.Tensor,
        labels: torch.Tensor,
        rec_info: dict,
        chunk_size: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        forward = self._chunked_causal_loss(h, labels, self.lm_head, chunk_size)
        loss = forward
        parts = {"forward": float(forward.detach())}

        if self.reverse_head is not None and self.reverse_loss_enabled:
            reverse = self._chunked_reverse_loss(
                h,
                labels,
                self.reverse_head,
                chunk_size,
            )
            loss = loss + self.cfg.reverse_loss_weight * reverse
            parts["reverse"] = float(reverse.detach())

        ponder = rec_info.get("ponder_cost")
        if self.cfg.use_act and isinstance(ponder, torch.Tensor):
            loss = loss + self.cfg.act_ponder_cost * ponder
            parts["ponder"] = float(ponder.detach())

        return loss, parts

    def _causal_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self._token_loss(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
        )

    def _reverse_loss(
        self,
        rev_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return self._token_loss(
            rev_logits[:, 1:].reshape(-1, rev_logits.size(-1)),
            labels[:, :-1].reshape(-1),
        )

    def _token_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = targets != self.cfg.ignore_index
        if logits.numel() == 0 or not bool(valid.any()):
            return logits.sum() * 0.0

        return F.cross_entropy(
            logits,
            targets,
            ignore_index=self.cfg.ignore_index,
        )

    def _chunked_causal_loss(
        self,
        h: torch.Tensor,
        labels: torch.Tensor,
        head: nn.Linear,
        chunk_size: int,
    ) -> torch.Tensor:
        return self._chunked_token_loss(
            h[:, :-1].reshape(-1, h.size(-1)),
            labels[:, 1:].reshape(-1),
            head,
            chunk_size,
        )

    def _chunked_reverse_loss(
        self,
        h: torch.Tensor,
        labels: torch.Tensor,
        head: nn.Linear,
        chunk_size: int,
    ) -> torch.Tensor:
        return self._chunked_token_loss(
            h[:, 1:].reshape(-1, h.size(-1)),
            labels[:, :-1].reshape(-1),
            head,
            chunk_size,
        )

    def _chunked_token_loss(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        head: nn.Linear,
        chunk_size: int,
    ) -> torch.Tensor:
        valid = targets != self.cfg.ignore_index
        if hidden.numel() == 0 or not bool(valid.any()):
            return hidden.sum() * 0.0

        hidden = hidden[valid]
        targets = targets[valid]
        loss_sum = hidden.new_zeros(())

        for start in range(0, hidden.size(0), chunk_size):
            end = min(start + chunk_size, hidden.size(0))
            logits = F.linear(hidden[start:end], head.weight, head.bias)
            loss_sum = loss_sum + F.cross_entropy(
                logits,
                targets[start:end],
                reduction="sum",
            )

        return loss_sum / targets.numel()

    def _maybe_ckpt(
        self,
        block: nn.Module,
        h: torch.Tensor,
        word_pos: torch.Tensor | None,
        morph_depth: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        causal: bool,
    ) -> torch.Tensor:
        if (
            getattr(self.cfg, "grad_ckpt_prelude_coda", False)
            and self.training
            and h.requires_grad
        ):
            def _fn(x):
                return block(
                    x,
                    word_pos=word_pos,
                    morph_depth=morph_depth,
                    attn_mask=attn_mask,
                    causal=causal,
                )

            return checkpoint(_fn, h, use_reentrant=False)
        return block(
            h,
            word_pos=word_pos,
            morph_depth=morph_depth,
            attn_mask=attn_mask,
            causal=causal,
        )

    def _default_morph_info(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Derive ``(word_pos, morph_depth)`` from boundary token IDs.

        This is a vectorized counterpart of
        :func:`Tokenizer.pretraining.derive_morph_info_from_boundary_ids`.
        Callers that already supply ``word_pos`` / ``morph_depth`` skip this.

        Semantics (see tokenizer-side reference for details):

        * ``word_boundary_id`` opens a new word at depth 0.
        * ``morpheme_boundary_id`` keeps the current word and bumps depth.
        * Other special tokens (ids in ``[0, 256)``) reset depth to 0 and
          stay on the current ``word_pos``; the next non-special content
          token will inherit ``word_pos`` until a new ``word_boundary``.
        * Padding positions (``attention_mask == 0``) keep their derived
          ``word_pos`` so downstream RoPE never sees ``-1``; the mask
          itself is what zeros out padded contributions.
        """

        device = input_ids.device
        bsz, seq_len = input_ids.shape
        cfg = self.cfg

        wb = int(cfg.word_boundary_id)
        mb = int(cfg.morpheme_boundary_id)
        special_hi = 256

        is_wb = input_ids == wb
        is_mb = input_ids == mb
        is_special = (input_ids >= 0) & (input_ids < special_hi)
        is_content = ~is_special
        is_other_special = is_special & ~is_wb & ~is_mb

        # word_pos: cumulative count of word_boundary occurrences with a
        # per-row shift. If the very first word_boundary appears before
        # any content token, the first wb anchors word 0 (shift = -1).
        # Otherwise content tokens implicitly open word 0 and the first
        # wb opens word 1 (shift = 0).
        wb_cum = is_wb.long().cumsum(dim=1)

        inf = seq_len + 1
        any_wb = is_wb.any(dim=1)
        any_content = is_content.any(dim=1)

        first_wb = torch.where(
            any_wb,
            is_wb.long().argmax(dim=1),
            torch.full((bsz,), inf, device=device, dtype=torch.long),
        )
        first_content = torch.where(
            any_content,
            is_content.long().argmax(dim=1),
            torch.full((bsz,), inf, device=device, dtype=torch.long),
        )
        shift = torch.where(
            first_wb < first_content,
            torch.full((bsz,), -1, device=device, dtype=torch.long),
            torch.zeros(bsz, device=device, dtype=torch.long),
        )

        word_pos = (wb_cum + shift.unsqueeze(-1)).clamp(min=0)

        # morph_depth: cumulative morpheme_boundary count since the most
        # recent reset (word_boundary or other special). The reset
        # position itself reads depth 0.
        cum_inc = is_mb.long().cumsum(dim=1)
        reset_positions = is_wb | is_other_special
        reset_value = torch.where(
            reset_positions, cum_inc, torch.full_like(cum_inc, -1)
        )
        last_reset, _ = reset_value.cummax(dim=1)
        depth = cum_inc - last_reset.clamp(min=0)
        depth = torch.where(reset_positions, torch.zeros_like(depth), depth)

        if cfg.max_morph_depth > 0:
            depth = depth.clamp(max=cfg.max_morph_depth - 1)

        return word_pos.to(torch.long), depth.to(torch.long)


    def _check_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        labels: torch.Tensor | None,
    ) -> None:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, L]")

        if input_ids.numel() == 0:
            raise ValueError("input_ids cannot be empty")

        # Dtype + shape checks only; value-range checks are skipped here
        # to avoid host syncs on the hot path. The embedding lookup will
        # raise an out-of-range error if vocab bounds are violated, and
        # we rely on the data pipeline to keep ids well-formed.
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must be int32 or int64")

        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have shape [B, L]")

        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must have shape [B, L]")

        if input_ids.shape[1] > self.cfg.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.init_std

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    # Last linear of every residual write: attention output, SwiGLU down
    # projection, and the SSM output projection (both NaiveSSM and the
    # official Mamba3 expose it as ``out_proj``).
    _RESIDUAL_PROJ_SUFFIXES = ("attn.o_proj", "ffn.w_down", "mamba.out_proj")

    @torch.no_grad()
    def _scale_residual_projections(self) -> None:
        """GPT-2-style depth-scaled init, extended to the unrolled loop.

        Every sublayer writes ``x + f(x)`` into the residual stream. With a
        weight-shared recurrent core the stream sees its core-aware
        ``effective_depth`` (including one-pass encoder/local stages and the
        repeated refinement stage), so leaving the output projections at
        ``init_std`` makes the hidden-state norm
        grow linearly across recurrent steps (and costs bf16 mantissa
        precision late in the loop). Scaling them by
        ``1/sqrt(2 * effective_depth)`` keeps the residual variance roughly
        depth-invariant at init.
        """

        scale = (2.0 * self.cfg.effective_depth) ** -0.5

        for container in (self.prelude, self.recurrent, self.coda):
            for name, module in container.named_modules():
                if isinstance(module, nn.Linear) and name.endswith(
                    self._RESIDUAL_PROJ_SUFFIXES
                ):
                    module.weight.mul_(scale)

    def _kl_exit_active(self, recurrent_steps: int | None) -> bool:
        """Whether zero-shot KL early-exit governs this decode step.

        Gated on ``cfg.kl_exit_threshold > 0`` and only on the cache-free path
        with no explicit ``recurrent_steps`` override; when disabled the decode
        is bit-exact with a single fixed-depth forward.
        """

        return (
            self.cfg.kl_exit_threshold > 0.0
            and recurrent_steps is None
            and self.cfg.core_type in {"two_stage", "segmented"}
        )

    @torch.no_grad()
    def _adaptive_depth_logits(
        self,
        window: torch.Tensor,
        pixel_values: torch.Tensor | None,
    ) -> torch.Tensor:
        """Last-position logits with per-token adaptive recurrent depth.

        Runs the refinement at increasing depth ``d = 1 .. recurrent_steps`` and
        stops as soon as the symmetric-free KL between the depth-``d`` and
        depth-``d-1`` last-token distributions drops below
        ``cfg.kl_exit_threshold`` for every row (Huginn arXiv:2502.05171 §6.1,
        zero-shot convergence of the latent recurrence). Saves test-time compute
        on "easy" tokens while spending full depth on hard / rare ones. Always
        returns the deepest distribution actually computed.
        """

        max_steps = self.cfg.recurrent_steps
        threshold = self.cfg.kl_exit_threshold
        prev_logp = None
        logits = None
        for d in range(1, max_steps + 1):
            if pixel_values is not None:
                out = self.forward(
                    window, steps=d, return_logits=True, pixel_values=pixel_values
                )
            else:
                out = self.forward(window, steps=d, return_logits=True)
            logits = out["logits"][:, -1, :].float()
            logp = F.log_softmax(logits, dim=-1)
            if prev_logp is not None:
                # KL(p_d || p_{d-1}) per row; exit when all rows have converged.
                kl = (logp.exp() * (logp - prev_logp)).sum(dim=-1)
                if bool((kl < threshold).all()):
                    break
            prev_logp = logp
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        greedy: bool = False,
        eos_id: int | None = None,
        pad_id: int | None = None,
        repetition_penalty: float = 1.0,
        use_cache: bool = False,
        stop_ids: Sequence[int] | None = None,
        on_token: Callable[[int, torch.Tensor], None] | None = None,
        recurrent_steps: int | None = None,
        pixel_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Autoregressively continue ``input_ids`` (``[B, L]``) with sampling.

        Sampling controls: ``temperature`` then ``min_p`` (ICLR 2025), ``top_k``
        and nucleus ``top_p`` truncation, plus CTRL-style ``repetition_penalty``;
        ``greedy=True`` takes the argmax.

        With ``use_cache=True`` (``core_type='two_stage'`` only) decoding runs an
        incremental KV/state cache that is numerically identical to the cache-
        free path but processes one token per step instead of re-running the
        whole prefix:

        * **Stage 1 (Mamba)** keeps a constant-size ``(conv_state, ssm_state)``
          per layer and steps the selective SSM once per new token -- O(1) time
          and memory per step (Gu & Dao, "Mamba", 2023/2024).
        * **Stage 2 (MLA)** appends the new token's ``K``/``V`` to a per-layer
          cache and attends over the frozen prefix; because each refinement pass
          is causal and earlier positions are frozen once computed, one cache is
          kept per ``(refinement step, layer)`` pair.

        The cached path keeps the full context (no sliding-window eviction), so
        ``L + max_new_tokens`` must stay within ``max_seq_len``; the cache-free
        path instead slides a ``max_seq_len`` window. ``use_cache`` is only
        supported for ``core_type='two_stage'``.

        Returns the full sequence ``[B, L + n]`` where ``n <= max_new_tokens``.
        Generation stops early for a row once it emits ``eos_id`` (subsequent
        positions are filled with ``pad_id``).

        ``stop_ids`` adds extra stop tokens: a row also finishes once it emits
        any id in ``stop_ids`` (the stop token itself is kept in the output, so
        a harness can see *which* delimiter — e.g. ``</think>`` or a tool-call
        close token — ended the segment, then resume by calling ``generate``
        again on the returned sequence). ``eos_id`` is always a stop token.

        ``on_token`` is an optional callback invoked as ``on_token(step, tok)``
        after each decoding step with the step index and the ``[B]`` tensor of
        ids that were just appended to the sequence; use it for streaming. Note
        rows that have already finished receive ``pad_id`` (not a freshly
        sampled id), matching exactly what is written to the output. Both are
        backward compatible: when unset, behaviour is identical to before.

        ``recurrent_steps`` overrides the recurrent-depth refinement count for
        this decode call (default ``None`` -> ``cfg.recurrent_steps``). Because
        RDT reasons in latent depth, raising this spends more test-time compute
        ("think harder") per token without emitting any extra tokens; lowering
        it trades quality for speed. The override is constant for the whole call
        so the incremental KV/state cache stays consistent across positions.

        ``pixel_values`` optionally supplies image features for prompts containing
        ``<image_patch>`` slots. Cached decoding consumes them only during the
        prefill step; cache-free decoding refuses sliding-window truncation with
        images because dropping patch slots would desync the visual payload.
        """

        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [B, L]")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when set")
        if top_p is not None and not (0.0 < top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1] when set")
        if min_p is not None and not (0.0 < min_p <= 1.0):
            raise ValueError("min_p must be in (0, 1] when set")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if recurrent_steps is not None and recurrent_steps <= 0:
            raise ValueError("recurrent_steps must be positive when set")
        if use_cache and self.cfg.core_type not in {"two_stage", "segmented"}:
            raise NotImplementedError(
                "use_cache=True is only supported for core_type="
                "'two_stage' / 'segmented'"
            )
        if use_cache and self.cfg.use_official_mamba:
            raise NotImplementedError(
                "use_cache=True requires the NaiveSSM fallback backend; "
                "official Mamba kernels are not steppable by the decode cache. "
                "Pass --mamba=naive for cached decoding or use_cache=False."
            )

        cfg = self.cfg
        eos_id = cfg.eos_id if eos_id is None else eos_id
        pad_id = cfg.pad_id if pad_id is None else pad_id

        was_training = self.training
        self.eval()

        seq = input_ids
        device = seq.device
        finished = torch.zeros(seq.shape[0], dtype=torch.bool, device=device)

        stop_token_ids: list[int] = []
        if eos_id is not None:
            stop_token_ids.append(int(eos_id))
        if stop_ids is not None:
            stop_token_ids.extend(int(s) for s in stop_ids)
        stop_tensor = (
            torch.tensor(sorted(set(stop_token_ids)), device=device, dtype=seq.dtype)
            if stop_token_ids
            else None
        )

        decode_cache = None
        if use_cache:
            from Model.inference.cache import DecodeCache

            # The incremental cache prefills every prompt position into the
            # MLA/Mamba state with an all-ones mask, so a padded prompt would
            # fold pad embeddings into the cache and change logits for shorter
            # rows. Reject padded prompts (use the cache-free path for those).
            if pad_id is not None and bool((input_ids == pad_id).any()):
                raise ValueError(
                    "use_cache=True requires pad-free prompts; the incremental "
                    "cache cannot mask padding. Use use_cache=False for padded "
                    "batches."
                )
            # The cache has no eviction, so the whole run must fit in context.
            # Guard up front (counting the tokens about to be appended) instead
            # of generating one token past the limit before raising.
            if input_ids.shape[1] + max_new_tokens > cfg.max_seq_len:
                raise ValueError(
                    "cached generation requires L + max_new_tokens <= "
                    f"max_seq_len ({cfg.max_seq_len}); got L={input_ids.shape[1]}"
                    f" + max_new_tokens={max_new_tokens}. Reduce max_new_tokens "
                    "or use use_cache=False."
                )
            decode_cache = DecodeCache()

        try:
            for step in range(max_new_tokens):
                if use_cache:
                    if decode_cache.seq_len == 0:
                        step_ids = seq
                        step_pixels = pixel_values
                    else:
                        step_ids = seq[:, -1:]
                        # The image was folded into the cache at prefill; later
                        # steps process only the new token and carry no
                        # <image_patch> slots, so pixel_values must be dropped.
                        step_pixels = None
                    mask = (seq != pad_id).long()
                    word_pos, morph_depth = self._default_morph_info(seq, mask)
                    m = step_ids.shape[1]
                    logits = self._forward_decode(
                        step_ids,
                        word_pos=word_pos[:, -m:],
                        morph_depth=morph_depth[:, -m:],
                        cache=decode_cache,
                        pixel_values=step_pixels,
                        steps=recurrent_steps,
                    )[:, -1, :].float()
                else:
                    window = seq
                    if window.shape[1] > cfg.max_seq_len:
                        window = window[:, -cfg.max_seq_len:]
                        if pixel_values is not None:
                            # Left-truncation could drop <image_patch> slots while
                            # pixel_values still holds the full visual payload,
                            # desyncing the injector. Refuse loudly.
                            raise ValueError(
                                "cache-free image generation cannot truncate the "
                                "context (L + generated tokens exceeded "
                                "max_seq_len); the <image_patch> slots would "
                                "desync from pixel_values. Use use_cache=True or "
                                "keep L + max_new_tokens <= max_seq_len."
                            )

                    if self._kl_exit_active(recurrent_steps):
                        logits = self._adaptive_depth_logits(
                            window, pixel_values
                        )
                    elif pixel_values is not None:
                        out = self.forward(
                            window,
                            steps=recurrent_steps,
                            return_logits=True,
                            pixel_values=pixel_values,
                        )
                        logits = out["logits"][:, -1, :].float()
                    else:
                        out = self.forward(
                            window, steps=recurrent_steps, return_logits=True
                        )
                        logits = out["logits"][:, -1, :].float()

                if repetition_penalty != 1.0:
                    logits = self._apply_repetition_penalty(
                        logits, seq, repetition_penalty
                    )

                if greedy:
                    next_token = torch.argmax(logits, dim=-1)
                else:
                    logits = logits / temperature
                    logits = self._filter_logits(logits, top_k, top_p, min_p)
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

                next_token = torch.where(
                    finished, torch.full_like(next_token, pad_id), next_token
                )
                next_token = next_token.to(seq.dtype)
                seq = torch.cat([seq, next_token.unsqueeze(1)], dim=1)
                if on_token is not None:
                    on_token(step, next_token)
                if stop_tensor is not None:
                    finished = finished | torch.isin(next_token, stop_tensor)
                else:
                    finished = finished | (next_token == eos_id)
                if bool(finished.all()):
                    break
        finally:
            if was_training:
                self.train()

        return seq

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor, seq: torch.Tensor, penalty: float
    ) -> torch.Tensor:
        """Divide logits of already-seen tokens by ``penalty`` (CTRL-style)."""

        for row in range(seq.shape[0]):
            seen = torch.unique(seq[row])
            row_logits = logits[row, seen]
            logits[row, seen] = torch.where(
                row_logits > 0, row_logits / penalty, row_logits * penalty
            )
        return logits

    @staticmethod
    def _filter_logits(
        logits: torch.Tensor,
        top_k: int | None,
        top_p: float | None,
        min_p: float | None = None,
    ) -> torch.Tensor:
        """Apply top-k, nucleus (top-p) and min-p masking to ``[B, V]`` logits.

        min-p (Nguyen et al., "Turning Up the Heat: Min-p Sampling for
        Creative and Coherent LLM Outputs", ICLR 2025) keeps only tokens whose
        probability is at least ``min_p * p_max`` where ``p_max`` is the top
        token's probability. The candidate pool scales with the model's own
        confidence: sharp distributions prune hard, flat ones stay permissive.
        It is applied before top-k/top-p so it can act as the primary truncation.
        """

        if min_p is not None:
            probs = F.softmax(logits, dim=-1)
            p_max = probs.max(dim=-1, keepdim=True).values
            logits = logits.masked_fill(probs < min_p * p_max, float("-inf"))

        if top_k is not None:
            k = min(top_k, logits.shape[-1])
            kth = torch.topk(logits, k, dim=-1).values[:, -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))

        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            remove_idx = remove.scatter(1, sorted_idx, remove)
            logits = logits.masked_fill(remove_idx, float("-inf"))

        return logits

    def count_params(self, trainable_only: bool = False) -> int:
        seen: set[int] = set()
        total = 0

        for param in self.parameters():
            if trainable_only and not param.requires_grad:
                continue

            ptr = param.data_ptr()
            if ptr in seen:
                continue

            seen.add(ptr)
            total += param.numel()

        return total


def _check() -> None:
    from Model.config import tiny_config

    torch.manual_seed(0)

    cfg = tiny_config()
    model = RDTForCausalLM(cfg)

    print("RDTForCausalLM")
    print(f"  params: {model.count_params():,}")
    print(f"  actual_layers: {cfg.actual_layers}")
    print(f"  effective_depth: {cfg.effective_depth}")

    bsz, seq_len = 2, 32
    input_ids = torch.randint(256, 24576, (bsz, seq_len))
    input_ids[:, 0] = cfg.bos_id

    attention_mask = torch.ones(bsz, seq_len, dtype=torch.long)
    labels = input_ids.clone()

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    print("Text")
    print(f"  logits: {tuple(out['logits'].shape)}")
    print(f"  loss: {out['loss'].item():.6f}")
    print(f"  loss_parts: {out['loss_parts']}")
    print(f"  steps_used: {out['rec_info'].get('steps_used')}")

    out["loss"].backward()
    grad_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_sq += p.grad.norm().item() ** 2

    print(f"  grad_norm: {grad_sq**0.5:.6f}")

    ids2 = torch.tensor(
        [
            [
                cfg.bos_id,
                300,
                cfg.image_start_id,
                cfg.image_patch_id,
                cfg.image_patch_id,
                cfg.image_end_id,
                301,
                cfg.eos_id,
            ]
        ]
    )

    labels2 = ids2.clone()
    labels2[ids2 == cfg.image_start_id] = cfg.ignore_index
    labels2[ids2 == cfg.image_patch_id] = cfg.ignore_index
    labels2[ids2 == cfg.image_end_id] = cfg.ignore_index

    pixel_values = torch.randn(2, model.patch_pixels)

    out2 = model(
        input_ids=ids2,
        labels=labels2,
        pixel_values=pixel_values,
    )

    print("ImageText")
    print(f"  loss: {out2['loss'].item():.6f}")

    model.eval()
    with torch.no_grad():
        for steps in [2, 8]:
            out_step = model(input_ids=input_ids, steps=steps)
            print(
                f"  steps={steps}, logits_norm={out_step['logits'].norm().item():.6f}"
            )


if __name__ == "__main__":
    _check()
