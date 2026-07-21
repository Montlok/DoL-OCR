# -*- coding: utf-8 -*-

"""Generative OCR evaluation for VLM-aligned RDT checkpoints.

``scripts.train_vlm_align`` optimizes teacher-forced cross-entropy; this CLI
measures what actually matters for the OCR program: free decoding. It replays
held-out alignment rows (the :func:`Model.ocr.data.build_ocr_row` contract:
``[BOS] <image_start> <image_patch>*N <image_end> <instruction...>`` + target),
feeds each row's masked prompt plus its page image to
:meth:`RDTForCausalLM.generate`, and scores the sampled transcription with
:func:`Model.ocr.metrics.ocr_report` (grapheme CER as the headline number).

``--blank-baseline`` re-runs the same prompts with an all-white page so the
visual contribution can be isolated: a model that only learned the language
prior scores the same with and without the real pixels.

The model is rebuilt exactly like the training CLI (same config name, same
tower geometry flags), then the full VLM state (RDT + projector + tower) is
loaded from a ``train_vlm_align`` output. Decoding is cache-free — the
recurrent core has no incremental cache — so wall-clock is O(steps * L); use
``--limit`` to score a subset while a run is still hot.

Usage (mirrors the A10/GB10 training flags)::

    PYTHONPATH=. python3 -m scripts.eval_vlm_ocr \
        --checkpoint /root/llm/vlm_align_scratch2 \
        --data /root/llm/vlm_ocr_v1/eval.jsonl \
        --tokenizer-bundle /root/llm/tok_bundle \
        --config tiny --image-size 448 --d-vision 512 --patch-preset prod \
        --limit 96 --blank-baseline --out /root/llm/vlm_eval/preds.jsonl

    python3 -m scripts.eval_vlm_ocr --smoke   # CPU self-check, no artifacts
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

from Model.config import EOS_ID, IMAGE_PATCH_ID, OMVTConfig, PAD_ID, RDTConfig
from Model.model import RDTForCausalLM
from Model.ocr.data import build_ocr_row, split_ocr_row
from Model.ocr.metrics import ocr_report
from Model.omvt import OMVTInjector
from Model.omvt.patcher import collate_omvt_batch
from Model.training import load_checkpoint_metadata, resolve_checkpoint_dir
from Model.training.multimodal_cli import make_omvt_cfg
from Tokenizer.multimodal import PILImageProcessor
from scripts.train_rdt import CONFIG_CHOICES, _resolve_mamba_backend


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generative OCR eval (CER) for VLM checkpoints")
    p.add_argument("--checkpoint", default="", help="train_vlm_align output root, step dir, or model.pt")
    p.add_argument("--data", default="", help="alignment-row JSONL (build_vlm_ocr_data output)")
    p.add_argument("--tokenizer-bundle", default="", help="unified tokenizer bundle dir (id -> text)")
    p.add_argument("--config", choices=list(CONFIG_CHOICES), default="tiny")
    p.add_argument("--mamba", choices=["auto", "official", "naive"], default="auto")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument(
        "--precision",
        choices=("auto", "fp32", "bf16"),
        default="auto",
        help="'auto' = bf16 autocast on cuda (matches training), fp32 elsewhere",
    )
    # Tower geometry: must mirror the training CLI so the checkpoint state fits.
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--d-vision", type=int, default=512)
    p.add_argument("--n-image-tokens", type=int, default=None)
    p.add_argument("--patch-preset", choices=("derived", "prod"), default="prod")
    p.add_argument("--max-new-tokens", type=int, default=480)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--recurrent-steps",
        type=int,
        default=None,
        help="fixed decode depth; defaults to the checkpoint's trained depth",
    )
    p.add_argument("--limit", type=int, default=0, help="evaluate at most N rows (0 = all)")
    p.add_argument("--blank-baseline", action="store_true",
                   help="also decode with an all-white page (language-prior-only CER)")
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--out", default="", help="write {image, ref, pred[, pred_blank]} JSONL here")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="synthetic CPU self-check")
    return p.parse_args(argv)


def _load_model_state(path: str):
    raw = Path(path)
    if raw.is_file() and raw.name != "model.pt":
        state = torch.load(raw, map_location="cpu", weights_only=False)
        p = raw
    else:
        p = resolve_checkpoint_dir(raw)
        state = torch.load(p / "model.pt", map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state and "embed.weight" not in state:
        state = state["model"]
    return state, p


def _checkpoint_metadata(path: str) -> dict:
    if not path:
        return {}
    p = Path(path)
    if p.is_file() and p.name != "model.pt":
        return {}
    return load_checkpoint_metadata(path)


def _restore_omvt_geometry(args, metadata: dict | None = None) -> OMVTConfig | None:
    """Apply a VLM checkpoint's authoritative image geometry to ``args``.

    Deployment callers must do this before building prompts or letterboxing;
    otherwise a checkpoint with a non-default image-token count could be
    rebuilt correctly but paired with a stale prompt/pixel preprocessing path.
    """

    metadata = (
        _checkpoint_metadata(getattr(args, "checkpoint", ""))
        if metadata is None
        else metadata
    )
    raw_omvt = metadata.get("omvt_config")
    if not isinstance(raw_omvt, dict):
        return None
    omvt_cfg = OMVTConfig(**raw_omvt)
    args.image_size = omvt_cfg.image_size
    args.n_image_tokens = omvt_cfg.compress_to
    return omvt_cfg


def _build_model(args, device: torch.device) -> RDTForCausalLM:
    metadata = _checkpoint_metadata(getattr(args, "checkpoint", ""))
    raw_rdt = metadata.get("rdt_config")
    if isinstance(raw_rdt, dict):
        rdt_cfg = RDTConfig(**raw_rdt)
        print("[eval] using RDTConfig from checkpoint metadata")
    else:
        rdt_cfg = CONFIG_CHOICES[args.config]()
    rdt_cfg = replace(rdt_cfg, max_seq_len=args.seq_len)
    if args.recurrent_steps is not None:
        rdt_cfg = replace(rdt_cfg, recurrent_steps=args.recurrent_steps)
    rdt_cfg = _resolve_mamba_backend(
        rdt_cfg, args.mamba, device=device, context="scripts.eval_vlm_ocr"
    )
    omvt_cfg = _restore_omvt_geometry(args, metadata)
    if omvt_cfg is not None:
        print("[eval] using OMVTConfig from checkpoint metadata")
    else:
        omvt_cfg = make_omvt_cfg(
            args.image_size,
            args.d_vision,
            args.n_image_tokens,
            preset=args.patch_preset,
        )
    model = RDTForCausalLM(rdt_cfg).to(device)
    model.vision._omvt_cfg = omvt_cfg
    model.vision.omvt = OMVTInjector(rdt_cfg, omvt_cfg).to(device)
    return model.eval()


def _load_rows(path: str, limit: int) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def _blank_pages(n: int, image_size: int):
    from PIL import Image

    return [Image.new("RGB", (image_size, image_size), "white") for _ in range(n)]


def _pixel_batch(images, processor, omvt_cfg, device) -> dict:
    batch = dict(collate_omvt_batch(processor(images), omvt_cfg))
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


def _cut_continuation(seq: list[int]) -> list[int]:
    """Trim a generated continuation at EOS and drop finished-row padding."""
    if EOS_ID in seq:
        seq = seq[: seq.index(EOS_ID)]
    return [t for t in seq if t != PAD_ID]


def print_script_cer(tag: str, rep) -> None:
    """Print one compact line per non-empty script bucket in ``rep.script_cer``.

    Shared by :mod:`scripts.eval_vlm_ocr` and :mod:`scripts.eval_l2_deploy` so
    the two CLIs report per-script CER in the same format. Buckets with
    ``n_ref == 0`` are already omitted from ``rep.script_cer`` (see
    :func:`Model.ocr.metrics.script_bucket_cer`), so nothing to skip here.
    """
    if not rep.script_cer:
        return
    for bucket, stats in rep.script_cer.items():
        print(
            f"{tag} script={bucket} grapheme_cer={stats['cer']:.4f} "
            f"n_ref={stats['n_ref']}"
        )


def visual_contribution(real_report, blank_report) -> float:
    """Return the visual gain measured on the headline grapheme CER."""

    return float(blank_report.grapheme_cer - real_report.grapheme_cer)


@torch.no_grad()
def _decode_batches(model, prompts, pixel_fn, args, device, autocast_ctx):
    """Generate continuations for uniform-length prompt rows.

    ``pixel_fn(start, end)`` supplies the pixel batch for rows [start, end) —
    real pages for the main pass, white pages for the baseline pass.
    """
    preds_ids: list[list[int]] = []
    n = len(prompts)
    prompt_len = len(prompts[0])
    recurrent_steps = (
        args.recurrent_steps
        if args.recurrent_steps is not None
        else model.cfg.recurrent_steps
    )
    t0 = time.time()
    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        ids = torch.tensor(prompts[start:end], dtype=torch.long, device=device)
        with autocast_ctx():
            out = model.generate(
                ids,
                max_new_tokens=args.max_new_tokens,
                greedy=True,
                eos_id=EOS_ID,
                pad_id=PAD_ID,
                repetition_penalty=args.repetition_penalty,
                recurrent_steps=recurrent_steps,
                pixel_values=pixel_fn(start, end),
            )
        for row in out[:, prompt_len:].tolist():
            preds_ids.append(_cut_continuation(row))
        done = min(end, n)
        rate = (time.time() - t0) / done
        print(f"[eval] decoded {done}/{n} rows ({rate:.1f}s/row)", flush=True)
    return preds_ids


def _smoke(args) -> int:
    """End-to-end self-check on CPU: tiny tower, random weights, two rows."""
    from PIL import Image

    args.config = "tiny"
    args.image_size, args.d_vision, args.patch_preset = 56, 64, "derived"
    args.n_image_tokens = None
    args.max_new_tokens, args.batch_size = 4, 2
    n_img = make_omvt_cfg(56, 64, None, preset="derived").compress_to
    rows = [
        build_ocr_row(
            [300 + i, 301 + i, 302 + i],
            n_img,
            f"smoke_{i}",
            bos_id=1, image_start_id=8, image_patch_id=9, image_end_id=10,
            eos_id=EOS_ID,
        )
        for i in range(2)
    ]
    prompts, refs = [], []
    for row in rows:
        prompt, target, _ = split_ocr_row(row, eos_id=EOS_ID)
        prompts.append(prompt)
        refs.append(" ".join(map(str, target)))
    args.seq_len = len(prompts[0]) + args.max_new_tokens + 1
    device = torch.device("cpu")
    model = _build_model(args, device)
    processor = PILImageProcessor(image_size=args.image_size)
    omvt_cfg = model.vision._omvt_cfg
    pages = [Image.new("RGB", (56, 56), c) for c in ("white", "black")]

    def pixels(start, end):
        return _pixel_batch(pages[start:end], processor, omvt_cfg, device)

    preds_ids = _decode_batches(
        model, prompts, pixels, args, device, torch.no_grad
    )
    preds = [" ".join(map(str, ids)) for ids in preds_ids]
    rep = ocr_report(preds, refs, backend="python")
    print(f"[smoke] n={rep.n} norm_cer={rep.norm_cer:.4f} (random model; high CER expected)")
    print("VLM OCR eval smoke OK")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.recurrent_steps is not None and args.recurrent_steps <= 0:
        print(
            "scripts/eval_vlm_ocr: --recurrent-steps must be positive",
            file=sys.stderr,
        )
        return 2
    if args.smoke:
        return _smoke(args)
    for flag in ("checkpoint", "data", "tokenizer_bundle"):
        if not getattr(args, flag):
            print(f"scripts/eval_vlm_ocr: --{flag.replace('_', '-')} is required "
                  "(or use --smoke)", file=sys.stderr)
            return 2

    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    rows = _load_rows(args.data, args.limit)
    prompts: list[list[int]] = []
    refs_ids: list[list[int]] = []
    images: list[str] = []
    for row in rows:
        prompt, target, image_ref = split_ocr_row(row, eos_id=EOS_ID)
        if image_ref is None:
            raise ValueError("eval rows must carry an image reference")
        prompts.append(prompt)
        refs_ids.append(target)
        images.append(image_ref)
    prompt_lens = {len(p) for p in prompts}
    if len(prompt_lens) != 1:
        raise ValueError(f"prompts must share one length, got {sorted(prompt_lens)}")
    n_img = prompts[0].count(IMAGE_PATCH_ID)
    args.seq_len = len(prompts[0]) + args.max_new_tokens + 1

    model = _build_model(args, device)
    omvt_cfg = model.vision._omvt_cfg
    if omvt_cfg.compress_to != n_img:
        raise ValueError(
            f"tower compress_to={omvt_cfg.compress_to} but rows carry {n_img} "
            "<image_patch> slots; tower geometry flags must mirror training"
        )
    state, ckpt_path = _load_model_state(args.checkpoint)
    model.load_state_dict(state)
    print(f"[eval] loaded {ckpt_path} on {device}")

    from Tokenizer.unified.bundle import TokenizerBundle

    bundle = TokenizerBundle.from_dir(args.tokenizer_bundle)
    decode = bundle.tokenizer.decode

    if args.precision == "auto":
        precision = "bf16" if device.type == "cuda" else "fp32"
    else:
        precision = args.precision
    if precision == "bf16":
        def autocast_ctx():
            return torch.autocast(device.type, dtype=torch.bfloat16)
    else:
        autocast_ctx = torch.no_grad

    processor = PILImageProcessor(image_size=args.image_size)

    def real_pixels(start, end):
        return _pixel_batch(images[start:end], processor, omvt_cfg, device)

    preds = [decode(ids) for ids in _decode_batches(
        model, prompts, real_pixels, args, device, autocast_ctx
    )]
    refs = [decode(ids) for ids in refs_ids]
    rep = ocr_report(preds, refs)
    print(
        f"[eval] n={rep.n} grapheme_cer={rep.grapheme_cer:.4f} "
        f"norm_cer={rep.norm_cer:.4f} raw_cer={rep.raw_cer:.4f} "
        f"wer={rep.wer:.4f} line_exact={rep.line_exact:.4f}"
    )
    print_script_cer("[eval]", rep)

    preds_blank = None
    if args.blank_baseline:
        def blank_pixels(start, end):
            return _pixel_batch(
                _blank_pages(end - start, args.image_size), processor, omvt_cfg, device
            )

        preds_blank = [decode(ids) for ids in _decode_batches(
            model, prompts, blank_pixels, args, device, autocast_ctx
        )]
        rep_blank = ocr_report(preds_blank, refs)
        print(
            f"[eval/blank] n={rep_blank.n} grapheme_cer={rep_blank.grapheme_cer:.4f} "
            f"norm_cer={rep_blank.norm_cer:.4f} "
            f"raw_cer={rep_blank.raw_cer:.4f} wer={rep_blank.wer:.4f}"
        )
        print_script_cer("[eval/blank]", rep_blank)
        print(
            "[eval] visual contribution (blank_cer - real_cer): "
            f"{visual_contribution(rep, rep_blank):+.4f}"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for i in range(len(rows)):
                obj = {"image": images[i], "ref": refs[i], "pred": preds[i]}
                if preds_blank is not None:
                    obj["pred_blank"] = preds_blank[i]
                fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"[eval] wrote {len(rows)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
