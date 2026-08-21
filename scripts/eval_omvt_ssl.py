# -*- coding: utf-8 -*-

"""Evaluate an OMVT SSL checkpoint on one or more page datasets.

Purpose: quantify the synthetic-to-real domain gap before fine-tuning on
scanned books. Run it with a held-out slice of the synthetic render set and
the ingested scan set (see :mod:`scripts.ingest_scan_pdfs`); the SSL metrics
are directly comparable across datasets:

- masked-patch reconstruction loss (label-free)
- orientation accuracy on synthetic 0/90/180/270 rotations (label-free)
- OCR token loss / top-1 accuracy (only for rows carrying ``ocr_labels``)

Usage::

    PYTHONPATH=. python3 -m scripts.eval_omvt_ssl \
        --checkpoint ~/llm/omvt_ssl_v1 \
        --data synth=~/llm/omvt_synth_v1/ssl.holdout.jsonl \
        --data scan=~/llm/omvt_scan_v1/ssl.jsonl \
        --batch-size 16 --max-batches 40
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from Model.config import OMVTConfig  # noqa: E402
from Model.omvt import (  # noqa: E402
    LayoutOrderHead,
    MaskedPatchHead,
    OCRReconstructionHead,
    OMVTVisionTower,
    OrientationHead,
    collate_omvt_batch,
)
from Model.omvt.losses import (  # noqa: E402
    masked_patch_loss,
    ocr_reconstruction_loss,
    orientation_loss,
)
from Tokenizer.multimodal import PILImageProcessor  # noqa: E402

from Model.training.omvt_checkpoint import (  # noqa: E402
    load_omvt_payload,
    tower_state_from_payload,
)
from scripts.train_omvt_ssl import _first_seq, _rotate_batch  # noqa: E402


def _iter_batches(path: str, batch_size: int, proc: PILImageProcessor, limit: int):
    """One finite pass over a JSONL, yielding image batches + optional labels."""

    def _batch(rows: list[dict]) -> dict:
        return {
            "images": proc([r["images"][0] for r in rows]),
            "ocr_labels": [_first_seq(r.get("ocr_labels")) for r in rows],
        }

    buf: list[dict] = []
    yielded = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("images"):
                continue
            buf.append(row)
            if len(buf) < batch_size:
                continue
            yield _batch(buf)
            buf = []
            yielded += 1
            if limit and yielded >= limit:
                return
    if buf:
        # Flush the final partial batch: on a small eval set, dropping it
        # would silently discard up to batch_size-1 rows from the metrics.
        yield _batch(buf)


@torch.no_grad()
def _eval_dataset(
    name: str,
    path: str,
    tower: OMVTVisionTower,
    heads: dict,
    cfg: OMVTConfig,
    args,
    device: torch.device,
) -> dict:
    proc = PILImageProcessor(image_size=cfg.image_size)
    gen = torch.Generator().manual_seed(args.seed)
    sums = {"mp": 0.0, "ori_loss": 0.0, "ocr_loss": 0.0}
    counts = {"batches": 0, "ori_ok": 0, "ori_n": 0, "ocr_ok": 0, "ocr_n": 0,
              "ocr_batches": 0}

    for batch in _iter_batches(path, args.batch_size, proc, args.max_batches):
        images = batch["images"].to(device)
        bsz = images.shape[0]

        # Orientation: known synthetic rotations, accuracy is meaningful.
        ori_labels = torch.randint(0, 4, (bsz,), generator=gen)
        rotated = _rotate_batch(images, ori_labels).to(device)
        ori_labels = ori_labels.to(device)
        feats = tower(rotated)["compressed"]
        ori_logits = heads["ori"](feats)
        sums["ori_loss"] += float(orientation_loss(ori_logits, ori_labels))
        counts["ori_ok"] += int((ori_logits.argmax(-1) == ori_labels).sum())
        counts["ori_n"] += bsz

        # Masked patch on the un-rotated pages, deterministic mask per batch.
        omvt_batch = collate_omvt_batch(images, cfg)
        target = omvt_batch["square_patches"]
        bbox = omvt_batch["square_bbox"]
        mask = (
            torch.rand(target.shape[0], target.shape[1], generator=gen)
            < cfg.mask_ratio
        ).to(device)
        masked_in = target.clone()
        masked_in[mask] = 0.0
        mp_feats = tower.encoders["square"](masked_in, bbox)
        sums["mp"] += float(
            masked_patch_loss(heads["mp"](mp_feats), target, mask=mask)
        )

        # OCR only when labels exist (synthetic rows).
        labels = [seq for seq in batch["ocr_labels"] if seq]
        if labels and len(labels) == bsz:
            feats_plain = tower(images)["compressed"]
            logits = heads["ocr"](feats_plain)
            padded = torch.full(
                (bsz, cfg.compress_to), -100, dtype=torch.long, device=device
            )
            for i, seq in enumerate(batch["ocr_labels"]):
                seq_t = torch.tensor(seq[: cfg.compress_to], dtype=torch.long)
                padded[i, : seq_t.numel()] = seq_t.to(device)
            sums["ocr_loss"] += float(ocr_reconstruction_loss(logits, padded))
            valid = padded != -100
            counts["ocr_ok"] += int((logits.argmax(-1)[valid] == padded[valid]).sum())
            counts["ocr_n"] += int(valid.sum())
            counts["ocr_batches"] += 1

        counts["batches"] += 1

    b = max(counts["batches"], 1)
    out = {
        "dataset": name,
        "batches": counts["batches"],
        "masked_patch_loss": round(sums["mp"] / b, 4),
        "orientation_acc": round(counts["ori_ok"] / max(counts["ori_n"], 1), 4),
        "orientation_loss": round(sums["ori_loss"] / b, 4),
    }
    if counts["ocr_batches"]:
        out["ocr_loss"] = round(sums["ocr_loss"] / counts["ocr_batches"], 4)
        out["ocr_token_acc"] = round(counts["ocr_ok"] / max(counts["ocr_n"], 1), 4)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", action="append", required=True,
                    help="NAME=path/to/ssl.jsonl (repeatable)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-batches", type=int, default=40, help="0 = full pass")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", default="", help="optional JSON output path")
    args = ap.parse_args(argv)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    payload = load_omvt_payload(args.checkpoint)
    cfg = OMVTConfig(**payload["omvt_config"])
    # The vocab dim is the largest output dim among the head's linear layers
    # (vocab >> any hidden width by construction).
    ocr_vocab = max(
        t.shape[0] for t in payload["ocr_head"].values() if t.dim() == 2
    )

    tower = OMVTVisionTower(cfg).to(device)
    use_ema = bool(payload.get("tower_ema"))
    if use_ema:
        print("[eval] using EMA tower weights")
    tower.load_state_dict(tower_state_from_payload(payload, use_ema=use_ema))
    heads = {
        "ocr": OCRReconstructionHead(cfg.d_vision, ocr_vocab).to(device),
        "mp": MaskedPatchHead(
            cfg.d_vision,
            patch_pixels=cfg.square_patch[0] * cfg.square_patch[1] * cfg.in_channels,
        ).to(device),
        "ori": OrientationHead(cfg.d_vision).to(device),
        # Loaded only to validate the checkpoint is complete; no layout metric
        # is computed because page datasets carry no reading-order labels.
        "layout": LayoutOrderHead(cfg.d_vision, max_positions=cfg.compress_to).to(device),
    }
    heads["ocr"].load_state_dict(payload["ocr_head"])
    heads["mp"].load_state_dict(payload["masked_patch_head"])
    heads["ori"].load_state_dict(payload["orientation_head"])
    heads["layout"].load_state_dict(payload["layout_head"])
    tower.eval()
    for h in heads.values():
        h.eval()

    print(f"[eval] checkpoint step={payload.get('step')} from {args.checkpoint}")
    results = []
    for spec in args.data:
        name, _, path = spec.partition("=")
        if not path:
            name, path = Path(spec).stem, spec
        path = os.path.expanduser(path)
        res = _eval_dataset(name, path, tower, heads, cfg, args, device)
        results.append(res)
        print(json.dumps(res, ensure_ascii=False))

    if args.report:
        with open(os.path.expanduser(args.report), "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        print(f"[eval] report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
