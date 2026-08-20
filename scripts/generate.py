# -*- coding: utf-8 -*-
"""Autoregressive text generation from a trained RDT checkpoint.

Closes the train -> infer loop: load a ``model.pt`` saved by
``scripts/train_rdt`` plus the tokenizer bundle, encode a prompt, sample a
continuation and decode it back to text.

Example::

    python3 -m scripts.generate \
        --config two_stage_pretrain \
        --checkpoint outputs/run/step_00010000 \
        --tokenizer-bundle outputs/tok_build/tokenizer/bundle \
        --prompt "ᠮᠣᠩᠭᠣᠯ" --max-new-tokens 64 --top-p 0.9
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from scripts.train_rdt import CONFIG_CHOICES, _resolve_mamba_backend  # noqa: E402
from Model.model import RDTForCausalLM  # noqa: E402
from Model.ocr.tokenization import tokenizer_morphology_track_table  # noqa: E402
from Tokenizer.unified.bundle import TokenizerBundle  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text from an RDT checkpoint")
    p.add_argument("--config", choices=list(CONFIG_CHOICES), required=True)
    p.add_argument(
        "--checkpoint",
        required=True,
        help="A step dir (containing model.pt) or a direct path to model.pt.",
    )
    p.add_argument("--tokenizer-bundle", required=True)
    p.add_argument("--prompt", default="")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--min-p", type=float, default=None)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--greedy", action="store_true")
    p.add_argument(
        "--recurrent-steps",
        type=int,
        default=None,
        help="Latent refinement depth override for this generation run.",
    )
    p.add_argument(
        "--use-cache",
        action="store_true",
        help=(
            "Incremental KV/state decoding (core_type='two_stage' or "
            "'segmented'); requires --mamba naive and a NaiveSSM checkpoint."
        ),
    )
    p.add_argument(
        "--mamba",
        choices=["auto", "official", "naive"],
        default="auto",
        help=(
            "Mamba backend. auto uses official CUDA Mamba on CUDA/Linux and "
            "NaiveSSM on macOS/CPU; cached decode requires naive."
        ),
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _resolve_model_pt(path: str) -> Path:
    p = Path(path)
    if p.is_dir():
        p = p / "model.pt"
    if not p.exists():
        raise SystemExit(f"scripts/generate: checkpoint not found: {p}")
    return p


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.recurrent_steps is not None and args.recurrent_steps <= 0:
        raise SystemExit("scripts/generate: --recurrent-steps must be positive")

    bundle = TokenizerBundle.from_dir(args.tokenizer_bundle)

    cfg = CONFIG_CHOICES[args.config]()
    try:
        cfg = _resolve_mamba_backend(
            cfg,
            args.mamba,
            use_cache=args.use_cache,
            device=args.device,
            context="scripts.generate",
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    model = RDTForCausalLM(cfg)
    state = torch.load(_resolve_model_pt(args.checkpoint), map_location="cpu")
    if isinstance(state, dict) and "model" in state and "embed.weight" not in state:
        state = state["model"]
    model.load_state_dict(state)
    model.to(args.device)
    model.eval()

    prompt_ids = bundle.encode(args.prompt, add_bos=True)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
    morphology_track_table = torch.tensor(
        tokenizer_morphology_track_table(bundle.tokenizer),
        dtype=torch.long,
        device=args.device,
    )

    out = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        greedy=args.greedy,
        repetition_penalty=args.repetition_penalty,
        use_cache=args.use_cache,
        recurrent_steps=args.recurrent_steps,
        morphology_track_table=morphology_track_table,
    )

    full = out[0].tolist()
    completion = full[len(prompt_ids):]
    print("=== prompt ===")
    print(args.prompt)
    print("=== completion ===")
    print(bundle.tokenizer.decode(completion))


if __name__ == "__main__":
    main()
