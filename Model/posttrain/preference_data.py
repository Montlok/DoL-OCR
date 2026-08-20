# -*- coding: utf-8 -*-

"""Preference / prompt datasets for DPO and GRPO training.

Reuses the SFT chat renderer/masking so completion masks are identical to the
ones validated by ``test_sft_data`` (assistant content + EOS supervised,
prompt / role headers / tool-results masked). This keeps offline preference
(DPO) and online RL (GRPO) scoring consistent with SFT.

JSONL formats
-------------
DPO (``PreferenceDataset``)::

    {"messages": [{"role": "user", "content": "..."}], "chosen": "...", "rejected": "..."}
    # or a bare prompt string instead of messages:
    {"prompt": "...", "chosen": "...", "rejected": "..."}

GRPO (``PromptDataset``)::

    {"messages": [...]} | {"prompt": "..."}            # required
    {"reference": "..."}                                # optional, for verifiable reward

Image-conditioned OCR GRPO (``OCRPromptDataset``)::

    {"id": "camera-0001", "split": "rl_train",
     "image": "photos/camera-0001.jpg", "reference": "ᠮᠣᠩᠭᠤᠯ"}
"""

from __future__ import annotations

import json
import hashlib
import re
from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import Dataset

from Model.config import IGNORE_INDEX

from .sft_data import build_sft_example

Encode = Callable[[str], list[int]]
CanonicalizeReference = Callable[[str], str]
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")


def _as_messages(obj: dict) -> list[dict[str, str]]:
    if "messages" in obj:
        return list(obj["messages"])
    if "prompt" in obj:
        return [{"role": "user", "content": obj["prompt"]}]
    raise KeyError("preference/prompt row needs 'messages' or 'prompt'")


def build_preference_example(
    context: list[dict[str, str]],
    chosen: str,
    rejected: str,
    encode: Encode,
    eos_id: int,
    bos_id: int | None = None,
    max_seq_len: int | None = None,
) -> dict[str, list[int]]:
    """Tokenize a (context, chosen, rejected) triple into two masked sequences.

    Both branches share the same context; the completion mask is 1 exactly on
    the appended assistant response tokens (+ EOS), reusing the SFT builder.
    """
    def _one(answer: str) -> tuple[list[int], list[int]]:
        msgs = [*context, {"role": "assistant", "content": answer}]
        ex = build_sft_example(
            msgs, encode, eos_id=eos_id, bos_id=bos_id,
            ignore_index=IGNORE_INDEX, max_seq_len=max_seq_len,
        )
        ids = ex["input_ids"]
        mask = [0 if label == IGNORE_INDEX else 1 for label in ex["labels"]]
        return ids, mask

    chosen_ids, chosen_mask = _one(chosen)
    rejected_ids, rejected_mask = _one(rejected)
    if not any(chosen_mask):
        raise ValueError("chosen response has no supervised completion tokens")
    if not any(rejected_mask):
        raise ValueError("rejected response has no supervised completion tokens")
    return {
        "chosen_input_ids": chosen_ids,
        "chosen_completion_mask": chosen_mask,
        "rejected_input_ids": rejected_ids,
        "rejected_completion_mask": rejected_mask,
    }


def _iter_jsonl(path: str | Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


class PreferenceDataset(Dataset):
    """Map-style dataset of DPO preference examples from JSONL."""

    def __init__(
        self,
        path: str | Path,
        encode: Encode,
        eos_id: int,
        bos_id: int | None = None,
        max_seq_len: int | None = None,
    ) -> None:
        self._rows = [
            build_preference_example(
                _as_messages(obj), obj["chosen"], obj["rejected"],
                encode, eos_id=eos_id, bos_id=bos_id, max_seq_len=max_seq_len,
            )
            for obj in _iter_jsonl(path)
        ]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self._rows[idx]


def _pad(seqs: list[list[int]], pad_value: int) -> torch.Tensor:
    width = max((len(s) for s in seqs), default=1)
    return torch.tensor(
        [s + [pad_value] * (width - len(s)) for s in seqs], dtype=torch.long
    )


def preference_collate(
    rows: list[dict[str, list[int]]], pad_id: int
) -> dict[str, torch.Tensor]:
    """Pad a batch of preference examples (chosen/rejected padded independently).

    Returns ``chosen_input_ids``, ``chosen_completion_mask``,
    ``chosen_attention_mask`` and the ``rejected_*`` counterparts.
    """
    out: dict[str, torch.Tensor] = {}
    for side in ("chosen", "rejected"):
        ids = _pad([r[f"{side}_input_ids"] for r in rows], pad_id)
        comp = _pad([r[f"{side}_completion_mask"] for r in rows], 0).float()
        attn = (ids != pad_id).long()
        out[f"{side}_input_ids"] = ids
        out[f"{side}_completion_mask"] = comp
        out[f"{side}_attention_mask"] = attn
    return out


class PromptDataset(Dataset):
    """Map-style dataset of GRPO prompts (+ optional reference answer)."""

    def __init__(
        self,
        path: str | Path,
        encode: Encode,
        bos_id: int | None = None,
        max_prompt_len: int | None = None,
    ) -> None:
        from .sft_data import generation_prompt_ids

        self._rows: list[dict] = []
        for obj in _iter_jsonl(path):
            ids = generation_prompt_ids(_as_messages(obj), encode, bos_id=bos_id)
            if max_prompt_len is not None:
                ids = ids[-max_prompt_len:]
            self._rows.append(
                {"prompt_ids": ids, "reference": obj.get("reference")}
            )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        return self._rows[idx]


class OCRPromptDataset(Dataset):
    """Strict real-photo OCR prompt dataset for image-conditioned GRPO.

    The prompt geometry exactly matches :func:`Model.ocr.data.build_ocr_row`:
    ``BOS, image_start, image_patch * N, image_end, optional instruction``.
    The reference is never appended to the prompt.  Split, unique id, and image
    checks make accidental golden-set leakage or a silently missing NAS mount a
    startup error instead of an online-training corruption.
    """

    def __init__(
        self,
        path: str | Path,
        encode: Encode,
        *,
        n_image_tokens: int,
        bos_id: int,
        image_start_id: int,
        image_patch_id: int,
        image_end_id: int,
        encode_reference: Encode | None = None,
        canonicalize_reference: CanonicalizeReference | None = None,
        image_root: str | Path | None = None,
        max_prompt_len: int | None = None,
        max_completion_len: int | None = None,
        max_seq_len: int | None = None,
        required_split: str | None = "rl_train",
        excluded_ids: set[str] | None = None,
        excluded_images: set[str] | None = None,
        excluded_sha256: set[str] | None = None,
        excluded_groups: set[str] | None = None,
        validate_images: bool = True,
        verify_image_decode: bool = False,
        require_sha256: bool = True,
        require_group_id: bool = False,
        require_domain: bool = False,
        verify_sha256: bool = True,
        inspect_reference_tokens: bool = True,
        retain_reference: bool = True,
        inspect_prompt_tokens: bool = True,
        allow_instruction: bool = True,
    ) -> None:
        if n_image_tokens <= 0:
            raise ValueError("n_image_tokens must be positive")
        source = Path(path)
        root = Path(image_root) if image_root is not None else source.parent
        excluded_ids = excluded_ids or set()
        excluded_images = excluded_images or set()
        excluded_sha256 = {value.lower() for value in (excluded_sha256 or set())}
        excluded_groups = excluded_groups or set()
        encode_reference = encode if encode_reference is None else encode_reference

        self._rows: list[dict] = []
        seen_ids: set[str] = set()
        seen_images: set[str] = set()
        seen_sha256: set[str] = set()
        for line_no, obj in enumerate(_iter_jsonl(source), start=1):
            sample_id = obj.get("id")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(f"{source}:{line_no}: non-empty string 'id' is required")
            sample_id = sample_id.strip()
            if sample_id in seen_ids:
                raise ValueError(f"{source}:{line_no}: duplicate id {sample_id!r}")
            if sample_id in excluded_ids:
                raise ValueError(
                    f"{source}:{line_no}: id {sample_id!r} overlaps the locked golden set"
                )

            group_id = obj.get("group_id")
            if group_id is None and not require_group_id:
                group_id = sample_id
            if not isinstance(group_id, str) or not group_id.strip():
                raise ValueError(
                    f"{source}:{line_no}: non-empty string 'group_id' is required "
                    "to prevent same-document leakage across splits"
                )
            group_id = group_id.strip()
            if group_id in excluded_groups:
                raise ValueError(
                    f"{source}:{line_no}: group_id {group_id!r} overlaps another split"
                )

            split = obj.get("split")
            if required_split is not None and split != required_split:
                raise ValueError(
                    f"{source}:{line_no}: split must be {required_split!r}, got {split!r}"
                )

            image_spec = obj.get("image", obj.get("image_path"))
            if image_spec is None and isinstance(obj.get("images"), list):
                images = obj["images"]
                if len(images) != 1:
                    raise ValueError(
                        f"{source}:{line_no}: OCR GRPO requires exactly one image"
                    )
                image_spec = images[0]
            if not isinstance(image_spec, str) or not image_spec:
                raise ValueError(f"{source}:{line_no}: string 'image' is required")
            image_path = Path(image_spec)
            if not image_path.is_absolute():
                image_path = root / image_path
            image_key = str(image_path.absolute())
            if image_key in seen_images:
                raise ValueError(f"{source}:{line_no}: duplicate image {image_key!r}")
            if image_key in excluded_images or image_spec in excluded_images:
                raise ValueError(
                    f"{source}:{line_no}: image overlaps the locked golden set: {image_key}"
                )
            if validate_images and not image_path.is_file():
                raise FileNotFoundError(f"{source}:{line_no}: image not found: {image_path}")
            if validate_images and verify_image_decode:
                try:
                    from PIL import Image

                    with Image.open(image_path) as image:
                        image.load()
                        if image.width <= 0 or image.height <= 0:
                            raise ValueError("image has zero width or height")
                except Exception as exc:
                    raise ValueError(
                        f"{source}:{line_no}: image cannot be decoded: "
                        f"{image_path}: {exc}"
                    ) from exc

            digest = obj.get("sha256")
            if digest is None and require_sha256:
                raise ValueError(f"{source}:{line_no}: sha256 is required")
            if digest is not None:
                if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                    raise ValueError(f"{source}:{line_no}: sha256 must be 64 hex characters")
                digest = digest.lower()
                if digest in seen_sha256:
                    raise ValueError(
                        f"{source}:{line_no}: duplicate image sha256 {digest}"
                    )
                if digest in excluded_sha256:
                    raise ValueError(
                        f"{source}:{line_no}: image hash overlaps the locked golden set"
                    )
                if verify_sha256:
                    hasher = hashlib.sha256()
                    with image_path.open("rb") as image_handle:
                        for chunk in iter(lambda: image_handle.read(1024 * 1024), b""):
                            hasher.update(chunk)
                    actual = hasher.hexdigest()
                    if actual != digest:
                        raise ValueError(
                            f"{source}:{line_no}: image sha256 mismatch: "
                            f"manifest={digest} actual={actual}"
                        )

            raw_reference = obj.get("reference")
            if not isinstance(raw_reference, str):
                raise ValueError(f"{source}:{line_no}: string 'reference' is required")
            if not raw_reference.strip():
                raise ValueError(f"{source}:{line_no}: reference must not be empty")
            if "\ufffd" in raw_reference:
                raise ValueError(
                    f"{source}:{line_no}: reference contains U+FFFD, which is "
                    "reserved as the reward-safe invalid-token sentinel"
                )
            reference = (
                canonicalize_reference(raw_reference)
                if canonicalize_reference is not None
                else raw_reference
            )
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError(
                    f"{source}:{line_no}: canonical reference must be a non-empty string"
                )
            if "\ufffd" in reference:
                raise ValueError(
                    f"{source}:{line_no}: canonical reference contains U+FFFD"
                )
            instruction = obj.get("instruction", "")
            if not isinstance(instruction, str):
                raise ValueError(f"{source}:{line_no}: instruction must be a string")
            if instruction and not allow_instruction:
                raise ValueError(
                    f"{source}:{line_no}: locked golden instructions are "
                    "forbidden because they can leak the reference"
                )
            domain = obj.get("domain")
            if domain is None and not require_domain:
                domain = "unknown"
            if not isinstance(domain, str) or not domain.strip():
                raise ValueError(
                    f"{source}:{line_no}: non-empty string 'domain' is required"
                )
            domain = domain.strip()
            instruction_ids = (
                [int(token) for token in encode(instruction)]
                if instruction and inspect_prompt_tokens
                else []
            )
            reference_ids = (
                [int(token) for token in encode_reference(reference)]
                if inspect_reference_tokens
                else None
            )
            # Leave one decode position for EOS.  A row that cannot possibly be
            # completed under max_new_tokens would otherwise receive a permanent
            # truncation penalty and poison group-relative advantages.
            if (
                max_completion_len is not None
                and reference_ids is not None
                and len(reference_ids) + 1 > max_completion_len
            ):
                raise ValueError(
                    f"{source}:{line_no}: reference needs {len(reference_ids) + 1} "
                    f"tokens including EOS, exceeds max_completion_len="
                    f"{max_completion_len}"
                )
            if image_patch_id in instruction_ids:
                raise ValueError(
                    f"{source}:{line_no}: instruction encodes an image_patch token"
                )

            prompt_ids = None
            if inspect_prompt_tokens:
                prompt_ids = (
                    [int(bos_id), int(image_start_id)]
                    + [int(image_patch_id)] * int(n_image_tokens)
                    + [int(image_end_id)]
                    + instruction_ids
                )
            if (
                prompt_ids is not None
                and max_prompt_len is not None
                and len(prompt_ids) > max_prompt_len
            ):
                raise ValueError(
                    f"{source}:{line_no}: OCR prompt length {len(prompt_ids)} exceeds "
                    f"max_prompt_len={max_prompt_len}; image slots must never be truncated"
                )
            if (
                max_seq_len is not None
                and max_completion_len is not None
                and prompt_ids is not None
                and len(prompt_ids) + max_completion_len > max_seq_len
            ):
                raise ValueError(
                    f"{source}:{line_no}: prompt length {len(prompt_ids)} + "
                    f"max_completion_len {max_completion_len} exceeds model "
                    f"max_seq_len={max_seq_len}; image-conditioned generation "
                    "cannot slide away visual slots"
                )
            if (
                max_seq_len is not None
                and prompt_ids is not None
                and len(prompt_ids) > max_seq_len
            ):
                raise ValueError(
                    f"{source}:{line_no}: OCR prompt length {len(prompt_ids)} "
                    f"exceeds model max_seq_len={max_seq_len}"
                )

            seen_ids.add(sample_id)
            seen_images.add(image_key)
            if digest is not None:
                seen_sha256.add(digest)
            row = {
                "id": sample_id,
                "group_id": group_id,
                "split": split,
                "reference_token_count": (
                    len(reference_ids) if reference_ids is not None else None
                ),
                "image": image_key,
                "sha256": digest,
                "domain": domain,
            }
            if prompt_ids is not None:
                row["prompt_ids"] = prompt_ids
            if retain_reference:
                row["reference"] = reference
                row["raw_reference"] = raw_reference
                row["reference_canonicalized"] = reference != raw_reference
            self._rows.append(row)

        if not self._rows:
            raise ValueError(f"OCR prompt dataset is empty: {source}")

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        return self._rows[idx]


__all__ = [
    "PreferenceDataset",
    "OCRPromptDataset",
    "PromptDataset",
    "build_preference_example",
    "preference_collate",
]
