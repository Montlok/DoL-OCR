# -*- coding: utf-8 -*-

"""Image IO for the multimodal pretraining path.

Loads file paths / bytes / PIL images into ``[C, H, W]`` float tensors
that the OMVT multi-scale patcher can consume. The implementation is
**pure PIL + ``torch.frombuffer``** — no torchvision and no NumPy
dependency — so installing the ``[image]`` extra (just ``Pillow``) is
enough to enable the multimodal training path.

Designed to plug into ``MultimodalProcessor(image_processor=...)`` and
``PretrainingCollator(image_processor=...)`` — both call us with a list
of opaque image specs and expect a stacked ``[B, C, H, W]`` tensor back.
"""

from __future__ import annotations

import io
import os
from typing import Any, Sequence

import torch

try:
    from PIL import Image, ImageOps  # type: ignore
except ImportError as exc:  # pragma: no cover - hard fail at first use
    Image = None  # type: ignore
    ImageOps = None  # type: ignore
    _PIL_IMPORT_ERROR: Exception | None = exc
else:
    _PIL_IMPORT_ERROR = None


# ImageNet defaults — matches what most pretrained vision towers expect
# and gives a sensible normalization even when downstream OMVT trains
# from scratch (zero-mean unit-variance preconditions optimization).
_DEFAULT_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
_DEFAULT_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def _require_pil() -> None:
    if Image is None:
        raise ImportError(
            "Pillow is required for PILImageProcessor; install with "
            "`pip install Pillow` (it's the [image] extra of this repo)."
        ) from _PIL_IMPORT_ERROR


def _open_to_rgb(spec: Any, *, channels: int = 3) -> "Image.Image":
    """Materialise ``spec`` into a loaded RGB/L ``PIL.Image``.

    We wrap every disk/bytes opener in a context manager and call
    ``raw.load()`` before returning, so the underlying file descriptor
    closes deterministically. Without this the OS fd table can fill up
    when streaming tens of thousands of images per epoch. EXIF orientation is
    applied before conversion so phone photos are presented as users see them.
    """

    _require_pil()
    mode = "L" if channels == 1 else "RGB"
    if Image is not None and isinstance(spec, Image.Image):
        # Already an in-memory PIL image — convert eagerly so the caller
        # can drop the original reference.
        return ImageOps.exif_transpose(spec).convert(mode)
    if isinstance(spec, (bytes, bytearray, memoryview)):
        with Image.open(io.BytesIO(bytes(spec))) as raw:
            raw.load()
            return ImageOps.exif_transpose(raw).convert(mode)
    if isinstance(spec, (str, os.PathLike)):
        with Image.open(spec) as raw:
            raw.load()
            return ImageOps.exif_transpose(raw).convert(mode)
    if isinstance(spec, dict) and "bytes" in spec:
        with Image.open(io.BytesIO(bytes(spec["bytes"]))) as raw:
            raw.load()
            return ImageOps.exif_transpose(raw).convert(mode)
    if isinstance(spec, dict) and "path" in spec:
        with Image.open(spec["path"]) as raw:
            raw.load()
            return ImageOps.exif_transpose(raw).convert(mode)
    raise TypeError(
        f"unsupported image spec type {type(spec).__name__}; "
        "expected path / bytes / PIL.Image / dict with 'path' or 'bytes'."
    )


class PILImageProcessor:
    """File-path / bytes → ``[C, H, W]`` float tensor.

    Parameters
    ----------
    image_size:
        Final spatial size; the image is resized to a square so the OMVT
        patch shapes (which are fixed at config time) can divide it.
    in_channels:
        ``1`` (grayscale) or ``3`` (RGB).
    mean, std:
        Per-channel normalization. ``None`` ⇒ no normalization.
    resample:
        PIL resampling filter (default ``BILINEAR``).
    """

    def __init__(
        self,
        image_size: int = 224,
        *,
        in_channels: int = 3,
        mean: Sequence[float] | None = _DEFAULT_MEAN,
        std: Sequence[float] | None = _DEFAULT_STD,
        resample: int | None = None,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if in_channels not in (1, 3):
            raise ValueError("in_channels must be 1 or 3")
        if mean is not None and len(mean) != in_channels:
            raise ValueError("mean must have one entry per channel")
        if std is not None and len(std) != in_channels:
            raise ValueError("std must have one entry per channel")
        if (mean is None) ^ (std is None):
            raise ValueError("mean and std must be set together or both None")
        self.image_size = int(image_size)
        self.in_channels = int(in_channels)
        self.mean = tuple(float(v) for v in mean) if mean is not None else None
        self.std = tuple(float(v) for v in std) if std is not None else None
        self._resample = resample  # resolved lazily so import errors stay deferred

    def _resolve_resample(self) -> int:
        if self._resample is not None:
            return self._resample
        # Pillow ≥9 uses Image.Resampling; older builds expose the constants
        # directly on Image. Support both transparently.
        if hasattr(Image, "Resampling"):
            return Image.Resampling.BILINEAR
        return getattr(Image, "BILINEAR", 2)

    def _single(self, spec: Any) -> torch.Tensor:
        img = _open_to_rgb(spec, channels=self.in_channels)
        if img.size != (self.image_size, self.image_size):
            img = img.resize(
                (self.image_size, self.image_size), self._resolve_resample()
            )
        # Avoid pulling numpy in if torch tensor-from-bytes is enough.
        # ``Image.tobytes()`` returns row-major HWC for RGB or HW for L.
        raw = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
        if self.in_channels == 3:
            tensor = raw.reshape(self.image_size, self.image_size, 3).permute(2, 0, 1)
        else:
            tensor = raw.reshape(1, self.image_size, self.image_size)
        tensor = tensor.to(torch.float32) / 255.0
        if self.mean is not None and self.std is not None:
            mean_t = torch.tensor(self.mean, dtype=tensor.dtype).view(-1, 1, 1)
            std_t = torch.tensor(self.std, dtype=tensor.dtype).view(-1, 1, 1)
            tensor = (tensor - mean_t) / std_t
        return tensor

    def __call__(self, images: Sequence[Any]) -> torch.Tensor:
        if not images:
            return torch.empty(0, self.in_channels, self.image_size, self.image_size)
        return torch.stack([self._single(spec) for spec in images], dim=0)


__all__ = ["PILImageProcessor"]
