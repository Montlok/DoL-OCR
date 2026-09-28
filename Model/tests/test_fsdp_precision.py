from unittest import mock

import pytest
import torch

from Model.config import TrainingConfig
from Model.layers.rope import MorphologicalRoPE
from Model.training.dist import wrap_fsdp


@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp16"])
def test_fsdp_preserves_frequency_buffers_and_reduces_in_fp32(precision):
    module = MorphologicalRoPE(8)
    expected = module.word_freqs.clone()
    cfg = TrainingConfig(fsdp_mixed_precision=precision)
    with (
        mock.patch("torch.distributed.fsdp.FullyShardedDataParallel") as fsdp,
        mock.patch("torch.cuda.is_available", return_value=False),
    ):
        wrap_fsdp(module, cfg)
    policy = fsdp.call_args.kwargs["mixed_precision"]
    assert policy.param_dtype == {
        "fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16,
    }[precision]
    assert policy.reduce_dtype == torch.float32
    assert policy.buffer_dtype == torch.float32
    assert policy.keep_low_precision_grads is False
    torch.testing.assert_close(module.word_freqs.to(policy.buffer_dtype), expected, rtol=0, atol=0)
