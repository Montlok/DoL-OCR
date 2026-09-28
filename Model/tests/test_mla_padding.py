from unittest import mock

import pytest
import torch

from Model.config import tiny_config
from Model.inference.cache import MLACache
from Model.layers.mla import MLA


def make_mla(sdpa=True):
    cfg = tiny_config()
    cfg.use_sdpa_attention = sdpa
    cfg.dropout = 0.0
    return MLA(cfg)


@pytest.mark.parametrize("causal", [True, False])
def test_padding_preserves_outputs_and_gradients(causal):
    torch.manual_seed(12)
    fast, reference = make_mla(), make_mla(False)
    reference.load_state_dict(fast.state_dict())
    mask = torch.tensor([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ])
    x = torch.randn(5, 6, fast.d_model, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    actual = fast(x, attn_mask=mask, causal=causal)
    expected = reference(ref_x, attn_mask=mask, causal=causal)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad, atol=2e-6, rtol=2e-5)
    for (name, param), (_, ref_param) in zip(
        fast.named_parameters(), reference.named_parameters(),
    ):
        assert param.grad is not None, name
        torch.testing.assert_close(param.grad, ref_param.grad, atol=3e-5, rtol=2e-4)


def test_causal_padding_uses_no_dense_mask_or_host_sync():
    layer = make_mla()
    x = torch.randn(2, 128, layer.d_model)
    mask = torch.ones(2, 128, dtype=torch.long)
    mask[0, ::3] = 0
    original = torch.nn.functional.scaled_dot_product_attention
    with (
        mock.patch("Model.layers.mla.F.scaled_dot_product_attention", wraps=original) as sdpa,
        mock.patch("Model.layers.mla.torch.ones", side_effect=AssertionError("dense mask")),
        mock.patch.object(torch.Tensor, "item", side_effect=AssertionError("host sync")),
        mock.patch.object(torch.Tensor, "nonzero", side_effect=AssertionError("host sync")),
        mock.patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("host sync")),
    ):
        result = layer(x, attn_mask=mask)
    assert result.shape == x.shape
    assert sdpa.call_count == 1
    assert sdpa.call_args.kwargs.get("attn_mask") is None
    assert sdpa.call_args.kwargs["is_causal"] is True


def test_million_token_padding_bookkeeping_on_meta_device():
    # Shape-only regression, not a million-token GPU capacity/quality test.
    layer = make_mla().to("meta")
    x = torch.empty(1, 1_048_576, layer.d_model, device="meta")
    mask = torch.ones(1, 1_048_576, dtype=torch.long, device="meta")
    with (
        mock.patch(
            "Model.layers.mla.F.scaled_dot_product_attention",
            side_effect=lambda q, k, v, **kwargs: torch.empty_like(q),
        ) as sdpa,
        mock.patch("Model.layers.mla.torch.ones", side_effect=AssertionError("dense mask")),
    ):
        output = layer(x, attn_mask=mask)
    assert output.shape == x.shape
    assert sdpa.call_args.kwargs.get("attn_mask") is None


def test_cached_multitoken_continuations_preserve_causality_with_bounded_masks():
    torch.manual_seed(3)
    layer = make_mla().eval()
    x = torch.randn(2, 11, layer.d_model)
    with torch.no_grad():
        expected = layer(x)
        cache = MLACache()
        original = torch.nn.functional.scaled_dot_product_attention
        with (
            mock.patch("Model.layers.mla.F.scaled_dot_product_attention", wraps=original) as sdpa,
            mock.patch("Model.layers.mla.torch.ones", side_effect=AssertionError("dense mask")),
        ):
            pieces = [
                layer(x[:, start:end], cache=cache, pos_offset=start)
                for start, end in [(0, 4), (4, 7), (7, 8), (8, 11)]
            ]
    torch.testing.assert_close(torch.cat(pieces, 1), expected, atol=2e-6, rtol=2e-5)
    assert sdpa.call_count == 4
    assert all(
        call.kwargs.get("attn_mask") is None
        or call.kwargs["attn_mask"].numel() <= 8 * 1024 * 1024
        for call in sdpa.call_args_list
    )


def test_cached_fallback_batches_queries_and_bounds_each_tile():
    torch.manual_seed(9)
    fast, reference = make_mla(), make_mla(False)
    q = torch.randn(1, 2, 128, 16)
    k, v = torch.randn(1, 2, 1152, 16), torch.randn(1, 2, 1152, 16)
    expected = reference._attention_cached(q, k, v, past_len=1024)
    original = torch.nn.functional.scaled_dot_product_attention
    with mock.patch("Model.layers.mla.F.scaled_dot_product_attention", wraps=original) as sdpa:
        actual = fast._attention_cached(q, k, v, past_len=1024)
    assert sdpa.call_count == 1
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    budget = 16 * k.shape[-2]
    with (
        mock.patch("Model.layers.mla._MAX_CACHED_MASK_ELEMENTS", budget),
        mock.patch("Model.layers.mla.F.scaled_dot_product_attention", wraps=original) as sdpa,
    ):
        tiled = fast._attention_cached(q, k, v, past_len=1024)
    assert sdpa.call_count == 8
    assert all(c.kwargs["attn_mask"].numel() <= budget for c in sdpa.call_args_list)
    torch.testing.assert_close(tiled, expected, atol=2e-6, rtol=2e-5)
