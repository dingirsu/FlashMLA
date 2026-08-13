import pytest
import torch

from small_head_test_utils import get_extension
from quant import FP8KVCacheLayout, dequantize_k_cache, quantize_k_cache


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires an NVIDIA Blackwell GPU",
)


def _reference(q, kv, indices, sm_scale, attn_sink, topk_length):
    s_q, h_q, _ = q.shape
    topk = indices.shape[-1]
    flat_indices = indices[:, 0].clone()
    positions = torch.arange(topk, device=q.device).unsqueeze(0)
    valid = (
        (positions < topk_length.unsqueeze(1))
        & (flat_indices >= 0)
        & (flat_indices < kv.shape[0])
    )
    flat_indices.masked_fill_(~valid, 0)

    gathered = kv[:, 0].index_select(0, flat_indices.flatten()).view(s_q, topk, -1).float()
    scores = torch.einsum("shd,std->sht", q.float(), gathered) * sm_scale
    scores.masked_fill_(~valid.unsqueeze(1), -torch.inf)

    lse = torch.logsumexp(scores, dim=-1)
    max_logits = scores.amax(dim=-1)
    denominator = torch.logaddexp(lse, attn_sink.unsqueeze(0).expand(s_q, h_q))
    probabilities = torch.exp(scores - denominator.unsqueeze(-1))
    out = torch.einsum("sht,std->shd", probabilities, gathered)
    return out, max_logits, lse


@torch.inference_mode()
def test_sm100_sparse_fwd_for_small_topk_head128():
    torch.manual_seed(1234)
    device = torch.device("cuda")
    s_q, s_kv, h_q, d_qk, topk = 3, 79, 128, 512, 64
    sm_scale = d_qk**-0.5

    q = (torch.randn(s_q, h_q, d_qk, device=device) * 0.25).to(torch.bfloat16)
    kv = (torch.randn(s_kv, 1, d_qk, device=device) * 0.25).to(torch.bfloat16)
    indices = torch.randint(0, s_kv, (s_q, 1, topk), device=device, dtype=torch.int32)
    indices[0, 0, 3] = -1
    indices[1, 0, 7] = s_kv + 5
    indices[2, 0, 8] = indices[2, 0, 2]
    topk_length = torch.tensor([64, 47, 23], device=device, dtype=torch.int32)
    attn_sink = torch.linspace(-1.0, 1.0, h_q, device=device, dtype=torch.float32)

    out, max_logits, lse = get_extension().small_topk_fwd(
        q,
        kv,
        indices,
        sm_scale,
        attn_sink,
        topk_length,
    )
    ref_out, ref_max_logits, ref_lse = _reference(
        q, kv, indices, sm_scale, attn_sink, topk_length
    )

    torch.testing.assert_close(out.float(), ref_out, atol=2e-3, rtol=3e-2)
    torch.testing.assert_close(max_logits, ref_max_logits, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(lse, ref_lse, atol=2e-5, rtol=2e-5)


@torch.inference_mode()
@pytest.mark.parametrize("topk", [64, 192])
def test_sm100_sparse_fwd_for_small_topk_paired_head64(topk):
    torch.manual_seed(5678 + topk)
    device = torch.device("cuda")
    s_q, s_kv, h_q, d_qk = 6, 211, 64, 512
    sm_scale = d_qk**-0.5

    q = (torch.randn(s_q, h_q, d_qk, device=device) * 0.25).to(torch.bfloat16)
    kv = (torch.randn(s_kv, 1, d_qk, device=device) * 0.25).to(torch.bfloat16)
    pair_indices = torch.randint(
        0, s_kv, (s_q // 2, 1, topk), device=device, dtype=torch.int32
    )
    pair_indices[0, 0, 3] = -1
    pair_indices[1, 0, 7] = s_kv + 5
    pair_indices[2, 0, 8] = pair_indices[2, 0, 2]
    pair_topk_length = torch.tensor(
        [topk, max(topk - 17, 0), max(topk - 41, 0)],
        device=device,
        dtype=torch.int32,
    )
    attn_sink = torch.linspace(-1.0, 1.0, h_q, device=device, dtype=torch.float32)

    out, max_logits, lse = get_extension().small_topk_head64_fwd(
        q,
        kv,
        pair_indices,
        sm_scale,
        attn_sink,
        pair_topk_length,
    )

    token_to_pair = torch.arange(s_q, device=device) // 2
    token_indices = pair_indices.index_select(0, token_to_pair)
    token_topk_length = pair_topk_length.index_select(0, token_to_pair)
    ref_out, ref_max_logits, ref_lse = _reference(
        q, kv, token_indices, sm_scale, attn_sink, token_topk_length
    )

    torch.testing.assert_close(out.float(), ref_out, atol=2e-3, rtol=3e-2)
    torch.testing.assert_close(max_logits, ref_max_logits, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(lse, ref_lse, atol=2e-5, rtol=2e-5)


@torch.inference_mode()
def test_sm100_sparse_small_topk_head64_decode_odd_s_q():
    torch.manual_seed(9013)
    device = torch.device("cuda")
    batch, s_q, s_kv, page_size, topk = 2, 3, 191, 64, 128
    num_blocks = (s_kv + page_size - 1) // page_size
    sm_scale = 512**-0.5

    q = (torch.randn(batch, s_q, 64, 512, device=device) * 0.1).to(torch.bfloat16)
    kv_bf16 = (
        torch.randn(num_blocks, page_size, 1, 512, device=device) * 0.1
    ).to(torch.bfloat16)
    kv = quantize_k_cache(kv_bf16, FP8KVCacheLayout.MODEL1_FP8Sparse)
    pair_indices = torch.randint(
        0,
        s_kv,
        (batch, (s_q + 1) // 2, topk),
        device=device,
        dtype=torch.int32,
    )
    topk_length = torch.tensor([93, 128], device=device, dtype=torch.int32)
    attn_sink = torch.linspace(-1.0, 1.0, 64, device=device)

    scheduler_metadata = torch.zeros((1, 8), device=device, dtype=torch.int32)
    scheduler_metadata[0, 1] = batch - 1
    scheduler_metadata[0, 3] = topk // 64
    num_splits = torch.zeros(batch + 1, device=device, dtype=torch.int32)
    out, lse = get_extension().small_topk_head64_decode(
        q,
        kv,
        pair_indices,
        sm_scale,
        attn_sink,
        topk_length,
        scheduler_metadata,
        num_splits,
    )

    dequantized_kv = dequantize_k_cache(
        kv, FP8KVCacheLayout.MODEL1_FP8Sparse
    ).flatten(0, 2).float()
    token_to_pair = torch.arange(s_q, device=device) // 2
    token_indices = pair_indices.index_select(1, token_to_pair).long()
    gathered = dequantized_kv[token_indices]
    scores = torch.einsum("bshd,bstd->bsht", q.float(), gathered) * sm_scale
    positions = torch.arange(topk, device=device)[None, None, None, :]
    scores.masked_fill_(positions >= topk_length[:, None, None, None], -torch.inf)
    ref_lse = torch.logsumexp(scores, dim=-1)
    denominator = torch.logaddexp(ref_lse, attn_sink[None, None, :])
    probabilities = torch.exp(scores - denominator.unsqueeze(-1))
    ref_out = torch.einsum("bsht,bstd->bshd", probabilities, gathered)

    assert out.shape == (batch, s_q, 64, 512)
    assert lse.shape == (batch, s_q, 64)
    torch.testing.assert_close(out.float(), ref_out, atol=2e-3, rtol=3e-2)
    torch.testing.assert_close(lse, ref_lse, atol=3e-5, rtol=5e-5)
