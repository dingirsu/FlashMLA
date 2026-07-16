import pytest
import torch

from small_head_test_utils import get_extension


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
