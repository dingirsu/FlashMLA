import pytest
import torch

from small_head_test_utils import get_extension


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires an NVIDIA Blackwell GPU",
)


def _assert_gradient_close(name, actual, expected):
    try:
        torch.testing.assert_close(actual.float(), expected.float(), atol=4e-3, rtol=3e-2)
    except AssertionError as error:
        difference = (actual.float() - expected.float()).abs()
        raise AssertionError(
            f"{name}: max_abs={difference.max().item():.6g}, "
            f"mean_abs={difference.mean().item():.6g}, "
            f"actual_absmax={actual.float().abs().max().item():.6g}, "
            f"expected_absmax={expected.float().abs().max().item():.6g}"
        ) from error


def _forward_and_backward_reference(
    q, kv, indices, d_out, sm_scale, attn_sink, topk_length
):
    s_q, h_q, d_qk = q.shape
    s_kv = kv.shape[0]
    d_v = d_out.shape[-1]
    topk = indices.shape[-1]

    flat_indices = indices[:, 0].clone()
    positions = torch.arange(topk, device=q.device).unsqueeze(0)
    valid = (
        (positions < topk_length.unsqueeze(1))
        & (flat_indices >= 0)
        & (flat_indices < s_kv)
    )
    flat_indices.masked_fill_(~valid, 0)

    q_float = q.float()
    kv_float = kv[:, 0].float()
    d_out_float = d_out.float()
    gathered = kv_float.index_select(0, flat_indices.flatten()).view(s_q, topk, d_qk)
    scores = torch.einsum("shd,std->sht", q_float, gathered) * sm_scale
    scores.masked_fill_(~valid.unsqueeze(1), -torch.inf)
    lse = torch.logsumexp(scores, dim=-1)
    denominator = torch.logaddexp(lse, attn_sink.unsqueeze(0).expand(s_q, h_q))
    probability = torch.exp(scores - denominator.unsqueeze(-1))

    # The CUDA API receives BF16 forward output, so dS uses the rounded O.
    out = torch.einsum("sht,std->shd", probability, gathered[..., :d_v]).to(torch.bfloat16)
    sum_odo = torch.sum(out.float() * d_out_float, dim=-1)
    dp = torch.einsum("shd,std->sht", d_out_float, gathered[..., :d_v])
    d_score = probability * (dp - sum_odo.unsqueeze(-1)) * sm_scale

    dq = torch.einsum("sht,std->shd", d_score, gathered)
    dk = torch.zeros(s_kv, 1, d_qk, device=q.device, dtype=torch.float32)
    dv = torch.zeros(s_kv, 1, d_v, device=q.device, dtype=torch.float32)
    for s in range(s_q):
        for k in range(topk_length[s].item()):
            index = indices[s, 0, k].item()
            if 0 <= index < s_kv:
                dk[index, 0] += torch.einsum("h,hd->d", d_score[s, :, k], q_float[s])
                dv[index, 0] += torch.einsum("h,hd->d", probability[s, :, k], d_out_float[s])

    sink_probability = torch.exp(attn_sink.unsqueeze(0) - denominator)
    d_attn_sink = torch.sum(-sink_probability * sum_odo, dim=0)
    return out, lse, tuple(x.to(torch.bfloat16) for x in (dq, dk, dv)), d_attn_sink


@pytest.mark.parametrize("d_qk,h_q", [(128, 8), (128, 16), (128, 32)])
def test_sm100_sparse_head_small_backward(d_qk, h_q):
    torch.manual_seed(5678 + d_qk)
    device = torch.device("cuda")
    s_q, s_kv, d_v, topk = 3, 73, 128, 64
    sm_scale = d_qk**-0.5

    q = (torch.randn(s_q, h_q, d_qk, device=device) * 0.2).to(torch.bfloat16)
    kv = (torch.randn(s_kv, 1, d_qk, device=device) * 0.2).to(torch.bfloat16)
    d_out = (torch.randn(s_q, h_q, d_v, device=device) * 0.2).to(torch.bfloat16)
    indices = torch.randint(0, s_kv, (s_q, 1, topk), device=device, dtype=torch.int32)
    indices[0, 0, 5] = -1
    indices[1, 0, 9] = s_kv + 2
    indices[2, 0, 11] = indices[2, 0, 3]
    topk_length = torch.tensor([64, 43, 19], device=device, dtype=torch.int32)
    attn_sink = torch.linspace(-0.75, 0.75, h_q, device=device, dtype=torch.float32)

    out, lse, expected_grads, expected_d_sink = _forward_and_backward_reference(
        q, kv, indices, d_out, sm_scale, attn_sink, topk_length
    )
    actual_grads = get_extension().head_small_bwd(
        d_out,
        q,
        kv,
        out,
        lse,
        indices,
        sm_scale,
        attn_sink,
        topk_length,
    )

    for name, actual, expected in zip(("dq", "dk", "dv"), actual_grads[:3], expected_grads):
        _assert_gradient_close(name, actual, expected)
    torch.testing.assert_close(
        actual_grads[3], expected_d_sink, atol=2e-4, rtol=2e-4
    )
