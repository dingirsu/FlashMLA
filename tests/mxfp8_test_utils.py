import math
from typing import Optional, Tuple

import torch


D_HEAD = 512
FP8_MAX = 448.0
Q_GROUP_SIZE = 32
KV_GROUP_SIZE = 64
Q_BYTES_PER_TOKEN = D_HEAD + D_HEAD // Q_GROUP_SIZE
KV_BYTES_PER_TOKEN = D_HEAD + D_HEAD // KV_GROUP_SIZE


def require_sm100_family() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("MXFP8 precision tests require an SM100-family CUDA device")
    major, _ = torch.cuda.get_device_capability()
    if major != 10:
        raise RuntimeError("MXFP8 precision tests require an SM100-family CUDA device")


def _quantize_groups(
    x: torch.Tensor, group_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.shape[-1] == D_HEAD
    grouped = x.float().reshape(*x.shape[:-1], D_HEAD // group_size, group_size)
    scale = grouped.abs().amax(dim=-1) / FP8_MAX
    scale = torch.pow(2.0, torch.ceil(torch.log2(scale.clamp_min(2.0**-126))))
    scale_e8m0 = scale.to(torch.float8_e8m0fnu)
    scale_fp32 = scale_e8m0.float()
    quantized = (grouped / scale_fp32.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    quantized = quantized.to(torch.float8_e4m3fn)
    dequantized = (quantized.float() * scale_fp32.unsqueeze(-1)).reshape_as(x.float())
    return (
        quantized.reshape_as(x).view(torch.uint8),
        scale_e8m0.view(torch.uint8),
        dequantized,
    )


def pack_q(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    data, scales, dequantized = _quantize_groups(q, Q_GROUP_SIZE)
    packed = torch.empty(
        (*q.shape[:-1], Q_BYTES_PER_TOKEN), dtype=torch.uint8, device=q.device
    )
    packed[..., :D_HEAD] = data
    packed[..., D_HEAD:] = scales
    return packed, dequantized


def pack_prefill_kv(kv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack the complete prefill KV allocation as one page-tail-scale page."""
    assert kv.ndim == 3
    data, scales, dequantized = _quantize_groups(kv, KV_GROUP_SIZE)
    num_tokens = kv.shape[0] * kv.shape[1]
    storage = torch.empty(
        num_tokens * KV_BYTES_PER_TOKEN, dtype=torch.uint8, device=kv.device
    )
    storage[: num_tokens * D_HEAD] = data.reshape(-1)
    storage[num_tokens * D_HEAD :] = scales.reshape(-1)
    return storage.view(*kv.shape[:-1], KV_BYTES_PER_TOKEN), dequantized


def pack_prefill_kv_rank1(
    kv: torch.Tensor,
    w_exponents: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize KV with VSF(token, group) = U(token) * W(group)."""
    assert kv.ndim == 3 and kv.shape[-1] == D_HEAD
    num_groups = D_HEAD // KV_GROUP_SIZE
    grouped = kv.float().reshape(*kv.shape[:-1], num_groups, KV_GROUP_SIZE)

    if w_exponents is None:
        w_exponents = torch.tensor(
            [2, 1, 3, 0, 4, 2, 1, 3],
            dtype=torch.int32,
            device=kv.device,
        )
    else:
        w_exponents = w_exponents.to(device=kv.device, dtype=torch.int32)
    assert tuple(w_exponents.shape) == (num_groups,)

    raw_scale = grouped.abs().amax(dim=-1) / FP8_MAX
    required_exp = torch.ceil(
        torch.log2(raw_scale.clamp_min(2.0**-126))
    ).to(torch.int32)
    u_exponents = (required_exp - w_exponents).amax(dim=-1, keepdim=True)
    product_exponents = u_exponents + w_exponents
    if product_exponents.amin().item() < -126 or product_exponents.amax().item() > 127:
        raise ValueError("rank-1 product scale is outside the UE8M0 test range")

    product_scale = torch.exp2(product_exponents.float()).to(torch.float8_e8m0fnu)
    product_scale_f32 = product_scale.float()
    quantized = (
        grouped / product_scale_f32.unsqueeze(-1)
    ).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    dequantized = (
        quantized.float() * product_scale_f32.unsqueeze(-1)
    ).reshape_as(kv.float())

    w_scale = torch.exp2(w_exponents.float()).to(torch.float8_e8m0fnu)
    u_scale = torch.exp2(u_exponents.squeeze(-1).float())
    num_tokens = kv.shape[0] * kv.shape[1]
    storage = torch.empty(
        num_tokens * KV_BYTES_PER_TOKEN, dtype=torch.uint8, device=kv.device
    )
    storage[: num_tokens * D_HEAD] = quantized.reshape_as(kv).view(torch.uint8).reshape(-1)
    storage[num_tokens * D_HEAD :] = product_scale.view(torch.uint8).reshape(-1)
    return (
        storage.view(*kv.shape[:-1], KV_BYTES_PER_TOKEN),
        dequantized,
        w_scale,
        u_scale,
        quantized.reshape_as(kv).float(),
    )


def pack_decode_kv_pages(kv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack every decode cache page as [all data rows][all scale rows]."""
    assert kv.ndim == 4 and kv.shape[2] == 1
    data, scales, dequantized = _quantize_groups(kv, KV_GROUP_SIZE)
    num_pages, page_size, h_kv, _ = kv.shape
    tokens_per_page = page_size * h_kv
    storage = torch.empty(
        (num_pages, tokens_per_page * KV_BYTES_PER_TOKEN),
        dtype=torch.uint8,
        device=kv.device,
    )
    storage[:, : tokens_per_page * D_HEAD] = data.reshape(num_pages, -1)
    storage[:, tokens_per_page * D_HEAD :] = scales.reshape(num_pages, -1)
    return storage.view(num_pages, page_size, h_kv, KV_BYTES_PER_TOKEN), dequantized


def make_indices(
    rows: int, topk: int, s_kv: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert rows >= 3 and topk == 128 and s_kv >= topk
    indices = torch.empty((rows, topk), dtype=torch.int32, device=device)
    indices[0] = torch.randperm(s_kv, device=device)[:topk].to(torch.int32)
    indices[1] = torch.randperm(s_kv, device=device)[:topk].to(torch.int32)
    indices[1, 7] = -1
    indices[1, 19] = s_kv + 11
    indices[2] = -1
    if rows > 3:
        indices[3:] = indices[0]
    topk_length = torch.full((rows,), topk, dtype=torch.int32, device=device)
    topk_length[1] = 73
    return indices, topk_length


def attention_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    sm_scale: float,
    attn_sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference for MQA inputs with arbitrary leading Q dimensions."""
    assert kv.shape[-2] == 1
    q_leading = q.shape[:-2]
    h_q = q.shape[-2]
    topk = indices.shape[-1]
    q_flat = q.float().reshape(-1, h_q, D_HEAD)
    indices_flat = indices.reshape(-1, topk)
    lengths_flat = topk_length.reshape(-1)
    kv_flat = kv.float().reshape(-1, D_HEAD)

    positions = torch.arange(topk, device=q.device).view(1, topk)
    valid = (
        (indices_flat >= 0)
        & (indices_flat < kv_flat.shape[0])
        & (positions < lengths_flat.view(-1, 1))
    )
    safe_indices = indices_flat.clamp(0, kv_flat.shape[0] - 1).to(torch.long)
    gathered = kv_flat.index_select(0, safe_indices.reshape(-1)).reshape(
        q_flat.shape[0], topk, D_HEAD
    )
    scores = torch.matmul(q_flat, gathered.transpose(-1, -2)) * sm_scale
    scores.masked_fill_(~valid.unsqueeze(1), -math.inf)

    max_logits = scores.amax(dim=-1)
    raw_lse = torch.logsumexp(scores, dim=-1)
    finite_lse = torch.isfinite(raw_lse)
    weights = torch.zeros_like(scores)
    weights[finite_lse] = torch.exp(scores[finite_lse] - raw_lse[finite_lse, None])
    out = torch.matmul(weights, gathered)
    if attn_sink is not None:
        sink_factor = torch.sigmoid(raw_lse - attn_sink.float().view(1, h_q))
        out *= torch.where(finite_lse, sink_factor, 0.0).unsqueeze(-1)

    returned_lse = torch.where(finite_lse, raw_lse, math.inf)
    return (
        out.reshape(*q_leading, h_q, D_HEAD),
        max_logits.reshape(*q_leading, h_q),
        returned_lse.reshape(*q_leading, h_q),
    )


def assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
) -> None:
    actual_f = actual.float()
    expected_f = expected.float()
    assert torch.equal(torch.isposinf(actual_f), torch.isposinf(expected_f)), (
        f"{name}: +inf mask mismatch"
    )
    assert torch.equal(torch.isneginf(actual_f), torch.isneginf(expected_f)), (
        f"{name}: -inf mask mismatch"
    )
    finite = torch.isfinite(expected_f)
    if not torch.allclose(actual_f[finite], expected_f[finite], atol=atol, rtol=rtol):
        diff = (actual_f[finite] - expected_f[finite]).abs()
        raise AssertionError(
            f"{name}: max_abs={diff.max().item():.6g}, "
            f"mean_abs={diff.mean().item():.6g}, atol={atol}, rtol={rtol}"
        )
