import math
from typing import Optional, Tuple

import torch


D_HEAD = 512
FP8_MAX = 448.0
Q_GROUP_SIZE = 32
KV_GROUP_SIZE = 64
Q_BYTES_PER_TOKEN = D_HEAD + D_HEAD // Q_GROUP_SIZE
KV_BYTES_PER_TOKEN = D_HEAD + D_HEAD // KV_GROUP_SIZE


def pack_e8m0x4_as_uint32(scales: torch.Tensor) -> int:
    """Pack four UE8M0 bytes into one little-endian uint32 argument."""
    bits = scales.detach().to(device="cpu", dtype=torch.uint8).tolist()
    assert len(bits) == 4
    return int.from_bytes(bytes(bits), byteorder="little", signed=False)


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


def pack_q(
    q: torch.Tensor, group_size: int = Q_GROUP_SIZE, scale_slot_bytes: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    data, scales, dequantized = _quantize_groups(q, group_size)
    num_scale_bytes = D_HEAD // group_size
    if scale_slot_bytes == 0:
        scale_slot_bytes = num_scale_bytes
    assert scale_slot_bytes >= num_scale_bytes
    q_bytes_per_token = D_HEAD + scale_slot_bytes
    packed = torch.empty(
        (*q.shape[:-1], q_bytes_per_token), dtype=torch.uint8, device=q.device
    )
    packed[..., :D_HEAD] = data
    packed[..., D_HEAD : D_HEAD + num_scale_bytes] = scales
    packed[..., D_HEAD + num_scale_bytes :] = 0
    return packed, dequantized


def pack_dual_q64(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack Q with one scale for corresponding 32-D chunks in both halves."""
    assert q.shape[-1] == D_HEAD
    paired = (
        q.float()
        .reshape(*q.shape[:-1], 2, D_HEAD // 64, 32)
        .transpose(-3, -2)
        .reshape(*q.shape[:-1], D_HEAD // 64, 64)
    )
    scale = paired.abs().amax(dim=-1) / FP8_MAX
    scale = torch.pow(2.0, torch.ceil(torch.log2(scale.clamp_min(2.0**-126))))
    scale_e8m0 = scale.to(torch.float8_e8m0fnu)
    scale_fp32 = scale_e8m0.float()
    quantized = (paired / scale_fp32.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    quantized = quantized.to(torch.float8_e4m3fn)
    data = quantized.view(torch.uint8)
    scales = scale_e8m0.view(torch.uint8)
    dequantized = quantized.float() * scale_fp32.unsqueeze(-1)
    data = (
        data.reshape(*q.shape[:-1], D_HEAD // 64, 2, 32)
        .transpose(-3, -2)
        .reshape_as(q)
    )
    dequantized = (
        dequantized.reshape(*q.shape[:-1], D_HEAD // 64, 2, 32)
        .transpose(-3, -2)
        .reshape_as(q.float())
    )
    packed = torch.empty(
        (*q.shape[:-1], D_HEAD + 16), dtype=torch.uint8, device=q.device
    )
    packed[..., :D_HEAD] = data
    packed[..., D_HEAD:] = torch.cat((scales, scales), dim=-1)
    return packed, dequantized


def pack_dual_prefill_kv64(kv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack dual-MXFP8 KV with replicated 16B scale slots in the tail plane."""
    assert kv.ndim == 3
    data, scales, dequantized = _quantize_groups(kv, KV_GROUP_SIZE)
    scale_slots = torch.cat((scales, scales), dim=-1)
    num_tokens = kv.shape[0] * kv.shape[1]
    bytes_per_token = D_HEAD + scale_slots.shape[-1]
    storage = torch.empty(
        num_tokens * bytes_per_token, dtype=torch.uint8, device=kv.device
    )
    storage[: num_tokens * D_HEAD] = data.reshape(-1)
    storage[num_tokens * D_HEAD :] = scale_slots.reshape(-1)
    return storage.view(*kv.shape[:-1], bytes_per_token), dequantized


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
            [0, 1, 3, 0, 4, 2, 1, 3],
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


def pack_decode_kv_pages_rank1(
    kv: torch.Tensor,
    w_exponents: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack decode pages with VSF(token, group) = U(token) * W(group)."""
    assert kv.ndim == 4 and kv.shape[2] == 1 and kv.shape[-1] == D_HEAD
    num_groups = D_HEAD // KV_GROUP_SIZE
    grouped = kv.float().reshape(*kv.shape[:-1], num_groups, KV_GROUP_SIZE)

    if w_exponents is None:
        w_exponents = torch.tensor(
            [0, 1, 3, 0, 4, 2, 1, 3],
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
    num_pages, page_size, h_kv, _ = kv.shape
    tokens_per_page = page_size * h_kv
    storage = torch.empty(
        (num_pages, tokens_per_page * KV_BYTES_PER_TOKEN),
        dtype=torch.uint8,
        device=kv.device,
    )
    storage[:, : tokens_per_page * D_HEAD] = quantized.reshape(num_pages, -1).view(torch.uint8)
    storage[:, tokens_per_page * D_HEAD :] = product_scale.view(torch.uint8).reshape(num_pages, -1)
    return (
        storage.view(num_pages, page_size, h_kv, KV_BYTES_PER_TOKEN),
        dequantized,
        w_scale,
        u_scale,
        quantized.reshape_as(kv).float(),
    )


def pack_dual_decode_kv_pages_rank1(
    kv: torch.Tensor,
    w_exponents: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack head64 dual-MXFP8 pages with replicated 16B scale slots.

    The physical page is laid out as one contiguous 512B E4M3 data plane
    followed by one 16B ``[scale[0:8], scale[0:8]]`` slot per token.  The
    tensor's last dimension is therefore a 528B storage envelope; it is not
    an interleaved per-row representation of the page.
    """
    packed, dequantized, w_scale, u_scale, quantized = pack_decode_kv_pages_rank1(
        kv, w_exponents=w_exponents
    )
    num_pages, page_size, h_kv, _ = kv.shape
    num_tokens = page_size * h_kv
    flat = packed.reshape(num_pages, -1)
    data_bytes = num_tokens * D_HEAD
    scale_bytes = num_tokens * (D_HEAD // KV_GROUP_SIZE)
    scales = flat[:, data_bytes : data_bytes + scale_bytes].reshape(
        num_pages, page_size, h_kv, D_HEAD // KV_GROUP_SIZE
    )
    dual_storage = torch.empty(
        (num_pages, num_tokens * (D_HEAD + 16)),
        dtype=torch.uint8,
        device=kv.device,
    )
    dual_storage[:, :data_bytes] = flat[:, :data_bytes]
    # The kernel addresses one 16B replicated slot per token from the page
    # tail. Replicate within each token slot, then flatten the page.
    scale_slots = torch.cat((scales, scales), dim=-1).reshape(num_pages, -1)
    dual_storage[:, data_bytes:] = scale_slots
    return (
        dual_storage.view(num_pages, page_size, h_kv, D_HEAD + 16),
        dequantized,
        w_scale,
        u_scale,
        quantized,
    )


def pack_dual_decode_kv_pages(
    kv: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack decode pages with the same independent group quantization as prefill."""
    assert kv.ndim == 4 and kv.shape[2] == 1 and kv.shape[-1] == D_HEAD
    data, scales, dequantized = _quantize_groups(kv, KV_GROUP_SIZE)
    num_pages, page_size, h_kv, _ = kv.shape
    num_tokens = page_size * h_kv
    flat_data = data.reshape(num_pages, -1)
    # Decode addresses one replicated 16B slot per token from the page tail.
    # Keep the scale bytes adjacent to their token, unlike the legacy packed
    # representation whose tail was two contiguous copies of the whole page.
    scale_slots = torch.cat((scales, scales), dim=-1).reshape(num_pages, -1)
    storage = torch.empty(
        (num_pages, num_tokens * (D_HEAD + 16)), dtype=torch.uint8, device=kv.device
    )
    storage[:, : num_tokens * D_HEAD] = flat_data
    storage[:, num_tokens * D_HEAD :] = scale_slots
    return (
        storage.view(num_pages, page_size, h_kv, D_HEAD + 16),
        dequantized,
        scales.view(num_pages, page_size, h_kv, D_HEAD // KV_GROUP_SIZE),
        data.view(num_pages, page_size, h_kv, D_HEAD),
    )


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


def attention_reference_dual_mxfp8(
    q: torch.Tensor,
    k_v: torch.Tensor,
    v_e4m3: torch.Tensor,
    token_u: torch.Tensor,
    w_scale: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    sm_scale: float,
    attn_sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference dual-MXFP8, including its 64-token online-softmax tiles."""
    q_leading = q.shape[:-2]
    h_q = q.shape[-2]
    topk = indices.shape[-1]
    q_flat = q.float().reshape(-1, h_q, D_HEAD)
    k_flat = k_v.float().reshape(-1, D_HEAD)
    v_flat = v_e4m3.float().reshape(-1, D_HEAD)
    u_flat = token_u.float().reshape(-1)
    w_flat = w_scale.float().repeat_interleave(KV_GROUP_SIZE)
    idx_flat = indices.reshape(-1, topk)
    len_flat = topk_length.reshape(-1)
    safe = idx_flat.clamp(0, k_flat.shape[0] - 1).long()
    gathered_k = k_flat.index_select(0, safe.reshape(-1)).reshape(-1, topk, D_HEAD)
    gathered_v = v_flat.index_select(0, safe.reshape(-1)).reshape(-1, topk, D_HEAD)
    gathered_u = u_flat.index_select(0, safe.reshape(-1)).reshape(-1, topk)
    pos = torch.arange(topk, device=q.device).view(1, topk)
    valid = (idx_flat >= 0) & (idx_flat < k_flat.shape[0]) & (pos < len_flat[:, None])
    scores_log2 = (
        torch.matmul(q_flat, gathered_k.transpose(-1, -2))
        * (sm_scale * math.log2(math.e))
    )
    scores_log2.masked_fill_(~valid[:, None, :], -math.inf)

    local_out = []
    local_lse = []
    for tile_begin in range(0, topk, 64):
        tile_end = tile_begin + 64
        tile_scores = scores_log2[..., tile_begin:tile_end]
        tile_valid = valid[:, tile_begin:tile_end]
        tile_max = tile_scores.amax(dim=-1)
        softmax = torch.exp2(tile_scores - tile_max[..., None])
        softmax = torch.where(tile_valid[:, None, :], softmax, 0.0)
        li = softmax.sum(dim=-1)
        s_fp8 = (
            softmax * gathered_u[:, None, tile_begin:tile_end]
        ).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
        numerator = torch.matmul(
            s_fp8,
            gathered_v[:, tile_begin:tile_end] * w_flat,
        )
        local_out.append(
            torch.where(li[..., None] != 0, numerator / li[..., None], 0.0)
        )
        local_lse.append(
            torch.where(li != 0, torch.log2(li) + tile_max, -math.inf)
        )

    stacked_out = torch.stack(local_out)
    stacked_lse = torch.stack(local_lse)
    max_lse = stacked_lse.amax(dim=0)
    safe_max = torch.where(torch.isneginf(max_lse), 0.0, max_lse)
    sum_lse = torch.exp2(stacked_lse - safe_max).sum(dim=0)
    global_lse = torch.where(
        sum_lse != 0, torch.log2(sum_lse) + safe_max, math.inf
    )
    output_lse = global_lse
    if attn_sink is not None:
        sink_log2 = attn_sink.float().view(1, h_q) * math.log2(math.e)
        output_lse = torch.where(
            torch.isfinite(global_lse),
            torch.log2(torch.exp2(global_lse) + torch.exp2(sink_log2)),
            sink_log2,
        )
    combine_weights = torch.exp2(stacked_lse - output_lse)
    out = (stacked_out * combine_weights[..., None]).sum(dim=0)
    returned_lse = global_lse / math.log2(math.e)
    return (
        out.reshape(*q_leading, h_q, D_HEAD),
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
