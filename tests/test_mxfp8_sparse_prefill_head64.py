import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_mxfp8_prefill import mxfp8_sparse_prefill
from mxfp8_test_utils import (
    D_HEAD,
    FP8_MAX,
    KV_GROUP_SIZE,
    Q_GROUP_SIZE,
    assert_close,
    attention_reference,
    pack_prefill_kv,
    pack_q,
    require_sm100_family,
)


B_TOPK = 128
MMA_K = 32
MAX_INIT_VAL = -1.0e30
LOG2_E = math.log2(math.e)


@dataclass(frozen=True)
class TileTrace:
    q_idx: int
    tile_idx: int
    valid_count: int
    tile_max_min: float
    tile_max_max: float
    mi_min: float
    mi_max: float
    li_min: float
    li_max: float
    old_o_scale_min: float
    old_o_scale_max: float
    s_scale_min: float
    s_scale_max: float
    o_absmax: float


def _unpack_q(packed_q: torch.Tensor) -> torch.Tensor:
    q_fp8 = (
        packed_q[..., :D_HEAD]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
    )
    q_scale = (
        packed_q[..., D_HEAD:]
        .contiguous()
        .view(torch.float8_e8m0fnu)
        .float()
        .repeat_interleave(Q_GROUP_SIZE, dim=-1)
    )
    return q_fp8 * q_scale


def _unpack_page_tail_kv(packed_kv: torch.Tensor) -> torch.Tensor:
    num_tokens = packed_kv.shape[0] * packed_kv.shape[1]
    flat = packed_kv.reshape(-1)
    data_end = num_tokens * D_HEAD
    kv_fp8 = (
        flat[:data_end]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
        .reshape(*packed_kv.shape[:-1], D_HEAD)
    )
    kv_scale = (
        flat[data_end:]
        .contiguous()
        .view(torch.float8_e8m0fnu)
        .float()
        .reshape(*packed_kv.shape[:-1], D_HEAD // KV_GROUP_SIZE)
        .repeat_interleave(KV_GROUP_SIZE, dim=-1)
    )
    return kv_fp8 * kv_scale


def _round_up_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """Match CUTLASS float_ue8m0_t's cvt.rp conversion."""
    positive = x > 0
    safe_x = torch.where(positive, x, torch.ones_like(x))
    exponent = torch.ceil(torch.log2(safe_x)).clamp(-127, 127)
    rounded = torch.exp2(exponent)
    rounded = torch.where(positive, rounded, torch.ones_like(rounded))
    return rounded.to(torch.float8_e8m0fnu).float()


def _qk_mma_tiles(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Accumulate QK exactly in the kernel's 32-element block-scale K tiles."""
    h_q = q.shape[0]
    scores = torch.zeros((h_q, B_TOPK), dtype=torch.float32, device=q.device)
    for d_start in range(0, D_HEAD, MMA_K):
        d_end = d_start + MMA_K
        scores.add_(q[:, d_start:d_end] @ k[:, d_start:d_end].transpose(0, 1))
    return scores


def _sv_mma_tiles(
    o: torch.Tensor,
    s: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """Accumulate S@V using CUDA's 128-wide D tiles and 32-wide K tiles."""
    for dv_start in range(0, D_HEAD, B_TOPK):
        dv_end = dv_start + B_TOPK
        o_tile = o[:, dv_start:dv_end]
        for k_start in range(0, B_TOPK, MMA_K):
            k_end = k_start + MMA_K
            o_tile.add_(s[:, k_start:k_end] @ v[k_start:k_end, dv_start:dv_end])


def _minmax(x: torch.Tensor) -> Tuple[float, float]:
    return x.amin().item(), x.amax().item()


def phase1_tiled_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    sm_scale: float,
    attn_sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[TileTrace]]:
    """Mirror phase1.cuh's tiled QK, online softmax, S quantization, and SV."""
    assert indices.ndim == 2 and indices.shape[1] % B_TOPK == 0

    q = q.float()
    kv = kv.float().reshape(-1, D_HEAD)
    indices = indices.reshape(q.shape[0], -1)
    topk_length = topk_length.reshape(-1)
    topk = indices.shape[1]
    h_q = q.shape[1]

    out_rows = []
    max_rows = []
    lse_rows = []
    traces: list[TileTrace] = []

    for q_idx in range(q.shape[0]):
        requested_length = int(topk_length[q_idx].item())
        length = max(0, min(requested_length, topk))
        num_k_blocks = max((length + B_TOPK - 1) // B_TOPK, 1)

        mi = torch.full((h_q,), MAX_INIT_VAL, dtype=torch.float32, device=q.device)
        li = torch.zeros((h_q,), dtype=torch.float32, device=q.device)
        real_mi = torch.full((h_q,), -math.inf, dtype=torch.float32, device=q.device)
        o = torch.zeros((h_q, D_HEAD), dtype=torch.float32, device=q.device)

        for tile_idx in range(num_k_blocks):
            tile_start = tile_idx * B_TOPK
            tile_indices = indices[q_idx, tile_start : tile_start + B_TOPK]
            positions = torch.arange(
                tile_start, tile_start + B_TOPK, device=q.device
            )
            valid = (
                (tile_indices >= 0)
                & (tile_indices < kv.shape[0])
                & (positions < length)
            )
            safe_indices = tile_indices.clamp(0, kv.shape[0] - 1).long()
            gathered_kv = kv.index_select(0, safe_indices)

            p = _qk_mma_tiles(q[q_idx], gathered_kv)
            p.mul_(sm_scale * LOG2_E)
            p.masked_fill_(~valid.unsqueeze(0), -math.inf)

            group_max = p.reshape(h_q, B_TOPK // MMA_K, MMA_K).amax(dim=-1)
            cur_pi_max = group_max.amax(dim=-1)
            old_mi = mi
            should_scale_o = cur_pi_max - old_mi > 6.0
            new_mi = torch.where(
                should_scale_o, torch.maximum(cur_pi_max, old_mi), old_mi
            )
            old_o_scale = torch.where(
                should_scale_o,
                torch.exp2(old_mi - new_mi),
                torch.ones_like(new_mi),
            )
            real_mi = torch.maximum(real_mi, cur_pi_max)

            s = torch.exp2(p - new_mi.unsqueeze(-1))
            group_s_max = torch.exp2(group_max - new_mi.unsqueeze(-1))
            raw_s_scale = torch.where(
                group_s_max > 0,
                group_s_max / FP8_MAX,
                torch.ones_like(group_s_max),
            )
            s_scale = _round_up_ue8m0(raw_s_scale)
            s_fp8 = (
                s.reshape(h_q, B_TOPK // MMA_K, MMA_K)
                / s_scale.unsqueeze(-1)
            ).to(torch.float8_e4m3fn)
            dequant_s = (
                s_fp8.float() * s_scale.unsqueeze(-1)
            ).reshape(h_q, B_TOPK)

            li = torch.add(li * old_o_scale, s.sum(dim=-1))
            if tile_idx > 0:
                o.mul_(old_o_scale.unsqueeze(-1))
            _sv_mma_tiles(o, dequant_s, gathered_kv)
            mi = new_mi

            tile_max_min, tile_max_max = _minmax(cur_pi_max)
            mi_min, mi_max = _minmax(mi)
            li_min, li_max = _minmax(li)
            old_scale_min, old_scale_max = _minmax(old_o_scale)
            s_scale_min, s_scale_max = _minmax(s_scale)
            traces.append(
                TileTrace(
                    q_idx=q_idx,
                    tile_idx=tile_idx,
                    valid_count=int(valid.sum().item()),
                    tile_max_min=tile_max_min,
                    tile_max_max=tile_max_max,
                    mi_min=mi_min,
                    mi_max=mi_max,
                    li_min=li_min,
                    li_max=li_max,
                    old_o_scale_min=old_scale_min,
                    old_o_scale_max=old_scale_max,
                    s_scale_min=s_scale_min,
                    s_scale_max=s_scale_max,
                    o_absmax=o.abs().amax().item(),
                )
            )

        all_invalid = torch.isneginf(real_mi)
        mi = torch.where(all_invalid, torch.full_like(mi, -math.inf), mi)
        li = torch.where(all_invalid, torch.zeros_like(li), li)

        max_logits = real_mi * math.log(2.0)
        lse = mi * math.log(2.0) + torch.log(li)
        lse = torch.where(torch.isneginf(lse), torch.full_like(lse, math.inf), lse)

        denominator = li
        if attn_sink is not None:
            sink_base2 = attn_sink.float() * LOG2_E
            denominator = denominator + torch.exp2(sink_base2 - mi)
        output_scale = torch.where(
            li == 0, torch.zeros_like(li), denominator.reciprocal()
        )

        out_rows.append(o * output_scale.unsqueeze(-1))
        max_rows.append(max_logits)
        lse_rows.append(lse)

    return (
        torch.stack(out_rows),
        torch.stack(max_rows),
        torch.stack(lse_rows),
        traces,
    )


def _format_traces(traces: list[TileTrace]) -> str:
    lines = []
    for trace in traces:
        lines.append(
            "q={q_idx} tile={tile_idx} valid={valid_count:3d} "
            "tile_max=[{tile_max_min:.6g},{tile_max_max:.6g}] "
            "mi=[{mi_min:.6g},{mi_max:.6g}] "
            "li=[{li_min:.6g},{li_max:.6g}] "
            "old_O_scale=[{old_o_scale_min:.6g},{old_o_scale_max:.6g}] "
            "S_scale=[{s_scale_min:.6g},{s_scale_max:.6g}] "
            "O_absmax={o_absmax:.6g}".format(**trace.__dict__)
        )
    return "\n".join(lines)


def _assert_kernel_stage(
    stage: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    traces: list[TileTrace],
    *,
    atol: float,
    rtol: float,
) -> None:
    try:
        assert_close(stage, actual, expected, atol=atol, rtol=rtol)
    except AssertionError as error:
        raise AssertionError(
            f"{stage} failed: {error}\nExpected per-tile phase1 state:\n"
            f"{_format_traces(traces)}"
        ) from None


def _make_tiled_indices(
    s_q: int,
    topk: int,
    s_kv: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randint(
        s_kv, (s_q, topk), dtype=torch.int32, device=device
    )
    topk_length = torch.full(
        (s_q,), topk, dtype=torch.int32, device=device
    )

    if s_q > 1:
        invalid_positions = [7, 129, 258]
        invalid_values = [-1, s_kv + 11, -1]
        for position, value in zip(invalid_positions, invalid_values):
            if position < topk:
                indices[1, position] = value
        topk_length[1] = min(topk, 301)
    if s_q > 2:
        indices[2] = -1
    if s_q > 3:
        topk_length[3] = 0
    return indices, topk_length


@torch.inference_mode()
def test_mxfp8_sparse_prefill_head64_precision() -> None:
    require_sm100_family()
    torch.manual_seed(20260715)
    device = torch.device("cuda")
    s_q = 1
    s_kv = 640
    h_q = 64
    topk = 128
    sm_scale = D_HEAD**-0.5

    q_gain = torch.linspace(0.6, 1.4, h_q, device=device).view(1, h_q, 1)
    kv_gain = torch.linspace(0.5, 1.5, s_kv, device=device).view(s_kv, 1, 1)
    q = torch.randn((s_q, h_q, D_HEAD), device=device) * (0.35 * q_gain)
    kv = torch.randn((s_kv, 1, D_HEAD), device=device) * (0.35 * kv_gain)

    packed_q, packed_q_reference = pack_q(q)
    packed_kv, packed_kv_reference = pack_prefill_kv(kv)
    tiled_q = _unpack_q(packed_q)
    tiled_kv = _unpack_page_tail_kv(packed_kv)
    torch.testing.assert_close(tiled_q, packed_q_reference, atol=0, rtol=0)
    torch.testing.assert_close(tiled_kv, packed_kv_reference, atol=0, rtol=0)

    indices, topk_length = _make_tiled_indices(s_q, topk, s_kv, device)
    ref_out, ref_max_logits, ref_lse, traces = phase1_tiled_reference(
        tiled_q, tiled_kv, indices, topk_length, sm_scale
    )

    if os.getenv("MXFP8_PRINT_TILE_TRACE") == "1":
        print(_format_traces(traces))

    out, max_logits, lse = mxfp8_sparse_prefill(
        packed_q,
        packed_kv,
        indices.unsqueeze(1),
        sm_scale,
        d_qk=D_HEAD,
        d_v=D_HEAD,
        topk_length=topk_length,
    )
    print(out)
    _assert_kernel_stage(
        "Q/K gather, block scales, QK MMA, or validity mask",
        max_logits,
        ref_max_logits,
        traces,
        atol=2.0e-2,
        rtol=2.0e-2,
    )
    _assert_kernel_stage(
        "online softmax mi/li update",
        lse,
        ref_lse,
        traces,
        atol=2.0e-2,
        rtol=5.0e-3,
    )
    _assert_kernel_stage(
        "S MXFP8 quantization, old-O rescale, V scales, or SV MMA",
        out,
        ref_out,
        traces,
        atol=3.0e-2,
        rtol=1.2e-1,
    )
    if s_q > 2:
        assert torch.count_nonzero(out[2]) == 0
    if s_q > 3:
        assert torch.count_nonzero(out[3]) == 0


if __name__ == "__main__":
    test_mxfp8_sparse_prefill_head64_precision()
    print("MXFP8 sparse prefill head64 tiled precision test passed")
