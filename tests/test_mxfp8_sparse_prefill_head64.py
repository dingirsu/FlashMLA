import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lib
from lib import TestParam
from run_mxfp8_prefill import mxfp8_sparse_prefill
from mxfp8_test_utils import (
    D_HEAD,
    FP8_MAX,
    KV_GROUP_SIZE,
    Q_GROUP_SIZE,
    assert_close,
    pack_prefill_kv_rank1,
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


def _unpack_page_tail_kv_rank1(
    packed_kv: torch.Tensor,
    kv_scale_w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    product_scale = (
        flat[data_end:]
        .contiguous()
        .view(torch.float8_e8m0fnu)
        .float()
        .reshape(*packed_kv.shape[:-1], D_HEAD // KV_GROUP_SIZE)
    )
    w_scale = kv_scale_w.contiguous().view(torch.float8_e8m0fnu).float()
    u_scale = product_scale[..., 0] / w_scale[0]
    torch.testing.assert_close(
        product_scale, u_scale.unsqueeze(-1) * w_scale, atol=0, rtol=0
    )
    dequantized = kv_fp8 * product_scale.repeat_interleave(
        KV_GROUP_SIZE, dim=-1
    )
    return dequantized, kv_fp8, u_scale, w_scale


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
    kv_fp8: torch.Tensor,
    kv_u_scale: torch.Tensor,
    kv_w_scale: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    sm_scale: float,
    attn_sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[TileTrace]]:
    """Mirror phase1.cuh's tiled QK, online softmax, S quantization, and SV."""
    assert indices.ndim == 2 and indices.shape[1] % B_TOPK == 0

    q = q.float()
    kv = kv.float().reshape(-1, D_HEAD)
    kv_fp8 = kv_fp8.float().reshape(-1, D_HEAD)
    kv_u_scale = kv_u_scale.float().reshape(-1)
    kv_w_scale = kv_w_scale.float().reshape(D_HEAD // KV_GROUP_SIZE)
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
            gathered_v_fp8 = kv_fp8.index_select(0, safe_indices)
            gathered_u_scale = kv_u_scale.index_select(0, safe_indices)

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
            s_for_sv = s * gathered_u_scale.unsqueeze(0)
            grouped_s_for_sv = s_for_sv.reshape(
                h_q, B_TOPK // MMA_K, MMA_K
            )
            group_s_max = grouped_s_for_sv.amax(dim=-1)
            raw_s_scale = torch.where(
                group_s_max > 0,
                group_s_max / FP8_MAX,
                torch.ones_like(group_s_max),
            )
            s_scale = _round_up_ue8m0(raw_s_scale)
            s_fp8 = (
                grouped_s_for_sv / s_scale.unsqueeze(-1)
            ).to(torch.float8_e4m3fn)
            dequant_s_for_sv = (
                s_fp8.float() * s_scale.unsqueeze(-1)
            ).reshape(h_q, B_TOPK)

            li = torch.add(li * old_o_scale, s.sum(dim=-1))
            if tile_idx > 0:
                o.mul_(old_o_scale.unsqueeze(-1))
            _sv_mma_tiles(o, dequant_s_for_sv, gathered_v_fp8)
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

        w_per_d = kv_w_scale.repeat_interleave(KV_GROUP_SIZE)
        out_rows.append(o * output_scale.unsqueeze(-1) * w_per_d.unsqueeze(0))
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


def _make_correctness_cases() -> list[TestParam]:
    # These are the head64/d512 subset of test_flash_mla_sparse_prefill.py.
    shape_cases = [
        (1, 128, 128),
        (1, 1840, 256),
        (1, 1592, 384),
        (1, 1521, 512),
        (1, 95, 128),
        (1, 153, 256),
        (1, 114, 384),
    ]
    cases = [
        TestParam(
            s_q,
            s_kv,
            topk,
            h_q=64,
            d_qk=512,
            seed=20260715 + case_idx,
            num_runs=0,
        )
        for case_idx, (s_q, s_kv, topk) in enumerate(shape_cases)
    ]
    cases.append(
        TestParam(
            62,
            592,
            128,
            h_q=64,
            d_qk=512,
            seed=20260722,
            num_runs=0,
            have_attn_sink=True,
            have_topk_length=True,
        )
    )
    return cases


def _run_precision_case(p: TestParam) -> None:
    torch.cuda.empty_cache()
    with torch.device("cuda"):
        testcase = lib.generate_testcase(p)

    packed_q, packed_q_reference = pack_q(testcase.q)
    (
        packed_kv,
        packed_kv_reference,
        kv_scale_w,
        packed_u_reference,
        packed_v_fp8_reference,
    ) = pack_prefill_kv_rank1(testcase.kv)
    tiled_q = _unpack_q(packed_q)
    tiled_kv, tiled_v_fp8, tiled_u_scale, tiled_w_scale = (
        _unpack_page_tail_kv_rank1(packed_kv, kv_scale_w)
    )
    torch.testing.assert_close(tiled_q, packed_q_reference, atol=0, rtol=0)
    torch.testing.assert_close(tiled_kv, packed_kv_reference, atol=0, rtol=0)
    torch.testing.assert_close(tiled_v_fp8, packed_v_fp8_reference, atol=0, rtol=0)
    torch.testing.assert_close(tiled_u_scale, packed_u_reference, atol=0, rtol=0)

    indices = testcase.indices[:, 0, :]
    reference_topk_length = testcase.topk_length
    if reference_topk_length is None:
        reference_topk_length = torch.full(
            (p.s_q,), p.topk, dtype=torch.int32, device=indices.device
        )
    ref_out, ref_max_logits, ref_lse, traces = phase1_tiled_reference(
        tiled_q,
        tiled_kv,
        tiled_v_fp8,
        tiled_u_scale,
        tiled_w_scale,
        indices,
        reference_topk_length,
        testcase.sm_scale,
        testcase.attn_sink,
    )

    if os.getenv("MXFP8_PRINT_TILE_TRACE") == "1":
        print(_format_traces(traces))

    out, max_logits, lse = mxfp8_sparse_prefill(
        packed_q,
        packed_kv,
        kv_scale_w,
        testcase.indices,
        testcase.sm_scale,
        d_qk=p.d_qk,
        d_v=p.d_v,
        attn_sink=testcase.attn_sink,
        topk_length=testcase.topk_length,
    )
    _assert_kernel_stage(
        f"{p}: Q/K gather, block scales, QK MMA, or validity mask",
        max_logits,
        ref_max_logits,
        traces,
        atol=2.0e-2,
        rtol=2.0e-2,
    )
    _assert_kernel_stage(
        f"{p}: online softmax mi/li update",
        lse,
        ref_lse,
        traces,
        atol=2.0e-2,
        rtol=5.0e-3,
    )
    _assert_kernel_stage(
        f"{p}: S MXFP8 quantization, old-O rescale, V scales, or SV MMA",
        out,
        ref_out,
        traces,
        atol=3.0e-2,
        rtol=1.2e-1,
    )

    positions = torch.arange(p.topk, device=indices.device).unsqueeze(0)
    valid = (indices >= 0) & (indices < p.s_kv)
    valid &= positions < reference_topk_length.unsqueeze(1)
    all_invalid = ~valid.any(dim=1)
    if all_invalid.any():
        assert torch.count_nonzero(out[all_invalid]) == 0


@torch.inference_mode()
def test_mxfp8_sparse_prefill_head64_precision() -> None:
    require_sm100_family()
    old_matmul_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        for case in _make_correctness_cases():
            print(f"Running MXFP8 sparse prefill precision case: {case}")
            _run_precision_case(case)
    finally:
        torch.set_float32_matmul_precision(old_matmul_precision)


if __name__ == "__main__":
    test_mxfp8_sparse_prefill_head64_precision()
    print("MXFP8 sparse prefill head64 tiled precision test passed")
