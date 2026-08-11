import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_fp8_prefill import fp8_sparse_prefill
from fp8_test_utils import (
    D_HEAD,
    FP8_MAX,
    H_Q,
    KV_BYTES_PER_TOKEN,
    KV_GROUP_SIZE,
    Q_DATA_BYTES,
    assert_close,
    pack_kv_rank1,
    pack_kv_raw,
    pack_q_per_head,
    pack_q_raw,
    require_sm100_family,
    round_up_ue8m0,
)


B_TOPK = 128
MMA_K = 32
SV_M = 128
MAX_INIT_VAL = -1.0e30
LOG2_E = math.log2(math.e)


@dataclass(frozen=True)
class TileTrace:
    q_idx: int
    tile_idx: int
    valid_count: int
    qk_raw_min: float
    qk_raw_max: float
    p_min: float
    p_max: float
    mi_min: float
    mi_max: float
    li_min: float
    li_max: float
    s_scale_min: float
    s_scale_max: float
    o_absmax: float


@dataclass(frozen=True)
class TileSnapshot:
    q_idx: int
    tile_idx: int
    indices: torch.Tensor
    valid: torch.Tensor
    q_raw: torch.Tensor
    k_raw: torch.Tensor
    kv_token_scale: torch.Tensor
    qk_parts: torch.Tensor
    qk_raw: torch.Tensor
    p_scaled: torch.Tensor
    p_masked: torch.Tensor
    softmax_s: torch.Tensor
    s_for_sv: torch.Tensor
    s_scale: torch.Tensor
    s_fp8: torch.Tensor
    o_tmem_units: torch.Tensor


@dataclass(frozen=True)
class Phase1Reference:
    out: torch.Tensor
    max_logits: torch.Tensor
    lse: torch.Tensor
    traces: list[TileTrace]
    snapshots: list[TileSnapshot]


def _unpack_q(packed_q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    q_raw = packed_q[:, :Q_DATA_BYTES].contiguous().view(
        torch.float8_e4m3fn
    ).float().reshape(packed_q.shape[0], H_Q, D_HEAD)
    q_scale = packed_q[:, Q_DATA_BYTES:].contiguous().view(
        torch.float8_e8m0fnu
    ).float()
    return q_raw, q_scale


def _unpack_kv(packed_kv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert packed_kv.shape[1] == KV_BYTES_PER_TOKEN
    kv_raw = packed_kv[:, :D_HEAD].contiguous().view(
        torch.float8_e4m3fn
    ).float().reshape(packed_kv.shape[0], D_HEAD)
    token_scale = packed_kv[:, D_HEAD].contiguous().view(
        torch.float8_e8m0fnu
    ).float()
    return kv_raw, token_scale


def _qk_dual_mma(q_raw: torch.Tensor, k_raw: torch.Tensor) -> torch.Tensor:
    """Return the two ordinary-FP8 dual-GEMM partials before FP32 scaling."""
    parts = []
    for starts in ((0, 256), (128, 384)):
        partial = torch.zeros(
            (H_Q, B_TOPK), dtype=torch.float32, device=q_raw.device
        )
        for block_start in starts:
            for d_start in range(block_start, block_start + 128, MMA_K):
                partial.add_(
                    q_raw[:, d_start : d_start + MMA_K]
                    @ k_raw[:, d_start : d_start + MMA_K].transpose(0, 1)
                )
        parts.append(partial)
    return torch.stack(parts)


def _sv_mma(
    o: torch.Tensor,
    s_fp8: torch.Tensor,
    v_fp8: torch.Tensor,
) -> None:
    s_raw = s_fp8.float()
    for dv_start in range(0, D_HEAD, SV_M):
        o_tile = o[:, dv_start : dv_start + SV_M]
        for token_start in range(0, B_TOPK, MMA_K):
            o_tile.add_(
                s_raw[:, token_start : token_start + MMA_K]
                @ v_fp8[
                    token_start : token_start + MMA_K,
                    dv_start : dv_start + SV_M,
                ]
            )


def _finite_minmax(x: torch.Tensor) -> Tuple[float, float]:
    finite = x[torch.isfinite(x)]
    if finite.numel() == 0:
        return -math.inf, -math.inf
    return finite.amin().item(), finite.amax().item()


def _update_online_max(
    old_mi: torch.Tensor,
    tile_max: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mirror the kernel's warp-uniform >6 rescale decision for 32 heads."""
    new_mi = old_mi.clone()
    scale_for_old = torch.ones_like(old_mi)
    trigger = tile_max - old_mi > 6.0
    for head_start in (0, 32):
        head_slice = slice(head_start, head_start + 32)
        if bool(trigger[head_slice].any().item()):
            new_mi[head_slice] = torch.maximum(
                tile_max[head_slice], old_mi[head_slice]
            )
            scale_for_old[head_slice] = torch.exp2(
                old_mi[head_slice] - new_mi[head_slice]
            )
    return new_mi, scale_for_old


def phase1_token_tile_reference(
    packed_q: torch.Tensor,
    packed_kv: torch.Tensor,
    kv_scale_w: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    sm_scale: float,
    attn_sink: Optional[torch.Tensor] = None,
    capture: Optional[Iterable[Tuple[int, int]]] = None,
) -> Phase1Reference:
    """Execute the CUDA algorithm one query token and one 64-token tile at a time."""
    assert packed_q.ndim == 2
    assert packed_kv.ndim == 2
    assert indices.ndim == 3 and indices.shape[1] == 1
    assert indices.shape[2] % B_TOPK == 0

    q_raw, q_scale = _unpack_q(packed_q)
    kv_raw, kv_token_scale = _unpack_kv(packed_kv)
    w_scale = kv_scale_w.contiguous().view(torch.float8_e8m0fnu).float()
    w_per_d = w_scale.repeat_interleave(KV_GROUP_SIZE)
    indices_2d = indices[:, 0, :]
    if topk_length is None:
        topk_length = torch.full(
            (packed_q.shape[0],),
            indices.shape[2],
            dtype=torch.int32,
            device=indices.device,
        )

    capture_set = set(capture or ())
    traces: list[TileTrace] = []
    snapshots: list[TileSnapshot] = []
    out_rows = []
    max_rows = []
    lse_rows = []

    for q_idx in range(packed_q.shape[0]):
        length = max(
            0,
            min(int(topk_length[q_idx].item()), indices.shape[2]),
        )
        num_tiles = max((length + B_TOPK - 1) // B_TOPK, 1)
        mi = torch.full(
            (H_Q,), MAX_INIT_VAL, dtype=torch.float32, device=packed_q.device
        )
        real_mi = torch.full_like(mi, -math.inf)
        li_halves = torch.zeros(
            (H_Q, 2), dtype=torch.float32, device=packed_q.device
        )
        o_tmem_units = torch.zeros(
            (H_Q, D_HEAD), dtype=torch.float32, device=packed_q.device
        )
        previous_s_scale = torch.ones_like(mi)

        for tile_idx in range(num_tiles):
            token_start = tile_idx * B_TOPK
            tile_indices = indices_2d[
                q_idx, token_start : token_start + B_TOPK
            ]
            positions = torch.arange(
                token_start,
                token_start + B_TOPK,
                device=packed_q.device,
            )
            valid = (
                (tile_indices >= 0)
                & (tile_indices < packed_kv.shape[0])
                & (positions < length)
            )
            safe_indices = tile_indices.clamp(0, packed_kv.shape[0] - 1).long()
            gathered_k_raw = kv_raw.index_select(0, safe_indices)
            gathered_u = kv_token_scale.index_select(0, safe_indices)
            gathered_u = torch.where(valid, gathered_u, torch.ones_like(gathered_u))

            qk_parts = _qk_dual_mma(q_raw[q_idx], gathered_k_raw)
            qk_raw = qk_parts.sum(dim=0)
            p_scaled = qk_raw * q_scale[q_idx].unsqueeze(-1)
            p_scaled *= gathered_u.unsqueeze(0)
            p_scaled *= sm_scale * LOG2_E
            p_masked = p_scaled.masked_fill(~valid.unsqueeze(0), -math.inf)

            tile_max = p_masked.amax(dim=-1)
            real_mi = torch.maximum(real_mi, tile_max)
            new_mi, scale_for_old = _update_online_max(mi, tile_max)
            softmax_s = torch.exp2(p_masked - new_mi.unsqueeze(-1))
            cur_sum_halves = torch.stack(
                (
                    softmax_s[:, : B_TOPK // 2].sum(dim=-1),
                    softmax_s[:, B_TOPK // 2 :].sum(dim=-1),
                ),
                dim=-1,
            )
            li_halves = li_halves * scale_for_old.unsqueeze(-1) + cur_sum_halves

            s_for_sv = softmax_s * gathered_u.unsqueeze(0)
            s_scale = torch.full_like(mi, 1.0 / FP8_MAX)
            s_fp8 = (s_for_sv / s_scale.unsqueeze(-1)).to(
                torch.float8_e4m3fn
            )

            if tile_idx > 0:
                o_tmem_units *= (
                    scale_for_old * previous_s_scale / s_scale
                ).unsqueeze(-1)
            _sv_mma(o_tmem_units, s_fp8, gathered_k_raw)
            previous_s_scale = s_scale
            mi = new_mi

            qk_min, qk_max = _finite_minmax(qk_raw)
            p_min, p_max = _finite_minmax(p_masked)
            traces.append(
                TileTrace(
                    q_idx=q_idx,
                    tile_idx=tile_idx,
                    valid_count=int(valid.sum().item()),
                    qk_raw_min=qk_min,
                    qk_raw_max=qk_max,
                    p_min=p_min,
                    p_max=p_max,
                    mi_min=mi.amin().item(),
                    mi_max=mi.amax().item(),
                    li_min=li_halves.sum(dim=-1).amin().item(),
                    li_max=li_halves.sum(dim=-1).amax().item(),
                    s_scale_min=s_scale.amin().item(),
                    s_scale_max=s_scale.amax().item(),
                    o_absmax=o_tmem_units.abs().amax().item(),
                )
            )
            if (q_idx, tile_idx) in capture_set:
                snapshots.append(
                    TileSnapshot(
                        q_idx=q_idx,
                        tile_idx=tile_idx,
                        indices=tile_indices.clone(),
                        valid=valid.clone(),
                        q_raw=q_raw[q_idx].clone(),
                        k_raw=gathered_k_raw.clone(),
                        kv_token_scale=gathered_u.clone(),
                        qk_parts=qk_parts.clone(),
                        qk_raw=qk_raw.clone(),
                        p_scaled=p_scaled.clone(),
                        p_masked=p_masked.clone(),
                        softmax_s=softmax_s.clone(),
                        s_for_sv=s_for_sv.clone(),
                        s_scale=s_scale.clone(),
                        s_fp8=s_fp8.clone(),
                        o_tmem_units=o_tmem_units.clone(),
                    )
                )

        all_invalid = torch.isneginf(real_mi)
        mi = torch.where(all_invalid, torch.full_like(mi, -math.inf), mi)
        li = li_halves.sum(dim=-1)
        li = torch.where(all_invalid, torch.zeros_like(li), li)
        max_logits = real_mi * math.log(2.0)
        lse = mi * math.log(2.0) + torch.log(li)
        lse = torch.where(torch.isneginf(lse), torch.full_like(lse, math.inf), lse)

        denominator = li
        if attn_sink is not None:
            denominator = denominator + torch.exp2(
                attn_sink.float() * LOG2_E - mi
            )
        output_scale = torch.where(
            li == 0, torch.zeros_like(li), denominator.reciprocal()
        )
        out = o_tmem_units * output_scale.unsqueeze(-1)
        out *= previous_s_scale.unsqueeze(-1)
        out *= w_per_d.unsqueeze(0)
        out_rows.append(out.to(torch.bfloat16))
        max_rows.append(max_logits)
        lse_rows.append(lse)

    return Phase1Reference(
        out=torch.stack(out_rows),
        max_logits=torch.stack(max_rows),
        lse=torch.stack(lse_rows),
        traces=traces,
        snapshots=snapshots,
    )


def _format_traces(traces: list[TileTrace]) -> str:
    return "\n".join(
        (
            f"q={t.q_idx} tile={t.tile_idx} valid={t.valid_count:2d} "
            f"rawQK=[{t.qk_raw_min:.6g},{t.qk_raw_max:.6g}] "
            f"P=[{t.p_min:.6g},{t.p_max:.6g}] "
            f"mi=[{t.mi_min:.6g},{t.mi_max:.6g}] "
            f"li=[{t.li_min:.6g},{t.li_max:.6g}] "
            f"Sscale=[{t.s_scale_min:.6g},{t.s_scale_max:.6g}] "
            f"Oabs={t.o_absmax:.6g}"
        )
        for t in traces
    )


def _print_snapshots(snapshots: list[TileSnapshot]) -> None:
    torch.set_printoptions(precision=5, linewidth=180, sci_mode=False)
    for s in snapshots:
        print(f"Snapshot q={s.q_idx}, tile={s.tile_idx}")
        print("indices:", s.indices.cpu())
        print("valid:", s.valid.cpu())
        print("Q raw h0 d0:32:", s.q_raw[0, :32].cpu())
        print("K raw token0 d0:32:", s.k_raw[0, :32].cpu())
        print("U:", s.kv_token_scale.cpu())
        print("QK part0 h0:", s.qk_parts[0, 0].cpu())
        print("QK part1 h0:", s.qk_parts[1, 0].cpu())
        print("QK raw h0:", s.qk_raw[0].cpu())
        print("P scaled h0:", s.p_scaled[0].cpu())
        print("softmax S h0:", s.softmax_s[0].cpu())
        print("S*U h0:", s.s_for_sv[0].cpu())
        print("S scale h0:8:", s.s_scale[:8].cpu())
        print("S fp8 h0:", s.s_fp8[0].float().cpu())
        print("O TMEM units h0 d0:32:", s.o_tmem_units[0, :32].cpu())


def _run_and_check(
    name: str,
    packed_q: torch.Tensor,
    packed_kv: torch.Tensor,
    kv_scale_w: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    topk_length: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor] = None,
) -> None:
    capture = {(0, 0)} if os.getenv("FP8_DEBUG_DUMP") == "1" else set()
    reference = phase1_token_tile_reference(
        packed_q,
        packed_kv,
        kv_scale_w,
        indices,
        topk_length,
        sm_scale,
        attn_sink,
        capture,
    )
    if os.getenv("FP8_DEBUG_TRACE") == "1":
        print(_format_traces(reference.traces))
    if capture:
        _print_snapshots(reference.snapshots)

    out, max_logits, lse = fp8_sparse_prefill(
        packed_q,
        packed_kv,
        kv_scale_w,
        indices,
        sm_scale,
        attn_sink,
        topk_length,
    )
    torch.cuda.synchronize()

    try:
        assert_close(
            f"{name}: packed Q/K load, dual QK MMA, or Q/K FP32 scale",
            max_logits,
            reference.max_logits,
            atol=2.0e-2,
            rtol=2.0e-2,
        )
        assert_close(
            f"{name}: online softmax mi/li",
            lse,
            reference.lse,
            atol=2.0e-2,
            rtol=5.0e-3,
        )
        assert_close(
            f"{name}: S head quant, SV MMA, O rescale, or W(g) epilogue",
            out,
            reference.out,
            atol=4.0e-2,
            rtol=1.2e-1,
        )
    except AssertionError as error:
        raise AssertionError(
            f"{error}\nPer-token/tile reference state:\n"
            f"{_format_traces(reference.traces)}"
        ) from None


def _case_all_ones() -> None:
    device = torch.device("cuda")
    q_raw = torch.ones((1, H_Q, D_HEAD), device=device)
    q_scale = torch.ones((1, H_Q), device=device)
    kv_raw = torch.ones((B_TOPK, 1, D_HEAD), device=device)
    kv_scale = torch.ones((B_TOPK,), device=device)
    packed_q, _, _ = pack_q_raw(q_raw, q_scale)
    packed_kv, _, _ = pack_kv_raw(kv_raw, kv_scale)
    w = torch.ones(8, device=device).to(torch.float8_e8m0fnu).view(torch.uint8)
    indices = torch.arange(B_TOPK, device=device, dtype=torch.int32).view(1, 1, -1)
    length = torch.tensor([B_TOPK], device=device, dtype=torch.int32)
    _run_and_check(
        "all ones",
        packed_q,
        packed_kv,
        w,
        indices,
        D_HEAD**-0.5,
        length,
    )


def _case_single_token_load_and_scale() -> None:
    device = torch.device("cuda")
    selected = torch.tensor(
        [0, 1, 2, 15, 16, 31, 32, 63], device=device, dtype=torch.int64
    )
    s_q = selected.numel()
    q_raw = torch.ones((s_q, H_Q, D_HEAD), device=device)
    q_scale = torch.exp2(
        ((torch.arange(H_Q, device=device) % 7) - 3).float()
    ).expand(s_q, -1).contiguous()
    kv_values = 0.5 + (torch.arange(B_TOPK, device=device) % 8).float() * 0.25
    kv_raw = kv_values.view(-1, 1, 1).expand(-1, 1, D_HEAD).contiguous()
    kv_scale = torch.exp2(
        ((torch.arange(B_TOPK, device=device) % 5) - 2).float()
    )
    packed_q, _, _ = pack_q_raw(q_raw, q_scale)
    packed_kv, _, _ = pack_kv_raw(kv_raw, kv_scale)
    w = torch.ones(8, device=device).to(torch.float8_e8m0fnu).view(torch.uint8)
    indices = selected.to(torch.int32).view(s_q, 1, 1).expand(-1, 1, B_TOPK).contiguous()
    length = torch.ones(s_q, device=device, dtype=torch.int32)
    _run_and_check(
        "single selected token with head/token scales",
        packed_q,
        packed_kv,
        w,
        indices,
        D_HEAD**-0.5,
        length,
    )


def _case_random_two_tiles() -> None:
    device = torch.device("cuda")
    torch.manual_seed(20260724)
    s_q, s_kv, topk = 3, 384, 256
    q = torch.randn((s_q, H_Q, D_HEAD), device=device) * 0.35
    kv = torch.randn((s_kv, 1, D_HEAD), device=device) * 0.35
    packed_q, _, _, _ = pack_q_per_head(q)
    packed_kv, _, _, w_scale, _ = pack_kv_rank1(kv)
    w = w_scale.to(torch.float8_e8m0fnu).view(torch.uint8)
    indices = torch.stack(
        [torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]
    ).to(torch.int32).unsqueeze(1)
    length = torch.tensor([256, 173, 128], device=device, dtype=torch.int32)
    attn_sink = torch.linspace(-1.0, 1.0, H_Q, device=device)
    _run_and_check(
        "random rank-1 KV, two tiles",
        packed_q,
        packed_kv,
        w,
        indices,
        D_HEAD**-0.5,
        length,
        attn_sink,
    )


def _case_pipeline_reuse() -> None:
    device = torch.device("cuda")
    torch.manual_seed(20260725)
    s_q, s_kv, topk = 2, 512, 384
    q = torch.randn((s_q, H_Q, D_HEAD), device=device) * 0.3
    kv = torch.randn((s_kv, 1, D_HEAD), device=device) * 0.3
    packed_q, _, _, _ = pack_q_per_head(q)
    packed_kv, _, _, w_scale, _ = pack_kv_rank1(kv)
    w = w_scale.to(torch.float8_e8m0fnu).view(torch.uint8)
    indices = torch.stack(
        [torch.randperm(s_kv, device=device)[:topk] for _ in range(s_q)]
    ).to(torch.int32).unsqueeze(1)
    length = torch.tensor([384, 257], device=device, dtype=torch.int32)
    _run_and_check(
        "three tiles with KV stage reuse",
        packed_q,
        packed_kv,
        w,
        indices,
        D_HEAD**-0.5,
        length,
    )


def _case_invalid_mask() -> None:
    device = torch.device("cuda")
    torch.manual_seed(20260726)
    s_q, s_kv, topk = 3, 96, B_TOPK
    q = torch.randn((s_q, H_Q, D_HEAD), device=device) * 0.25
    kv = torch.randn((s_kv, 1, D_HEAD), device=device) * 0.25
    packed_q, _, _, _ = pack_q_per_head(q)
    packed_kv, _, _, w_scale, _ = pack_kv_rank1(kv)
    w = w_scale.to(torch.float8_e8m0fnu).view(torch.uint8)

    indices = torch.randint(
        s_kv, (s_q, 1, topk), device=device, dtype=torch.int32
    )
    indices[1, 0, 7] = -1
    indices[1, 0, 19] = s_kv + 11
    indices[2].fill_(-1)
    length = torch.tensor([128, 91, 0], device=device, dtype=torch.int32)
    _run_and_check(
        "invalid indices and zero topk_length",
        packed_q,
        packed_kv,
        w,
        indices,
        D_HEAD**-0.5,
        length,
    )


@torch.inference_mode()
def test_fp8_sparse_prefill_head64_precision() -> None:
    require_sm100_family()
    old_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        _case_all_ones()
        _case_single_token_load_and_scale()
        _case_random_two_tiles()
        _case_pipeline_reuse()
        _case_invalid_mask()
    finally:
        torch.set_float32_matmul_precision(old_precision)


if __name__ == "__main__":
    test_fp8_sparse_prefill_head64_precision()
    print("FP8 sparse prefill head64 token/tile precision test passed")
