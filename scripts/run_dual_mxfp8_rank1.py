#!/usr/bin/env python3
"""Exercise dual-MXFP8 prefill and decode with an explicit rank-1 KV scale.

The KV data is generated as random E4M3 values.  The first and second 256-D
halves each use an independent rank-1 UE8M0 scale matrix.  Their dimension
anchors are ``W[0] == W[4] == 1``:

    scale[t, g] = U_half[t] * W[g]

Each output is compared with both ordinary Torch attention (without an extra
S quantize/dequantize) and a Torch simulation of the CUTLASS kernel flow.
This keeps the test independent of the input quantizer while exposing the
error introduced by the kernel's E4M3 S operand.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from quant import FP8KVCacheLayout, dequantize_k_cache, quantize_k_cache
EXTENSION_PATH = Path(
    os.environ.get("DUAL_MXFP8_EXTENSION_PATH", ROOT / "build/dual_mxfp8_test_ext.so")
)
BF16_EXTENSION_PATH = Path(
    os.environ.get("BF16_EXTENSION_PATH", ROOT / "build/bf16_test_ext.so")
)
HEAD64_DECODE_EXTENSION_PATH = Path(
    os.environ.get(
        "HEAD64_DECODE_EXTENSION_PATH", ROOT / "build/head64_decode_test_ext.so"
    )
)
D_HEAD = 512
H_Q = 64
KV_GROUP_SIZE = 64
KV_GROUPS = D_HEAD // KV_GROUP_SIZE
MMA_K = 32
FP8_E4M3_MAX = 448.0
KV_SCALE_SLOT_BYTES = 16
KV_RECORD_BYTES = D_HEAD + KV_SCALE_SLOT_BYTES
SM_SCALE = D_HEAD ** -0.5
PRINT_S_ROW = 0
PRINT_S_HEAD = 24

# All factors are exactly representable UE8M0 powers of two.  The two-anchor
# case has an independent unit anchor in each 256-D half; the single-anchor
# case retains the original full-512-D rank-1 construction.
W_EXPONENTS_TWO_ANCHORS = torch.tensor([0, 8, 8, 8, 0, 8, 8, 8], dtype=torch.int32)
W_EXPONENTS_SINGLE_ANCHOR = torch.tensor([0, 8, 8, 8, 8, 8, 8, 8], dtype=torch.int32)


def _load_extension():
    if not EXTENSION_PATH.exists():
        raise RuntimeError(
            f"{EXTENSION_PATH} does not exist; run "
            "PYTHON=/usr/bin/python3 ./compile_sm100.sh dual_mxfp8"
        )
    spec = importlib.util.spec_from_file_location("dual_mxfp8_test_ext", EXTENSION_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {EXTENSION_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_bf16_extension():
    if not BF16_EXTENSION_PATH.exists():
        raise RuntimeError(
            f"{BF16_EXTENSION_PATH} does not exist; run "
            "PYTHON=/usr/bin/python3 ./compile_sm100.sh bf16"
        )
    spec = importlib.util.spec_from_file_location("bf16_test_ext", BF16_EXTENSION_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BF16_EXTENSION_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_head64_decode_extension():
    if not HEAD64_DECODE_EXTENSION_PATH.exists():
        raise RuntimeError(
            f"{HEAD64_DECODE_EXTENSION_PATH} does not exist; run "
            "PYTHON=/usr/bin/python3 ./compile_sm100.sh decode_head64"
        )
    spec = importlib.util.spec_from_file_location(
        "head64_decode_test_ext", HEAD64_DECODE_EXTENSION_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {HEAD64_DECODE_EXTENSION_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pack_scale_bytes(bits: torch.Tensor) -> int:
    values = bits.to(device="cpu", dtype=torch.uint8).tolist()
    assert len(values) == 4
    return int.from_bytes(bytes(values), byteorder="little", signed=False)


def _w_arguments(
    device: torch.device, w_exponents: torch.Tensor, two_anchors: bool
) -> Tuple[int, int, torch.Tensor]:
    assert int(w_exponents[0]) == 0, "W[0] must be the fixed unit anchor"
    if two_anchors:
        assert int(w_exponents[4]) == 0, (
            "two-anchor mode requires W[4] to be the fixed unit anchor"
        )
    bits = (w_exponents + 127).to(device=device, dtype=torch.uint8)
    w1 = _pack_scale_bytes(bits[:4])
    w2 = _pack_scale_bytes(bits[4:])
    return w1, w2, torch.exp2(w_exponents.to(device=device, dtype=torch.float32))


def _make_rank1_storage(
    num_tokens: int,
    device: torch.device,
    seed: int,
    w_exponents: torch.Tensor,
    two_anchors: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return raw data bytes, replicated scale slots, dequantized KV, U, FP8."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    # Match CUTLASS's cvt.rn.satfinite conversion explicitly. Random normal
    # data is already well inside the endpoint; the clamp documents the rule.
    fp8 = torch.randn(
        (num_tokens, D_HEAD), device=device, generator=generator
    ).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    if not torch.isfinite(fp8.float()).all():
        raise AssertionError("random E4M3 generation produced a non-finite value")
    if two_anchors:
        # Relaxed construction: U_lo/W_lo and U_hi/W_hi are rank-1
        # independently within their respective 256-D halves.
        u_exp = torch.randint(
            -3, 4, (num_tokens, 2), device=device, dtype=torch.int32,
            generator=generator,
        )
    else:
        # Original construction: one U[t] is shared by all 512 dimensions.
        u_one = torch.randint(
            -3, 4, (num_tokens,), device=device, dtype=torch.int32,
            generator=generator,
        )
        u_exp = u_one[:, None].expand(-1, 2)
    w_device = w_exponents.to(device=device)
    product_exp = torch.cat(
        (
            u_exp[:, 0, None] + w_device[None, :4],
            u_exp[:, 1, None] + w_device[None, 4:],
        ), dim=1
    )
    if int(product_exp.amin()) < -126 or int(product_exp.amax()) > 127:
        raise AssertionError("rank-1 product scale is outside UE8M0 range")

    product_scale = torch.exp2(product_exp.float()).to(torch.float8_e8m0fnu)
    scale_slots = torch.cat(
        (product_scale.view(torch.uint8), product_scale.view(torch.uint8)), dim=-1
    )
    data_bytes = fp8.view(torch.uint8)
    u_scale = torch.exp2(u_exp.float())
    dequant = fp8.float() * product_scale.float().repeat_interleave(KV_GROUP_SIZE, dim=-1)
    # The 528-byte interface envelope represents a page's average stride.  The
    # physical allocation is [all data rows][all 16-byte scale slots].
    expected_product = torch.cat(
        (
            u_scale[:, 0, None] * torch.exp2(w_device[:4].float())[None, :],
            u_scale[:, 1, None] * torch.exp2(w_device[4:].float())[None, :],
        ),
        dim=1,
    )
    torch.testing.assert_close(product_scale.float(), expected_product, atol=0, rtol=0)
    return data_bytes, scale_slots, dequant, u_scale, fp8.float()


def _pack_page_storage(
    data_bytes: torch.Tensor,
    scale_slots: torch.Tensor,
    *,
    num_pages: int,
    page_size: int,
) -> torch.Tensor:
    """Pack page storage as contiguous data plane followed by scale plane."""
    data = data_bytes[: num_pages * page_size].reshape(num_pages, -1)
    scales = scale_slots[: num_pages * page_size].reshape(num_pages, -1)
    # Each physical page owns its data plane and its scale plane.  The final
    # 528-byte row is only an interface envelope after this flatten/view.
    return torch.cat((data, scales), dim=1).view(
        num_pages, page_size, 1, KV_RECORD_BYTES
    )


def _pack_prefill_storage(
    data_bytes: torch.Tensor, scale_slots: torch.Tensor
) -> torch.Tensor:
    """Pack the prefill allocation using the same single-page layout."""
    flat = torch.cat((data_bytes.reshape(-1), scale_slots.reshape(-1)))
    return flat.view(data_bytes.shape[0], 1, KV_RECORD_BYTES)


def _make_prefill_q(s_q: int, device: torch.device, seed: int):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    q = torch.randn((s_q, H_Q, D_HEAD), device=device, generator=generator) * 0.25
    # Q uses the kernel's normal dual-64 packing; the reference uses its exact
    # dequantized result.
    from mxfp8_test_utils import pack_dual_q64

    return pack_dual_q64(q)


def _make_indices(rows: int, topk: int, s_kv: int, device: torch.device, seed: int):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    indices = torch.stack(
        [torch.randperm(s_kv, device=device, generator=generator)[:topk] for _ in range(rows)]
    ).to(torch.int32)
    return indices


def _torch_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    pair_indices: bool,
    topk_length: Optional[torch.Tensor] = None,
    dtype = torch.bfloat16
):
    """Ordinary Torch attention without quantizing/dequantizing S."""
    qf = q.to(dtype)
    kvf = kv.to(dtype).reshape(-1, D_HEAD)
    if pair_indices:
        # Prefill has one index row for each adjacent pair of query tokens.
        idx = indices[:, 0]
        row_for_q = torch.arange(q.shape[0], device=q.device) // 2
        idx = idx.index_select(0, row_for_q)
        lengths = (
            torch.full((idx.shape[0],), idx.shape[1], device=q.device, dtype=torch.int32)
            if topk_length is None
            else topk_length.index_select(0, row_for_q)
        )
    else:
        idx = indices.reshape(-1, indices.shape[-1])
        lengths = (
            torch.full((idx.shape[0],), idx.shape[1], device=q.device, dtype=torch.int32)
            if topk_length is None
            else topk_length.reshape(-1)
        )

    safe = idx.clamp(0, kvf.shape[0] - 1).long()
    gathered = kvf.index_select(0, safe.reshape(-1)).reshape(idx.shape[0], idx.shape[1], D_HEAD)
    positions = torch.arange(idx.shape[1], device=q.device)[None, :]
    valid = (idx >= 0) & (idx < kvf.shape[0]) & (positions < lengths[:, None])
    q_flat = qf.reshape(-1, H_Q, D_HEAD)
    logits = torch.matmul(q_flat, gathered.transpose(-1, -2)) * SM_SCALE
    logits.masked_fill_(~valid[:, None, :], -math.inf)
    max_logits = logits.amax(dim=-1)
    lse = torch.logsumexp(logits, dim=-1)
    weights = torch.softmax(logits, dim=-1)
    weights = torch.where(torch.isfinite(lse)[..., None], weights, 0.0).to(dtype)
    out = torch.matmul(weights, gathered)
    out = out.reshape_as(qf).to(torch.bfloat16)
    return out, max_logits, lse


def _summary(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_f = actual.float()
    expected_f = expected.float()
    both_finite = torch.isfinite(actual_f) & torch.isfinite(expected_f)
    same_nan = torch.isnan(actual_f) & torch.isnan(expected_f)
    same_inf = (
        torch.isinf(actual_f)
        & torch.isinf(expected_f)
        & (torch.signbit(actual_f) == torch.signbit(expected_f))
    )
    nonfinite_mismatch = ~(both_finite | same_nan | same_inf)
    if nonfinite_mismatch.any():
        raise AssertionError(
            f"{name}: {int(nonfinite_mismatch.sum().item())} mismatched NaN/Inf values"
        )
    if not both_finite.any():
        print(f"{name}: all values are matching NaN/Inf")
        return
    diff = (actual_f - expected_f).abs()

    def print_part(
        label: str,
        part_actual: torch.Tensor,
        part_diff: torch.Tensor,
        part_expected: torch.Tensor,
        part_mask: torch.Tensor,
    ) -> None:
        values = part_diff[part_mask]
        actual_values = part_actual[part_mask]
        expected_values = part_expected[part_mask]
        relative = values / expected_values.abs().clamp_min(1.0e-8)
        flat_position = int(values.argmax().item())
        print(
            f"{name}[{label}]: max_abs={values.max().item():.6g} "
            f"max_position={flat_position} mean_abs={values.mean().item():.6g} "
            f"actual={actual_values[flat_position].item():.6g} "
            f"reference={expected_values[flat_position].item():.6g} "
            f"max_rel={relative.max().item():.6g} "
            f"mean_rel={relative.mean().item():.6g}"
        )

    if actual_f.shape[-1] == D_HEAD:
        print_part(
            "0:256",
            actual_f[..., :256],
            diff[..., :256],
            expected_f[..., :256],
            both_finite[..., :256],
        )
        print_part(
            "256:512",
            actual_f[..., 256:],
            diff[..., 256:],
            expected_f[..., 256:],
            both_finite[..., 256:],
        )
    else:
        print_part("all", actual_f, diff, expected_f, both_finite)


def _gather_reference_inputs(
    q_dequant: torch.Tensor,
    kv_dequant: torch.Tensor,
    kv_fp8: torch.Tensor,
    indices: torch.Tensor,
    u_scale: torch.Tensor,
    *,
    pair_indices: bool,
    topk_length: Optional[torch.Tensor],
):
    qf = q_dequant.float().reshape(-1, H_Q, D_HEAD)
    kf = kv_dequant.reshape(-1, D_HEAD).float()
    vf = kv_fp8.reshape(-1, D_HEAD).float()
    uf = u_scale.reshape(-1, 2).float()

    if pair_indices:
        idx = indices[:, 0]
        row_for_q = torch.arange(qf.shape[0], device=qf.device) // 2
        idx = idx.index_select(0, row_for_q)
        lengths = (
            torch.full(
                (idx.shape[0],), idx.shape[1], device=qf.device, dtype=torch.int32
            )
            if topk_length is None
            else topk_length.index_select(0, row_for_q)
        )
    else:
        idx = indices.reshape(-1, indices.shape[-1])
        lengths = (
            torch.full(
                (idx.shape[0],), idx.shape[1], device=qf.device, dtype=torch.int32
            )
            if topk_length is None
            else topk_length.reshape(-1)
        )

    safe = idx.clamp(0, kf.shape[0] - 1).long()
    gathered_k = kf.index_select(0, safe.reshape(-1)).reshape(
        idx.shape[0], idx.shape[1], D_HEAD
    )
    gathered_v = vf.index_select(0, safe.reshape(-1)).reshape_as(gathered_k)
    gathered_u = uf.index_select(0, safe.reshape(-1)).reshape(*idx.shape, 2)
    positions = torch.arange(idx.shape[1], device=qf.device)[None, :]
    valid = (idx >= 0) & (idx < kf.shape[0]) & (positions < lengths[:, None])
    return qf, gathered_k, gathered_v, gathered_u, valid


def _cutlass_segment(
    qf: torch.Tensor,
    gathered_k: torch.Tensor,
    gathered_v: torch.Tensor,
    gathered_u: torch.Tensor,
    valid: torch.Tensor,
    w_scale: torch.Tensor,
    begin_block: int,
    end_block: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simulate one kernel segment, including its online mi/li state."""
    rows = qf.shape[0]
    mi = torch.full((rows, H_Q), -1.0e30, device=qf.device, dtype=torch.float32)
    li = torch.zeros_like(mi)
    out = torch.zeros((rows, H_Q, D_HEAD), device=qf.device, dtype=torch.float32)
    v_scale = w_scale.float().repeat_interleave(KV_GROUP_SIZE)
    sm_scale_log2 = SM_SCALE * math.log2(math.e)

    for block_idx in range(begin_block, end_block):
        begin = block_idx * 64
        end = begin + 64
        block_k = gathered_k[:, begin:end]
        block_v = gathered_v[:, begin:end] * v_scale[None, None, :]
        # The kernel's S path reads the token factor from scale group 0.  The
        # second U half remains present in K/V dequantization, exposing any
        # incorrect attempt to use one half's factor for the other half.
        block_u = gathered_u[:, begin:end, 0]
        block_valid = valid[:, begin:end]

        # tcgen05 accumulates QK as sixteen K=32 atoms.  Preserve that
        # accumulation order instead of replacing it with one K=512 GEMM.
        p = torch.zeros(
            (rows, H_Q, block_k.shape[1]), device=qf.device, dtype=torch.float32
        )
        for k_begin in range(0, D_HEAD, MMA_K):
            k_end = k_begin + MMA_K
            p.add_(
                torch.matmul(
                    qf[..., k_begin:k_end],
                    block_k[..., k_begin:k_end].transpose(-1, -2),
                )
            )
        p.masked_fill_(~block_valid[:, None, :], -math.inf)
        p_log2 = p * sm_scale_log2
        cur_max = p_log2.amax(dim=-1)

        # __any_sync makes the rescale decision jointly for heads 0..31 and
        # 32..63. Once selected, each head still updates to its own maximum.
        should_scale = torch.empty_like(cur_max, dtype=torch.bool)
        for head_begin in (0, 32):
            head_end = head_begin + 32
            trigger = (
                cur_max[:, head_begin:head_end] - mi[:, head_begin:head_end] > 6.0
            ).any(dim=-1, keepdim=True)
            should_scale[:, head_begin:head_end] = trigger

        new_max = torch.where(should_scale, torch.maximum(cur_max, mi), mi)
        scale_old = torch.where(should_scale, torch.exp2(mi - new_max), 1.0)
        softmax = torch.exp2(p_log2 - new_max[..., None])
        softmax = torch.where(block_valid[:, None, :], softmax, 0.0)

        # The kernel transfers the rank-1 token factor U from V into S before
        # E4M3 conversion; S is then normalized by a per-row UE8M0 scale and
        # the O MMA supplies that scale together with W as SFA/SFB.
        # CUTLASS uses cvt.rn.satfinite.e4m3; Torch's direct float8 cast
        # returns NaN on overflow, so apply the finite E4M3 endpoint first.
        s_value = (softmax * block_u[:, None, :])
        s_grouped = s_value.reshape(*s_value.shape[:-1], 2, MMA_K)
        s_abs_max = s_grouped.amax(dim=-1)
        raw_s_scale = torch.where(
            s_abs_max > 0,
            s_abs_max / FP8_E4M3_MAX,
            torch.ones_like(s_abs_max),
        )
        raw_s_scale = raw_s_scale.clamp_min(1.0e-20)
        # UE8M0 conversion uses round-up (cvt.rp), i.e. the next power of two.
        s_scale = torch.exp2(torch.ceil(torch.log2(raw_s_scale)))
        s_quantized = (s_grouped / s_scale[..., None]).to(torch.float8_e4m3fn)
        s_e4m3 = s_quantized.float()
        # Match the kernel's single-thread S dump: row 0, head 0, first
        # K=32 atom.
        for print_atom in range(2):
            s_values = s_e4m3[PRINT_S_ROW, PRINT_S_HEAD, print_atom].detach().cpu().tolist()
            s_scale_value = float(s_scale[PRINT_S_ROW, PRINT_S_HEAD, print_atom].item())
            entries = ", ".join(
                f"{value:g}"
                for value in s_values
            )
            print(
                f"python cutlass k={block_idx} head={PRINT_S_HEAD} atom={print_atom} "
                f"s_scale={s_scale_value:g} s=[{entries}]",
                flush=True,
            )
        # The S/V tile is likewise two K=32 atoms, with FP32 accumulation
        # retained between them.
        block_out = torch.zeros_like(out)
        for k_begin in range(0, block_k.shape[1], MMA_K):
            k_end = k_begin + MMA_K
            block_out.add_(
                torch.matmul(
                    s_e4m3[..., k_begin // MMA_K, :].float()
                    * s_scale[..., k_begin // MMA_K, None],
                    block_v[:, k_begin:k_end],
                )
            )
        out = out * scale_old[..., None] + block_out
        li = li * scale_old + softmax.sum(dim=-1)
        mi = new_max

    normalized = torch.where(li[..., None] != 0, out / li[..., None], 0.0)
    lse_log2 = torch.where(li != 0, torch.log2(li) + mi, math.inf)
    return normalized, lse_log2


def _decode_split_ranges(
    scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    topk: int,
) -> list[Tuple[int, int]]:
    """Recover the batch-0 block ranges consumed by decode's split kernel."""
    first_split = int(num_splits[0].item())
    split_count = int((num_splits[1] - num_splits[0]).item())
    num_blocks = (topk + 63) // 64
    if split_count == 1:
        return [(0, num_blocks)]

    ranges: list[Optional[Tuple[int, int]]] = [None] * split_count
    for row in scheduler_metadata.cpu().tolist():
        begin_req, end_req, begin_block, end_block, begin_split = row[:5]
        if begin_req != 0 or end_req != 0:
            continue
        split_idx = begin_split - first_split
        if 0 <= split_idx < split_count:
            ranges[split_idx] = (begin_block, end_block)
    if any(block_range is None for block_range in ranges):
        raise AssertionError(
            f"could not recover all decode split ranges from scheduler metadata: {ranges}"
        )
    result = [block_range for block_range in ranges if block_range is not None]
    if result[0][0] != 0 or result[-1][1] != num_blocks:
        raise AssertionError(f"decode split ranges do not cover all blocks: {result}")
    if any(left[1] != right[0] for left, right in zip(result, result[1:])):
        raise AssertionError(f"decode split ranges are not contiguous: {result}")
    return result


def _cutlass_flow_reference(
    q_dequant: torch.Tensor,
    kv_dequant: torch.Tensor,
    kv_fp8: torch.Tensor,
    indices: torch.Tensor,
    u_scale: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    pair_indices: bool,
    topk_length: Optional[torch.Tensor] = None,
    split_ranges: Optional[list[Tuple[int, int]]] = None,
) -> torch.Tensor:
    """Mirror CUTLASS QK/softmax/SV, plus decode split combine when given."""
    qf, gathered_k, gathered_v, gathered_u, valid = _gather_reference_inputs(
        q_dequant,
        kv_dequant,
        kv_fp8,
        indices,
        u_scale,
        pair_indices=pair_indices,
        topk_length=topk_length,
    )
    num_blocks = indices.shape[-1] // 64
    # Prefill is one persistent online-softmax segment. Decode receives the
    # scheduler's exact split ranges and then follows the combine kernel.
    ranges = [(0, num_blocks)] if split_ranges is None else split_ranges
    local_out, local_lse = zip(*(
        _cutlass_segment(
            qf,
            gathered_k,
            gathered_v,
            gathered_u,
            valid,
            w_scale,
            begin_block,
            end_block,
        )
        for begin_block, end_block in ranges
    ))

    if len(local_out) == 1:
        out = local_out[0]
    else:
        stacked_out = torch.stack(local_out)
        stacked_lse = torch.stack(local_lse)
        max_lse = stacked_lse.amax(dim=0)
        global_lse = torch.log2(torch.exp2(stacked_lse - max_lse).sum(dim=0)) + max_lse
        weights = torch.exp2(stacked_lse - global_lse)
        out = (stacked_out * weights[..., None]).sum(dim=0)
    out = out.reshape_as(q_dequant)
    out_flat = out.reshape(-1, H_Q, D_HEAD)
    print(
        "python O before bf16 q=0 head=59 dim=122 "
        f"value={out_flat[0, 59, 122].float().item():.9g}",
        flush=True,
    )
    print(
        "python O before bf16 q=0 head=59 dim=432 "
        f"value={out_flat[0, 59, 432].float().item():.9g}",
        flush=True,
    )
    return out.to(torch.bfloat16)


@torch.inference_mode()
def run_prefill(
    ext,
    device: torch.device,
    w_exponents: torch.Tensor,
    bf16_ext,
    two_anchors: bool,
) -> None:
    s_q, s_kv, topk = 2, 256, 128
    packed_q, q_dequant = _make_prefill_q(s_q, device, 1001)
    data_bytes, scale_slots, kv_dequant, u_scale, kv_fp8 = _make_rank1_storage(
        s_kv, device, 1002, w_exponents, two_anchors
    )
    packed_kv = _pack_prefill_storage(data_bytes, scale_slots)
    indices = _make_indices(s_q // 2, topk, s_kv, device, 1003).unsqueeze(1)
    w1, w2, w_scale = _w_arguments(device, w_exponents, two_anchors)
    actual, actual_max, actual_lse = ext.dual_mxfp8_head64_sparse_prefill_fwd(
        packed_q,
        packed_kv,
        indices,
        SM_SCALE,
        w1,
        w2,
        None,
        None,
    )
    
    bf16_indices = indices.expand(s_q, -1, -1).contiguous()
    bf16_actual, bf16_max, bf16_lse = bf16_ext.bf16_sparse_prefill_fwd(
        q_dequant.to(torch.bfloat16),
        kv_dequant.reshape(s_kv, 1, D_HEAD).to(torch.bfloat16),
        bf16_indices,
        SM_SCALE,
        None,
        None,
    )
    torch_expected, expected_max, expected_lse = _torch_reference(
        q_dequant,
        kv_dequant,
        indices,
        pair_indices=True,
        dtype=torch.float32
    )
    cutlass_expected = _cutlass_flow_reference(
        q_dequant,
        kv_dequant,
        kv_fp8,
        indices,
        u_scale,
        w_scale,
        pair_indices=True,
    )
    torch.cuda.synchronize()
    # _summary("mxfp8 kernel vs bf16 kernel", actual, bf16_actual)
    # breakpoint()
    _summary("lse", actual_lse, expected_lse)
    _summary("mxfp8 torch vs mxfp8 kernel", actual, cutlass_expected)
    # _summary("torch vs bf16", bf16_actual, cutlass_expected)
    # _summary("mxfp8 torch vs fp32 torch", cutlass_expected, torch_expected)
    # _summary("mxfp8 torch vs bf16 kernel", torch_expected, bf16_actual)

@torch.inference_mode()
def run_decode(
    ext,
    device: torch.device,
    topk: int,
    w_exponents: torch.Tensor,
    bf16_decode_ext,
    two_anchors: bool,
) -> None:
    batch, s_q, page_size = 1, 1, 64
    num_pages = (topk + page_size - 1) // page_size
    packed_q, q_dequant = _make_prefill_q(batch * s_q, device, 2001)
    packed_q = packed_q.reshape(batch, s_q, H_Q, D_HEAD + 16)
    q_dequant = q_dequant.reshape(batch, s_q, H_Q, D_HEAD)
    data_bytes, scale_slots, kv_dequant, u_scale, kv_fp8 = _make_rank1_storage(
        num_pages * page_size, device, 2002, w_exponents, two_anchors
    )
    packed_kv = _pack_page_storage(
        data_bytes, scale_slots, num_pages=num_pages, page_size=page_size
    )
    kv_dequant = kv_dequant.reshape(num_pages, page_size, 1, D_HEAD)
    indices = torch.arange(topk, device=device, dtype=torch.int32).view(batch, s_q, topk)
    lengths = torch.full((batch,), topk, device=device, dtype=torch.int32)
    
    # The ordinary decode kernel consumes the Model1 FP8 cache format,
    # so quantize the same BF16 KV values and use its dequantized form as
    # the matching Torch reference.
    model1_kv_quantized = quantize_k_cache(
        kv_dequant.to(torch.bfloat16), FP8KVCacheLayout.MODEL1_FP8Sparse
    )
    model1_kv_dequant = dequantize_k_cache(
        model1_kv_quantized, FP8KVCacheLayout.MODEL1_FP8Sparse
    )
    # The Model1 kernel's TMA descriptor requires each page stride to be
    # a multiple of 576 bytes.  quantize_k_cache retains that padding in
    # its storage, but a single-page view can expose the unpadded stride.
    model1_kv = model1_kv_quantized.view(torch.uint8)
    page_stride = ((page_size * 584 + 575) // 576) * 576
    model1_kv = model1_kv.as_strided(
        model1_kv.shape, (page_stride, 584, 584, 1)
    )
    w1, w2, w_scale = _w_arguments(device, w_exponents, two_anchors)
    actual, actual_lse, scheduler_metadata, splits = ext.dual_mxfp8_sparse_decode_fwd(
        packed_q,
        packed_kv,
        indices,
        lengths,
        None,
        None,
        None,
        None,
        None,
        None,
        D_HEAD,
        D_HEAD,
        SM_SCALE,
        w1,
        w2,
    )

    split_ranges = _decode_split_ranges(scheduler_metadata, splits, topk)
    cutlass_expected = _cutlass_flow_reference(
        q_dequant,
        kv_dequant,
        kv_fp8,
        indices,
        u_scale,
        w_scale,
        pair_indices=False,
        topk_length=lengths,
        split_ranges=split_ranges,
    )
    
    bf16_actual, bf16_lse = bf16_decode_ext.head64_decode(
        q_dequant.to(torch.bfloat16), model1_kv, indices, SM_SCALE
    )
    bf16_actual = bf16_actual.reshape_as(actual)

    torch.cuda.synchronize()

    _summary("mxfp8 torch vs mxfp8 decode kernel", actual, cutlass_expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--one", "-o",
        action="store_true",
        help="use W=1 for KV generation, kernel arguments, and both Torch references",
    )
    anchor_group = parser.add_mutually_exclusive_group()
    anchor_group.add_argument(
        "--two-anchors",
        dest="two_anchors",
        action="store_true",
        help="use independent rank-1 U/W factors for the two 256-D halves (default)",
    )
    anchor_group.add_argument(
        "--single-anchor",
        dest="two_anchors",
        action="store_false",
        help="use one U factor and one W anchor for all 512 dimensions",
    )
    parser.set_defaults(two_anchors=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("this test requires an NVIDIA SM100-family GPU")
    torch.cuda.set_device(args.device)
    ext = _load_extension()
    bf16_ext = _load_bf16_extension()
    bf16_decode_ext = _load_head64_decode_extension()
    base_w = (
        W_EXPONENTS_TWO_ANCHORS if args.two_anchors else W_EXPONENTS_SINGLE_ANCHOR
    )
    w_exponents = torch.zeros_like(base_w) if args.one else base_w
    run_prefill(ext, torch.device("cuda"), w_exponents, bf16_ext, args.two_anchors)
    run_decode(
        ext, torch.device("cuda"), 256, w_exponents, bf16_decode_ext, args.two_anchors
    )


if __name__ == "__main__":
    main()
