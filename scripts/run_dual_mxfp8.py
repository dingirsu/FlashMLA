#!/usr/bin/env python3
"""Precision checks for the two-token, 2-SM MXFP8 head64 prefill kernel.

Q uses one real UE8M0 scale per 32 values and K uses one per 64 values.
The kernel duplicates each K scale for tcgen05's 32-value scale vectors.
S is converted directly to E4M3 and both S and V use unit UE8M0 scales.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from mxfp8_test_utils import (  # noqa: E402
    D_HEAD,
    KV_GROUP_SIZE,
    Q_GROUP_SIZE,
    pack_prefill_kv,
    pack_q,
)


EXTENSION_PATH = Path(
    os.environ.get(
        "DUAL_MXFP8_EXTENSION_PATH", ROOT / "build/dual_mxfp8_test_ext.so"
    )
)
B_TOPK = 64
MMA_K = 32
LOG2_E = math.log2(math.e)
MAX_INIT_VAL = -1.0e30
UE8M0_ONE_BITS = 0x7F


def load_extension():
    if not EXTENSION_PATH.exists():
        raise RuntimeError(
            f"{EXTENSION_PATH} does not exist; run "
            "PYTHON=/usr/bin/python3 ./compile_sm100.sh dual_mxfp8"
        )
    spec = importlib.util.spec_from_file_location(
        "dual_mxfp8_test_ext", EXTENSION_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {EXTENSION_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unpack_v_with_unit_scale(packed_kv: torch.Tensor) -> torch.Tensor:
    """Read only E4M3 data; V's UE8M0 scale is exactly one in the kernel."""
    num_tokens = packed_kv.shape[0] * packed_kv.shape[1]
    data = packed_kv.reshape(-1)[: num_tokens * D_HEAD]
    return data.contiguous().view(torch.float8_e4m3fn).float().reshape(
        *packed_kv.shape[:-1], D_HEAD
    )


def scale_bits(packed_q: torch.Tensor, packed_kv: torch.Tensor):
    q_scales = packed_q[..., D_HEAD:]
    num_tokens = packed_kv.shape[0] * packed_kv.shape[1]
    k_scales = packed_kv.reshape(-1)[num_tokens * D_HEAD :]
    k_scales = k_scales.reshape(*packed_kv.shape[:-1], D_HEAD // KV_GROUP_SIZE)
    return q_scales, k_scales


def qk_mma_reference(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Mirror the 2-SM QK result layout and its K=32 accumulation steps."""
    q_dup = torch.cat((q, q), dim=0)
    p_dup = torch.zeros(
        (2 * q.shape[0], B_TOPK), dtype=torch.float32, device=q.device
    )
    for d_start in range(0, D_HEAD, MMA_K):
        d_end = d_start + MMA_K
        p_dup.add_(
            q_dup[:, d_start:d_end]
            @ k[:, d_start:d_end].transpose(0, 1)
        )
    return torch.cat(
        (p_dup[: q.shape[0], :MMA_K], p_dup[q.shape[0] :, MMA_K:]),
        dim=1,
    )


def sv_mma_reference(
    out: torch.Tensor, s_e4m3: torch.Tensor, v_e4m3: torch.Tensor
) -> None:
    """Accumulate S@V in the same K=32 chunks as tcgen05."""
    for k_start in range(0, B_TOPK, MMA_K):
        k_end = k_start + MMA_K
        out.add_(s_e4m3[:, k_start:k_end] @ v_e4m3[k_start:k_end])


def tiled_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v_unit_scaled: torch.Tensor,
    pair_indices: torch.Tensor,
    sm_scale: float,
    attn_sink: Optional[torch.Tensor],
    pair_topk_length: Optional[torch.Tensor],
):
    """Emulate Q/K block scales, online softmax, and unit-scaled S/V MMA."""
    s_q, h_q, _ = q.shape
    topk = pair_indices.shape[-1]
    k = k[:, 0].float()
    v_unit_scaled = v_unit_scaled[:, 0].float()
    pair_indices = pair_indices[:, 0]

    out_rows = []
    max_rows = []
    lse_rows = []
    for q_idx in range(s_q):
        pair_idx = q_idx // 2
        length = (
            topk
            if pair_topk_length is None
            else int(pair_topk_length[pair_idx].item())
        )
        length = max(0, min(length, topk))
        num_k_blocks = max((length + B_TOPK - 1) // B_TOPK, 1)

        mi = torch.full((h_q,), MAX_INIT_VAL, dtype=torch.float32, device=q.device)
        li = torch.zeros((h_q,), dtype=torch.float32, device=q.device)
        real_mi = torch.full((h_q,), -math.inf, dtype=torch.float32, device=q.device)
        out = torch.zeros((h_q, D_HEAD), dtype=torch.float32, device=q.device)

        for tile_idx in range(num_k_blocks):
            tile_start = tile_idx * B_TOPK
            tile_indices = pair_indices[pair_idx, tile_start : tile_start + B_TOPK]
            positions = torch.arange(
                tile_start, tile_start + B_TOPK, device=q.device
            )
            valid = (
                (tile_indices >= 0)
                & (tile_indices < k.shape[0])
                & (positions < length)
            )
            safe_indices = tile_indices.clamp(0, k.shape[0] - 1).long()
            gathered_k = k.index_select(0, safe_indices)
            gathered_v = v_unit_scaled.index_select(0, safe_indices)

            p = qk_mma_reference(q[q_idx].float(), gathered_k)
            p.mul_(sm_scale * LOG2_E)
            p.masked_fill_(~valid.unsqueeze(0), -math.inf)

            cur_pi_max = p.amax(dim=-1)
            real_mi = torch.maximum(real_mi, cur_pi_max)
            should_scale_out = cur_pi_max - mi > 6.0
            new_mi = torch.where(
                should_scale_out, torch.maximum(cur_pi_max, mi), mi
            )
            old_out_scale = torch.where(
                should_scale_out,
                torch.exp2(mi - new_mi),
                torch.ones_like(mi),
            )

            s = torch.exp2(p - new_mi.unsqueeze(-1))
            li = li * old_out_scale + s.sum(dim=-1)
            if tile_idx > 0:
                out.mul_(old_out_scale.unsqueeze(-1))

            # The requested S path is registers -> E4M3 with scale=1.
            s_e4m3 = s.to(torch.float8_e4m3fn).float()
            sv_mma_reference(out, s_e4m3, gathered_v)
            mi = new_mi

        all_invalid = torch.isneginf(real_mi)
        mi = torch.where(all_invalid, torch.full_like(mi, -math.inf), mi)
        li = torch.where(all_invalid, torch.zeros_like(li), li)
        denominator = li
        if attn_sink is not None:
            denominator = denominator + torch.exp2(attn_sink.float() * LOG2_E - mi)
        output_scale = torch.where(
            li == 0, torch.zeros_like(li), denominator.reciprocal()
        )

        out_rows.append(out * output_scale.unsqueeze(-1))
        max_rows.append(real_mi * math.log(2.0))
        lse = mi * math.log(2.0) + torch.log(li)
        lse_rows.append(
            torch.where(torch.isneginf(lse), torch.full_like(lse, math.inf), lse)
        )

    return torch.stack(out_rows), torch.stack(max_rows), torch.stack(lse_rows)


def error_summary(actual: torch.Tensor, expected: torch.Tensor) -> str:
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    if not finite.any():
        return "no finite values"
    error = (actual.float() - expected.float()).abs()[finite]
    return f"max_abs={error.max().item():.4g} mean_abs={error.mean().item():.4g}"


@torch.inference_mode()
def run_case(
    extension,
    *,
    name: str,
    s_q: int,
    s_kv: int,
    topk: int,
    use_sink: bool,
    use_lengths: bool,
) -> None:
    torch.manual_seed(20260813 + s_q + s_kv + topk)
    device = torch.device("cuda")

    q_group_gain = torch.exp2(
        torch.linspace(-3.0, 3.0, D_HEAD // Q_GROUP_SIZE, device=device)
    ).repeat_interleave(Q_GROUP_SIZE)
    k_group_gain = torch.exp2(
        torch.tensor([-4, 1, -2, 3, 0, -3, 2, -1], device=device).float()
    ).repeat_interleave(KV_GROUP_SIZE)
    q = torch.randn(s_q, 64, D_HEAD, device=device) * 0.08 * q_group_gain
    kv = torch.randn(s_kv, 1, D_HEAD, device=device) * 0.08 * k_group_gain

    packed_q, dequantized_q = pack_q(q)
    packed_kv, dequantized_k = pack_prefill_kv(kv)
    v_unit_scaled = unpack_v_with_unit_scale(packed_kv)
    q_scale_bits, k_scale_bits = scale_bits(packed_q, packed_kv)
    assert torch.unique(q_scale_bits).numel() > 1
    assert torch.unique(k_scale_bits).numel() > 1
    assert torch.any(q_scale_bits != UE8M0_ONE_BITS)
    assert torch.any(k_scale_bits != UE8M0_ONE_BITS)

    pair_indices = torch.randint(
        0, s_kv, (s_q // 2, 1, topk), device=device, dtype=torch.int32
    )
    if topk >= 128:
        pair_indices[0, 0, 7] = -1
        pair_indices[-1, 0, 19] = s_kv + 11
    pair_topk_length = None
    if use_lengths:
        pair_topk_length = torch.tensor(
            [topk, topk - 37][: s_q // 2], device=device, dtype=torch.int32
        )
    attn_sink = (
        torch.linspace(-1.0, 1.0, 64, device=device, dtype=torch.float32)
        if use_sink
        else None
    )
    sm_scale = D_HEAD**-0.5

    actual = extension.dual_mxfp8_head64_sparse_prefill_fwd(
        packed_q,
        packed_kv,
        pair_indices,
        sm_scale,
        attn_sink,
        pair_topk_length,
    )
    expected = tiled_reference(
        dequantized_q,
        dequantized_k,
        v_unit_scaled,
        pair_indices,
        sm_scale,
        attn_sink,
        pair_topk_length,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual[1], expected[1], atol=2.0e-2, rtol=2.0e-2, equal_nan=True,
        msg=lambda msg: f"{name}: Q/K scale or QK MMA mismatch\n{msg}",
    )
    torch.testing.assert_close(
        actual[2], expected[2], atol=2.0e-2, rtol=5.0e-3, equal_nan=True,
        msg=lambda msg: f"{name}: online softmax mismatch\n{msg}",
    )
    torch.testing.assert_close(
        actual[0].float(), expected[0], atol=5.0e-1, rtol=4.0e-2,
        msg=lambda msg: f"{name}: unit-scaled S/V path mismatch\n{msg}",
    )
    print(
        f"PASS {name:14s} q_scales={torch.unique(q_scale_bits).numel():2d} "
        f"k_scales={torch.unique(k_scale_bits).numel():2d} "
        f"out[{error_summary(actual[0], expected[0])}] "
        f"lse[{error_summary(actual[2], expected[2])}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("this test requires an NVIDIA SM100-family GPU")
    torch.cuda.set_device(args.device)
    torch.set_float32_matmul_precision("highest")
    extension = load_extension()

    cases = [
        dict(
            name="one-tile",
            s_q=2,
            s_kv=80,
            topk=64,
            use_sink=False,
            use_lengths=False,
        ),
        dict(
            name="two-tile-pairs",
            s_q=4,
            s_kv=194,
            topk=128,
            use_sink=True,
            use_lengths=True,
        ),
    ]
    for case in cases:
        run_case(extension, **case)
    print(f"All {len(cases)} dual MXFP8 precision cases passed.")


if __name__ == "__main__":
    main()
