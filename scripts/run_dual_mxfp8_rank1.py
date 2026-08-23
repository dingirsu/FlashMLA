#!/usr/bin/env python3
"""Exercise dual-MXFP8 prefill and decode with an explicit rank-1 KV scale.

The KV data is generated as random E4M3 values.  A fixed dimension scale
``W[g]`` (with ``W[0] == 1``) and independently generated token scales ``U[t]``
form the stored UE8M0 scale:

    scale[t, g] = U[t] * W[g]

The BF16 reference is built by dequantizing that exact E4M3 data and scale,
then running ordinary attention.  This keeps the test independent of the
kernel's input quantizer and makes the rank-1 construction explicit.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import struct
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
EXTENSION_PATH = Path(
    os.environ.get("DUAL_MXFP8_EXTENSION_PATH", ROOT / "build/dual_mxfp8_test_ext.so")
)
D_HEAD = 512
H_Q = 64
KV_GROUP_SIZE = 64
KV_GROUPS = D_HEAD // KV_GROUP_SIZE
KV_SCALE_SLOT_BYTES = 16
KV_RECORD_BYTES = D_HEAD + KV_SCALE_SLOT_BYTES
SM_SCALE = D_HEAD ** -0.5

# W[0] is the anchor.  All factors are exactly representable UE8M0 powers of
# two and are deliberately shared by every token and by both kernels.
W_EXPONENTS = torch.tensor([0, 1, 3, 0, 4, 2, 1, 3], dtype=torch.int32)


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


def _pack_float_bytes(bits: torch.Tensor) -> float:
    values = bits.to(device="cpu", dtype=torch.uint8).tolist()
    assert len(values) == 4
    return struct.unpack("<f", bytes(values))[0]


def _w_arguments(device: torch.device) -> Tuple[float, float, torch.Tensor]:
    assert int(W_EXPONENTS[0]) == 0, "rank-1 W[0] must be the fixed unit anchor"
    bits = (W_EXPONENTS + 127).to(device=device, dtype=torch.uint8)
    w1 = _pack_float_bytes(bits[:4])
    w2 = _pack_float_bytes(bits[4:])
    return w1, w2, torch.exp2(W_EXPONENTS.to(device=device, dtype=torch.float32))


def _make_rank1_storage(
    num_tokens: int,
    device: torch.device,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return raw data bytes, replicated scale slots, dequantized KV, U, FP8."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    # Directly generate finite E4M3 values.  Casting a normal distribution
    # avoids the two E4M3 NaN bit patterns while retaining realistic values.
    fp8 = torch.randn((num_tokens, D_HEAD), device=device, generator=generator).to(
        torch.float8_e4m3fn
    )
    if not torch.isfinite(fp8.float()).all():
        raise AssertionError("random E4M3 generation produced a non-finite value")
    u_exp = torch.randint(
        -3, 4, (num_tokens,), device=device, dtype=torch.int32, generator=generator
    )
    product_exp = u_exp[:, None] + W_EXPONENTS.to(device=device)[None, :]
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
    expected_product = u_scale[:, None] * torch.exp2(
        W_EXPONENTS.to(device=device, dtype=torch.float32)
    )[None, :]
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


def _bf16_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    pair_indices: bool,
    topk_length: Optional[torch.Tensor] = None,
):
    """Reference attention from dequantized values, with BF16 output."""
    # Materialize the reference operands as BF16 after exact FP8*scale
    # dequantization; use FP32 accumulation for a stable baseline.
    qf = q.to(torch.bfloat16).float()
    kvf = kv.to(torch.bfloat16).float().reshape(-1, D_HEAD)
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
    weights = torch.where(torch.isfinite(lse)[..., None], weights, 0.0)
    out = torch.matmul(weights, gathered)
    out = out.reshape_as(qf).to(torch.bfloat16)
    return out, max_logits, lse


def _summary(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    diff = (actual.float() - expected.float()).abs()
    finite = torch.isfinite(diff)
    if not finite.any():
        print(f"{name}: no finite values (actual or expected contains only NaN/Inf)")
        return
    values = diff[finite]
    print(
        f"{name}: max_abs={values.max().item():.6g} "
        f"mean_abs={values.mean().item():.6g} "
        f"actual=[{actual.float().amin().item():.6g},{actual.float().amax().item():.6g}] "
        f"reference=[{expected.float().amin().item():.6g},{expected.float().amax().item():.6g}]"
    )


@torch.inference_mode()
def run_prefill(ext, device: torch.device, strict: bool) -> None:
    s_q, s_kv, topk = 2, 256, 128
    packed_q, q_dequant = _make_prefill_q(s_q, device, 1001)
    data_bytes, scale_slots, kv_dequant, u_scale, _ = _make_rank1_storage(
        s_kv, device, 1002
    )
    packed_kv = _pack_prefill_storage(data_bytes, scale_slots)
    indices = _make_indices(s_q // 2, topk, s_kv, device, 1003).unsqueeze(1)
    w1, w2, w_scale = _w_arguments(device)
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
    expected, expected_max, expected_lse = _bf16_reference(
        q_dequant,
        kv_dequant,
        indices,
        pair_indices=True,
    )
    torch.cuda.synchronize()
    _summary("prefill.out vs BF16", actual, expected)
    _summary("prefill.max_logits vs BF16", actual_max, expected_max)
    _summary("prefill.lse vs BF16", actual_lse, expected_lse)
    print(
        "prefill rank1 scales: "
        f"U range={u_scale.min().item():.4g}..{u_scale.max().item():.4g}, "
        f"W={w_scale.tolist()}"
    )
    if strict:
        torch.testing.assert_close(actual, expected, atol=0.5, rtol=0.12)
        torch.testing.assert_close(actual_max, expected_max, atol=0.08, rtol=0.02)
        torch.testing.assert_close(actual_lse, expected_lse, atol=0.08, rtol=0.02)


@torch.inference_mode()
def run_decode(ext, device: torch.device, topk: int, strict: bool) -> None:
    batch, s_q, page_size = 1, 1, 64
    num_pages = (topk + page_size - 1) // page_size
    packed_q, q_dequant = _make_prefill_q(batch * s_q, device, 2001)
    packed_q = packed_q.reshape(batch, s_q, H_Q, D_HEAD + 16)
    q_dequant = q_dequant.reshape(batch, s_q, H_Q, D_HEAD)
    data_bytes, scale_slots, kv_dequant, u_scale, _ = _make_rank1_storage(
        num_pages * page_size, device, 2002
    )
    packed_kv = _pack_page_storage(
        data_bytes, scale_slots, num_pages=num_pages, page_size=page_size
    )
    kv_dequant = kv_dequant.reshape(num_pages, page_size, 1, D_HEAD)
    indices = torch.arange(topk, device=device, dtype=torch.int32).view(batch, s_q, topk)
    lengths = torch.full((batch,), topk, device=device, dtype=torch.int32)
    w1, w2, w_scale = _w_arguments(device)
    actual, actual_lse, _, splits = ext.dual_mxfp8_sparse_decode_fwd(
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
    expected, _, expected_lse = _bf16_reference(
        q_dequant,
        kv_dequant,
        indices,
        pair_indices=False,
        topk_length=lengths,
    )
    expected_lse = expected_lse.view(batch, s_q, H_Q).transpose(1, 2)
    torch.cuda.synchronize()
    _summary(f"decode[{topk}].out vs BF16", actual, expected)
    _summary(f"decode[{topk}].lse vs BF16", actual_lse, expected_lse)
    print(
        f"decode[{topk}] rank1 scales: U range={u_scale.min().item():.4g}.."
        f"{u_scale.max().item():.4g}, W={w_scale.tolist()}, splits={splits.tolist()}"
    )
    if strict:
        torch.testing.assert_close(actual, expected, atol=0.5, rtol=0.12)
        torch.testing.assert_close(actual_lse, expected_lse, atol=0.08, rtol=0.02)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--strict", action="store_true", help="fail on BF16-reference mismatch")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("this test requires an NVIDIA SM100-family GPU")
    torch.cuda.set_device(args.device)
    torch.set_float32_matmul_precision("high")
    ext = _load_extension()
    run_prefill(ext, torch.device("cuda"), args.strict)
    run_decode(ext, torch.device("cuda"), 64, args.strict)
    run_decode(ext, torch.device("cuda"), 256, args.strict)


if __name__ == "__main__":
    main()
