#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import torch

import flash_mla


EXTENSION_PATH = Path("/tmp/fp8_test_ext.so")
if not EXTENSION_PATH.exists():
    raise ImportError(f"{EXTENSION_PATH} does not exist; run ./compile_fp8.sh first")
sys.path.insert(0, str(EXTENSION_PATH.parent))
import fp8_test_ext as ext  # noqa: E402


BATCH = 64
S_Q = 1
S_KV = 32768
TOPK = 512
PAGE_SIZE = 64
H_Q = 64
D_HEAD = 512
INPUT_STD = 0.35
SEED = 20260803
WARMUP = 10
ITERS = 30
FP8_MAX = 448.0


def fp8_sparse_decode(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_scale_w: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    tile_scheduler_metadata: Optional[torch.Tensor] = None,
    num_splits: Optional[torch.Tensor] = None,
    extra_kv: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    d_qk: int = D_HEAD,
    d_v: int = D_HEAD,
    sm_scale: Optional[float] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    if sm_scale is None:
        sm_scale = d_qk**-0.5
    return ext.fp8_sparse_decode_fwd(
        q,
        kv,
        kv_scale_w,
        indices,
        topk_length,
        attn_sink,
        tile_scheduler_metadata,
        num_splits,
        extra_kv,
        extra_indices,
        extra_topk_length,
        d_qk,
        d_v,
        sm_scale,
    )


def _quantize_unit_scale(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    quantized = x.float().clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return quantized, quantized.float().to(torch.bfloat16)


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    return (
        torch.linalg.vector_norm(actual_f32 - expected_f32)
        / torch.linalg.vector_norm(expected_f32).clamp_min(1.0e-12)
    ).item()


def _print_accuracy(
    fp8_outputs: Sequence[torch.Tensor],
    bf16_outputs: Sequence[torch.Tensor],
) -> None:
    print("\nAccuracy (FP8 decode vs BF16 with quant-dequant inputs):")
    print(f"{'tensor':<12} {'max_abs':>12} {'mean_abs':>12} {'relative_l2':>14}")
    for name, actual, expected in zip(("out", "lse"), fp8_outputs, bf16_outputs):
        if not torch.isfinite(actual).all():
            raise RuntimeError(f"FP8 {name} contains NaN or Inf")
        error = (actual.float() - expected.float()).abs()
        print(
            f"{name:<12} {error.amax().item():>12.6g} "
            f"{error.mean().item():>12.6g} {_relative_l2(actual, expected):>14.6g}"
        )


def _time_us(fn: Callable[[], object]) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1.0e3 / ITERS


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--s-q", type=int, default=S_Q)
    parser.add_argument("--s-kv", type=int, default=S_KV)
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--page-size", type=int, default=PAGE_SIZE)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, _ = torch.cuda.get_device_capability()
    if major != 10:
        raise RuntimeError("the FP8 decode kernel requires an SM100-family GPU")
    if args.batch <= 0 or args.s_q <= 0:
        raise ValueError("batch and s_q must be positive")
    if args.s_kv <= 0 or args.s_kv % args.page_size != 0:
        raise ValueError("s_kv must be positive and divisible by page_size")
    if args.topk < 128 or args.topk % 128 != 0 or args.topk > args.s_kv:
        raise ValueError("topk must be a multiple of 128 in [128, s_kv]")

    torch.manual_seed(SEED)
    device = torch.device("cuda")
    num_pages = args.s_kv // args.page_size
    q_data = (
        torch.randn(
            args.batch,
            args.s_q,
            H_Q * D_HEAD + H_Q,
            device=device,
            dtype=torch.bfloat16,
        )
        * INPUT_STD
    )
    kv_data = (
        torch.randn(
            num_pages,
            args.page_size,
            1,
            D_HEAD + 16,
            device=device,
            dtype=torch.bfloat16,
        )
        * INPUT_STD
    )
    q_fp8_data, q_dequant_data = _quantize_unit_scale(
        q_data[..., : H_Q * D_HEAD]
    )
    q_fp8 = torch.zeros_like(q_data, dtype=torch.float8_e4m3fn).view(torch.uint8)
    q_fp8[..., : H_Q * D_HEAD] = q_fp8_data.view(torch.uint8)
    q_fp8[..., H_Q * D_HEAD :] = 0x7F
    q_dequant = torch.zeros_like(q_data, dtype=torch.float32)
    q_dequant[..., : H_Q * D_HEAD] = q_dequant_data

    kv_fp8_data, kv_dequant_data = _quantize_unit_scale(
        kv_data[..., :D_HEAD]
    )
    kv_fp8 = torch.zeros_like(kv_data, dtype=torch.float8_e4m3fn).view(torch.uint8)
    kv_fp8[..., :D_HEAD] = kv_fp8_data.view(torch.uint8)
    kv_fp8[..., D_HEAD] = 0x7F
    kv_dequant = torch.zeros_like(kv_data, dtype=torch.float32)
    kv_dequant[..., :D_HEAD] = kv_dequant_data
    kv_scale_w = torch.ones(8, device=device, dtype=torch.float8_e8m0fnu)
    indices = torch.randint(
        args.s_kv,
        (args.batch, args.s_q, args.topk),
        dtype=torch.int32,
        device=device,
    )
    topk_length = torch.full(
        (args.batch,), args.topk, dtype=torch.int32, device=device
    )
    sm_scale = D_HEAD**-0.5

    def invoke_fp8(
        metadata: Optional[torch.Tensor], splits: Optional[torch.Tensor]
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        return fp8_sparse_decode(
            q_fp8,
            kv_fp8,
            kv_scale_w,
            indices,
            topk_length=topk_length,
            tile_scheduler_metadata=metadata,
            num_splits=splits,
            sm_scale=sm_scale,
        )

    if args.profile:
        invoke_fp8(None, None)
        torch.cuda.synchronize()
        return

    q_ref = q_dequant[..., : H_Q * D_HEAD].reshape(
        -1, H_Q, D_HEAD
    ).to(torch.bfloat16)
    kv_ref = kv_dequant[..., :D_HEAD].reshape(
        -1, 1, D_HEAD
    ).to(torch.bfloat16)
    indices_ref = indices.reshape(-1, args.topk).unsqueeze(1)
    topk_length_ref = topk_length.repeat_interleave(args.s_q)

    def run_bf16() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return flash_mla.flash_mla_sparse_fwd(
            q_ref,
            kv_ref,
            indices_ref,
            sm_scale,
            D_HEAD,
            None,
            topk_length_ref,
        )

    bf16_out, _, bf16_lse = run_bf16()
    out, lse, metadata, splits = invoke_fp8(None, None)
    torch.cuda.synchronize()
    if metadata is None or splits is None:
        raise RuntimeError("FP8 decode did not return scheduling metadata")

    bf16_out = bf16_out.reshape(args.batch, args.s_q, H_Q, D_HEAD)
    bf16_lse = bf16_lse.reshape(args.batch, args.s_q, H_Q).transpose(1, 2)

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        f"shape: batch={args.batch}, s_q={args.s_q}, s_kv={args.s_kv}, "
        f"topk={args.topk}, page_size={args.page_size}, "
        f"h_q={H_Q}, d_qk={D_HEAD}, d_v={D_HEAD}, seed={SEED}"
    )
    print("input format: raw E4M3 with unit Q/K/V scales")
    print(
        "Q quant-dequant relative L2: "
        f"{_relative_l2(q_dequant[..., : H_Q * D_HEAD], q_data[..., : H_Q * D_HEAD]):.6g}"
    )
    print(
        "KV quant-dequant relative L2: "
        f"{_relative_l2(kv_dequant[..., :D_HEAD], kv_data[..., :D_HEAD]):.6g}"
    )
    _print_accuracy((out, lse), (bf16_out, bf16_lse))

    def run_fp8() -> object:
        return invoke_fp8(metadata, splits)

    bf16_us = _time_us(run_bf16)
    fp8_us = _time_us(run_fp8)
    print(f"\nTiming ({WARMUP} warmups, {ITERS} iterations):")
    print(f"BF16 sparse prefill reference: {bf16_us:.2f} us")
    print(f"FP8  sparse decode:            {fp8_us:.2f} us")
    print(f"FP8 speedup vs reference:      {bf16_us / fp8_us:.3f}x")


if __name__ == "__main__":
    main()
