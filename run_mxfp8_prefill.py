#!/usr/bin/env python3
import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import torch

import flash_mla


ROOT = Path(__file__).resolve().parent
EXTENSION_PATH = Path("/tmp/mxfp8_test_ext.so")
H_Q = 64
D_HEAD = 512
DEFAULT_SEED = 20260729


def _load_extension():
    if not EXTENSION_PATH.exists():
        raise ImportError(f"{EXTENSION_PATH} does not exist; run ./compile.sh first")
    spec = importlib.util.spec_from_file_location("mxfp8_test_ext", EXTENSION_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load MXFP8 test extension from {EXTENSION_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ext = _load_extension()


def mxfp8_sparse_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_scale_w: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_qk: int,
    d_v: int = D_HEAD,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return ext.mxfp8_sparse_prefill_fwd(
        q,
        kv,
        kv_scale_w,
        indices,
        sm_scale,
        d_qk,
        d_v,
        attn_sink,
        topk_length,
    )


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    return (
        torch.linalg.vector_norm(actual_f32 - expected_f32)
        / torch.linalg.vector_norm(expected_f32).clamp_min(1.0e-12)
    ).item()


def _print_accuracy(
    mxfp8_outputs: Sequence[torch.Tensor],
    bf16_outputs: Sequence[torch.Tensor],
) -> None:
    print("\nAccuracy (MXFP8 vs BF16 with quant-dequant Q/KV):")
    print(f"{'tensor':<12} {'max_abs':>12} {'mean_abs':>12} {'relative_l2':>14}")
    for name, actual, expected in zip(
        ("out", "max_logits", "lse"), mxfp8_outputs, bf16_outputs
    ):
        if not torch.isfinite(actual).all():
            raise RuntimeError(f"MXFP8 {name} contains NaN or Inf")
        error = (actual.float() - expected.float()).abs()
        print(
            f"{name:<12} {error.amax().item():>12.6g} "
            f"{error.mean().item():>12.6g} {_relative_l2(actual, expected):>14.6g}"
        )


def _time_us(
    fn: Callable[[], object],
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1.0e3 / iterations


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check SM100 MXFP8 sparse-prefill accuracy and speed."
    )
    parser.add_argument("--s-q", type=int, default=4096)
    parser.add_argument("--s-kv", type=int, default=32768)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--input-std", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--trace-only",
        action="store_true",
        help="launch MXFP8 once; use with MXFP8_BARRIER_TIMING=1 builds",
    )
    args = parser.parse_args()
    if args.s_q <= 0 or args.s_kv <= 0:
        parser.error("--s-q and --s-kv must be positive")
    if args.topk <= 0 or args.topk % 64 != 0 or args.topk > args.s_kv:
        parser.error("--topk must be a multiple of 64 in [64, s-kv]")
    if args.input_std <= 0:
        parser.error("--input-std must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be non-negative and --iterations positive")
    return args


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, _ = torch.cuda.get_device_capability()
    if major != 10:
        raise RuntimeError("the MXFP8 kernel requires an SM100-family GPU")

    sys.path.insert(0, str(ROOT / "tests"))
    from mxfp8_test_utils import pack_prefill_kv_rank1, pack_q

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    q = (
        torch.randn(
            args.s_q,
            H_Q,
            D_HEAD,
            device=device,
            dtype=torch.bfloat16,
        )
        * args.input_std
    )
    kv = (
        torch.randn(
            args.s_kv,
            1,
            D_HEAD,
            device=device,
            dtype=torch.bfloat16,
        )
        * args.input_std
    )
    indices = torch.randint(
        args.s_kv,
        (args.s_q, 1, args.topk),
        dtype=torch.int32,
        device=device,
    )
    packed_q, q_dequant = pack_q(q)
    packed_kv, kv_dequant, kv_scale_w, _, _ = pack_prefill_kv_rank1(kv)
    q_reference = q_dequant.to(torch.bfloat16)
    kv_reference = kv_dequant.to(torch.bfloat16)
    sm_scale = D_HEAD**-0.5

    def run_bf16() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return flash_mla.flash_mla_sparse_fwd(
            q_reference,
            kv_reference,
            indices,
            sm_scale,
            D_HEAD,
        )

    def run_mxfp8() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return mxfp8_sparse_prefill(
            packed_q,
            packed_kv,
            kv_scale_w,
            indices,
            sm_scale,
            D_HEAD,
            D_HEAD,
        )

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        f"shape: s_q={args.s_q}, s_kv={args.s_kv}, topk={args.topk}, "
        f"h_q={H_Q}, d={D_HEAD}, seed={args.seed}"
    )
    print(f"W[g]: {kv_scale_w.float().cpu().tolist()}")

    if args.trace_only:
        outputs = run_mxfp8()
        torch.cuda.synchronize()
        if any(not torch.isfinite(output).all() for output in outputs):
            raise RuntimeError("MXFP8 trace launch produced a non-finite tensor")
        print("MXFP8 trace launch completed")
        return

    bf16_outputs = run_bf16()
    mxfp8_outputs = run_mxfp8()
    torch.cuda.synchronize()
    print(f"Q quant-dequant relative L2:  {_relative_l2(q_dequant, q):.6g}")
    print(f"KV quant-dequant relative L2: {_relative_l2(kv_dequant, kv):.6g}")
    _print_accuracy(mxfp8_outputs, bf16_outputs)

    bf16_us = _time_us(run_bf16, args.warmup, args.iterations)
    mxfp8_us = _time_us(run_mxfp8, args.warmup, args.iterations)
    print(f"\nTiming ({args.warmup} warmups, {args.iterations} iterations):")
    print(f"BF16  sparse prefill: {bf16_us:.2f} us")
    print(f"MXFP8 sparse prefill: {mxfp8_us:.2f} us")
    print(f"MXFP8 speedup:        {bf16_us / mxfp8_us:.3f}x")


if __name__ == "__main__":
    main()
