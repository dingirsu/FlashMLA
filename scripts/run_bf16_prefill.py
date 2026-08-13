#!/usr/bin/env python3
"""Run or time the standalone SM100 BF16 head64 k512 sparse-prefill extension."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
EXTENSION_PATH = Path(os.environ.get("BF16_EXTENSION_OUTPUT", ROOT / "build/bf16_test_ext.so"))


def _load_extension():
    if not EXTENSION_PATH.exists():
        raise RuntimeError(f"{EXTENSION_PATH} does not exist; run ./compile_sm100.sh bf16")
    spec = importlib.util.spec_from_file_location("bf16_test_ext", EXTENSION_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {EXTENSION_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bf16_sparse_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
):
    return _load_extension().bf16_sparse_prefill_fwd(
        q, kv, indices, sm_scale, attn_sink, topk_length
    )


def _make_inputs(args: argparse.Namespace):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    q = torch.randn(
        args.s_q, 64, 512, dtype=torch.bfloat16, device=device
    ) * args.input_std
    kv = torch.randn(
        args.s_kv, 1, 512, dtype=torch.bfloat16, device=device
    ) * args.input_std
    indices = torch.randint(
        args.s_kv, (args.s_q, 1, args.topk), dtype=torch.int32, device=device
    )
    attn_sink = torch.linspace(-1.0, 1.0, 64, device=device) if args.attn_sink else None
    topk_length = (
        torch.full((args.s_q,), args.active_topk, dtype=torch.int32, device=device)
        if args.active_topk is not None
        else None
    )
    return q, kv, indices, attn_sink, topk_length


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--s-q", type=int, default=32768)
    parser.add_argument("--s-kv", type=int, default=32768)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--active-topk", type=int)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--input-std", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--attn-sink", action="store_true")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.s_q <= 0 or args.s_kv <= 0 or args.topk < 64 or args.topk % 64:
        raise ValueError("require positive s_q/s_kv and topk a multiple of 64")
    if args.topk > args.s_kv:
        raise ValueError("topk must not exceed s_kv")
    if args.active_topk is not None and not 0 < args.active_topk <= args.topk:
        raise ValueError("active-topk must be in [1, topk]")

    q, kv, indices, attn_sink, topk_length = _make_inputs(args)

    def run():
        return bf16_sparse_prefill(
            q, kv, indices, 512 ** -0.5, attn_sink, topk_length
        )

    outputs = run()
    torch.cuda.synchronize()
    if args.trace:
        return

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
    for start, end in zip(starts, ends):
        start.record()
        run()
        end.record()
    torch.cuda.synchronize()
    samples_us = sorted(
        start.elapsed_time(end) * 1.0e3 for start, end in zip(starts, ends)
    )
    median_us = samples_us[len(samples_us) // 2]
    print(
        f"BF16 sparse prefill: s_q={args.s_q} s_kv={args.s_kv} "
        f"topk={args.topk} median={median_us:.3f} us "
        f"p20={samples_us[max(0, int(.2 * (len(samples_us)-1)))]:.3f} us "
        f"p80={samples_us[min(len(samples_us)-1, int(.8 * (len(samples_us)-1)))]:.3f} us"
    )


if __name__ == "__main__":
    main()
