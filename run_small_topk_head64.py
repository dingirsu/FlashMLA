#!/usr/bin/env python3
"""Benchmark the paired-token SM100 small-topk head64 prefill kernel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tests"))

from small_head_test_utils import get_extension  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--s-q", type=int, default=4096)
    parser.add_argument("--s-kv", type=int, default=32768)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--active-topk", type=int)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.s_q <= 0 or args.s_q % 2:
        raise ValueError("--s-q must be positive and even")
    if args.s_kv <= 0:
        raise ValueError("--s-kv must be positive")
    if args.topk < 64 or args.topk % 64 or args.topk > 1280:
        raise ValueError("--topk must be a multiple of 64 in [64, 1280]")
    if args.active_topk is not None and not 0 < args.active_topk <= args.topk:
        raise ValueError("--active-topk must be in [1, topk]")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("require --warmup >= 0 and --iterations > 0")

    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    extension = get_extension()

    q = torch.randn(
        args.s_q, 64, 512, device="cuda", dtype=torch.bfloat16
    )
    kv = torch.randn(
        args.s_kv, 1, 512, device="cuda", dtype=torch.bfloat16
    )
    indices = torch.randint(
        args.s_kv,
        (args.s_q // 2, 1, args.topk),
        device="cuda",
        dtype=torch.int32,
    )
    attn_sink = torch.zeros(64, device="cuda", dtype=torch.float32)
    topk_length = (
        torch.full(
            (args.s_q // 2,),
            args.active_topk,
            device="cuda",
            dtype=torch.int32,
        )
        if args.active_topk is not None
        else None
    )

    def launch():
        return extension.small_topk_head64_fwd(
            q, kv, indices, 512**-0.5, attn_sink, topk_length
        )

    launch()
    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iterations)]
    for start, end in zip(starts, ends):
        start.record()
        launch()
        end.record()
    torch.cuda.synchronize()

    samples_us = sorted(
        start.elapsed_time(end) * 1.0e3 for start, end in zip(starts, ends)
    )
    p20 = samples_us[int(0.2 * (len(samples_us) - 1))]
    median = samples_us[len(samples_us) // 2]
    p80 = samples_us[int(0.8 * (len(samples_us) - 1))]
    print(
        f"small-topk head64 2-SM: q=({args.s_q},64,512) "
        f"kv=({args.s_kv},1,512) "
        f"indices=({args.s_q // 2},1,{args.topk}) "
        f"active_topk={args.active_topk or args.topk} "
        f"median={median:.3f} us p20={p20:.3f} us p80={p80:.3f} us"
    )


if __name__ == "__main__":
    main()
