#!/usr/bin/env python3
"""Numerical checks for the two-token, 2-SM BF16 head64 sparse kernel."""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parent
EXTENSION_PATH = Path(os.environ.get("DUAL_EXTENSION_PATH", ROOT / "build/dual_head64_test_ext.so"))


def load_extension():
    if not EXTENSION_PATH.exists():
        raise RuntimeError(f"{EXTENSION_PATH} does not exist; run ./compile_dual.sh")
    spec = importlib.util.spec_from_file_location("dual_head64_test_ext", EXTENSION_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {EXTENSION_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    attn_sink: Optional[torch.Tensor],
    topk_length: Optional[torch.Tensor],
):
    s_q, h_q, _ = q.shape
    s_kv = kv.shape[0]
    topk = indices.shape[-1]

    pair_for_token = torch.arange(s_q, device=q.device, dtype=torch.long) // 2
    token_indices = indices[:, 0].index_select(0, pair_for_token).long()
    safe_indices = token_indices.clamp(0, max(s_kv - 1, 0))
    gathered = kv[:, 0][safe_indices].float()

    scores = torch.einsum("shd,skd->shk", q.float(), gathered) * sm_scale
    positions = torch.arange(topk, device=q.device)[None, :]
    if topk_length is None:
        lengths = torch.full((s_q,), topk, device=q.device, dtype=torch.int32)
    else:
        lengths = topk_length.index_select(0, pair_for_token)
    valid = (token_indices >= 0) & (token_indices < s_kv)
    valid &= positions < lengths.unsqueeze(1)
    scores = scores.masked_fill(~valid.unsqueeze(1), -math.inf)

    raw_lse = torch.logsumexp(scores, dim=-1)
    max_logits = scores.amax(dim=-1)
    denominator = raw_lse
    if attn_sink is not None:
        denominator = torch.logaddexp(raw_lse, attn_sink.float().unsqueeze(0))
    probability = torch.exp(scores - denominator.unsqueeze(-1)).nan_to_num(0.0)
    out = torch.einsum("shk,skd->shd", probability, gathered[..., :512])

    reported_lse = raw_lse.clone()
    reported_lse[reported_lse == -math.inf] = math.inf
    return out, max_logits, reported_lse


def max_finite_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    if not finite.any():
        return 0.0
    return (actual[finite] - expected[finite]).abs().max().item()


def run_case(
    extension,
    *,
    name: str,
    s_q: int,
    s_kv: int,
    d_qk: int,
    topk: int,
    use_sink: bool,
    use_lengths: bool,
    invalid: bool,
    all_invalid: bool = False,
):
    torch.manual_seed(20260812 + d_qk + topk + s_q)
    q = (torch.randn(s_q, 64, d_qk, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    kv = (torch.randn(s_kv, 1, d_qk, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    indices = torch.randint(s_kv, (s_q // 2, 1, topk), device="cuda", dtype=torch.int32)
    if all_invalid:
        indices.fill_(-1)
    elif invalid:
        indices[..., ::11] = -1
        indices[..., 5::17] = s_kv + 7
    attn_sink = (
        torch.linspace(-1.0, 1.0, 64, device="cuda", dtype=torch.float32)
        if use_sink
        else None
    )
    topk_length = None
    if use_lengths:
        topk_length = torch.tensor(
            [(pair * 37) % (topk + 1) for pair in range(s_q // 2)],
            device="cuda",
            dtype=torch.int32,
        )

    sm_scale = d_qk**-0.5
    actual = extension.dual_head64_sparse_prefill_fwd(
        q, kv, indices, sm_scale, attn_sink, topk_length
    )
    expected = reference(q, kv, indices, sm_scale, attn_sink, topk_length)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual[0].float(), expected[0], atol=8e-4, rtol=3.01 / 128,
        msg=lambda msg: f"{name}: output mismatch\n{msg}",
    )
    torch.testing.assert_close(
        actual[1], expected[1], atol=3e-5, rtol=5e-5, equal_nan=True,
        msg=lambda msg: f"{name}: max_logits mismatch\n{msg}",
    )
    torch.testing.assert_close(
        actual[2], expected[2], atol=3e-5, rtol=5e-5, equal_nan=True,
        msg=lambda msg: f"{name}: lse mismatch\n{msg}",
    )
    print(
        f"PASS {name:18s} q={tuple(q.shape)} indices={tuple(indices.shape)} "
        f"out_err={max_finite_error(actual[0].float(), expected[0]):.3e} "
        f"lse_err={max_finite_error(actual[2], expected[2]):.3e}"
    )


def benchmark(extension, args: argparse.Namespace) -> None:
    if args.bench_s_q <= 0 or args.bench_s_q % 2:
        raise ValueError("--bench-s-q must be positive and even")
    if args.bench_s_kv <= 0:
        raise ValueError("--bench-s-kv must be positive")
    if args.bench_topk < 128 or args.bench_topk % 128:
        raise ValueError("--bench-topk must be a positive multiple of 128")
    if args.bench_dim not in (512, 576):
        raise ValueError("--bench-dim must be 512 or 576")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("require --warmup >= 0 and --iterations > 0")

    torch.manual_seed(20260812)
    q = torch.randn(
        args.bench_s_q, 64, args.bench_dim,
        device="cuda", dtype=torch.bfloat16,
    ).contiguous()
    kv = torch.randn(
        args.bench_s_kv, 1, args.bench_dim,
        device="cuda", dtype=torch.bfloat16,
    ).contiguous()
    indices = torch.randint(
        args.bench_s_kv,
        (args.bench_s_q // 2, 1, args.bench_topk),
        device="cuda",
        dtype=torch.int32,
    )
    attn_sink = torch.zeros(64, device="cuda", dtype=torch.float32)
    sm_scale = args.bench_dim**-0.5

    def launch():
        return extension.dual_head64_sparse_prefill_fwd(
            q, kv, indices, sm_scale, attn_sink, None
        )

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
        f"BENCH q=({args.bench_s_q},64,{args.bench_dim}) "
        f"kv=({args.bench_s_kv},1,{args.bench_dim}) "
        f"indices=({args.bench_s_q // 2},1,{args.bench_topk}) "
        f"median={median:.3f} us p20={p20:.3f} us p80={p80:.3f} us"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--bench-s-q", type=int, default=32768)
    parser.add_argument("--bench-s-kv", type=int, default=32768)
    parser.add_argument("--bench-topk", type=int, default=512)
    parser.add_argument("--bench-dim", type=int, default=512)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.device)
    torch.set_float32_matmul_precision("highest")
    extension = load_extension()

    if not args.skip_correctness:
        cases = [
            dict(name="d512-basic", s_q=2, s_kv=257, d_qk=512, topk=128,
                 use_sink=False, use_lengths=False, invalid=False),
            dict(name="d512-pairs", s_q=6, s_kv=521, d_qk=512, topk=256,
                 use_sink=True, use_lengths=False, invalid=False),
            dict(name="d512-masked", s_q=8, s_kv=193, d_qk=512, topk=256,
                 use_sink=True, use_lengths=True, invalid=True),
            dict(name="d576-basic", s_q=4, s_kv=389, d_qk=576, topk=128,
                 use_sink=False, use_lengths=False, invalid=False),
            dict(name="d576-masked", s_q=6, s_kv=271, d_qk=576, topk=256,
                 use_sink=True, use_lengths=True, invalid=True),
            dict(name="topk512-stress", s_q=16, s_kv=1021, d_qk=512, topk=512,
                 use_sink=True, use_lengths=True, invalid=True),
            dict(name="all-invalid", s_q=4, s_kv=129, d_qk=512, topk=128,
                 use_sink=True, use_lengths=True, invalid=False, all_invalid=True),
        ]
        for case in cases:
            run_case(extension, **case)
        print(f"All {len(cases)} dual-head64 numerical cases passed.")

    benchmark(extension, args)


if __name__ == "__main__":
    main()
