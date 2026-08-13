#!/usr/bin/env python3
import argparse
import dataclasses
import math
import os
import sys
from pathlib import Path
from typing import Callable, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
FP8_EXTENSION = Path(os.environ.get("FP8_EXTENSION_OUTPUT", ROOT / "build/fp8_test_ext.so"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import flash_mla  # noqa: E402
from fp8_test_utils import (  # noqa: E402
    D_HEAD,
    H_Q,
    pack_kv_rank1,
    pack_q_per_head,
    require_sm100_family,
)


H_KV = 1


@dataclasses.dataclass(frozen=True)
class Timing:
    median_us: float
    p20_us: float
    p80_us: float


@dataclasses.dataclass(frozen=True)
class Result:
    name: str
    timing: Timing
    effective_tflops: float


class CudaTimer:
    def __init__(
        self,
        warmup: int,
        iterations: int,
        flush_l2_mb: int,
        device: torch.device,
    ) -> None:
        self.warmup = warmup
        self.iterations = iterations
        self.l2_buffer = None
        if flush_l2_mb > 0:
            self.l2_buffer = torch.empty(
                flush_l2_mb * 1024 * 1024,
                dtype=torch.uint8,
                device=device,
            )

    def measure(self, fn: Callable[[], object]) -> Timing:
        for _ in range(self.warmup):
            fn()
        torch.cuda.synchronize()

        starts = [
            torch.cuda.Event(enable_timing=True) for _ in range(self.iterations)
        ]
        ends = [
            torch.cuda.Event(enable_timing=True) for _ in range(self.iterations)
        ]
        for start, end in zip(starts, ends):
            if self.l2_buffer is not None:
                self.l2_buffer.zero_()
            start.record()
            fn()
            end.record()
        torch.cuda.synchronize()

        samples_us = sorted(
            start.elapsed_time(end) * 1.0e3
            for start, end in zip(starts, ends)
        )
        return Timing(
            median_us=_percentile(samples_us, 0.50),
            p20_us=_percentile(samples_us, 0.20),
            p80_us=_percentile(samples_us, 0.80),
        )


def _percentile(values: Sequence[float], quantile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = quantile * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _make_indices(
    rows: int,
    topk: int,
    s_kv: int,
    device: torch.device,
) -> torch.Tensor:
    stride = max(104729 % s_kv, 1)
    while math.gcd(stride, s_kv) != 1:
        stride += 1
    offsets = torch.randint(0, s_kv, (rows, 1), device=device)
    positions = torch.arange(topk, device=device) * stride
    return ((offsets + positions) % s_kv).to(torch.int32).unsqueeze(1)


def _effective_flops(rows: int, logical_topk: int) -> int:
    # QK and SV each perform one multiply-add over D_HEAD.
    return 2 * rows * H_Q * logical_topk * (D_HEAD + D_HEAD)


def _make_result(name: str, timing: Timing, flops: int) -> Result:
    tflops = flops / (timing.median_us * 1.0e-6) / 1.0e12
    return Result(name, timing, tflops)


def _print_results(results: Sequence[Result]) -> None:
    bf16_us = next(
        result.timing.median_us
        for result in results
        if result.name == "BF16 sparse prefill"
    )
    print(
        f"{'implementation':<25} {'median(us)':>11} {'p20(us)':>10} "
        f"{'p80(us)':>10} {'TFLOP/s':>10} {'vs BF16':>10}"
    )
    for result in results:
        speedup = bf16_us / result.timing.median_us
        print(
            f"{result.name:<25} "
            f"{result.timing.median_us:>11.2f} "
            f"{result.timing.p20_us:>10.2f} "
            f"{result.timing.p80_us:>10.2f} "
            f"{result.effective_tflops:>10.2f} "
            f"{speedup:>9.2f}x"
        )


def _check_output(name: str, outputs: object) -> None:
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 3:
        raise RuntimeError(f"{name} returned an unexpected result")
    for output_name, output in zip(("out", "max_logits", "lse"), outputs):
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(f"{name} returned a non-tensor {output_name}")
        if not torch.isfinite(output).all():
            raise RuntimeError(f"{name} produced non-finite {output_name}")
    torch.cuda.synchronize()


def _print_accuracy_sample(
    bf16_outputs: Sequence[torch.Tensor],
    fp8_outputs: Sequence[torch.Tensor],
) -> None:
    rows = min(8, bf16_outputs[0].shape[0])
    print(f"Accuracy sample (first {rows} query rows, FP8 vs BF16):")
    for name, bf16_value, fp8_value in zip(
        ("out", "max_logits", "lse"), bf16_outputs, fp8_outputs
    ):
        reference = bf16_value[:rows].float()
        actual = fp8_value[:rows].float()
        error = (actual - reference).abs()
        relative_l2 = (
            torch.linalg.vector_norm(actual - reference)
            / torch.linalg.vector_norm(reference).clamp_min(1.0e-12)
        )
        print(
            f"  {name:<10} max_abs={error.max().item():.6g}, "
            f"mean_abs={error.mean().item():.6g}, "
            f"relative_l2={relative_l2.item():.6g}"
        )


def _load_fp8_prefill() -> Callable[..., object]:
    if not FP8_EXTENSION.exists():
        raise RuntimeError(
            f"{FP8_EXTENSION} does not exist; run ./compile_fp8.sh first"
        )
    from run_fp8_prefill import fp8_sparse_prefill

    return fp8_sparse_prefill


@torch.inference_mode()
def benchmark(args: argparse.Namespace) -> None:
    torch.cuda.set_device(args.device)
    require_sm100_family()
    torch.manual_seed(args.seed)
    device = torch.device("cuda", args.device)
    fp8_sparse_prefill = _load_fp8_prefill()

    q = torch.randn(
        (args.s_q, H_Q, D_HEAD), dtype=torch.bfloat16, device=device
    ) * args.input_std
    kv = torch.randn(
        (args.s_kv, H_KV, D_HEAD), dtype=torch.bfloat16, device=device
    ) * args.input_std
    indices = _make_indices(args.s_q, args.topk, args.s_kv, device)

    packed_q, q_fp8, q_scale, q_dequantized = pack_q_per_head(q)
    packed_kv, kv_fp8, kv_token_scale, kv_scale_w, kv_dequantized = (
        pack_kv_rank1(kv)
    )
    kv_scale_w = kv_scale_w.to(torch.float8_e8m0fnu)
    del q_fp8, q_scale, q_dequantized
    del kv_fp8, kv_token_scale, kv_dequantized

    topk_length = None
    if args.active_topk is not None:
        topk_length = torch.full(
            (args.s_q,),
            args.active_topk,
            dtype=torch.int32,
            device=device,
        )
    attn_sink = None
    if args.attn_sink:
        attn_sink = torch.linspace(-1.0, 1.0, H_Q, device=device)

    sm_scale = D_HEAD**-0.5

    def bf16_prefill() -> object:
        return flash_mla.flash_mla_sparse_fwd(
            q,
            kv,
            indices,
            sm_scale,
            D_HEAD,
            attn_sink,
            topk_length,
        )

    def fp8_prefill() -> object:
        return fp8_sparse_prefill(
            packed_q,
            packed_kv,
            kv_scale_w,
            indices,
            sm_scale,
            attn_sink,
            topk_length,
        )

    bf16_outputs = bf16_prefill()
    fp8_outputs = fp8_prefill()
    _check_output("BF16 sparse prefill", bf16_outputs)
    _check_output("FP8 sparse prefill", fp8_outputs)
    if args.check_accuracy:
        _print_accuracy_sample(bf16_outputs, fp8_outputs)
    del bf16_outputs, fp8_outputs

    timer = CudaTimer(
        args.warmup,
        args.iterations,
        args.flush_l2_mb,
        device,
    )
    logical_topk = args.active_topk or args.topk
    flops = _effective_flops(args.s_q, logical_topk)
    results = [
        _make_result(
            "BF16 sparse prefill",
            timer.measure(bf16_prefill),
            flops,
        ),
        _make_result(
            "FP8 sparse prefill",
            timer.measure(fp8_prefill),
            flops,
        ),
    ]

    feature_desc = []
    if args.active_topk is not None:
        feature_desc.append(f"active_topk={args.active_topk}")
    if args.attn_sink:
        feature_desc.append("attn_sink")
    suffix = f", {', '.join(feature_desc)}" if feature_desc else ""

    print(f"GPU: {torch.cuda.get_device_name(args.device)}")
    print(f"PyTorch: {torch.__version__}, CUDA runtime: {torch.version.cuda}")
    print(
        f"Shape: s_q={args.s_q}, s_kv={args.s_kv}, h_q={H_Q}, "
        f"h_kv={H_KV}, topk={args.topk}, d_qk=d_v={D_HEAD}{suffix}"
    )
    print(
        f"Timing: {args.warmup} warmups, {args.iterations} samples, "
        f"L2 flush={args.flush_l2_mb} MiB"
    )
    print(
        "FP8 layout: Q has one UE8M0 scale per head; KV token is "
        "512B E4M3 + 1B UE8M0 + 15B padding."
    )
    print("Input generation, quantization, and packing are outside the timed region.")
    print(
        f"Input storage: BF16={((q.nbytes + kv.nbytes) / 2**20):.1f} MiB, "
        f"FP8={((packed_q.nbytes + packed_kv.nbytes + kv_scale_w.nbytes) / 2**20):.1f} MiB"
    )
    _print_results(results)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the SM100 head64 FP8 sparse prefill kernel against "
            "the BF16 sparse prefill kernel."
        )
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--s-q", type=int, default=4096)
    parser.add_argument("--s-kv", type=int, default=32768)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--active-topk", type=int)
    parser.add_argument("--attn-sink", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--flush-l2-mb", type=int, default=256)
    parser.add_argument("--input-std", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--check-accuracy", action="store_true")
    args = parser.parse_args()

    if args.device < 0:
        parser.error("--device must be non-negative")
    if args.s_q <= 0 or args.s_kv <= 0:
        parser.error("--s-q and --s-kv must be positive")
    if args.topk < 64 or args.topk % 64 != 0:
        parser.error("--topk must be a positive multiple of 64")
    if args.topk > args.s_kv:
        parser.error("--topk must not exceed --s-kv")
    if args.active_topk is not None and not 0 < args.active_topk <= args.topk:
        parser.error("--active-topk must be in [1, topk]")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be non-negative and --iterations positive")
    if args.flush_l2_mb < 0:
        parser.error("--flush-l2-mb must be non-negative")
    if args.input_std <= 0:
        parser.error("--input-std must be positive")
    return args


def main() -> None:
    benchmark(_parse_args())


if __name__ == "__main__":
    main()
