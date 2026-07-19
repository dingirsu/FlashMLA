#!/usr/bin/env python3
import argparse
import dataclasses
import math
import sys
from pathlib import Path
from typing import Callable, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import flash_mla  # noqa: E402
from mxfp8_test_utils import (  # noqa: E402
    D_HEAD,
    pack_decode_kv_pages_rank1,
    pack_prefill_kv_rank1,
    pack_q,
    require_sm100_family,
)
from quant import FP8KVCacheLayout, quantize_k_cache  # noqa: E402


H_Q = 64
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
    def __init__(self, warmup: int, iterations: int, flush_l2_mb: int) -> None:
        self.warmup = warmup
        self.iterations = iterations
        self.l2_buffer = None
        if flush_l2_mb > 0:
            self.l2_buffer = torch.empty(
                flush_l2_mb * 1024 * 1024,
                dtype=torch.uint8,
                device="cuda",
            )

    def measure(self, fn: Callable[[], object]) -> Timing:
        for _ in range(self.warmup):
            fn()
        torch.cuda.synchronize()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(self.iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(self.iterations)]
        for start, end in zip(starts, ends):
            if self.l2_buffer is not None:
                self.l2_buffer.zero_()
            start.record()
            fn()
            end.record()
        torch.cuda.synchronize()

        samples_us = sorted(start.elapsed_time(end) * 1.0e3 for start, end in zip(starts, ends))
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


def _make_indices(rows: int, topk: int, s_kv: int, device: torch.device) -> torch.Tensor:
    stride = 104729 % s_kv
    stride = max(stride, 1)
    while math.gcd(stride, s_kv) != 1:
        stride += 1
    offsets = torch.randint(0, s_kv, (rows, 1), device=device)
    positions = torch.arange(topk, device=device) * stride
    return ((offsets + positions) % s_kv).to(torch.int32).unsqueeze(1)


def _effective_flops(rows: int, topk: int) -> int:
    return 2 * rows * H_Q * topk * (D_HEAD + D_HEAD)


def _make_result(name: str, timing: Timing, flops: int) -> Result:
    tflops = flops / (timing.median_us * 1.0e-6) / 1.0e12
    return Result(name, timing, tflops)


def _print_results(title: str, results: Sequence[Result], reference: str) -> None:
    reference_us = next(result.timing.median_us for result in results if result.name == reference)
    print(f"\n{title}")
    print(f"{'implementation':<35} {'median(us)':>11} {'p20(us)':>10} {'p80(us)':>10} {'TFLOP/s':>10} {'vs ref':>9}")
    for result in results:
        speedup = reference_us / result.timing.median_us
        print(
            f"{result.name:<35} "
            f"{result.timing.median_us:>11.2f} "
            f"{result.timing.p20_us:>10.2f} "
            f"{result.timing.p80_us:>10.2f} "
            f"{result.effective_tflops:>10.2f} "
            f"{speedup:>8.2f}x"
        )


@torch.inference_mode()
def bench_prefill(args: argparse.Namespace, timer: CudaTimer, device: torch.device) -> None:
    s_q = args.prefill_s_q
    s_kv = args.prefill_s_kv
    topk = args.prefill_topk
    q = torch.randn((s_q, H_Q, D_HEAD), dtype=torch.bfloat16, device=device) * 0.1
    kv = torch.randn((s_kv, H_KV, D_HEAD), dtype=torch.bfloat16, device=device) * 0.1
    indices = _make_indices(s_q, topk, s_kv, device)
    sm_scale = D_HEAD**-0.5

    packed_q, _ = pack_q(q)
    packed_kv, _, kv_scale_w, _, _ = pack_prefill_kv_rank1(kv)

    def bf16_prefill() -> object:
        return flash_mla.flash_mla_sparse_fwd(q, kv, indices, sm_scale, D_HEAD)

    def mxfp8_prefill() -> object:
        return flash_mla.flash_mla_mxfp8_sparse_prefill(
            packed_q,
            packed_kv,
            kv_scale_w,
            indices,
            sm_scale,
            D_HEAD,
            D_HEAD,
        )

    _check_output("BF16 prefill", bf16_prefill())
    _check_output("MXFP8 prefill", mxfp8_prefill())
    flops = _effective_flops(s_q, topk)
    results = [
        _make_result("BF16 sparse prefill", timer.measure(bf16_prefill), flops),
        _make_result("MXFP8 sparse prefill", timer.measure(mxfp8_prefill), flops),
    ]
    _print_results(
        f"Prefill: s_q={s_q}, s_kv={s_kv}, h_q={H_Q}, topk={topk}, d=512",
        results,
        reference="BF16 sparse prefill",
    )


@torch.inference_mode()
def bench_decode(args: argparse.Namespace, timer: CudaTimer, device: torch.device) -> None:
    batch = args.decode_batch
    s_q = args.decode_s_q
    s_kv = args.decode_s_kv
    topk = args.decode_topk
    page_size = args.page_size
    num_pages = s_kv // page_size
    rows = batch * s_q
    sm_scale = D_HEAD**-0.5

    q = torch.randn(
        (batch, s_q, H_Q, D_HEAD), dtype=torch.bfloat16, device=device
    ) * 0.1
    kv = torch.randn(
        (num_pages, page_size, H_KV, D_HEAD),
        dtype=torch.bfloat16,
        device=device,
    ) * 0.1
    indices = _make_indices(rows, topk, s_kv, device).reshape(batch, s_q, topk)

    packed_q, _ = pack_q(q)
    packed_kv, _, kv_scale_w, _, _ = pack_decode_kv_pages_rank1(kv)
    legacy_kv = quantize_k_cache(kv, FP8KVCacheLayout.MODEL1_FP8Sparse)

    extra_kv = None
    packed_extra_kv = None
    legacy_extra_kv = None
    extra_indices = None
    extra_s_kv = args.decode_extra_s_kv
    extra_topk = args.decode_extra_topk
    if extra_topk > 0:
        extra_num_pages = extra_s_kv // page_size
        extra_kv = torch.randn(
            (extra_num_pages, page_size, H_KV, D_HEAD),
            dtype=torch.bfloat16,
            device=device,
        ) * 0.1
        extra_indices = _make_indices(rows, extra_topk, extra_s_kv, device).reshape(
            batch, s_q, extra_topk
        )
        packed_extra_kv, _, extra_scale_w, _, _ = pack_decode_kv_pages_rank1(extra_kv)
        if not torch.equal(kv_scale_w.view(torch.uint8), extra_scale_w.view(torch.uint8)):
            raise RuntimeError("main and extra MXFP8 caches must share W(g)")
        legacy_extra_kv = quantize_k_cache(
            extra_kv, FP8KVCacheLayout.MODEL1_FP8Sparse
        )

    bf16_q = q.reshape(rows, H_Q, D_HEAD)
    bf16_kv_parts = [kv.reshape(-1, H_KV, D_HEAD)]
    bf16_index_parts = [indices.reshape(rows, 1, topk)]
    if extra_kv is not None and extra_indices is not None:
        bf16_kv_parts.append(extra_kv.reshape(-1, H_KV, D_HEAD))
        bf16_index_parts.append(
            extra_indices.reshape(rows, 1, extra_topk) + s_kv
        )
    bf16_kv = torch.cat(bf16_kv_parts, dim=0)
    bf16_indices = torch.cat(bf16_index_parts, dim=-1)

    legacy_meta, _ = flash_mla.get_mla_metadata()
    mxfp8_meta, _ = flash_mla.get_mla_metadata()

    def bf16_decode_shape() -> object:
        return flash_mla.flash_mla_sparse_fwd(
            bf16_q,
            bf16_kv,
            bf16_indices,
            sm_scale,
            D_HEAD,
        )

    def legacy_decode() -> object:
        return flash_mla.flash_mla_with_kvcache(
            q,
            legacy_kv,
            None,
            None,
            D_HEAD,
            legacy_meta,
            None,
            sm_scale,
            False,
            True,
            indices,
            None,
            legacy_extra_kv,
            extra_indices,
        )

    def mxfp8_decode() -> object:
        return flash_mla.flash_mla_mxfp8_with_kvcache(
            packed_q,
            packed_kv,
            kv_scale_w,
            indices,
            D_HEAD,
            D_HEAD,
            mxfp8_meta,
            None,
            sm_scale,
            None,
            packed_extra_kv,
            extra_indices,
        )

    _check_output("BF16 decode-shaped prefill", bf16_decode_shape())
    _check_output("legacy FP8 decode", legacy_decode())
    _check_output("MXFP8 decode", mxfp8_decode())
    total_topk = topk + extra_topk
    flops = _effective_flops(rows, total_topk)
    results = [
        _make_result(
            "BF16 sparse prefill (decode shape)",
            timer.measure(bf16_decode_shape),
            flops,
        ),
        _make_result(
            "legacy FP8 decode (BF16 Q)", timer.measure(legacy_decode), flops
        ),
        _make_result("MXFP8 sparse decode", timer.measure(mxfp8_decode), flops),
    ]
    print(
        "\nThe repository has no dedicated BF16 sparse decode kernel; "
        "the BF16 row uses sparse prefill on the same decode-shaped queries."
    )
    extra_desc = f", extra_topk={extra_topk}" if extra_topk else ""
    _print_results(
        f"Decode: batch={batch}, s_q={s_q}, s_kv={s_kv}, h_q={H_Q}, "
        f"topk={topk}{extra_desc}, d=512",
        results,
        reference="BF16 sparse prefill (decode shape)",
    )
    legacy_us = next(
        result.timing.median_us
        for result in results
        if result.name == "legacy FP8 decode (BF16 Q)"
    )
    mxfp8_us = next(
        result.timing.median_us
        for result in results
        if result.name == "MXFP8 sparse decode"
    )
    print(f"MXFP8 vs dedicated legacy decode: {legacy_us / mxfp8_us:.2f}x")


def _check_output(name: str, outputs: object) -> None:
    if not isinstance(outputs, (list, tuple)) or not outputs:
        raise RuntimeError(f"{name} returned an unexpected result")
    output = outputs[0]
    if not isinstance(output, torch.Tensor) or not torch.isfinite(output).all():
        raise RuntimeError(f"{name} produced a non-finite output")
    torch.cuda.synchronize()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the SM100 MXFP8 sparse kernels with repository baselines."
    )
    parser.add_argument("--mode", choices=("all", "prefill", "decode"), default="all")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--flush-l2-mb", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260719)

    parser.add_argument("--prefill-s-q", type=int, default=4096)
    parser.add_argument("--prefill-s-kv", type=int, default=32768)
    parser.add_argument("--prefill-topk", type=int, default=512)

    parser.add_argument("--decode-batch", type=int, default=128)
    parser.add_argument("--decode-s-q", type=int, default=1)
    parser.add_argument("--decode-s-kv", type=int, default=16384)
    parser.add_argument("--decode-topk", type=int, default=128)
    parser.add_argument("--decode-extra-s-kv", type=int, default=0)
    parser.add_argument("--decode-extra-topk", type=int, default=0)
    parser.add_argument("--page-size", type=int, default=64)
    args = parser.parse_args()

    if args.warmup < 1 or args.iterations < 1:
        parser.error("--warmup and --iterations must be positive")
    if args.flush_l2_mb < 0:
        parser.error("--flush-l2-mb must be non-negative")
    positive_shapes = (
        args.prefill_s_q,
        args.prefill_s_kv,
        args.prefill_topk,
        args.decode_batch,
        args.decode_s_q,
        args.decode_s_kv,
        args.decode_topk,
        args.page_size,
    )
    if any(value <= 0 for value in positive_shapes):
        parser.error("all non-extra shapes and page size must be positive")
    if args.prefill_topk % 128 != 0 or args.decode_topk % 128 != 0:
        parser.error("MXFP8 topk values must be multiples of 128")
    if args.prefill_topk > args.prefill_s_kv or args.decode_topk > args.decode_s_kv:
        parser.error("topk must not exceed its KV sequence length")
    if args.decode_s_kv % args.page_size != 0:
        parser.error("decode s_kv must be a multiple of page size")
    if (args.decode_extra_topk == 0) != (args.decode_extra_s_kv == 0):
        parser.error("extra s_kv and extra topk must either both be zero or both be set")
    if args.decode_extra_topk and (
        args.decode_extra_topk < 0
        or args.decode_extra_s_kv < 0
        or args.decode_extra_topk > args.decode_extra_s_kv
        or args.decode_extra_topk % 128 != 0
        or args.decode_extra_s_kv % args.page_size != 0
    ):
        parser.error("extra topk must be a multiple of 128 and extra s_kv a page multiple")
    return args


def main() -> None:
    args = _parse_args()
    torch.cuda.set_device(args.device)
    require_sm100_family()
    torch.manual_seed(args.seed)
    device = torch.device("cuda", args.device)
    timer = CudaTimer(args.warmup, args.iterations, args.flush_l2_mb)

    print(f"GPU: {torch.cuda.get_device_name(args.device)}")
    print(f"PyTorch: {torch.__version__}, CUDA runtime: {torch.version.cuda}")
    print(
        f"Timing: {args.warmup} warmups, {args.iterations} samples, "
        f"L2 flush={args.flush_l2_mb} MiB"
    )
    print("Input quantization and rank-1 packing are outside the timed region.")
    if args.mode in ("all", "prefill"):
        bench_prefill(args, timer, device)
    if args.mode in ("all", "decode"):
        bench_decode(args, timer, device)


if __name__ == "__main__":
    main()
