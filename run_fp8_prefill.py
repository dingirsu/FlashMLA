#!/usr/bin/env python3
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


# Keep the benchmark configuration here so kernel-edit/compile/test is one command.
S_Q = 4096
S_KV = 32768
TOPK = 512
H_Q = 64
D_HEAD = 512
INPUT_STD = 0.1
SEED = 20260728
WARMUP = 10
ITERS = 30

FP8_MAX = 448.0
KV_GROUP_SIZE = 64
# UE8M0 absorbs this exact power-of-two slack before the FP16 QK MMA.
Q_ACCUM_HEADROOM_BITS = 3
KV_ACCUM_HEADROOM_BITS = 2
Q_DATA_BYTES = H_Q * D_HEAD
Q_BYTES_PER_TOKEN = Q_DATA_BYTES + H_Q
KV_BYTES_PER_TOKEN = D_HEAD + 16

# W[g] is deliberately non-uniform so all eight epilogue scales are exercised.
KV_W_EXPONENTS = (0, 1, -1, 2, -2, 1, 0, -1)


def fp8_sparse_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_scale_w: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return ext.fp8_sparse_prefill_fwd(
        q,
        kv,
        kv_scale_w,
        indices,
        sm_scale,
        attn_sink,
        topk_length,
    )


def _round_up_e8m0(x: torch.Tensor) -> torch.Tensor:
    positive = x > 0
    safe = torch.where(positive, x, torch.ones_like(x))
    exponent = torch.ceil(torch.log2(safe)).clamp(-127, 127)
    value = torch.where(positive, torch.exp2(exponent), torch.ones_like(x))
    return value.to(torch.float8_e8m0fnu)


def _pack_q(
    q: torch.Tensor, w_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Q absorbs W[g] because QK restores only the per-token U scale.
    w_per_d = w_scale.float().repeat_interleave(KV_GROUP_SIZE)
    q_for_kernel = q.float() * w_per_d
    scale = _round_up_e8m0(
        q_for_kernel.abs().amax(dim=-1)
        / FP8_MAX
        * (2.0**Q_ACCUM_HEADROOM_BITS)
    )
    q_fp8 = (q_for_kernel / scale.float().unsqueeze(-1)).clamp(
        -FP8_MAX, FP8_MAX
    ).to(torch.float8_e4m3fn)

    packed = torch.zeros(
        (q.shape[0], Q_BYTES_PER_TOKEN), dtype=torch.uint8, device=q.device
    )
    packed[:, :Q_DATA_BYTES] = q_fp8.view(torch.uint8).reshape(q.shape[0], -1)
    packed[:, Q_DATA_BYTES:] = scale.view(torch.uint8)
    q_dequant = q_fp8.float() * scale.float().unsqueeze(-1) / w_per_d
    return packed, q_dequant.to(torch.bfloat16)


def _pack_kv(
    kv: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    groups = D_HEAD // KV_GROUP_SIZE
    num_tokens = kv.shape[0]
    grouped = kv.float().reshape(num_tokens, groups, KV_GROUP_SIZE)
    w_exp = torch.tensor(KV_W_EXPONENTS, dtype=torch.int32, device=kv.device)

    required_exp = torch.ceil(
        torch.log2((grouped.abs().amax(dim=-1) / FP8_MAX).clamp_min(2.0**-127))
    ).to(torch.int32)
    u_exp = (required_exp - w_exp).amax(dim=-1)
    u_exp = torch.where(
        grouped.abs().amax(dim=(1, 2)) == 0,
        torch.zeros_like(u_exp),
        u_exp + KV_ACCUM_HEADROOM_BITS,
    )
    product_exp = u_exp.unsqueeze(-1) + w_exp.unsqueeze(0)
    if product_exp.amin().item() < -127 or product_exp.amax().item() > 127:
        raise ValueError("U[token] * W[group] is outside the E8M0 range")

    u_scale = torch.exp2(u_exp.float()).to(torch.float8_e8m0fnu)
    w_scale = torch.exp2(w_exp.float()).to(torch.float8_e8m0fnu)
    product_scale = u_scale.float().unsqueeze(-1) * w_scale.float().unsqueeze(0)
    kv_fp8 = (grouped / product_scale.unsqueeze(-1)).clamp(
        -FP8_MAX, FP8_MAX
    ).to(torch.float8_e4m3fn)

    packed = torch.zeros(
        (num_tokens, KV_BYTES_PER_TOKEN), dtype=torch.uint8, device=kv.device
    )
    packed[:, :D_HEAD] = kv_fp8.view(torch.uint8).reshape(num_tokens, D_HEAD)
    packed[:, D_HEAD] = u_scale.view(torch.uint8)
    kv_dequant = (kv_fp8.float() * product_scale.unsqueeze(-1)).reshape_as(kv)
    return packed, w_scale, kv_dequant.to(torch.bfloat16)


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
    print("\nAccuracy (FP8 vs BF16 with quant-dequant KV):")
    print(f"{'tensor':<12} {'max_abs':>12} {'mean_abs':>12} {'relative_l2':>14}")
    for name, actual, expected in zip(
        ("out", "max_logits", "lse"), fp8_outputs, bf16_outputs
    ):
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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, _ = torch.cuda.get_device_capability()
    if major != 10:
        raise RuntimeError("the FP8 kernel requires an SM100-family GPU")
    if TOPK < 64 or TOPK % 64 != 0 or TOPK > S_KV:
        raise ValueError("TOPK must be a multiple of 64 in [64, S_KV]")

    torch.manual_seed(SEED)
    device = torch.device("cuda")
    q = (
        torch.randn(S_Q, H_Q, D_HEAD, device=device, dtype=torch.bfloat16)
        * INPUT_STD
    )
    kv = (
        torch.randn(S_KV, 1, D_HEAD, device=device, dtype=torch.bfloat16)
        * INPUT_STD
    )
    indices = torch.randint(
        S_KV, (S_Q, 1, TOPK), dtype=torch.int32, device=device
    )

    packed_kv, kv_scale_w, kv_dequant = _pack_kv(kv)
    packed_q, q_dequant = _pack_q(q, kv_scale_w)
    sm_scale = D_HEAD**-0.5

    def run_bf16() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return flash_mla.flash_mla_sparse_fwd(
            q, kv_dequant, indices, sm_scale, D_HEAD
        )

    def run_fp8() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return ext.fp8_sparse_prefill_fwd(
            packed_q, packed_kv, kv_scale_w, indices, sm_scale, None, None
        )

    bf16_outputs = run_bf16()
    fp8_outputs = run_fp8()
    torch.cuda.synchronize()

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        f"shape: s_q={S_Q}, s_kv={S_KV}, topk={TOPK}, "
        f"h_q={H_Q}, d={D_HEAD}, seed={SEED}"
    )
    print(f"W[g]: {kv_scale_w.float().cpu().tolist()}")
    print(f"Q quant-dequant relative L2: {_relative_l2(q_dequant, q):.6g}")
    _print_accuracy(fp8_outputs, bf16_outputs)

    bf16_us = _time_us(run_bf16)
    fp8_us = _time_us(run_fp8)
    print(f"\nTiming ({WARMUP} warmups, {ITERS} iterations):")
    print(f"BF16 sparse prefill: {bf16_us:.2f} us")
    print(f"FP8  sparse prefill: {fp8_us:.2f} us")
    print(f"FP8 speedup:         {bf16_us / fp8_us:.3f}x")


if __name__ == "__main__":
    main()
