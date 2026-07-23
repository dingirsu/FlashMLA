import math
from typing import Optional, Tuple

import torch


D_HEAD = 512
H_Q = 64
FP8_MAX = 448.0
KV_GROUP_SIZE = 64
Q_DATA_BYTES = H_Q * D_HEAD
Q_BYTES_PER_TOKEN = Q_DATA_BYTES + H_Q
KV_SCALE_SLOT_BYTES = 16
KV_BYTES_PER_TOKEN = D_HEAD + KV_SCALE_SLOT_BYTES


def require_sm100_family() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("FP8 precision tests require an SM100-family CUDA device")
    major, _ = torch.cuda.get_device_capability()
    if major != 10:
        raise RuntimeError("FP8 precision tests require an SM100-family CUDA device")


def round_up_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """Match CUTLASS float_ue8m0_t's positive round-up conversion."""
    positive = x > 0
    safe = torch.where(positive, x, torch.ones_like(x))
    exponent = torch.ceil(torch.log2(safe)).clamp(-127, 127)
    rounded = torch.where(positive, torch.exp2(exponent), torch.ones_like(x))
    return rounded.to(torch.float8_e8m0fnu)


def pack_q_per_head(
    q: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack [token, 64, 512] as data-plane followed by 64 head scales."""
    assert q.ndim == 3 and tuple(q.shape[1:]) == (H_Q, D_HEAD)
    q_f32 = q.float()
    raw_scale = q_f32.abs().amax(dim=-1) / FP8_MAX
    scale_e8m0 = round_up_ue8m0(raw_scale)
    scale_f32 = scale_e8m0.float()
    q_fp8 = (q_f32 / scale_f32.unsqueeze(-1)).clamp(
        -FP8_MAX, FP8_MAX
    ).to(torch.float8_e4m3fn)

    packed = torch.zeros(
        (q.shape[0], Q_BYTES_PER_TOKEN),
        dtype=torch.uint8,
        device=q.device,
    )
    packed[:, :Q_DATA_BYTES] = q_fp8.view(torch.uint8).reshape(q.shape[0], -1)
    packed[:, Q_DATA_BYTES:] = scale_e8m0.view(torch.uint8)
    dequantized = q_fp8.float() * scale_f32.unsqueeze(-1)
    return packed, q_fp8.float(), scale_f32, dequantized


def pack_q_raw(
    q_raw: torch.Tensor,
    q_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert q_raw.ndim == 3 and tuple(q_raw.shape[1:]) == (H_Q, D_HEAD)
    assert tuple(q_scale.shape) == tuple(q_raw.shape[:2])
    q_fp8 = q_raw.to(torch.float8_e4m3fn)
    q_scale_e8m0 = q_scale.float().to(torch.float8_e8m0fnu)
    packed = torch.zeros(
        (q_raw.shape[0], Q_BYTES_PER_TOKEN),
        dtype=torch.uint8,
        device=q_raw.device,
    )
    packed[:, :Q_DATA_BYTES] = q_fp8.view(torch.uint8).reshape(q_raw.shape[0], -1)
    packed[:, Q_DATA_BYTES:] = q_scale_e8m0.view(torch.uint8)
    return packed, q_fp8.float(), q_scale_e8m0.float()


def pack_kv_rank1(
    kv: torch.Tensor,
    w_exponents: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack KV with scale(token, group) = U(token) * W(group)."""
    assert kv.ndim == 3 and tuple(kv.shape[1:]) == (1, D_HEAD)
    device = kv.device
    num_groups = D_HEAD // KV_GROUP_SIZE
    grouped = kv.float().reshape(kv.shape[0], num_groups, KV_GROUP_SIZE)

    if w_exponents is None:
        w_exponents = torch.tensor(
            [0, 1, -1, 2, -2, 1, 0, -1],
            dtype=torch.int32,
            device=device,
        )
    else:
        w_exponents = w_exponents.to(device=device, dtype=torch.int32)
    assert tuple(w_exponents.shape) == (num_groups,)

    group_absmax = grouped.abs().amax(dim=-1)
    required_exp = torch.ceil(
        torch.log2((group_absmax / FP8_MAX).clamp_min(2.0**-127))
    ).to(torch.int32)
    u_exponents = (required_exp - w_exponents).amax(dim=-1)
    token_is_zero = group_absmax.amax(dim=-1) == 0
    u_exponents = torch.where(
        token_is_zero, torch.zeros_like(u_exponents), u_exponents
    )
    product_exponents = u_exponents.unsqueeze(-1) + w_exponents
    if product_exponents.amin().item() < -127 or product_exponents.amax().item() > 127:
        raise ValueError("rank-1 product scale is outside the UE8M0 range")

    u_e8m0 = torch.exp2(u_exponents.float()).to(torch.float8_e8m0fnu)
    w_e8m0 = torch.exp2(w_exponents.float()).to(torch.float8_e8m0fnu)
    product_scale = u_e8m0.float().unsqueeze(-1) * w_e8m0.float().unsqueeze(0)
    kv_fp8 = (grouped / product_scale.unsqueeze(-1)).clamp(
        -FP8_MAX, FP8_MAX
    ).to(torch.float8_e4m3fn)

    packed = torch.zeros(
        (kv.shape[0], KV_BYTES_PER_TOKEN),
        dtype=torch.uint8,
        device=device,
    )
    packed[:, :D_HEAD] = kv_fp8.view(torch.uint8).reshape(kv.shape[0], D_HEAD)
    packed[:, D_HEAD] = u_e8m0.view(torch.uint8)
    dequantized_v = (
        kv_fp8.float() * product_scale.unsqueeze(-1)
    ).reshape_as(kv.float())
    return (
        packed,
        kv_fp8.float().reshape_as(kv.float()),
        u_e8m0.float(),
        w_e8m0.float(),
        dequantized_v,
    )


def pack_kv_raw(
    kv_raw: torch.Tensor,
    token_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert kv_raw.ndim == 3 and tuple(kv_raw.shape[1:]) == (1, D_HEAD)
    assert tuple(token_scale.shape) == (kv_raw.shape[0],)
    kv_fp8 = kv_raw.to(torch.float8_e4m3fn)
    scale_e8m0 = token_scale.float().to(torch.float8_e8m0fnu)
    packed = torch.zeros(
        (kv_raw.shape[0], KV_BYTES_PER_TOKEN),
        dtype=torch.uint8,
        device=kv_raw.device,
    )
    packed[:, :D_HEAD] = kv_fp8.view(torch.uint8).reshape(kv_raw.shape[0], D_HEAD)
    packed[:, D_HEAD] = scale_e8m0.view(torch.uint8)
    return packed, kv_fp8.float(), scale_e8m0.float()


def assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> None:
    actual_f = actual.float()
    expected_f = expected.float()
    for predicate, label in (
        (torch.isposinf, "+inf"),
        (torch.isneginf, "-inf"),
        (torch.isnan, "NaN"),
    ):
        if not torch.equal(predicate(actual_f), predicate(expected_f)):
            raise AssertionError(f"{name}: {label} mask mismatch")

    finite = torch.isfinite(expected_f)
    if not torch.allclose(actual_f[finite], expected_f[finite], atol=atol, rtol=rtol):
        diff = (actual_f - expected_f).abs()
        diff = torch.where(finite, diff, torch.zeros_like(diff))
        flat_idx = int(diff.argmax().item())
        bad_idx = tuple(
            int(i) for i in torch.unravel_index(
                torch.tensor(flat_idx, device=diff.device), diff.shape
            )
        )
        raise AssertionError(
            f"{name}: max_abs={diff.max().item():.6g} at {bad_idx}, "
            f"actual={actual_f[bad_idx].item():.6g}, "
            f"expected={expected_f[bad_idx].item():.6g}, "
            f"atol={atol}, rtol={rtol}"
        )
