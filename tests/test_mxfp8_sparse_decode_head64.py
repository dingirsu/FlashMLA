import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_mxfp8_decode import mxfp8_sparse_decode
from mxfp8_test_utils import (
    D_HEAD,
    assert_close,
    attention_reference,
    make_indices,
    pack_decode_kv_pages_rank1,
    pack_q,
    require_sm100_family,
)


@torch.inference_mode()
def test_mxfp8_sparse_decode_head64_precision() -> None:
    require_sm100_family()
    torch.manual_seed(20260716)
    device = torch.device("cuda")
    b, s_q, h_q, topk = 3, 1, 64, 128
    num_pages, page_size = 3, 64
    s_kv = num_pages * page_size
    sm_scale = D_HEAD**-0.5

    q = torch.randn((b, s_q, h_q, D_HEAD), device=device, dtype=torch.float32) * 0.35
    kv = torch.randn(
        (num_pages, page_size, 1, D_HEAD), device=device, dtype=torch.float32
    ) * 0.35
    packed_q, dequant_q = pack_q(q)
    packed_kv, dequant_kv, kv_scale_w, _, _ = pack_decode_kv_pages_rank1(kv)
    indices, topk_length = make_indices(b, topk, s_kv, device)
    indices = indices.unsqueeze(1)
    attn_sink = torch.linspace(-1.0, 1.0, h_q, device=device, dtype=torch.float32)

    out, lse, _, _ = mxfp8_sparse_decode(
        packed_q,
        packed_kv,
        kv_scale_w,
        indices,
        topk_length=topk_length,
        attn_sink=attn_sink,
        d_qk=D_HEAD,
        d_v=D_HEAD,
        sm_scale=sm_scale,
    )
    ref_out, _, ref_lse = attention_reference(
        dequant_q,
        dequant_kv,
        indices,
        topk_length.view(b, 1),
        sm_scale,
        attn_sink=attn_sink,
    )

    assert_close("decode.out", out, ref_out, atol=3.0e-2, rtol=1.2e-1)
    assert_close("decode.lse", lse, ref_lse.transpose(1, 2), atol=2.0e-2, rtol=5.0e-3)
    assert torch.count_nonzero(out[2]) == 0


@torch.inference_mode()
def test_mxfp8_sparse_decode_head64_multitile_precision() -> None:
    require_sm100_family()
    torch.manual_seed(20260717)
    device = torch.device("cuda")
    b, s_q, h_q, topk = 2, 1, 64, 256
    num_pages, page_size = 5, 64
    s_kv = num_pages * page_size
    sm_scale = D_HEAD**-0.5

    q = torch.randn((b, s_q, h_q, D_HEAD), device=device) * 0.35
    kv = torch.randn((num_pages, page_size, 1, D_HEAD), device=device) * 0.35
    packed_q, dequant_q = pack_q(q)
    packed_kv, dequant_kv, kv_scale_w, _, _ = pack_decode_kv_pages_rank1(kv)
    indices = torch.stack(
        [torch.randperm(s_kv, device=device)[:topk] for _ in range(b)]
    ).to(torch.int32).unsqueeze(1)
    topk_length = torch.tensor([topk, 173], dtype=torch.int32, device=device)
    attn_sink = torch.linspace(-1.0, 1.0, h_q, device=device)

    out, lse, _, _ = mxfp8_sparse_decode(
        packed_q,
        packed_kv,
        kv_scale_w,
        indices,
        topk_length=topk_length,
        attn_sink=attn_sink,
        d_qk=D_HEAD,
        d_v=D_HEAD,
        sm_scale=sm_scale,
    )
    ref_out, _, ref_lse = attention_reference(
        dequant_q,
        dequant_kv,
        indices,
        topk_length.view(b, 1),
        sm_scale,
        attn_sink=attn_sink,
    )

    assert_close("decode.multitile.out", out, ref_out, atol=3.0e-2, rtol=1.2e-1)
    assert_close(
        "decode.multitile.lse",
        lse,
        ref_lse.transpose(1, 2),
        atol=2.0e-2,
        rtol=5.0e-3,
    )


if __name__ == "__main__":
    test_mxfp8_sparse_decode_head64_precision()
    test_mxfp8_sparse_decode_head64_multitile_precision()
    print("MXFP8 sparse decode head64 precision test passed")
