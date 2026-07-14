import torch

import flash_mla
from mxfp8_test_utils import (
    D_HEAD,
    assert_close,
    attention_reference,
    make_indices,
    pack_decode_kv_pages,
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
    packed_kv, dequant_kv = pack_decode_kv_pages(kv)
    indices, topk_length = make_indices(b, topk, s_kv, device)
    indices = indices.unsqueeze(1)
    attn_sink = torch.linspace(-1.0, 1.0, h_q, device=device, dtype=torch.float32)

    scheduler, _ = flash_mla.get_mla_metadata()
    out, lse = flash_mla.flash_mla_mxfp8_with_kvcache(
        packed_q,
        packed_kv,
        indices,
        d_qk=D_HEAD,
        head_dim_v=D_HEAD,
        tile_scheduler_metadata=scheduler,
        softmax_scale=sm_scale,
        attn_sink=attn_sink,
        topk_length=topk_length,
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


if __name__ == "__main__":
    test_mxfp8_sparse_decode_head64_precision()
    print("MXFP8 sparse decode head64 precision test passed")
