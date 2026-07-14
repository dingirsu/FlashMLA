import torch

import flash_mla
from mxfp8_test_utils import (
    D_HEAD,
    assert_close,
    attention_reference,
    make_indices,
    pack_prefill_kv,
    pack_q,
    require_sm100_family,
)


@torch.inference_mode()
def test_mxfp8_sparse_prefill_head64_precision() -> None:
    require_sm100_family()
    torch.manual_seed(20260715)
    device = torch.device("cuda")
    s_q, s_kv, h_q, topk = 3, 192, 64, 128
    sm_scale = D_HEAD**-0.5

    q = torch.randn((s_q, h_q, D_HEAD), device=device, dtype=torch.float32) * 0.35
    kv = torch.randn((s_kv, 1, D_HEAD), device=device, dtype=torch.float32) * 0.35
    packed_q, dequant_q = pack_q(q)
    packed_kv, dequant_kv = pack_prefill_kv(kv)
    indices, topk_length = make_indices(s_q, topk, s_kv, device)
    indices = indices.unsqueeze(1)

    out, max_logits, lse = flash_mla.flash_mla_mxfp8_sparse_prefill(
        packed_q,
        packed_kv,
        indices,
        sm_scale,
        d_qk=D_HEAD,
        d_v=D_HEAD,
        topk_length=topk_length,
    )
    ref_out, ref_max_logits, ref_lse = attention_reference(
        dequant_q, dequant_kv, indices.squeeze(1), topk_length, sm_scale
    )

    assert_close("prefill.out", out, ref_out, atol=3.0e-2, rtol=1.2e-1)
    assert_close("prefill.max_logits", max_logits, ref_max_logits, atol=2.0e-2, rtol=2.0e-2)
    assert_close("prefill.lse", lse, ref_lse, atol=2.0e-2, rtol=5.0e-3)
    assert torch.count_nonzero(out[2]) == 0


if __name__ == "__main__":
    test_mxfp8_sparse_prefill_head64_precision()
    print("MXFP8 sparse prefill head64 precision test passed")

