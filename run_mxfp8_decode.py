import importlib.util
import os
from pathlib import Path
from typing import Optional, Tuple

import torch


ROOT = Path(__file__).resolve().parent
EXTENSION_PATH = Path(os.environ.get("MXFP8_EXTENSION_OUTPUT", ROOT / "build/mxfp8_test_ext.so"))


def _load_extension():
    spec = importlib.util.spec_from_file_location("mxfp8_test_ext", EXTENSION_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load MXFP8 test extension from {EXTENSION_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ext = _load_extension()


def mxfp8_sparse_decode(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_scale_w: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    tile_scheduler_metadata: Optional[torch.Tensor] = None,
    num_splits: Optional[torch.Tensor] = None,
    extra_kv: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    d_qk: int = 512,
    d_v: int = 512,
    sm_scale: Optional[float] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    if sm_scale is None:
        sm_scale = d_qk**-0.5
    return ext.mxfp8_sparse_decode_fwd(
        q,
        kv,
        kv_scale_w,
        indices,
        topk_length,
        attn_sink,
        tile_scheduler_metadata,
        num_splits,
        extra_kv,
        extra_indices,
        extra_topk_length,
        d_qk,
        d_v,
        sm_scale,
    )


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "tests"))
    from mxfp8_test_utils import (
        assert_close,
        attention_reference,
        pack_decode_kv_pages_rank1,
        pack_q,
    )

    torch.manual_seed(20260717)
    d, h, topk = 512, 64, 128
    batch, s_q = 1, 1
    num_pages, page_size = 3, 64
    s_kv = num_pages * page_size

    use_ones = os.getenv("MXFP8_DECODE_ONES") == "1"
    if use_ones:
        q = torch.ones(batch, s_q, h, d, device="cuda")
        kv = torch.ones(num_pages, page_size, 1, d, device="cuda")
        w_exponents = torch.zeros(8, dtype=torch.int32, device="cuda")
    else:
        q = torch.randn(batch, s_q, h, d, device="cuda") * 0.35
        kv = torch.randn(num_pages, page_size, 1, d, device="cuda") * 0.35
        w_exponents = (
            torch.zeros(8, dtype=torch.int32, device="cuda")
            if os.getenv("MXFP8_DECODE_W_ONE") == "1"
            else None
        )
    packed_q, dequant_q = pack_q(q)
    packed_kv, dequant_kv, kv_scale_w, _, _ = pack_decode_kv_pages_rank1(
        kv, w_exponents=w_exponents
    )
    indices = (
        torch.randperm(s_kv, device="cuda")[:topk]
        .to(torch.int32)
        .view(batch, s_q, topk)
    )
    topk_length = torch.full((batch,), topk, dtype=torch.int32, device="cuda")

    out, lse, _, _ = mxfp8_sparse_decode(
        packed_q,
        packed_kv,
        kv_scale_w,
        indices,
        topk_length=topk_length,
        d_qk=d,
        d_v=d,
        sm_scale=d**-0.5,
    )
    torch.cuda.synchronize()
    ref_out, _, ref_lse = attention_reference(
        dequant_q,
        dequant_kv,
        indices,
        topk_length.view(batch, 1),
        d**-0.5,
    )
    if os.getenv("MXFP8_DECODE_TRACE") == "1":
        gathered = dequant_kv.reshape(-1, d).index_select(
            0, indices[0, 0].to(torch.long)
        )
        raw_p = torch.matmul(dequant_q[0, 0, 0].float(), gathered.T.float())
        print("reference P h0 t0..7=", raw_p[:8].cpu().tolist())
    print(
        f"actual out=[{out.float().amin().item():.6g}, {out.float().amax().item():.6g}] "
        f"lse=[{lse.amin().item():.6g}, {lse.amax().item():.6g}]"
    )
    print(
        f"reference out=[{ref_out.amin().item():.6g}, {ref_out.amax().item():.6g}] "
        f"lse=[{ref_lse.amin().item():.6g}, {ref_lse.amax().item():.6g}]"
    )
    assert_close("decode.lse", lse, ref_lse.transpose(1, 2), atol=2.0e-2, rtol=5.0e-3)
    assert_close("decode.out", out, ref_out, atol=3.0e-2, rtol=1.2e-1)
    print(f"out={tuple(out.shape)}, lse={tuple(lse.shape)}")
    print("MXFP8 rank-1 decode precision test passed")


if __name__ == "__main__":
    main()
