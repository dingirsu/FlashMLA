import importlib.util
import os
from pathlib import Path
from typing import Optional, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
EXTENSION_PATH = Path(os.environ.get("MXFP8_EXTENSION_OUTPUT", ROOT / "build/mxfp8_test_ext.so"))


def _load_extension():
    spec = importlib.util.spec_from_file_location("mxfp8_test_ext", EXTENSION_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load MXFP8 test extension from {EXTENSION_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ext = _load_extension()


def mxfp8_sparse_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_scale_w: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_qk: int,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return ext.mxfp8_sparse_prefill_fwd(
        q,
        kv,
        kv_scale_w,
        indices,
        sm_scale,
        d_qk,
        d_v,
        attn_sink,
        topk_length,
    )


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "tests"))
    from mxfp8_test_utils import pack_prefill_kv_rank1, pack_q

    d, h, s_q, s_kv, topk = 512, 64, 4096, 32768, 512
    q = torch.randn(s_q, h, d, device="cuda") * 0.35
    kv = torch.randn(s_kv, 1, d, device="cuda") * 0.35
    packed_q, _ = pack_q(q)
    packed_kv, _, kv_scale_w, _, _ = pack_prefill_kv_rank1(kv)
    indices = torch.randperm(s_kv, device="cuda")[:topk].to(torch.int32)
    indices = indices.view(1, 1, topk).expand(s_q, 1, topk).contiguous()
    o = mxfp8_sparse_prefill(
        packed_q, packed_kv, kv_scale_w, indices, d**-0.5, d, d
    )
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
