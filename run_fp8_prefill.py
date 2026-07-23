import importlib.util
from pathlib import Path
from typing import Optional, Tuple

import torch


EXTENSION_PATH = Path("/tmp/fp8_test_ext.so")


def _load_extension():
    spec = importlib.util.spec_from_file_location("fp8_test_ext", EXTENSION_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load FP8 test extension from {EXTENSION_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ext = _load_extension()


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
