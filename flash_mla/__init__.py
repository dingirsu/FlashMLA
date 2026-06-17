__version__ = "1.0.0"

from flash_mla.flash_mla_interface import (
    get_mla_metadata,
    flash_mla_with_kvcache,
    flash_attn_varlen_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_kvpacked_func,
    flash_mla_sparse_fwd,
    flash_mla_sparse_bwd,
    flash_mla_mxfp8_sparse_prefill,      # TODO: Uncomment when kernel is implemented
    flash_mla_mxfp8_with_kvcache,        # TODO: Uncomment when kernel is implemented
)

__all__ = [
    "get_mla_metadata",
    "flash_mla_with_kvcache",
    "flash_attn_varlen_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_varlen_kvpacked_func",
    "flash_mla_sparse_fwd",
    "flash_mla_sparse_bwd",
    "flash_mla_mxfp8_sparse_prefill",    # TODO: Uncomment when kernel is implemented
    "flash_mla_mxfp8_with_kvcache",      # TODO: Uncomment when kernel is implemented
]
