#pragma once

#include "params.h"

namespace sm100::decode::mxfp8_head64 {

// MXFP8 sparse decode with BF16 RoPE: Q and KV both use the same format as existing FP8 KV cache.
// NoPE part: e4m3 data + block scales, RoPE part: BF16 (not quantized).
template<ModelType MODEL_TYPE>
void run_flash_splitkv_mla_mxfp8_sparse_kernel(const MxFp8SparseAttnDecodeParams &params);

}
