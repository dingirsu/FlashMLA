#pragma once

#include "params.h"

namespace sm100::mxfp8_fwd::head64 {

// MXFP8 sparse prefill with BF16 RoPE: Q and KV both use the same format as existing FP8 KV cache.
// NoPE part: e4m3 data + block scales, RoPE part: BF16 (not quantized).
template<int D_QK>
void run_mxfp8_fwd_phase1_kernel(const MxFp8SparseAttnFwdParams &params);

}
