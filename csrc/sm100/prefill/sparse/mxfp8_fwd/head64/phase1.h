#pragma once

#include "params.h"

namespace sm100::mxfp8_fwd::head64 {

// Sparse prefill with MXFP8 Q and packed MXFP4 KV, both using UE8M0 block scales.
template<int D_QK>
void run_mxfp8_fwd_phase1_kernel(const MxFp8SparseAttnFwdParams &params);

}
