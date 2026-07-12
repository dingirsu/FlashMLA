#pragma once

#include "params.h"

namespace sm100::mxfp8_fwd::head64 {

// MXFP8 sparse prefill: Q and KV are stored as e4m3 data followed by e8m0 block scales.
template<int D_QK>
void run_mxfp8_fwd_phase1_kernel(const MxFp8SparseAttnFwdParams &params);

}
