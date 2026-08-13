#pragma once

#include "params.h"

namespace sm100::dual_mxfp8::head64 {

template<SparseAttnFwdMode FWD_MODE, int D_QK>
void run_dual_mxfp8_phase1_kernel(const SparseFwdArgT<FWD_MODE>& params);

}
