#pragma once

#include "params.h"

namespace sm100::fp8_fwd::head64 {

template<int D_QK>
void run_fp8_fwd_phase1_kernel(const Head64Fp8SparseAttnFwdParams &params);

}
