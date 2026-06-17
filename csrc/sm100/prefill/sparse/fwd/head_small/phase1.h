#pragma once

#include "params.h"

namespace sm100::fwd::head_small {

template<int D_QK, int H_Q>
void run_fwd_phase1_kernel(const SparseAttnFwdParams& params);

}
