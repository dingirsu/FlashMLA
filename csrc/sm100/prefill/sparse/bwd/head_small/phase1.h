#pragma once

#include "params.h"

namespace sm100::bwd::head_small {

template<int D_QK, int H_Q>
void run_bwd_phase1_kernel(const SparseAttnBwdParams& params);

}

