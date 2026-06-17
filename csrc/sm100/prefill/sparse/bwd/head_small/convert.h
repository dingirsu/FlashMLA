#pragma once

#include "params.h"

namespace sm100::bwd::head_small {

void run_convert_dkv_accum_kernel(const SparseAttnBwdParams& params);

}

