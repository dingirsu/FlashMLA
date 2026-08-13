#pragma once

#include "params.h"
#include <type_traits>

namespace sm100::dual_mxfp8::head64 {

template<SparseAttnFwdMode FWD_MODE, int D_QK>
void run_dual_mxfp8_phase1_kernel(
    const std::conditional_t<
        is_decode_v<FWD_MODE>,
        SparseAttnDecodeParams,
        MxFp8SparseAttnFwdParams
    >& params
);

}
