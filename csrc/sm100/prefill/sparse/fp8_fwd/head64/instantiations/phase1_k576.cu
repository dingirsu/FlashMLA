#define FP8_FWD_QK576 1

#include "../phase1.h"
#include "../phase1.cuh"

namespace sm100::fp8_fwd::head64 {

template void run_fp8_fwd_phase1_kernel<576>(
    const Head64Fp8SparseAttnFwdParams &params
);

}
