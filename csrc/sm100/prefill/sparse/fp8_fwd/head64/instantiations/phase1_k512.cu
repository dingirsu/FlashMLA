#include "../phase1.h"
#include "../phase1.cuh"

namespace sm100::fp8_fwd::head64 {

template void run_fp8_fwd_phase1_kernel<512>(
    const Head64Fp8SparseAttnFwdParams &params
);

}
