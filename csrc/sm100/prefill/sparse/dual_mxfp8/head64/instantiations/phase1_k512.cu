#include "../phase1.h"
#include "../phase1.cuh"

namespace sm100::dual_mxfp8::head64 {

template void run_dual_mxfp8_phase1_kernel<SparseAttnFwdMode::Prefill, 512>(const SparseAttnFwdParams& params);

}
