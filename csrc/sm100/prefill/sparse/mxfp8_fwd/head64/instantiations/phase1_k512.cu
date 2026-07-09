#include "../phase1.h"
#include "../phase1.cuh"

namespace sm100::mxfp8_fwd::head64 {

template void run_mxfp8_fwd_phase1_kernel<512>(const MxFp8SparseAttnFwdParams& params);

}
