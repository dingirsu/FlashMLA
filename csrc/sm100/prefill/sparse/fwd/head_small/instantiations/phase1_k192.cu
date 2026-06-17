#include "../phase1.h"
#include "../phase1.cuh"

namespace sm100::fwd::head_small {

template void run_fwd_phase1_kernel<192, 8>(const SparseAttnFwdParams& params);
template void run_fwd_phase1_kernel<192, 16>(const SparseAttnFwdParams& params);
template void run_fwd_phase1_kernel<192, 24>(const SparseAttnFwdParams& params);
template void run_fwd_phase1_kernel<192, 32>(const SparseAttnFwdParams& params);

}
