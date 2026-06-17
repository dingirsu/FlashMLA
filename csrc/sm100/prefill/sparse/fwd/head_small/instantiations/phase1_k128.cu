#include "../phase1.h"
#include "../phase1.cuh"

namespace sm100::fwd::head_small {

template void run_fwd_phase1_kernel<128, 8>(const SparseAttnFwdParams& params);
template void run_fwd_phase1_kernel<128, 16>(const SparseAttnFwdParams& params);
template void run_fwd_phase1_kernel<128, 24>(const SparseAttnFwdParams& params);
template void run_fwd_phase1_kernel<128, 32>(const SparseAttnFwdParams& params);

}
