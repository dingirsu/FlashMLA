#include "phase1.h"
#include "phase1.cuh"

namespace sm100::bwd::head_small {

template void run_bwd_phase1_kernel<128, 8>(const SparseAttnBwdParams& params);
template void run_bwd_phase1_kernel<128, 16>(const SparseAttnBwdParams& params);
template void run_bwd_phase1_kernel<128, 32>(const SparseAttnBwdParams& params);

}
