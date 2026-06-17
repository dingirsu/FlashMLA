#include "phase1.h"

#include <torch/extension.h>

namespace sm100::bwd::head_small {

template<int D_QK, int H_Q>
void run_bwd_phase1_kernel(const SparseAttnBwdParams& params) {
    static_assert(D_QK == 128 || D_QK == 192);
    static_assert(H_Q == 8 || H_Q == 16 || H_Q == 24 || H_Q == 32);

    TORCH_CHECK(params.h_kv == 1, "SM100 sparse head-small backward requires h_kv == 1, got ", params.h_kv);
    TORCH_CHECK(params.h_q == H_Q, "SM100 sparse head-small backward h_q mismatch: params.h_q=", params.h_q, ", H_Q=", H_Q);
    TORCH_CHECK(params.d_qk == D_QK, "SM100 sparse head-small backward d_qk mismatch: params.d_qk=", params.d_qk, ", D_QK=", D_QK);
    TORCH_CHECK(params.d_v == 128, "SM100 sparse head-small backward requires d_v == 128, got ", params.d_v);
    TORCH_CHECK(params.topk % 64 == 0, "SM100 sparse head-small backward requires topk % 64 == 0, got ", params.topk);

    TORCH_CHECK(
        false,
        "SM100 sparse head-small backward kernel body is not implemented yet. "
        "The C++/Python API and parameter plumbing are present; implement "
        "sm100::bwd::head_small::run_bwd_phase1_kernel in ",
        __FILE__
    );
}

template void run_bwd_phase1_kernel<128, 8>(const SparseAttnBwdParams& params);
template void run_bwd_phase1_kernel<128, 16>(const SparseAttnBwdParams& params);
template void run_bwd_phase1_kernel<128, 24>(const SparseAttnBwdParams& params);
template void run_bwd_phase1_kernel<128, 32>(const SparseAttnBwdParams& params);
template void run_bwd_phase1_kernel<192, 8>(const SparseAttnBwdParams& params);
template void run_bwd_phase1_kernel<192, 16>(const SparseAttnBwdParams& params);
template void run_bwd_phase1_kernel<192, 24>(const SparseAttnBwdParams& params);
template void run_bwd_phase1_kernel<192, 32>(const SparseAttnBwdParams& params);

}

