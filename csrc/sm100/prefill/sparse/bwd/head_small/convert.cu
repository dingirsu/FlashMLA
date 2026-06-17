#include "convert.h"

#include <cutlass/bfloat16.h>
#include <kerutils/host/host.h>

namespace sm100::bwd::head_small {

namespace {

using bf16 = cutlass::bfloat16_t;

__global__ void convert_fp32_to_bf16_kernel(
    const float* __restrict__ src,
    bf16* __restrict__ dst,
    int64_t total_elements
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        dst[idx] = bf16(src[idx]);
    }
}

void launch_convert(const float* src, bf16* dst, int64_t total_elements, cudaStream_t stream) {
    if (total_elements == 0) {
        return;
    }
    constexpr int kThreads = 256;
    int blocks = static_cast<int>((total_elements + kThreads - 1) / kThreads);
    convert_fp32_to_bf16_kernel<<<blocks, kThreads, 0, stream>>>(src, dst, total_elements);
    KU_CHECK_KERNEL_LAUNCH();
}

}

void run_convert_dkv_accum_kernel(const SparseAttnBwdParams& params) {
    launch_convert(
        params.d_k_acc,
        params.dk,
        static_cast<int64_t>(params.s_kv) * params.h_kv * params.d_qk,
        params.stream
    );
    launch_convert(
        params.d_v_acc,
        params.dv,
        static_cast<int64_t>(params.s_kv) * params.h_kv * params.d_v,
        params.stream
    );
}

}
