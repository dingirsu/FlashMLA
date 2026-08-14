#include <optional>
#include <vector>

#include <torch/extension.h>

#include "api/common.h"
#include "params.h"
#include "sm100/prefill/sparse/dual_mxfp8/head64/phase1.h"

namespace {

std::vector<at::Tensor> dual_mxfp8_head64_sparse_prefill_interface(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    double sm_scale,
    const std::optional<at::Tensor>& attn_sink,
    const std::optional<at::Tensor>& topk_length
) {
    using bf16 = cutlass::bfloat16_t;

    Arch arch;
    TORCH_CHECK(arch.is_sm100f(), "Dual MXFP8 head64 sparse prefill requires SM100f.");

    KU_CHECK_NDIM(q, 3);
    KU_CHECK_NDIM(kv, 3);
    KU_CHECK_NDIM(indices, 3);
    KU_CHECK_NDIM(attn_sink, 1);
    KU_CHECK_NDIM(topk_length, 1);

    const int s_q = static_cast<int>(q.size(0));
    const int s_kv = static_cast<int>(kv.size(0));
    const int h_q = static_cast<int>(q.size(1));
    const int h_kv = static_cast<int>(kv.size(1));
    const int topk = static_cast<int>(indices.size(2));
    const int num_q_pairs = s_q / 2;
    constexpr int d_qk = 512;
    constexpr int d_v = 512;
    constexpr int q_bytes_per_head = d_qk + 16;
    constexpr int kv_bytes_per_token = d_qk + d_qk / 64;

    TORCH_CHECK(s_q > 0 && s_q % 2 == 0,
                "q must contain an even, positive number of query tokens");
    TORCH_CHECK(h_q == 64, "q must have shape [s_q, 64, 528]");
    TORCH_CHECK(h_kv == 1, "kv must have shape [s_kv, 1, 520]");
    TORCH_CHECK(q.size(2) == q_bytes_per_head,
                "q must store 512 E4M3 bytes followed by 8 UE8M0 scale bytes and 8 padding bytes per head");
    TORCH_CHECK(kv.size(2) == kv_bytes_per_token,
                "kv must provide a 520-byte envelope per token for page-tail scales");
    TORCH_CHECK(topk > 0 && topk % 64 == 0,
                "topk must be a positive multiple of 64");

    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(kv);
    KU_CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(attn_sink);
    KU_CHECK_DEVICE(topk_length);
    KU_CHECK_DTYPE(q, torch::kUInt8);
    KU_CHECK_DTYPE(kv, torch::kUInt8);
    KU_CHECK_DTYPE(indices, torch::kInt32);
    KU_CHECK_DTYPE(attn_sink, torch::kFloat32);
    KU_CHECK_DTYPE(topk_length, torch::kInt32);
    KU_CHECK_LAST_DIM_CONTIGUOUS(q);
    KU_CHECK_CONTIGUOUS(kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    KU_CHECK_LAST_DIM_CONTIGUOUS(attn_sink);
    KU_CHECK_LAST_DIM_CONTIGUOUS(topk_length);
    KU_CHECK_SHAPE(q, s_q, h_q, q_bytes_per_head);
    KU_CHECK_SHAPE(kv, s_kv, h_kv, kv_bytes_per_token);
    KU_CHECK_SHAPE(indices, num_q_pairs, h_kv, topk);
    KU_CHECK_SHAPE(attn_sink, h_q);
    KU_CHECK_SHAPE(topk_length, num_q_pairs);

    at::cuda::CUDAGuard device_guard{static_cast<char>(q.get_device())};
    const auto opts = q.options();
    at::Tensor out = torch::empty({s_q, h_q, d_v}, opts.dtype(torch::kBFloat16));
    at::Tensor max_logits = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));
    at::Tensor lse = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));

    MxFp8SparseAttnFwdParams params = {
        s_q, s_kv, h_q, h_kv, d_qk, d_v, topk,
        static_cast<float>(sm_scale), static_cast<float>(sm_scale) * LOG_2_E,
        q.data_ptr(),
        kv.data_ptr(),
        nullptr,  // The dual kernel reads complete K scales from the page tail.
        indices.data_ptr<int>(),
        ku::get_optional_tensor_ptr<float>(attn_sink),
        ku::get_optional_tensor_ptr<int>(topk_length),
        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)),
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),
        reinterpret_cast<bf16*>(out.data_ptr()),
        max_logits.data_ptr<float>(),
        lse.data_ptr<float>(),
        arch.num_sms,
        at::cuda::getCurrentCUDAStream().stream()
    };

    sm100::dual_mxfp8::head64::run_dual_mxfp8_phase1_kernel<
        SparseAttnFwdMode::Prefill, 512
    >(params);
    return {out, max_logits, lse};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "dual_mxfp8_head64_sparse_prefill_fwd",
        &dual_mxfp8_head64_sparse_prefill_interface
    );
}
