#include <pybind11/pybind11.h>

#include "api/common.h"
#include "params.h"
#include "sm100/prefill/sparse/fwd/head64/phase1.h"

static std::vector<at::Tensor> bf16_sparse_attn_prefill_interface(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    float sm_scale,
    const std::optional<at::Tensor>& attn_sink,
    const std::optional<at::Tensor>& topk_length
) {
    using bf16 = cutlass::bfloat16_t;

    Arch arch = Arch();
    TORCH_CHECK(arch.is_sm100f(), "BF16 head64 sparse prefill requires SM100f.");

    KU_CHECK_NDIM(q, 3);
    KU_CHECK_NDIM(kv, 3);
    KU_CHECK_NDIM(indices, 3);
    KU_CHECK_NDIM(attn_sink, 1);
    KU_CHECK_NDIM(topk_length, 1);

    const int s_q = q.size(0);
    const int s_kv = kv.size(0);
    const int h_q = q.size(1);
    const int h_kv = kv.size(1);
    const int d_qk = q.size(2);
    const int d_v = d_qk;
    const int topk = indices.size(2);

    TORCH_CHECK(h_q == 64 && h_kv == 1, "BF16 test extension expects h_q=64 and h_kv=1");
    TORCH_CHECK(d_qk == 512 && d_v == 512, "BF16 test extension expects d_qk=d_v=512");
    TORCH_CHECK(topk >= 64 && topk % 64 == 0, "topk must be a positive multiple of 64");

    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(kv);
    KU_CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(attn_sink);
    KU_CHECK_DEVICE(topk_length);
    KU_CHECK_DTYPE(q, torch::kBFloat16);
    KU_CHECK_DTYPE(kv, torch::kBFloat16);
    KU_CHECK_DTYPE(indices, torch::kInt32);
    KU_CHECK_DTYPE(attn_sink, torch::kFloat32);
    KU_CHECK_DTYPE(topk_length, torch::kInt32);
    KU_CHECK_LAST_DIM_CONTIGUOUS(q);
    KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    KU_CHECK_LAST_DIM_CONTIGUOUS(attn_sink);
    KU_CHECK_LAST_DIM_CONTIGUOUS(topk_length);
    KU_CHECK_SHAPE(q, s_q, h_q, d_qk);
    KU_CHECK_SHAPE(kv, s_kv, h_kv, d_qk);
    KU_CHECK_SHAPE(indices, s_q, h_kv, topk);
    KU_CHECK_SHAPE(attn_sink, h_q);
    KU_CHECK_SHAPE(topk_length, s_q);

    at::cuda::CUDAGuard device_guard{static_cast<char>(q.get_device())};
    auto opts = q.options();
    at::Tensor out = torch::empty({s_q, h_q, d_v}, opts);
    at::Tensor max_logits = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));
    at::Tensor lse = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));

    SparseAttnFwdParams params = {
        s_q, s_kv, h_q, h_kv, d_qk, d_v, topk,
        sm_scale, sm_scale * LOG_2_E,
        reinterpret_cast<bf16*>(q.data_ptr()),
        reinterpret_cast<bf16*>(kv.data_ptr()),
        reinterpret_cast<int*>(indices.data_ptr()),
        ku::get_optional_tensor_ptr<float>(attn_sink),
        ku::get_optional_tensor_ptr<int>(topk_length),
        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)),
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),
        reinterpret_cast<bf16*>(out.data_ptr()),
        reinterpret_cast<float*>(max_logits.data_ptr()),
        reinterpret_cast<float*>(lse.data_ptr()),
        arch.num_sms,
        at::cuda::getCurrentCUDAStream().stream()
    };

    sm100::fwd::head64::run_fwd_phase1_kernel<512>(params);
    return {out, max_logits, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bf16_sparse_prefill_fwd", &bf16_sparse_attn_prefill_interface);
}
