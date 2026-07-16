#include <optional>
#include <vector>

#include <torch/extension.h>

#include "api/common.h"
#include "params.h"
#include "sm100/prefill/sparse/bwd/head_small/convert.h"
#include "sm100/prefill/sparse/bwd/head_small/phase1.h"
#include "sm100/prefill/sparse/fwd_for_small_topk/head128/phase1.h"

namespace {

using bf16 = cutlass::bfloat16_t;

std::vector<at::Tensor> small_topk_fwd(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    double sm_scale,
    const at::Tensor& attn_sink,
    const std::optional<at::Tensor>& topk_length
) {
    Arch arch;
    TORCH_CHECK(arch.is_sm100f(), "small-topk forward test requires an SM100-family GPU");
    TORCH_CHECK(q.is_cuda() && kv.is_cuda() && indices.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(attn_sink.is_cuda(), "attn_sink must be a CUDA tensor");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(kv.scalar_type() == at::kBFloat16, "kv must be bfloat16");
    TORCH_CHECK(indices.scalar_type() == at::kInt, "indices must be int32");
    TORCH_CHECK(attn_sink.scalar_type() == at::kFloat, "attn_sink must be float32");
    TORCH_CHECK(q.dim() == 3 && q.size(1) == 128 && q.size(2) == 512,
                "q must have shape [s_q, 128, 512]");
    TORCH_CHECK(kv.dim() == 3 && kv.size(1) == 1 && kv.size(2) == 512,
                "kv must have shape [s_kv, 1, 512]");
    TORCH_CHECK(indices.dim() == 3 && indices.size(1) == 1,
                "indices must have shape [s_q, 1, topk]");
    TORCH_CHECK(indices.size(0) == q.size(0), "indices and q must have the same s_q");
    TORCH_CHECK(indices.size(2) % 64 == 0, "topk must be divisible by 64");
    TORCH_CHECK(indices.size(2) <= 1280, "this test targets the small-topk path");
    TORCH_CHECK(attn_sink.numel() == 128, "attn_sink must contain 128 elements");
    TORCH_CHECK(q.stride(2) == 1 && kv.stride(2) == 1 && indices.stride(2) == 1,
                "last dimensions must be contiguous");

    int s_q = static_cast<int>(q.size(0));
    int s_kv = static_cast<int>(kv.size(0));
    int topk = static_cast<int>(indices.size(2));
    int* topk_length_ptr = nullptr;
    if (topk_length.has_value()) {
        TORCH_CHECK(topk_length->is_cuda() && topk_length->scalar_type() == at::kInt,
                    "topk_length must be a CUDA int32 tensor");
        TORCH_CHECK(topk_length->dim() == 1 && topk_length->numel() == s_q,
                    "topk_length must have shape [s_q]");
        topk_length_ptr = topk_length->data_ptr<int>();
    }

    at::cuda::CUDAGuard guard(q.device());
    at::Tensor out = torch::empty({s_q, 128, 512}, q.options());
    at::Tensor max_logits = torch::empty({s_q, 128}, q.options().dtype(at::kFloat));
    at::Tensor lse = torch::empty({s_q, 128}, q.options().dtype(at::kFloat));

    SparseAttnFwdParams params = {
        s_q, s_kv, 128, 1, 512, 512, topk,
        static_cast<float>(sm_scale), static_cast<float>(sm_scale) * LOG_2_E,
        reinterpret_cast<bf16*>(q.data_ptr()),
        reinterpret_cast<bf16*>(kv.data_ptr()),
        indices.data_ptr<int>(),
        attn_sink.data_ptr<float>(),
        topk_length_ptr,
        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)),
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),
        reinterpret_cast<bf16*>(out.data_ptr()),
        max_logits.data_ptr<float>(),
        lse.data_ptr<float>(),
        arch.num_sms,
        at::cuda::getCurrentCUDAStream().stream()
    };

    sm100::fwd_for_small_topk::head128::run_fwd_for_small_topk_phase1_kernel<
        SparseAttnFwdMode::Prefill, 512
    >(params);
    return {out, max_logits, lse};
}

std::vector<at::Tensor> head_small_bwd(
    const at::Tensor& d_out,
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& out,
    const at::Tensor& lse,
    const at::Tensor& indices,
    double sm_scale,
    const at::Tensor& attn_sink,
    const std::optional<at::Tensor>& topk_length
) {
    Arch arch;
    TORCH_CHECK(arch.is_sm100f(), "head-small backward test requires an SM100-family GPU");
    TORCH_CHECK(q.is_cuda() && kv.is_cuda() && d_out.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(out.is_cuda() && lse.is_cuda() && indices.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(attn_sink.is_cuda(), "attn_sink must be a CUDA tensor");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && kv.scalar_type() == at::kBFloat16,
                "q and kv must be bfloat16");
    TORCH_CHECK(out.scalar_type() == at::kBFloat16 && d_out.scalar_type() == at::kBFloat16,
                "out and d_out must be bfloat16");
    TORCH_CHECK(lse.scalar_type() == at::kFloat && attn_sink.scalar_type() == at::kFloat,
                "lse and attn_sink must be float32");
    TORCH_CHECK(indices.scalar_type() == at::kInt, "indices must be int32");
    TORCH_CHECK(q.dim() == 3 && q.size(2) == 128, "q must have shape [s_q, h_q, 128]");
    int h_q = static_cast<int>(q.size(1));
    TORCH_CHECK(h_q == 8 || h_q == 16 || h_q == 32,
                "backward test supports h_q in {8,16,32}");
    TORCH_CHECK(kv.dim() == 3 && kv.size(1) == 1 && kv.size(2) == 128,
                "kv must have shape [s_kv, 1, 128]");
    TORCH_CHECK(d_out.sizes() == at::IntArrayRef({q.size(0), q.size(1), 128}),
                "d_out must have shape [s_q, h_q, 128]");
    TORCH_CHECK(out.sizes() == d_out.sizes(), "out and d_out must have the same shape");
    TORCH_CHECK(lse.dim() == 2 && lse.size(0) == q.size(0) && lse.size(1) == h_q,
                "lse must have shape [s_q, h_q]");
    TORCH_CHECK(indices.dim() == 3 && indices.size(0) == q.size(0) && indices.size(1) == 1,
                "indices must have shape [s_q, 1, topk]");
    TORCH_CHECK(indices.size(2) % 64 == 0, "topk must be divisible by 64");
    TORCH_CHECK(attn_sink.numel() == h_q, "attn_sink must contain h_q elements");

    int s_q = static_cast<int>(q.size(0));
    int s_kv = static_cast<int>(kv.size(0));
    int topk = static_cast<int>(indices.size(2));
    int* topk_length_ptr = nullptr;
    if (topk_length.has_value()) {
        TORCH_CHECK(topk_length->is_cuda() && topk_length->scalar_type() == at::kInt,
                    "topk_length must be a CUDA int32 tensor");
        TORCH_CHECK(topk_length->dim() == 1 && topk_length->numel() == s_q,
                    "topk_length must have shape [s_q]");
        topk_length_ptr = topk_length->data_ptr<int>();
    }

    at::cuda::CUDAGuard guard(q.device());
    at::Tensor d_q = torch::empty_like(q);
    at::Tensor d_k = torch::empty_like(kv);
    at::Tensor d_v = torch::empty({s_kv, 1, 128}, q.options());
    at::Tensor d_k_acc = torch::zeros({s_kv, 1, 128}, q.options().dtype(at::kFloat));
    at::Tensor d_v_acc = torch::zeros({s_kv, 1, 128}, q.options().dtype(at::kFloat));
    at::Tensor d_sink = torch::zeros({h_q}, q.options().dtype(at::kFloat));

    SparseAttnBwdParams params = {
        s_q, s_kv, h_q, 1, 128, 128, topk,
        static_cast<float>(sm_scale), static_cast<float>(sm_scale) * LOG_2_E,
        reinterpret_cast<bf16*>(q.data_ptr()),
        reinterpret_cast<bf16*>(kv.data_ptr()),
        reinterpret_cast<bf16*>(out.data_ptr()),
        reinterpret_cast<bf16*>(d_out.data_ptr()),
        lse.data_ptr<float>(),
        indices.data_ptr<int>(),
        attn_sink.data_ptr<float>(),
        topk_length_ptr,
        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)),
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(out.stride(0)), int64_stride_to_int(out.stride(1)),
        int64_stride_to_int(d_out.stride(0)), int64_stride_to_int(d_out.stride(1)),
        int64_stride_to_int(lse.stride(0)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),
        reinterpret_cast<bf16*>(d_q.data_ptr()),
        reinterpret_cast<bf16*>(d_k.data_ptr()),
        reinterpret_cast<bf16*>(d_v.data_ptr()),
        d_k_acc.data_ptr<float>(),
        d_v_acc.data_ptr<float>(),
        d_sink.data_ptr<float>(),
        int64_stride_to_int(d_q.stride(0)), int64_stride_to_int(d_q.stride(1)),
        int64_stride_to_int(d_k.stride(0)), int64_stride_to_int(d_k.stride(1)),
        int64_stride_to_int(d_v.stride(0)), int64_stride_to_int(d_v.stride(1)),
        int64_stride_to_int(d_k_acc.stride(0)), int64_stride_to_int(d_k_acc.stride(1)),
        int64_stride_to_int(d_v_acc.stride(0)), int64_stride_to_int(d_v_acc.stride(1)),
        arch.num_sms,
        at::cuda::getCurrentCUDAStream().stream()
    };

    if (h_q == 8) {
        sm100::bwd::head_small::run_bwd_phase1_kernel<128, 8>(params);
    } else if (h_q == 16) {
        sm100::bwd::head_small::run_bwd_phase1_kernel<128, 16>(params);
    } else {
        sm100::bwd::head_small::run_bwd_phase1_kernel<128, 32>(params);
    }
    sm100::bwd::head_small::run_convert_dkv_accum_kernel(params);
    return {d_q, d_k, d_v, d_sink};
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("small_topk_fwd", &small_topk_fwd);
    m.def("head_small_bwd", &head_small_bwd);
}
