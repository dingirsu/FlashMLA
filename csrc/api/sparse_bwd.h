#pragma once

#include "common.h"

#include "params.h"

#include "sm100/prefill/sparse/bwd/head_small/convert.h"
#include "sm100/prefill/sparse/bwd/head_small/phase1.h"

enum class BwdFeatures : int {
    HEAD_SMALL,

    HEAD_DIM_192,
    HEAD_DIM_128,

    ATTN_SINK,
    TOPK_LENGTH
};

class BwdImplBase : public ImplBase<
    SparseAttnBwdParams,
    BwdFeatures
> {};

class Bwd_Sm100_HeadSmall_Impl : public BwdImplBase {
    DECLARE_SUPPORTED_FEATURES(
        BwdFeatures::HEAD_SMALL,
        BwdFeatures::HEAD_DIM_128,
        BwdFeatures::HEAD_DIM_192,
        BwdFeatures::ATTN_SINK,
        BwdFeatures::TOPK_LENGTH
    )

protected:
    void run_(const SparseAttnBwdParams &params, const std::vector<FeatureT> &required_features) override {
        auto run_with_h = [&]<int H_Q>() {
            if (params.d_qk == 128) {
                sm100::bwd::head_small::run_bwd_phase1_kernel<128, H_Q>(params);
            } else if (params.d_qk == 192) {
                sm100::bwd::head_small::run_bwd_phase1_kernel<192, H_Q>(params);
            } else {
                TORCH_CHECK(false, "Unsupported small-head d_qk for sparse backward: ", params.d_qk);
            }
        };
        if (params.h_q == 8) {
            run_with_h.template operator()<8>();
        } else if (params.h_q == 16) {
            run_with_h.template operator()<16>();
        } else if (params.h_q == 24) {
            run_with_h.template operator()<24>();
        } else if (params.h_q == 32) {
            run_with_h.template operator()<32>();
        } else {
            TORCH_CHECK(false, "Unsupported small-head h_q for sparse backward: ", params.h_q);
        }
    }
};

static std::vector<at::Tensor> sparse_attn_prefill_bwd_interface(
    const at::Tensor &d_out,
    const at::Tensor &q,
    const at::Tensor &kv,
    const at::Tensor &out,
    const at::Tensor &lse,
    const at::Tensor &indices,
    float sm_scale,
    const at::Tensor &attn_sink,
    const std::optional<at::Tensor> &topk_length
) {
    using bf16 = cutlass::bfloat16_t;

    Arch arch = Arch();
    TORCH_CHECK(arch.is_sm100f(), "Sparse Attention Backward Kernel is only supported on SM100f architectures.");

    KU_CHECK_NDIM(d_out, 3);
    KU_CHECK_NDIM(q, 3);
    KU_CHECK_NDIM(kv, 3);
    KU_CHECK_NDIM(out, 3);
    KU_CHECK_NDIM(lse, 2);
    KU_CHECK_NDIM(indices, 3);
    KU_CHECK_NDIM(attn_sink, 1);
    KU_CHECK_NDIM(topk_length, 1);

    int s_q = q.size(0);
    int s_kv = kv.size(0);
    int h_q = q.size(1);
    int h_kv = kv.size(1);
    int d_qk = q.size(2);
    int d_v = d_out.size(2);
    int topk = indices.size(2);
    bool have_topk_length = topk_length.has_value();

    TORCH_CHECK(d_qk == 128 || d_qk == 192, "Sparse backward head-small only supports d_qk 128 or 192, got ", d_qk);
    TORCH_CHECK(h_q == 8 || h_q == 16 || h_q == 24 || h_q == 32, "Sparse backward head-small only supports h_q in {8,16,24,32}, got ", h_q);
    TORCH_CHECK(h_kv == 1, "Sparse backward head-small requires h_kv == 1, got ", h_kv);
    TORCH_CHECK(d_v == 128, "Sparse backward head-small requires d_v == 128, got ", d_v);
    TORCH_CHECK(topk % 64 == 0, "Sparse backward head-small requires topk % 64 == 0, got ", topk);

    KU_CHECK_DEVICE(d_out);
    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(kv);
    KU_CHECK_DEVICE(out);
    KU_CHECK_DEVICE(lse);
    KU_CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(attn_sink);
    KU_CHECK_DEVICE(topk_length);

    KU_CHECK_DTYPE(d_out, torch::kBFloat16);
    KU_CHECK_DTYPE(q, torch::kBFloat16);
    KU_CHECK_DTYPE(kv, torch::kBFloat16);
    KU_CHECK_DTYPE(out, torch::kBFloat16);
    KU_CHECK_DTYPE(lse, torch::kFloat32);
    KU_CHECK_DTYPE(indices, torch::kInt32);
    KU_CHECK_DTYPE(attn_sink, torch::kFloat32);
    KU_CHECK_DTYPE(topk_length, torch::kInt32);

    KU_CHECK_SHAPE(d_out, s_q, h_q, d_v);
    KU_CHECK_SHAPE(q, s_q, h_q, d_qk);
    KU_CHECK_SHAPE(kv, s_kv, h_kv, d_qk);
    KU_CHECK_SHAPE(out, s_q, h_q, d_v);
    KU_CHECK_SHAPE(lse, s_q, h_q);
    KU_CHECK_SHAPE(indices, s_q, h_kv, topk);
    KU_CHECK_SHAPE(attn_sink, h_q);
    KU_CHECK_SHAPE(topk_length, s_q);

    KU_CHECK_LAST_DIM_CONTIGUOUS(d_out);
    KU_CHECK_LAST_DIM_CONTIGUOUS(q);
    KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(out);
    KU_CHECK_LAST_DIM_CONTIGUOUS(lse);
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    KU_CHECK_LAST_DIM_CONTIGUOUS(attn_sink);
    KU_CHECK_LAST_DIM_CONTIGUOUS(topk_length);

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto opts = q.options();
    at::Tensor d_q = torch::empty_like(q);
    at::Tensor d_k = torch::empty_like(kv);
    at::Tensor d_v_tensor = torch::empty({s_kv, h_kv, d_v}, opts);
    at::Tensor d_k_acc = torch::empty({s_kv, h_kv, d_qk}, opts.dtype(torch::kFloat));
    at::Tensor d_v_acc = torch::empty({s_kv, h_kv, d_v}, opts.dtype(torch::kFloat));
    at::Tensor d_attn_sink = torch::empty({h_q}, opts.dtype(torch::kFloat));

    d_k_acc.zero_();
    d_v_acc.zero_();
    d_attn_sink.zero_();

    SparseAttnBwdParams params = {
        s_q, s_kv, h_q, h_kv, d_qk, d_v, topk,
        sm_scale, sm_scale * LOG_2_E,

        (bf16*)q.data_ptr(),
        (bf16*)kv.data_ptr(),
        (bf16*)out.data_ptr(),
        (bf16*)d_out.data_ptr(),
        (float*)lse.data_ptr(),
        (int*)indices.data_ptr(),
        (float*)attn_sink.data_ptr(),
        ku::get_optional_tensor_ptr<int>(topk_length),

        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)),
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(out.stride(0)), int64_stride_to_int(out.stride(1)),
        int64_stride_to_int(d_out.stride(0)), int64_stride_to_int(d_out.stride(1)),
        int64_stride_to_int(lse.stride(0)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),

        (bf16*)d_q.data_ptr(),
        (bf16*)d_k.data_ptr(),
        (bf16*)d_v_tensor.data_ptr(),
        (float*)d_k_acc.data_ptr(),
        (float*)d_v_acc.data_ptr(),
        (float*)d_attn_sink.data_ptr(),

        int64_stride_to_int(d_q.stride(0)), int64_stride_to_int(d_q.stride(1)),
        int64_stride_to_int(d_k.stride(0)), int64_stride_to_int(d_k.stride(1)),
        int64_stride_to_int(d_v_tensor.stride(0)), int64_stride_to_int(d_v_tensor.stride(1)),
        int64_stride_to_int(d_k_acc.stride(0)), int64_stride_to_int(d_k_acc.stride(1)),
        int64_stride_to_int(d_v_acc.stride(0)), int64_stride_to_int(d_v_acc.stride(1)),

        arch.num_sms,
        at::cuda::getCurrentCUDAStream().stream()
    };

    std::vector<BwdFeatures> required_features;
    required_features.push_back(BwdFeatures::HEAD_SMALL);
    if (d_qk == 192) {
        required_features.push_back(BwdFeatures::HEAD_DIM_192);
    } else {
        required_features.push_back(BwdFeatures::HEAD_DIM_128);
    }
    required_features.push_back(BwdFeatures::ATTN_SINK);
    if (have_topk_length) {
        required_features.push_back(BwdFeatures::TOPK_LENGTH);
    }

    Bwd_Sm100_HeadSmall_Impl bwd_impl;
    bwd_impl.run(params, required_features);
    sm100::bwd::head_small::run_convert_dkv_accum_kernel(params);

    return {d_q, d_k, d_v_tensor, d_attn_sink};
}

