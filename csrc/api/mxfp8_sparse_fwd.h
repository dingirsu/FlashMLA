#pragma once

#include "common.h"
#include "params.h"

#include "sm100/prefill/sparse/mxfp8_fwd/head64/phase1.h"
#include "sm100/prefill/sparse/mxfp8_fwd/head128/phase1.h"

enum class MxFp8FwdFeatures : int {
    HEAD_64,
    HEAD_128,

    HEAD_DIM_576,
    HEAD_DIM_512,

    ATTN_SINK,
    TOPK_LENGTH
};

class MxFp8FwdImplBase : public ImplBase<
    MxFp8SparseAttnFwdParams,
    MxFp8FwdFeatures
> {};

class MxFp8Fwd_Sm100_Head64_Impl : public MxFp8FwdImplBase {
    DECLARE_SUPPORTED_FEATURES(
        MxFp8FwdFeatures::HEAD_64,
        MxFp8FwdFeatures::HEAD_DIM_512,
        MxFp8FwdFeatures::HEAD_DIM_576,
        MxFp8FwdFeatures::ATTN_SINK,
        MxFp8FwdFeatures::TOPK_LENGTH
    )

protected:
    void run_(const MxFp8SparseAttnFwdParams &params, const std::vector<FeatureT> &required_features) override {
        DISPATCH_HEAD_DIM(params.d_qk, HEAD_DIM_QK, [&]() {
            sm100::mxfp8_fwd::head64::run_mxfp8_fwd_phase1_kernel<HEAD_DIM_QK>(params);
        });
    }
};

class MxFp8Fwd_Sm100_Head128_Impl : public MxFp8FwdImplBase {
    DECLARE_SUPPORTED_FEATURES(
        MxFp8FwdFeatures::HEAD_128,
        MxFp8FwdFeatures::HEAD_DIM_512,
        MxFp8FwdFeatures::HEAD_DIM_576,
        MxFp8FwdFeatures::ATTN_SINK,
        MxFp8FwdFeatures::TOPK_LENGTH
    )

protected:
    void run_(const MxFp8SparseAttnFwdParams &params, const std::vector<FeatureT> &required_features) override {
        DISPATCH_HEAD_DIM(params.d_qk, HEAD_DIM_QK, [&]() {
            sm100::mxfp8_fwd::head128::run_mxfp8_fwd_phase1_kernel<HEAD_DIM_QK>(params);
        });
    }
};

/*
 * mxfp8_sparse_attn_prefill_interface
 *
 * MXFP8 sparse attention prefill with BF16 RoPE. Both Q and KV use the same format
 * as existing FP8 KV cache: NoPE part is e4m3 + block scales, RoPE part is BF16. SM100 only.
 *
 * Token layout (same as existing FP8 KV cache):
 *   - d_qk=512 (MODEL1): 448 bytes NoPE (e4m3) + 8 bytes scales + 128 bytes RoPE (BF16) = 584 bytes/token
 *   - d_qk=576 (V32): 512 bytes NoPE (e4m3) + 16 bytes scales + 128 bytes RoPE (BF16) = 656 bytes/token
 */
static std::vector<at::Tensor> mxfp8_sparse_attn_prefill_interface(
    const at::Tensor &q,        // [s_q, h_q, bytes_per_token_q]
    const at::Tensor &kv,       // [s_kv, h_kv, bytes_per_token_kv]
    const at::Tensor &indices,  // [s_q, h_kv, topk]
    float sm_scale,
    int d_qk,
    int d_v,
    const std::optional<at::Tensor> &attn_sink,
    const std::optional<at::Tensor> &topk_length
) {
    using bf16 = cutlass::bfloat16_t;

    Arch arch = Arch();
    TORCH_CHECK(arch.is_sm100f(), "MXFP8 sparse attention prefill is only supported on SM100f.");

    KU_CHECK_NDIM(q, 3);
    KU_CHECK_NDIM(kv, 3);
    KU_CHECK_NDIM(indices, 3);
    KU_CHECK_NDIM(attn_sink, 1);
    KU_CHECK_NDIM(topk_length, 1);

    int s_q = q.size(0);
    int s_kv = kv.size(0);
    int h_q = q.size(1);
    int h_kv = kv.size(1);
    int topk = indices.size(2);

    TORCH_CHECK(d_qk == 576 || d_qk == 512, "Invalid d_qk: ", d_qk);
    TORCH_CHECK(d_v == 512, "Invalid d_v: ", d_v);

    // Compute expected bytes per token (same as existing FP8 KV cache format)
    int bytes_per_token;
    if (d_qk == 576 && d_v == 512) {
        // V3.2 style: 512 bytes NoPE (e4m3) + 16 bytes scales + 128 bytes RoPE (BF16)
        bytes_per_token = 512 + 16 + 128;
    } else if (d_qk == 512 && d_v == 512) {
        // MODEL1 style: 448 bytes NoPE (e4m3) + 8 bytes scales + 128 bytes RoPE (BF16)
        bytes_per_token = 448 + 8 + 128;
    } else {
        TORCH_CHECK(false, "Unsupported head sizes for MXFP8");
    }
    TORCH_CHECK(q.size(2) == bytes_per_token,
        "q last dim must be ", bytes_per_token, " for MXFP8 with d_qk=", d_qk, ", got ", q.size(2));
    TORCH_CHECK(kv.size(2) == bytes_per_token,
        "kv last dim must be ", bytes_per_token, " for MXFP8 with d_qk=", d_qk, ", got ", kv.size(2));

    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(kv);
    KU_CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(attn_sink);
    KU_CHECK_DEVICE(topk_length);

    TORCH_CHECK(q.dtype() == torch::kFloat8_e4m3fn || q.dtype() == torch::kUInt8,
        "q must have dtype fp8_e4m3fn or uint8 for MXFP8 mode");
    TORCH_CHECK(kv.dtype() == torch::kFloat8_e4m3fn || kv.dtype() == torch::kUInt8,
        "kv must have dtype fp8_e4m3fn or uint8 for MXFP8 mode");
    KU_CHECK_DTYPE(indices, torch::kInt32);
    KU_CHECK_DTYPE(attn_sink, torch::kFloat32);
    KU_CHECK_DTYPE(topk_length, torch::kInt32);

    KU_CHECK_LAST_DIM_CONTIGUOUS(q);
    KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    KU_CHECK_LAST_DIM_CONTIGUOUS(attn_sink);
    KU_CHECK_LAST_DIM_CONTIGUOUS(topk_length);

    KU_CHECK_SHAPE(indices, s_q, h_kv, topk);
    KU_CHECK_SHAPE(attn_sink, h_q);
    KU_CHECK_SHAPE(topk_length, s_q);

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto opts = q.options();

    at::Tensor out = torch::empty({s_q, h_q, d_v}, opts.dtype(torch::kBFloat16));
    at::Tensor lse = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));
    at::Tensor max_logits = torch::empty({s_q, h_q}, opts.dtype(torch::kFloat));

    MxFp8SparseAttnFwdParams params = {
        s_q, s_kv, h_q, h_kv, d_qk, d_v, topk,
        sm_scale, sm_scale * LOG_2_E,

        q.data_ptr(),
        kv.data_ptr(),
        (int*)indices.data_ptr(),
        ku::get_optional_tensor_ptr<float>(attn_sink),
        ku::get_optional_tensor_ptr<int>(topk_length),

        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)),
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),

        (bf16*)out.data_ptr(),
        (float*)max_logits.data_ptr(),
        (float*)lse.data_ptr(),

        arch.num_sms,
        at::cuda::getCurrentCUDAStream().stream()
    };

    std::vector<MxFp8FwdFeatures> required_features;
    if (h_q == 64) {
        required_features.push_back(MxFp8FwdFeatures::HEAD_64);
    } else if (h_q == 128) {
        required_features.push_back(MxFp8FwdFeatures::HEAD_128);
    } else {
        TORCH_CHECK(false, "Unsupported h_q: ", h_q);
    }
    if (d_qk == 576) {
        required_features.push_back(MxFp8FwdFeatures::HEAD_DIM_576);
    } else if (d_qk == 512) {
        required_features.push_back(MxFp8FwdFeatures::HEAD_DIM_512);
    } else {
        TORCH_CHECK(false, "Unsupported d_qk: ", d_qk);
    }
    if (attn_sink.has_value()) {
        required_features.push_back(MxFp8FwdFeatures::ATTN_SINK);
    }
    if (topk_length.has_value()) {
        required_features.push_back(MxFp8FwdFeatures::TOPK_LENGTH);
    }

    if (h_q == 64) {
        MxFp8Fwd_Sm100_Head64_Impl fwd_impl;
        fwd_impl.run(params, required_features);
    } else if (h_q == 128) {
        MxFp8Fwd_Sm100_Head128_Impl fwd_impl;
        fwd_impl.run(params, required_features);
    } else {
        TORCH_CHECK(false, "Unsupported h_q: ", h_q);
    }

    return {out, max_logits, lse};
}
