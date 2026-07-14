#pragma once

#include "common.h"
#include "params.h"

#include "sm100/prefill/sparse/mxfp8_fwd/head64/phase1.h"

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

/*
 * mxfp8_sparse_attn_prefill_interface
 *
 * MXFP8 sparse attention prefill. Q stores its scales after each token;
 * KV is a packed page with all e4m3 rows followed by all page scales.
 *
 * Layout for d_qk=512:
 *   - Q: 512 e4m3 bytes + 16 UE8M0 scales per token (32-value groups)
 *   - KV page: s_kv*512 e4m3 bytes, then s_kv*8 UE8M0 scales (64-value groups)
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

    TORCH_CHECK(d_qk == 512, "MXFP8 sparse prefill head64 currently supports only d_qk=512, got ", d_qk);
    TORCH_CHECK(d_v == 512, "MXFP8 sparse prefill head64 currently supports only d_v=512, got ", d_v);

    constexpr int q_bytes_per_token = 512 + 16;
    constexpr int kv_bytes_per_token = 512 + 8;
    TORCH_CHECK(q.size(2) == q_bytes_per_token,
        "q last dim must be ", q_bytes_per_token, " for MXFP8 with d_qk=", d_qk, ", got ", q.size(2));
    TORCH_CHECK(kv.size(2) == kv_bytes_per_token,
        "kv storage envelope must provide ", kv_bytes_per_token,
        " bytes per token for page-tail scales, got ", kv.size(2));

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
    KU_CHECK_CONTIGUOUS(kv);
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
    } else {
        TORCH_CHECK(false, "Unsupported h_q for MXFP8 sparse prefill k512/dv512: ", h_q);
    }
    if (d_qk == 512) {
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
    } else {
        TORCH_CHECK(false, "Unsupported h_q for MXFP8 sparse prefill k512/dv512: ", h_q);
    }

    return {out, max_logits, lse};
}
