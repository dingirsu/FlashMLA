#pragma once

#include "common.h"
#include "params.h"

#include "sm100/prefill/sparse/fp8_fwd/head64/phase1.h"

/*
 * Test/debug interface for the head64 FP8 sparse prefill kernel.
 *
 * Physical layout:
 *   Q[token]  = [64 * d_qk E4M3 bytes][64 UE8M0 head scales]
 *   KV[token] = [d_qk E4M3 bytes][1 UE8M0 token scale][padding to 16B]
 *   W         = [8 UE8M0 dimension-group scales]
 */
static std::vector<at::Tensor> fp8_sparse_attn_prefill_interface(
    const at::Tensor &q,
    const at::Tensor &kv,
    const at::Tensor &kv_scale_w,
    const at::Tensor &indices,
    float sm_scale,
    const std::optional<at::Tensor> &attn_sink,
    const std::optional<at::Tensor> &topk_length
) {
    using bf16 = cutlass::bfloat16_t;

    Arch arch = Arch();
    TORCH_CHECK(
        arch.is_sm100f(),
        "FP8 sparse attention prefill is only supported on SM100f."
    );

    KU_CHECK_NDIM(q, 2);
    KU_CHECK_NDIM(kv, 2);
    KU_CHECK_NDIM(kv_scale_w, 1);
    KU_CHECK_NDIM(indices, 3);
    KU_CHECK_NDIM(attn_sink, 1);
    KU_CHECK_NDIM(topk_length, 1);

    constexpr int h_q = 64;
    constexpr int h_kv = 1;
    constexpr int d_v = 512;
    const int d_qk = static_cast<int>(kv.size(1)) - 16;
    TORCH_CHECK(
        d_qk == 512 || d_qk == 576,
        "kv last dim must encode d_qk 512 or 576 plus a 16-byte scale slot, got ",
        kv.size(1)
    );
    const int q_bytes_per_token = h_q * d_qk + h_q;
    const int kv_bytes_per_token = d_qk + 16;

    const int s_q = q.size(0);
    const int s_kv = kv.size(0);
    const int topk = indices.size(2);

    TORCH_CHECK(
        q.size(1) == q_bytes_per_token,
        "q last dim must be ", q_bytes_per_token, ", got ", q.size(1)
    );
    TORCH_CHECK(
        kv.size(1) == kv_bytes_per_token,
        "kv last dim must be ", kv_bytes_per_token, ", got ", kv.size(1)
    );
    TORCH_CHECK(
        topk >= 128 && topk % 128 == 0,
        "topk must be a positive multiple of 128"
    );

    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(kv);
    KU_CHECK_DEVICE(kv_scale_w);
    KU_CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(attn_sink);
    KU_CHECK_DEVICE(topk_length);

    KU_CHECK_DTYPE(q, torch::kUInt8);
    KU_CHECK_DTYPE(kv, torch::kUInt8);
    TORCH_CHECK(
        kv_scale_w.dtype() == at::kFloat8_e8m0fnu
            || kv_scale_w.dtype() == torch::kUInt8,
        "kv_scale_w must have dtype float8_e8m0fnu or uint8"
    );
    KU_CHECK_DTYPE(indices, torch::kInt32);
    KU_CHECK_DTYPE(attn_sink, torch::kFloat32);
    KU_CHECK_DTYPE(topk_length, torch::kInt32);

    KU_CHECK_CONTIGUOUS(q);
    KU_CHECK_CONTIGUOUS(kv);
    KU_CHECK_CONTIGUOUS(kv_scale_w);
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    KU_CHECK_LAST_DIM_CONTIGUOUS(attn_sink);
    KU_CHECK_LAST_DIM_CONTIGUOUS(topk_length);

    KU_CHECK_SHAPE(indices, s_q, h_kv, topk);
    KU_CHECK_SHAPE(kv_scale_w, d_v / 64);
    KU_CHECK_SHAPE(attn_sink, h_q);
    KU_CHECK_SHAPE(topk_length, s_q);

    at::cuda::CUDAGuard device_guard{static_cast<char>(q.get_device())};
    auto opts = q.options();
    at::Tensor out = torch::empty(
        {s_q, h_q, d_v}, opts.dtype(torch::kBFloat16)
    );
    at::Tensor max_logits = torch::empty(
        {s_q, h_q}, opts.dtype(torch::kFloat)
    );
    at::Tensor lse = torch::empty(
        {s_q, h_q}, opts.dtype(torch::kFloat)
    );

    Head64Fp8SparseAttnFwdParams params = {
        s_q, s_kv, h_q, h_kv, d_qk, d_v, topk,
        sm_scale, sm_scale * LOG_2_E,

        q.data_ptr(),
        kv.data_ptr(),
        reinterpret_cast<uint8_t *>(kv_scale_w.data_ptr()),
        reinterpret_cast<int *>(indices.data_ptr()),
        ku::get_optional_tensor_ptr<float>(attn_sink),
        ku::get_optional_tensor_ptr<int>(topk_length),

        int64_stride_to_int(q.stride(0)), d_qk,
        int64_stride_to_int(kv.stride(0)), kv_bytes_per_token,
        int64_stride_to_int(indices.stride(0)),
        int64_stride_to_int(indices.stride(1)),

        reinterpret_cast<bf16 *>(out.data_ptr()),
        reinterpret_cast<float *>(max_logits.data_ptr()),
        reinterpret_cast<float *>(lse.data_ptr()),

        arch.num_sms,
        at::cuda::getCurrentCUDAStream().stream()
    };

    if (d_qk == 512) {
        sm100::fp8_fwd::head64::run_fp8_fwd_phase1_kernel<512>(params);
    } else {
        sm100::fp8_fwd::head64::run_fp8_fwd_phase1_kernel<576>(params);
    }
    return {out, max_logits, lse};
}
