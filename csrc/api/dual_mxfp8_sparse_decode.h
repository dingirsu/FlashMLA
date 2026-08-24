#pragma once

#include "common.h"
#include "params.h"
#include "sm100/prefill/sparse/dual_mxfp8/head64/phase1.h"
#include "smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.h"
#include "smxx/decode/combine/combine.h"

// Decode entry point for the dual-MXFP8 head64 kernel. It is deliberately
// separate from mxfp8_sparse_attn_decode_interface: the latter is the legacy
// MXFP8 implementation and owns the kv_scale_w input.
static std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>>
dual_mxfp8_sparse_attn_decode_interface(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    const std::optional<at::Tensor>& topk_length,
    const std::optional<at::Tensor>& attn_sink,
    std::optional<at::Tensor>& tile_scheduler_metadata,
    std::optional<at::Tensor>& num_splits,
    const std::optional<at::Tensor>& extra_kv,
    const std::optional<at::Tensor>& extra_indices,
    const std::optional<at::Tensor>& extra_topk_length,
    int d_qk,
    int d_v,
    float sm_scale,
    uint32_t w1,
    uint32_t w2
) {
    using bf16 = cutlass::bfloat16_t;
    constexpr int d = 512;
    constexpr int q_bytes = d + 16;
    constexpr int kv_bytes = d + 16;

    Arch arch = Arch();
    TORCH_CHECK(arch.is_sm100f(), "Dual MXFP8 head64 decode requires SM100f.");
    KU_CHECK_NDIM(q, 4);
    KU_CHECK_NDIM(kv, 4);
    KU_CHECK_NDIM(indices, 3);

    const int b = static_cast<int>(q.size(0));
    const int s_q = static_cast<int>(q.size(1));
    const int h_q = static_cast<int>(q.size(2));
    const int num_blocks = static_cast<int>(kv.size(0));
    const int page_block_size = static_cast<int>(kv.size(1));
    const int h_kv = static_cast<int>(kv.size(2));
    const int topk = static_cast<int>(indices.size(2));
    const bool have_extra_kv = extra_kv.has_value();
    const int extra_num_blocks = have_extra_kv ? static_cast<int>(extra_kv->size(0)) : 0;
    const int extra_page_block_size = have_extra_kv ? static_cast<int>(extra_kv->size(1)) : 0;
    const int extra_topk = extra_indices.has_value() ? static_cast<int>(extra_indices->size(2)) : 0;

    TORCH_CHECK(b > 0 && s_q > 0);
    TORCH_CHECK(h_q == 64, "Dual MXFP8 decode supports head64 only; got h_q=", h_q);
    TORCH_CHECK(h_kv == 1, "Dual MXFP8 decode requires h_kv=1");
    TORCH_CHECK(d_qk == d && d_v == d, "Dual MXFP8 decode requires d_qk=d_v=512");
    TORCH_CHECK(topk > 0 && topk % 64 == 0, "topk must be a positive multiple of 64");
    TORCH_CHECK(q.size(3) == q_bytes, "q must use the 528-byte head64 envelope");
    TORCH_CHECK(kv.size(3) == kv_bytes, "kv must use the 528-byte head64 envelope");
    TORCH_CHECK(!extra_kv.has_value() || extra_indices.has_value(),
        "extra_indices must be provided with extra_kv");
    TORCH_CHECK(extra_kv.has_value() || (!extra_indices.has_value() && !extra_topk_length.has_value()),
        "extra indices require extra_kv");

    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(kv);
    KU_CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(topk_length);
    KU_CHECK_DEVICE(attn_sink);
    KU_CHECK_DEVICE(tile_scheduler_metadata);
    KU_CHECK_DEVICE(num_splits);
    KU_CHECK_DEVICE(extra_kv);
    KU_CHECK_DEVICE(extra_indices);
    KU_CHECK_DEVICE(extra_topk_length);
    TORCH_CHECK(q.dtype() == torch::kUInt8 || q.dtype() == torch::kFloat8_e4m3fn,
        "q must have E4M3 storage (uint8 or float8_e4m3fn)");
    TORCH_CHECK(kv.dtype() == torch::kUInt8 || kv.dtype() == torch::kFloat8_e4m3fn,
        "kv must have E4M3 storage (uint8 or float8_e4m3fn)");
    if (have_extra_kv) {
        TORCH_CHECK(extra_kv->dtype() == torch::kUInt8 || extra_kv->dtype() == torch::kFloat8_e4m3fn,
            "extra_kv must have E4M3 storage");
    }
    KU_CHECK_DTYPE(indices, torch::kInt32);
    KU_CHECK_DTYPE(topk_length, torch::kInt32);
    KU_CHECK_DTYPE(attn_sink, torch::kFloat32);
    KU_CHECK_DTYPE(tile_scheduler_metadata, torch::kInt32);
    KU_CHECK_DTYPE(num_splits, torch::kInt32);
    KU_CHECK_DTYPE(extra_indices, torch::kInt32);
    KU_CHECK_DTYPE(extra_topk_length, torch::kInt32);
    KU_CHECK_LAST_DIM_CONTIGUOUS(q);
    KU_CHECK_CONTIGUOUS(kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    KU_CHECK_CONTIGUOUS(topk_length);
    KU_CHECK_CONTIGUOUS(attn_sink);
    KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
    KU_CHECK_CONTIGUOUS(num_splits);
    KU_CHECK_LAST_DIM_CONTIGUOUS(extra_kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(extra_indices);
    KU_CHECK_CONTIGUOUS(extra_topk_length);
    KU_CHECK_SHAPE(q, b, s_q, h_q, q_bytes);
    KU_CHECK_SHAPE(kv, num_blocks, page_block_size, h_kv, kv_bytes);
    KU_CHECK_SHAPE(indices, b, s_q, topk);
    KU_CHECK_SHAPE(topk_length, b);
    KU_CHECK_SHAPE(attn_sink, h_q);
    KU_CHECK_SHAPE(extra_indices, b, s_q, extra_topk);
    KU_CHECK_SHAPE(extra_topk_length, b);
    TORCH_CHECK(kv.stride(1) == kv_bytes * h_kv,
        "dual KV row stride must be 528 bytes, got ", kv.stride(1));
    TORCH_CHECK(kv.stride(0) % 512 == 0,
        "dual KV page stride must be divisible by 512 bytes, got ", kv.stride(0));
    if (have_extra_kv) {
        TORCH_CHECK(extra_kv->stride(1) == kv_bytes * h_kv);
        TORCH_CHECK(extra_kv->stride(0) % 512 == 0);
    }

    at::cuda::CUDAGuard device_guard{static_cast<char>(q.get_device())};
    auto opts = q.options();
    at::Tensor out = torch::empty({b, s_q, h_q, d_v}, opts.dtype(torch::kBFloat16));
    at::Tensor lse = torch::empty({b, s_q, h_q}, opts.dtype(torch::kFloat));

    const int num_sm_parts = std::max(arch.num_sms / s_q, 1);
    constexpr int fixed_overhead_num_blocks = 5;
    constexpr int block_size_topk = 64;
    if (!tile_scheduler_metadata.has_value()) {
        tile_scheduler_metadata = torch::empty(
            {num_sm_parts, static_cast<int>(sizeof(DecodingSchedMeta) / sizeof(int))},
            opts.dtype(torch::kInt32));
        num_splits = torch::empty({b + 1}, opts.dtype(torch::kInt32));
        GetDecodeSchedMetaParams sched = {
            b, s_q, block_size_topk, fixed_overhead_num_blocks, topk, extra_topk,
            ku::get_optional_tensor_ptr<int>(topk_length),
            ku::get_optional_tensor_ptr<int>(extra_topk_length), nullptr,
            reinterpret_cast<DecodingSchedMeta*>(tile_scheduler_metadata->data_ptr()),
            num_splits->data_ptr<int>(), num_sm_parts,
            at::cuda::getCurrentCUDAStream().stream()
        };
        smxx::decode::run_get_decoding_sched_meta_kernel(sched);
    }
    KU_CHECK_SHAPE(tile_scheduler_metadata, num_sm_parts, sizeof(DecodingSchedMeta) / sizeof(int));
    KU_CHECK_SHAPE(num_splits, b + 1);

    SparseAttnDualMxfp8DecodeParams params = {
        b, s_q, h_q, h_kv, d_qk, d_v,
        sm_scale, sm_scale * LOG_2_E,
        num_blocks, page_block_size, topk, ModelType::MODEL1,
        q.data_ptr(), kv.data_ptr(), w1, w2,
        indices.data_ptr<int>(), ku::get_optional_tensor_ptr<int>(topk_length),
        ku::get_optional_tensor_ptr<float>(attn_sink),
        lse.data_ptr<float>(), reinterpret_cast<bf16*>(out.data_ptr()),
        extra_num_blocks, extra_page_block_size, extra_topk,
        have_extra_kv ? extra_kv->data_ptr() : nullptr,
        ku::get_optional_tensor_ptr<int>(extra_indices),
        ku::get_optional_tensor_ptr<int>(extra_topk_length),
        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)), int64_stride_to_int(q.stride(2)),
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),
        int64_stride_to_int(lse.stride(0)), int64_stride_to_int(lse.stride(1)),
        int64_stride_to_int(out.stride(0)), int64_stride_to_int(out.stride(1)), int64_stride_to_int(out.stride(2)),
        have_extra_kv ? int64_stride_to_int(extra_kv->stride(0)) : 0,
        have_extra_kv ? int64_stride_to_int(extra_kv->stride(1)) : 0,
        have_extra_kv ? int64_stride_to_int(extra_indices->stride(0)) : 0,
        have_extra_kv ? int64_stride_to_int(extra_indices->stride(1)) : 0,
        at::cuda::getCurrentCUDAStream().stream(),
        nullptr, nullptr, 0, 0, 0, 0, 0,
        reinterpret_cast<DecodingSchedMeta*>(tile_scheduler_metadata->data_ptr()),
        num_splits->data_ptr<int>(), num_sm_parts
    };

    const int total_num_splits = b + num_sm_parts;
    at::Tensor lse_accum = torch::empty({total_num_splits, s_q, h_q}, opts.dtype(torch::kFloat));
    at::Tensor o_accum = torch::empty({total_num_splits, s_q, h_q, d_v}, opts.dtype(torch::kFloat));
    params.lse_accum = lse_accum.data_ptr<float>();
    params.o_accum = o_accum.data_ptr<float>();
    params.stride_lse_accum_split = int64_stride_to_int(lse_accum.stride(0));
    params.stride_lse_accum_s_q = int64_stride_to_int(lse_accum.stride(1));
    params.stride_o_accum_split = int64_stride_to_int(o_accum.stride(0));
    params.stride_o_accum_s_q = int64_stride_to_int(o_accum.stride(1));
    params.stride_o_accum_h_q = int64_stride_to_int(o_accum.stride(2));

    sm100::dual_mxfp8::head64::run_dual_mxfp8_phase1_kernel<
        SparseAttnFwdMode::DecodeWithSplitKV, 512>(params);

    CombineParams combine = {
        b, s_q, h_q, d_v,
        lse.data_ptr<float>(), out.data_ptr(),
        int64_stride_to_int(lse.stride(0)), int64_stride_to_int(lse.stride(1)),
        int64_stride_to_int(out.stride(0)), int64_stride_to_int(out.stride(1)), int64_stride_to_int(out.stride(2)),
        lse_accum.data_ptr<float>(), o_accum.data_ptr<float>(),
        params.stride_lse_accum_split, params.stride_lse_accum_s_q,
        params.stride_o_accum_split, params.stride_o_accum_s_q, params.stride_o_accum_h_q,
        params.tile_scheduler_metadata_ptr, params.num_splits_ptr, num_sm_parts,
        ku::get_optional_tensor_ptr<float>(attn_sink),
        at::cuda::getCurrentCUDAStream().stream()
    };
    smxx::decode::run_flash_mla_combine_kernel<bf16>(combine);
    return {out, lse.transpose(1, 2), tile_scheduler_metadata, num_splits};
}
