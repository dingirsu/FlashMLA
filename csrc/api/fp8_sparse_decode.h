#pragma once

#include "common.h"
#include "params.h"

#include "sm100/decode/fp8_head64/kernel.h"
#include "smxx/decode/combine/combine.h"
#include "smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.h"

/*
 * FP8 sparse decode interface. Q is packed as
 * [64 * 512 E4M3 bytes][64 UE8M0 head scales] per query token. KV is packed as
 * [512 E4M3 bytes][one UE8M0 token scale][15B padding] per cache token.
 */
static std::tuple<
    at::Tensor,
    at::Tensor,
    std::optional<at::Tensor>,
    std::optional<at::Tensor>
> fp8_sparse_attn_decode_interface(
    const at::Tensor &q,       // [b, s_q, 64 * 512 + 64]
    const at::Tensor &kv,      // [num_blocks, page_block_size, 1, 512 + 16]
    const at::Tensor &kv_scale_w, // [8] UE8M0 dimension scales
    const at::Tensor &indices, // [b, s_q, topk]
    const std::optional<at::Tensor> &topk_length,
    const std::optional<at::Tensor> &attn_sink,
    std::optional<at::Tensor> &tile_scheduler_metadata,
    std::optional<at::Tensor> &num_splits,
    const std::optional<at::Tensor> &extra_kv,
    const std::optional<at::Tensor> &extra_indices,
    const std::optional<at::Tensor> &extra_topk_length,
    int d_qk,
    int d_v,
    float sm_scale
) {
    using bf16 = cutlass::bfloat16_t;

    Arch arch = Arch();
    TORCH_CHECK(
        arch.is_sm100f(),
        "FP8 sparse attention decode is only supported on SM100f."
    );

    KU_CHECK_NDIM(q, 3);
    KU_CHECK_NDIM(kv, 4);
    KU_CHECK_NDIM(kv_scale_w, 1);
    KU_CHECK_NDIM(indices, 3);

    const int b = q.size(0);
    const int s_q = q.size(1);
    constexpr int h_q = 64;
    const int num_blocks = kv.size(0);
    const int page_block_size = kv.size(1);
    const int h_kv = kv.size(2);
    const int topk = indices.size(2);

    const bool have_topk_length = topk_length.has_value();
    const bool have_attn_sink = attn_sink.has_value();
    const bool have_extra_kv = extra_kv.has_value();
    const bool have_extra_topk_length = extra_topk_length.has_value();

    int extra_num_blocks = 0;
    int extra_page_block_size = 0;
    int extra_topk = 0;
    if (have_extra_kv) {
        extra_num_blocks = extra_kv->size(0);
        extra_page_block_size = extra_kv->size(1);
    }
    if (extra_indices.has_value()) {
        extra_topk = extra_indices->size(2);
    }

    TORCH_CHECK(b > 0, "batch size must be positive");
    TORCH_CHECK(s_q > 0, "s_q must be positive");
    TORCH_CHECK(h_kv == 1, "FP8 decode currently supports only h_kv == 1");
    TORCH_CHECK(d_qk == 512, "FP8 decode currently supports only d_qk == 512");
    TORCH_CHECK(d_v == 512, "FP8 decode currently supports only d_v == 512");
    TORCH_CHECK(topk >= 128 && topk % 128 == 0,
        "topk must be a positive multiple of 128");

    if (have_extra_kv) {
        TORCH_CHECK(extra_indices.has_value(),
            "extra_indices must be provided with extra_kv");
        TORCH_CHECK(extra_topk >= 128 && extra_topk % 128 == 0,
            "extra_topk must be a positive multiple of 128");
    } else {
        TORCH_CHECK(!extra_indices.has_value(),
            "extra_indices must not be provided without extra_kv");
        TORCH_CHECK(!have_extra_topk_length,
            "extra_topk_length must not be provided without extra_kv");
    }
    TORCH_CHECK(
        tile_scheduler_metadata.has_value() == num_splits.has_value(),
        "tile_scheduler_metadata and num_splits must be provided together"
    );

    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(kv);
    KU_CHECK_DEVICE(kv_scale_w);
    KU_CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(topk_length);
    KU_CHECK_DEVICE(attn_sink);
    KU_CHECK_DEVICE(tile_scheduler_metadata);
    KU_CHECK_DEVICE(num_splits);
    KU_CHECK_DEVICE(extra_kv);
    KU_CHECK_DEVICE(extra_indices);
    KU_CHECK_DEVICE(extra_topk_length);

    KU_CHECK_DTYPE(q, torch::kUInt8);
    KU_CHECK_DTYPE(kv, torch::kUInt8);
    TORCH_CHECK(
        kv_scale_w.dtype() == at::kFloat8_e8m0fnu
            || kv_scale_w.dtype() == torch::kUInt8,
        "kv_scale_w must have dtype float8_e8m0fnu or uint8"
    );
    if (have_extra_kv) {
        KU_CHECK_DTYPE(extra_kv, torch::kUInt8);
    }
    KU_CHECK_DTYPE(indices, torch::kInt32);
    KU_CHECK_DTYPE(topk_length, torch::kInt32);
    KU_CHECK_DTYPE(attn_sink, torch::kFloat32);
    KU_CHECK_DTYPE(tile_scheduler_metadata, torch::kInt32);
    KU_CHECK_DTYPE(num_splits, torch::kInt32);
    KU_CHECK_DTYPE(extra_indices, torch::kInt32);
    KU_CHECK_DTYPE(extra_topk_length, torch::kInt32);

    KU_CHECK_LAST_DIM_CONTIGUOUS(q);
    KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
    KU_CHECK_CONTIGUOUS(kv_scale_w);
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    KU_CHECK_CONTIGUOUS(topk_length);
    KU_CHECK_CONTIGUOUS(attn_sink);
    KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
    KU_CHECK_CONTIGUOUS(num_splits);
    KU_CHECK_LAST_DIM_CONTIGUOUS(extra_kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(extra_indices);
    KU_CHECK_CONTIGUOUS(extra_topk_length);

    KU_CHECK_SHAPE(q, b, s_q, h_q * d_qk + h_q);
    KU_CHECK_SHAPE(kv, num_blocks, page_block_size, h_kv, d_qk + 16);
    KU_CHECK_SHAPE(kv_scale_w, d_v / 64);
    KU_CHECK_SHAPE(indices, b, s_q, topk);
    KU_CHECK_SHAPE(topk_length, b);
    KU_CHECK_SHAPE(attn_sink, h_q);
    TORCH_CHECK(kv.stride(1) == d_qk + 16,
        "FP8 KV rows must be contiguous; expected stride 528, got ",
        kv.stride(1));
    TORCH_CHECK(kv.stride(0) % (d_qk + 16) == 0,
        "FP8 KV block stride must be a multiple of 528, got ", kv.stride(0));

    if (have_extra_kv) {
        KU_CHECK_SHAPE(
            extra_kv,
            extra_num_blocks,
            extra_page_block_size,
            h_kv,
            d_qk + 16
        );
        KU_CHECK_SHAPE(extra_indices, b, s_q, extra_topk);
        KU_CHECK_SHAPE(extra_topk_length, b);
        TORCH_CHECK(extra_kv->stride(1) == d_qk + 16,
            "FP8 extra KV rows must be contiguous; expected stride 528, got ",
            extra_kv->stride(1));
        TORCH_CHECK(extra_kv->stride(0) % (d_qk + 16) == 0,
            "FP8 extra KV block stride must be a multiple of 528, got ",
            extra_kv->stride(0));
    }

    at::cuda::CUDAGuard device_guard{static_cast<char>(q.get_device())};
    auto opts = q.options();
    at::Tensor out = torch::empty(
        {b, s_q, h_q, d_v}, opts.dtype(torch::kBFloat16)
    );
    at::Tensor lse = torch::empty(
        {b, s_q, h_q}, opts.dtype(torch::kFloat)
    );

    SparseAttnFp8DecodeParams params = {
        b, s_q, h_q, h_kv, d_qk, d_v,
        sm_scale, sm_scale * LOG_2_E,
        num_blocks, page_block_size, topk,
        ModelType::MODEL1,

        q.data_ptr(),
        kv.data_ptr(),
        reinterpret_cast<uint8_t *>(kv_scale_w.data_ptr()),
        reinterpret_cast<int *>(indices.data_ptr()),
        ku::get_optional_tensor_ptr<int>(topk_length),
        ku::get_optional_tensor_ptr<float>(attn_sink),
        reinterpret_cast<float *>(lse.data_ptr()),
        reinterpret_cast<bf16 *>(out.data_ptr()),

        extra_num_blocks, extra_page_block_size, extra_topk,
        have_extra_kv ? extra_kv->data_ptr() : nullptr,
        ku::get_optional_tensor_ptr<int>(extra_indices),
        ku::get_optional_tensor_ptr<int>(extra_topk_length),

        int64_stride_to_int(q.stride(0)),
        int64_stride_to_int(q.stride(1)),
        d_qk,
        int64_stride_to_int(kv.stride(0)),
        int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(indices.stride(0)),
        int64_stride_to_int(indices.stride(1)),
        int64_stride_to_int(lse.stride(0)),
        int64_stride_to_int(lse.stride(1)),
        int64_stride_to_int(out.stride(0)),
        int64_stride_to_int(out.stride(1)),
        int64_stride_to_int(out.stride(2)),

        have_extra_kv ? int64_stride_to_int(extra_kv->stride(0)) : 0,
        have_extra_kv ? int64_stride_to_int(extra_kv->stride(1)) : 0,
        have_extra_kv ? int64_stride_to_int(extra_indices->stride(0)) : 0,
        have_extra_kv ? int64_stride_to_int(extra_indices->stride(1)) : 0,
        at::cuda::getCurrentCUDAStream().stream()
    };

    const int num_sm_parts = std::max(arch.num_sms / s_q, 1);
    constexpr int fixed_overhead_num_blocks = 5;
    constexpr int block_size_topk = 128;

    if (!tile_scheduler_metadata.has_value()) {
        tile_scheduler_metadata = torch::empty(
            {num_sm_parts, sizeof(DecodingSchedMeta) / sizeof(int)},
            opts.dtype(torch::kInt32)
        );
        num_splits = torch::empty({b + 1}, opts.dtype(torch::kInt32));

        GetDecodeSchedMetaParams sched_params = {
            b,
            s_q,
            block_size_topk,
            fixed_overhead_num_blocks,
            topk,
            extra_topk,
            ku::get_optional_tensor_ptr<int>(topk_length),
            ku::get_optional_tensor_ptr<int>(extra_topk_length),
            nullptr,
            reinterpret_cast<DecodingSchedMeta *>(
                tile_scheduler_metadata->data_ptr()
            ),
            num_splits->data_ptr<int>(),
            num_sm_parts,
            at::cuda::getCurrentCUDAStream().stream()
        };
        smxx::decode::run_get_decoding_sched_meta_kernel(sched_params);
    }

    KU_CHECK_DEVICE(tile_scheduler_metadata);
    KU_CHECK_DEVICE(num_splits);
    KU_CHECK_DTYPE(tile_scheduler_metadata, torch::kInt32);
    KU_CHECK_DTYPE(num_splits, torch::kInt32);
    KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
    KU_CHECK_CONTIGUOUS(num_splits);
    KU_CHECK_SHAPE(
        tile_scheduler_metadata,
        num_sm_parts,
        sizeof(DecodingSchedMeta) / sizeof(int)
    );
    KU_CHECK_SHAPE(num_splits, b + 1);

    params.tile_scheduler_metadata_ptr = reinterpret_cast<DecodingSchedMeta *>(
        tile_scheduler_metadata->data_ptr()
    );
    params.num_splits_ptr = num_splits->data_ptr<int>();
    params.num_sm_parts = num_sm_parts;

    const int total_num_splits = b + num_sm_parts;
    at::Tensor lse_accum = torch::empty(
        {total_num_splits, s_q, h_q}, opts.dtype(torch::kFloat)
    );
    at::Tensor o_accum = torch::empty(
        {total_num_splits, s_q, h_q, d_v}, opts.dtype(torch::kFloat)
    );
    params.lse_accum = lse_accum.data_ptr<float>();
    params.o_accum = o_accum.data_ptr<float>();
    params.stride_lse_accum_split = int64_stride_to_int(lse_accum.stride(0));
    params.stride_lse_accum_s_q = int64_stride_to_int(lse_accum.stride(1));
    params.stride_o_accum_split = int64_stride_to_int(o_accum.stride(0));
    params.stride_o_accum_s_q = int64_stride_to_int(o_accum.stride(1));
    params.stride_o_accum_h_q = int64_stride_to_int(o_accum.stride(2));

    sm100::decode::fp8_head64::run_flash_splitkv_mla_fp8_sparse_kernel(params);

    CombineParams combine_params = {
        b, s_q, h_q, d_v,

        params.lse,
        params.out,
        params.stride_lse_b,
        params.stride_lse_s_q,
        params.stride_o_b,
        params.stride_o_s_q,
        params.stride_o_h_q,

        params.lse_accum,
        params.o_accum,
        params.stride_lse_accum_split,
        params.stride_lse_accum_s_q,
        params.stride_o_accum_split,
        params.stride_o_accum_s_q,
        params.stride_o_accum_h_q,

        params.tile_scheduler_metadata_ptr,
        params.num_splits_ptr,
        params.num_sm_parts,

        ku::get_optional_tensor_ptr<float>(attn_sink),
        at::cuda::getCurrentCUDAStream().stream()
    };
    smxx::decode::run_flash_mla_combine_kernel<bf16>(combine_params);

    return {out, lse.transpose(1, 2), tile_scheduler_metadata, num_splits};
}
