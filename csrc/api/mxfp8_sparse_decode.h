#pragma once

#include "common.h"
#include "params.h"

#include "sm100/decode/mxfp8_head64/kernel.h"
#include "smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.h"
#include "smxx/decode/combine/combine.h"

enum class MxFp8DecodeFeatures : int {
    HEAD_64,
    HEAD_128,

    HEAD_DIM_576,
    HEAD_DIM_512,

    ATTN_SINK,
    TOPK_LENGTH,
    EXTRA_KVCACHE,
    EXTRA_TOPK_LENGTH
};

struct MxFp8DecodeImplMeta {
    int num_sm_parts;
    int fixed_overhead_num_blocks;
    int block_size_topk;
};

class MxFp8DecodeImplBase : public ImplBase<
    SparseAttnMxfp8DecodeParams,
    MxFp8DecodeFeatures
> {
public:
    virtual MxFp8DecodeImplMeta get_meta(int h_q, int s_q) = 0;
};

class MxFp8Decode_Sm100_Head64_Impl : public MxFp8DecodeImplBase {
    DECLARE_SUPPORTED_FEATURES(
        MxFp8DecodeFeatures::HEAD_64,
        MxFp8DecodeFeatures::HEAD_DIM_512,
        MxFp8DecodeFeatures::ATTN_SINK,
        MxFp8DecodeFeatures::TOPK_LENGTH,
        MxFp8DecodeFeatures::EXTRA_KVCACHE,
        MxFp8DecodeFeatures::EXTRA_TOPK_LENGTH
    )

public:
    MxFp8DecodeImplMeta get_meta(int h_q, int s_q) override {
        Arch arch = Arch();
        return {
            std::max(arch.num_sms / s_q, 1),
            5,
            64
        };
    }

protected:
    void run_(const SparseAttnMxfp8DecodeParams &params, const std::vector<FeatureT> &required_features) override {
        sm100::decode::mxfp8_head64::run_flash_splitkv_mla_mxfp8_sparse_kernel<ModelType::MODEL1>(params);
    }
};

class MxFp8Decode_Sm100_Head64x2_Impl : public MxFp8DecodeImplBase {
    DECLARE_SUPPORTED_FEATURES(
        MxFp8DecodeFeatures::HEAD_128,
        MxFp8DecodeFeatures::HEAD_DIM_512,
        MxFp8DecodeFeatures::ATTN_SINK,
        MxFp8DecodeFeatures::TOPK_LENGTH,
        MxFp8DecodeFeatures::EXTRA_KVCACHE,
        MxFp8DecodeFeatures::EXTRA_TOPK_LENGTH
    )

public:
    MxFp8DecodeImplMeta get_meta(int h_q, int s_q) override {
        Arch arch = Arch();
        return {
            std::max(arch.num_sms / s_q, 1),
            5,
            64
        };
    }

protected:
    void run_(const SparseAttnMxfp8DecodeParams &params, const std::vector<FeatureT> &required_features) override {
            for (int start_head_idx = 0; start_head_idx < 128; start_head_idx += 64) {
                SparseAttnMxfp8DecodeParams cur_params = params;
                cur_params.q = (char*)cur_params.q + start_head_idx * params.stride_q_h_q;
                if (cur_params.attn_sink) {
                    cur_params.attn_sink += start_head_idx;
                }
                cur_params.lse += start_head_idx;
                cur_params.out += start_head_idx * params.stride_o_h_q;
                cur_params.lse_accum += start_head_idx;
                cur_params.o_accum += start_head_idx * params.stride_o_accum_h_q;
                cur_params.h_q = 64;
                sm100::decode::mxfp8_head64::run_flash_splitkv_mla_mxfp8_sparse_kernel<ModelType::MODEL1>(cur_params);
            }
    }
};

/*
 * mxfp8_sparse_attn_decode_interface
 *
 * Pure MXFP8 sparse attention decode for d_qk=d_v=512. SM100 only.
 *
 * Q token layout: 512 e4m3 bytes followed by 16 UE8M0 scales (32-value groups).
 * Each KV page stores all 512-byte e4m3 rows first, then 8 UE8M0 scales per
 * row (64-value groups). The KV tensor's 520-byte last dimension is a storage
 * envelope; scales are not interleaved after individual tokens.
 */
static std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>>
mxfp8_sparse_attn_decode_interface(
    const at::Tensor &q,       // [b, s_q, h_q, bytes_per_token_q]
    const at::Tensor &kv,      // [num_blocks, page_block_size, h_kv, bytes_per_token_kv]
    const at::Tensor &indices, // [b, s_q, topk]
    const std::optional<at::Tensor> &topk_length,   // [b]
    const std::optional<at::Tensor> &attn_sink,     // [h_q]
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
    TORCH_CHECK(arch.is_sm100f(), "MXFP8 sparse attention decode is only supported on SM100f.");

    KU_CHECK_NDIM(q, 4);
    KU_CHECK_NDIM(kv, 4);
    KU_CHECK_NDIM(indices, 3);

    int b = q.size(0);
    int s_q = q.size(1);
    int h_q = q.size(2);
    int num_blocks = kv.size(0);
    int page_block_size = kv.size(1);
    int h_kv = kv.size(2);
    int topk = indices.size(2);

    bool have_topk_length = topk_length.has_value();
    bool have_extra_kcache = extra_kv.has_value();
    bool have_extra_topk_length = extra_topk_length.has_value();
    bool have_attn_sink = attn_sink.has_value();

    int extra_num_blocks = 0, extra_page_block_size = 0, extra_topk = 0;
    if (have_extra_kcache) {
        extra_num_blocks = extra_kv->size(0);
        extra_page_block_size = extra_kv->size(1);
    }
    if (extra_indices.has_value()) {
        extra_topk = extra_indices->size(-1);
    }

    TORCH_CHECK(b > 0);
    TORCH_CHECK(s_q > 0);
    TORCH_CHECK(h_q > 0);
    TORCH_CHECK(h_kv == 1, "Currently only MQA (h_kv == 1) is supported for MXFP8 sparse decoding");
    TORCH_CHECK(d_qk == 512, "MXFP8 decode head64 currently supports only d_qk == 512");
    TORCH_CHECK(d_v == 512, "Only d_v == 512 is supported");
    TORCH_CHECK(topk > 0);

    constexpr int q_bytes_per_token = 512 + 16;
    constexpr int kv_bytes_per_token = 512 + 8;

    if (have_extra_kcache) {
        TORCH_CHECK(extra_indices.has_value(), "extra_indices must be provided when extra_kv is provided");
    } else {
        TORCH_CHECK(!extra_indices.has_value(), "extra_indices must not be provided when extra_kv is not provided");
        TORCH_CHECK(!extra_topk_length.has_value(), "extra_topk_length must not be provided when extra_kv is not provided");
    }

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

    TORCH_CHECK(q.dtype() == torch::kFloat8_e4m3fn || q.dtype() == torch::kUInt8,
        "q must have dtype fp8_e4m3fn or uint8 for MXFP8 mode");
    TORCH_CHECK(kv.dtype() == torch::kFloat8_e4m3fn || kv.dtype() == torch::kUInt8,
        "kv must have dtype fp8_e4m3fn or uint8 for MXFP8 mode");
    if (extra_kv.has_value()) {
        TORCH_CHECK(extra_kv->dtype() == torch::kFloat8_e4m3fn || extra_kv->dtype() == torch::kUInt8,
            "extra_kv must have dtype fp8_e4m3fn or uint8 for MXFP8 mode");
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
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    KU_CHECK_CONTIGUOUS(topk_length);
    KU_CHECK_CONTIGUOUS(attn_sink);
    KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
    KU_CHECK_CONTIGUOUS(num_splits);
    KU_CHECK_LAST_DIM_CONTIGUOUS(extra_kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(extra_indices);
    KU_CHECK_CONTIGUOUS(extra_topk_length);

    KU_CHECK_SHAPE(q, b, s_q, h_q, q_bytes_per_token);
    KU_CHECK_SHAPE(kv, num_blocks, page_block_size, h_kv, kv_bytes_per_token);
    TORCH_CHECK(kv.is_contiguous(), "KV cache storage envelope must be contiguous");
    TORCH_CHECK(kv.stride(0) % 512 == 0,
        "packed KV page stride must be a multiple of 512 bytes, got ", kv.stride(0));
    if (have_extra_kcache) {
        KU_CHECK_SHAPE(extra_kv, extra_num_blocks, extra_page_block_size, h_kv, kv_bytes_per_token);
        TORCH_CHECK(extra_kv->is_contiguous(), "Extra KV cache storage envelope must be contiguous");
        TORCH_CHECK(extra_kv->stride(0) % 512 == 0,
            "packed extra KV page stride must be a multiple of 512 bytes, got ", extra_kv->stride(0));
    }
    KU_CHECK_SHAPE(indices, b, s_q, topk);
    KU_CHECK_SHAPE(topk_length, b);
    KU_CHECK_SHAPE(attn_sink, h_q);
    KU_CHECK_SHAPE(extra_indices, b, s_q, extra_topk);
    KU_CHECK_SHAPE(extra_topk_length, b);

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto opts = q.options();

    at::Tensor out = torch::empty({b, s_q, h_q, d_v}, opts.dtype(torch::kBFloat16));
    at::Tensor lse = torch::empty({b, s_q, h_q}, opts.dtype(at::kFloat));

    std::vector<MxFp8DecodeFeatures> features;
    if (h_q == 64) {
        features.push_back(MxFp8DecodeFeatures::HEAD_64);
    } else if (h_q == 128) {
        features.push_back(MxFp8DecodeFeatures::HEAD_128);
    } else {
        TORCH_CHECK(false, "Unsupported h_q: ", h_q);
    }
    features.push_back(MxFp8DecodeFeatures::HEAD_DIM_512);
    if (have_attn_sink) {
        features.push_back(MxFp8DecodeFeatures::ATTN_SINK);
    }
    if (have_topk_length) {
        features.push_back(MxFp8DecodeFeatures::TOPK_LENGTH);
    }
    if (have_extra_kcache) {
        features.push_back(MxFp8DecodeFeatures::EXTRA_KVCACHE);
    }
    if (have_extra_topk_length) {
        features.push_back(MxFp8DecodeFeatures::EXTRA_TOPK_LENGTH);
    }

    MxFp8DecodeImplBase* impl;
    if (h_q == 64) {
        impl = new MxFp8Decode_Sm100_Head64_Impl();
    } else if (h_q == 128) {
        impl = new MxFp8Decode_Sm100_Head64x2_Impl();
    } else {
        TORCH_CHECK(false, "Unsupported h_q: ", h_q);
    }

    MxFp8DecodeImplMeta impl_meta = impl->get_meta(h_q, s_q);

    SparseAttnMxfp8DecodeParams params = {
        b, s_q, h_q, h_kv, d_qk, d_v,
        sm_scale, sm_scale * LOG_2_E,
        num_blocks, page_block_size, topk,
        ModelType::MODEL1,

        q.data_ptr(),
        kv.data_ptr(),
        (int*)indices.data_ptr(),
        ku::get_optional_tensor_ptr<int>(topk_length),
        ku::get_optional_tensor_ptr<float>(attn_sink),
        (float*)lse.data_ptr(),
        (bf16*)out.data_ptr(),

        extra_num_blocks, extra_page_block_size, extra_topk,
        have_extra_kcache ? extra_kv->data_ptr() : nullptr,
        ku::get_optional_tensor_ptr<int>(extra_indices),
        ku::get_optional_tensor_ptr<int>(extra_topk_length),

        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)), int64_stride_to_int(q.stride(2)),
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),
        int64_stride_to_int(lse.stride(0)), int64_stride_to_int(lse.stride(1)),
        int64_stride_to_int(out.stride(0)), int64_stride_to_int(out.stride(1)), int64_stride_to_int(out.stride(2)),

        have_extra_kcache ? int64_stride_to_int(extra_kv->stride(0)) : 0,
        have_extra_kcache ? int64_stride_to_int(extra_kv->stride(1)) : 0,
        have_extra_kcache ? int64_stride_to_int(extra_indices->stride(0)) : 0,
        have_extra_kcache ? int64_stride_to_int(extra_indices->stride(1)) : 0,
        at::cuda::getCurrentCUDAStream().stream()
    };

    // Get scheduling metadata if necessary
    at::Tensor o_accum, lse_accum;
    if (!tile_scheduler_metadata.has_value()) {
        tile_scheduler_metadata = torch::empty({impl_meta.num_sm_parts, sizeof(DecodingSchedMeta)/4}, opts.dtype(torch::kInt32));
        num_splits = torch::empty({b+1}, opts.dtype(torch::kInt32));
        KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
        KU_CHECK_CONTIGUOUS(num_splits);

        GetDecodeSchedMetaParams get_sched_meta_params = {
            b, s_q,
            impl_meta.block_size_topk,
            impl_meta.fixed_overhead_num_blocks,
            topk,
            extra_topk,
            ku::get_optional_tensor_ptr<int>(topk_length),
            ku::get_optional_tensor_ptr<int>(extra_topk_length),
            nullptr,
            (DecodingSchedMeta*)tile_scheduler_metadata->data_ptr(),
            num_splits->data_ptr<int>(),
            impl_meta.num_sm_parts,
            at::cuda::getCurrentCUDAStream().stream()
        };
        smxx::decode::run_get_decoding_sched_meta_kernel(get_sched_meta_params);
    }

    KU_CHECK_DEVICE(tile_scheduler_metadata);
    KU_CHECK_DEVICE(num_splits);
    KU_CHECK_DTYPE(tile_scheduler_metadata, torch::kInt32);
    KU_CHECK_DTYPE(num_splits, torch::kInt32);
    KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
    KU_CHECK_CONTIGUOUS(num_splits);
    KU_CHECK_SHAPE(tile_scheduler_metadata, impl_meta.num_sm_parts, sizeof(DecodingSchedMeta)/sizeof(int));
    KU_CHECK_SHAPE(num_splits, b+1);
    params.tile_scheduler_metadata_ptr = (DecodingSchedMeta*)tile_scheduler_metadata->data_ptr();
    params.num_splits_ptr = num_splits->data_ptr<int>();
    params.num_sm_parts = impl_meta.num_sm_parts;

    const int total_num_splits = b + impl_meta.num_sm_parts;
    lse_accum = torch::empty({total_num_splits, s_q, h_q}, opts.dtype(at::kFloat));
    o_accum = torch::empty({total_num_splits, s_q, h_q, d_v}, opts.dtype(at::kFloat));
    KU_CHECK_CONTIGUOUS(lse_accum);
    KU_CHECK_CONTIGUOUS(o_accum);
    params.lse_accum = lse_accum.data_ptr<float>();
    params.o_accum = o_accum.data_ptr<float>();
    params.stride_lse_accum_split = int64_stride_to_int(lse_accum.stride(0));
    params.stride_lse_accum_s_q = int64_stride_to_int(lse_accum.stride(1));
    params.stride_o_accum_split = int64_stride_to_int(o_accum.stride(0));
    params.stride_o_accum_s_q = int64_stride_to_int(o_accum.stride(1));
    params.stride_o_accum_h_q = int64_stride_to_int(o_accum.stride(2));

    impl->run(params, features);

    CombineParams combine_params = {
        b, s_q, h_q, d_v,

        params.lse,
        params.out,
        params.stride_lse_b, params.stride_lse_s_q,
        params.stride_o_b, params.stride_o_s_q, params.stride_o_h_q,

        params.lse_accum,
        params.o_accum,
        params.stride_lse_accum_split, params.stride_lse_accum_s_q,
        params.stride_o_accum_split, params.stride_o_accum_s_q, params.stride_o_accum_h_q,

        params.tile_scheduler_metadata_ptr,
        params.num_splits_ptr,
        params.num_sm_parts,

        ku::get_optional_tensor_ptr<float>(attn_sink),
        at::cuda::getCurrentCUDAStream().stream()
    };
    smxx::decode::run_flash_mla_combine_kernel<bf16>(combine_params);

    delete impl;

    return {out, lse.transpose(1, 2), tile_scheduler_metadata, num_splits};
}
