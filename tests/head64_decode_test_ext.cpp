#include <optional>

#include <torch/extension.h>

#include "api/common.h"
#include "params.h"
#include "sm100/decode/head64/kernel.h"

namespace {

std::vector<at::Tensor> head64_decode(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    double sm_scale
) {
    using bf16 = cutlass::bfloat16_t;
    Arch arch;
    TORCH_CHECK(arch.is_sm100f(), "head64 decode test requires an SM100-family GPU");
    TORCH_CHECK(q.is_cuda() && kv.is_cuda() && indices.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(kv.scalar_type() == at::kByte, "kv must be uint8 (packed Model1 FP8 cache)");
    TORCH_CHECK(indices.scalar_type() == at::kInt, "indices must be int32");
    TORCH_CHECK(q.dim() == 4 && q.size(2) == 64 && q.size(3) == 512, "q must be [b, s_q, 64, 512]");
    TORCH_CHECK(kv.dim() == 4 && kv.size(2) == 1 && kv.size(3) == 584, "kv must be [blocks, page, 1, 584]");
    TORCH_CHECK(indices.dim() == 3 && indices.size(0) == q.size(0) && indices.size(1) == q.size(1), "indices must be [b, s_q, topk]");
    TORCH_CHECK(indices.size(2) >= 64 && indices.size(2) % 64 == 0, "topk must be a multiple of 64");
    TORCH_CHECK(q.stride(3) == 1 && kv.stride(3) == 1 && indices.stride(2) == 1, "last dimensions must be contiguous");
    TORCH_CHECK(kv.stride(1) == 584, "packed Model1 cache must have contiguous page rows");

    const int b = static_cast<int>(q.size(0));
    const int s_q = static_cast<int>(q.size(1));
    const int topk = static_cast<int>(indices.size(2));
    const int num_blocks = static_cast<int>(kv.size(0));
    const int page_size = static_cast<int>(kv.size(1));
    at::cuda::CUDAGuard guard(q.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto out = torch::empty({b, s_q, 64, 512}, q.options());
    auto lse = torch::empty({b, s_q, 64}, q.options().dtype(torch::kFloat));
    auto metadata = torch::zeros({1, sizeof(DecodingSchedMeta) / sizeof(int)}, q.options().dtype(torch::kInt32));
    auto splits = torch::zeros({b + 1}, q.options().dtype(torch::kInt32));

    auto* meta = reinterpret_cast<DecodingSchedMeta*>(metadata.data_ptr<int>());
    // One unsplit request handled by one CTA partition.  This bypasses the
    // scheduler/combine kernels so the trace is from head64 decode itself.
    (void)meta;
    auto meta_host = torch::zeros({sizeof(DecodingSchedMeta) / sizeof(int)}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto* hmeta = reinterpret_cast<DecodingSchedMeta*>(meta_host.data_ptr<int>());
    hmeta->begin_req_idx = 0;
    hmeta->end_req_idx = b - 1;
    hmeta->begin_block_idx = 0;
    hmeta->end_block_idx = topk / 64;
    hmeta->begin_split_idx = 0;
    hmeta->is_first_req_splitted = 0;
    hmeta->is_last_req_splitted = 0;
    metadata.copy_(meta_host);

    SparseAttnDecodeParams params{};
    params.b = b; params.s_q = s_q; params.h_q = 64; params.h_kv = 1;
    params.d_qk = 512; params.d_v = 512;
    params.sm_scale = static_cast<float>(sm_scale);
    params.sm_scale_div_log2 = static_cast<float>(sm_scale) * LOG_2_E;
    params.num_blocks = num_blocks; params.page_block_size = page_size; params.topk = topk;
    params.model_type = ModelType::MODEL1;
    params.q = reinterpret_cast<bf16*>(q.data_ptr());
    params.kv = reinterpret_cast<bf16*>(kv.data_ptr());
    params.indices = indices.data_ptr<int>();
    params.topk_length = nullptr; params.attn_sink = nullptr;
    params.lse = lse.data_ptr<float>(); params.out = reinterpret_cast<bf16*>(out.data_ptr());
    params.extra_num_blocks = 0; params.extra_page_block_size = 0; params.extra_topk = 0;
    params.extra_kv = nullptr; params.extra_indices = nullptr; params.extra_topk_length = nullptr;
    params.stride_q_b = int64_stride_to_int(q.stride(0));
    params.stride_q_s_q = int64_stride_to_int(q.stride(1));
    params.stride_q_h_q = int64_stride_to_int(q.stride(2));
    params.stride_kv_block = int64_stride_to_int(kv.stride(0));
    // The kernel's `stride_kv_row` is measured in bytes per token for Model1;
    // the tensor is uint8 and therefore its element stride is already bytes.
    params.stride_kv_row = 584;
    params.stride_indices_b = int64_stride_to_int(indices.stride(0));
    params.stride_indices_s_q = int64_stride_to_int(indices.stride(1));
    params.stride_lse_b = int64_stride_to_int(lse.stride(0));
    params.stride_lse_s_q = int64_stride_to_int(lse.stride(1));
    params.stride_o_b = int64_stride_to_int(out.stride(0));
    params.stride_o_s_q = int64_stride_to_int(out.stride(1));
    params.stride_o_h_q = int64_stride_to_int(out.stride(2));
    params.stride_extra_kv_block = 0; params.stride_extra_kv_row = 0;
    params.stride_extra_indices_b = 0; params.stride_extra_indices_s_q = 0;
    params.stream = stream;
    params.lse_accum = nullptr; params.o_accum = nullptr;
    params.stride_lse_accum_split = 0; params.stride_lse_accum_s_q = 0;
    params.stride_o_accum_split = 0; params.stride_o_accum_s_q = 0; params.stride_o_accum_h_q = 0;
    params.tile_scheduler_metadata_ptr = reinterpret_cast<DecodingSchedMeta*>(metadata.data_ptr<int>());
    params.num_splits_ptr = splits.data_ptr<int>(); params.num_sm_parts = 1;

    sm100::decode::head64::run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::MODEL1>(params);
    return {out, lse};
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("head64_decode", &head64_decode);
}
