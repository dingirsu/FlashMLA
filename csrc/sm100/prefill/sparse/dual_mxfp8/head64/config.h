#pragma once
#include "phase1.h"

#include <math_constants.h>
#include <cutlass/float8.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "defines.h"
#include "params.h"

namespace sm100::dual_mxfp8::head64 {

using namespace cute;

template<SparseAttnFwdMode FWD_MODE, int D_QK>
struct KernelTemplate {

static constexpr bool IS_DECODE = is_decode_v<FWD_MODE>;
static constexpr bool IS_PREFILL = !IS_DECODE;
using ArgT = std::conditional_t<
    IS_DECODE,
    SparseAttnDecodeParams,
    MxFp8SparseAttnFwdParams
>;
using fp8_e4m3 = cutlass::float_e4m3_t;
using fp8_e8m0 = cutlass::float_ue8m0_t;

struct TmaParamsForPrefill {
    CUtensorMap tensor_map_q;
    CUtensorMap tensor_map_q_scale;
    CUtensorMap tensor_map_kv;
    CUtensorMap tensor_map_kv_scale;
    CUtensorMap tensor_map_o;
};

struct TmaParamsForDecode {
    CUtensorMap tensor_map_q;
    CUtensorMap tensor_map_o;
    CUtensorMap tensor_map_o_accum;
    CUtensorMap tensor_map_kv;
    CUtensorMap tensor_map_extra_kv;   // Only available if extra_kv is enabled
};

using TmaParams = std::conditional_t<
    IS_DECODE,
    TmaParamsForDecode,
    TmaParamsForPrefill
>;

static_assert(D_QK == 512);

static constexpr int D_Q = D_QK;
static constexpr int D_K = D_QK;
static constexpr int D_V = 512;
static constexpr float MAX_INIT_VAL = -1e30;    // We use this number as the initial value for mi (max logits) to avoid -inf - (-inf) = nan

// The 2-SM MMA still has 128 logical rows, but each CTA contributes all 64
// heads from a different query token in the adjacent-token pair.
static constexpr int TOKEN_H_Q = 64;
static constexpr int H_Q = 2*TOKEN_H_Q;
static constexpr int B_TOPK = 64; // For 2 CTAs
static constexpr int NUM_THREADS = 128*4;
// Prefill run_outer_loop participants per CTA:
// WG0=128, KV producer elected lanes=4, validity lanes=8, K-scale
// warps=64, and softmax WG=128. Both CTA0/CTA1 warp8 scale-copy workers
// participate as well. Warp10 folds the CLC query into its K-scale loop.
static constexpr int NUM_WORKER_THREADS = IS_PREFILL
    ? (128 + 4 + (B_TOPK/8) + 64 + 128)*2 + 2
    : (128 + 128 + 1 + 32 + 2 + 128)*2;

// For non-decode mode, we have 4 (half-)KV buffers
// For decode mode, we have 3 (half-)KV buffers with two raw KV buffers
static constexpr int NUM_K_BUFS = IS_DECODE ? 3 : 4;
static constexpr int NUM_INDEX_BUFS = IS_DECODE ? 4 : 4;

static constexpr int D_NOPE = 448;
static constexpr int D_ROPE = 64;
static constexpr int TMA_K_STRIDE_FOR_DECODING = D_QK;
static constexpr int NUM_SCALES_EACH_TOKEN = 8; // 7 scales + 1 padding
static constexpr int MXFP8_SCALE_VEC_SIZE = 32;
static constexpr int Q_SCALE_BYTES = D_Q / MXFP8_SCALE_VEC_SIZE;
static constexpr int K_QUANT_GROUP_SIZE = 64;
static constexpr int K_SCALE_BYTES = D_K / K_QUANT_GROUP_SIZE;
static constexpr int K_SCALE_DUP = K_QUANT_GROUP_SIZE / MXFP8_SCALE_VEC_SIZE;
static constexpr int K_SCALE_TMA_BYTES = 16;
static constexpr int K_SCALE_GATHER_ROWS = 4;
static constexpr int K_SCALE_GATHER_SMEM_STRIDE = 128;
static constexpr int Q_BYTES_PER_HEAD = D_Q + Q_SCALE_BYTES;
static constexpr int KV_BYTES_PER_TOKEN = D_K + K_SCALE_BYTES;
static constexpr uint8_t UE8M0_ONE_BITS = 0x7f;
static_assert(Q_SCALE_BYTES == 16);
static_assert(K_SCALE_BYTES == 8);
static_assert(K_SCALE_DUP == 2);

static constexpr int B_EPI = 64;                // Epilogue block size for normal case (i.e. prefill or non-splitkv decoding)
static constexpr int B_EPI_SPLITKV = 32;        // Epilogue block size for splitkv decoding
static constexpr int NUM_EPI_SPLITKV_BUFS = 4;  // The number of epilogue buffers for splitkv decoding
static_assert((H_Q/2)*D_Q*sizeof(bf16) >= NUM_EPI_SPLITKV_BUFS*(H_Q/2)*(B_EPI_SPLITKV*2)*sizeof(float));

// Tensor memory columns
struct tmem_cols {
    //   0 ~ 256: Output accumulator
    // 256 ~ 384: Q
    // 384 ~ 448: P
    static constexpr int O = 0;
    static constexpr int Q = 256;
    static constexpr int Q_scale = 320;
    static constexpr int K_scale = 336;
    static constexpr int S_scale = 344;
    static constexpr int V_scale = 352;
    static constexpr int P = 384;
};

struct SharedMemoryPlan {
    // Q is reused by the BF16 output epilogue after the final MMA.
    array_aligned<fp8_e4m3, (H_Q/2)*D_Q*sizeof(bf16)> Q;
    array_aligned<fp8_e4m3, B_TOPK*(D_K/2)> K[NUM_K_BUFS];
    array_aligned<fp8_e4m3, (H_Q/2)*B_TOPK> S;
    // Q scales are loaded once per adjacent-token pair.  K scales are
    // double-buffered with K data and expanded from 64-value groups to the
    // 32-value groups consumed by tcgen05.
    CUTE_ALIGNAS(16) fp8_e8m0 q_scale_raw[TOKEN_H_Q][Q_SCALE_BYTES];
    array_aligned<fp8_e8m0, 8192> q_scale_mma;
    // A gather4 destination must be 128B aligned for cta_group::2. Each
    // issued copy writes four 16B token-pair rows into the first 64B.
    CUTE_ALIGNAS(128) fp8_e8m0
        k_scale_pair_raw[NUM_K_BUFS][B_TOPK / K_SCALE_GATHER_ROWS][K_SCALE_GATHER_SMEM_STRIDE];
    CUTE_ALIGNAS(16) int k_scale_token_idx[NUM_K_BUFS][B_TOPK];
    CUTE_ALIGNAS(16) fp8_e8m0 k_scale_expanded[B_TOPK][Q_SCALE_BYTES];
    array_aligned<fp8_e8m0, 8192> k_scale_mma;
    // V remains unit-scaled and uses the CUTLASS SMEM -> TMEM scale path.
    array_aligned<fp8_e8m0, 8192> v_scale_one;
    float P_exchange[4][(H_Q/2/2)*(B_TOPK/2)];
    float rowwise_max_buf[128], rowwise_li_buf[128];

    CUTE_ALIGNAS(16) char is_k_valid[NUM_INDEX_BUFS][B_TOPK/8];
    CUTE_ALIGNAS(16) int tma_coord[NUM_INDEX_BUFS][B_TOPK];
    CUTE_ALIGNAS(16) fp8_e8m0 scales[NUM_INDEX_BUFS][B_TOPK][NUM_SCALES_EACH_TOKEN/2];
    
    transac_bar_t bar_sQ_full, bar_sQ_scale_full;
    transac_bar_t bar_Q_scale_ready;
    transac_bar_t bar_tQ_empty, bar_tQ_full;
    transac_bar_t bar_tOut_full, bar_tOut_empty;
    transac_bar_t bar_KV_full[NUM_K_BUFS], bar_KV_empty[NUM_K_BUFS];
    transac_bar_t bar_K_scale_raw_full[NUM_K_BUFS];
    transac_bar_t bar_K_scale_layout_ready[NUM_K_BUFS];
    transac_bar_t bar_P_empty;
    transac_bar_t bar_QK_done, bar_SV_done;
    transac_bar_t bar_S_O_full;
    transac_bar_t bar_li_full, bar_li_empty;

    // The following barriers are prefill-only
    transac_bar_t bar_clc_full, bar_clc_empty;

    // The following barriers are decode-only
    transac_bar_t bar_valid_coord_scales_full[NUM_INDEX_BUFS], bar_valid_coord_scales_empty[NUM_INDEX_BUFS];

    ku::CLCResponseObj clc_response_obj;
    array_aligned<uint32_t, 1> tmem_start_addr;
};

using TiledMMA_P = decltype(make_tiled_mma(
    SM100_MMA_MXF8F6F4_2x1SM_TS_NOELECT<fp8_e4m3, fp8_e4m3, float, fp8_e8m0, H_Q, B_TOPK*2, UMMA::Major::K, UMMA::Major::K>{}
)); // *2 for dual gemm

using TiledMMA_O = decltype(make_tiled_mma(
    SM100_MMA_MXF8F6F4_2x1SM_SS_NOELECT<fp8_e4m3, fp8_e4m3, float, fp8_e8m0, H_Q, 256, UMMA::Major::K, UMMA::Major::MN>{},
    Layout<Shape<_1, _1, _1>>{},
    Tile<Int<128>, Layout<Shape<_128, _2, _2>, Stride<_1, _256, _128>>, _16>{}  // We use this permutation layout to let CTA0 takes V[:, 0:256] and CTA1 takes V[:, 256:512]
));

// CUTLASS's generic 2-CTA scale layout assumes a 128-row scale tile per CTA.
// This kernel uses the legal M=128 2-SM MMA shape, i.e. 64 rows per CTA, so
// use the MMA's 128-row physical scale atom and let UTCCP broadcast it across
// the two CTAs.  All current factors are one, so the duplicate logical rows
// need no distinct SMEM storage.
using SmemLayoutPScaleA = decltype(cutlass::detail::Sm1xxBlockScaledConfig<32>::deduce_smem_layoutSFA(
    TiledMMA_P{}, Shape<Int<H_Q*2>, Int<B_TOPK*2>, Int<D_Q>>{}
));
using SmemLayoutPScaleB = decltype(cutlass::detail::Sm1xxBlockScaledConfig<32>::deduce_smem_layoutSFB(
    TiledMMA_P{}, Shape<Int<H_Q*2>, Int<B_TOPK*2>, Int<D_K>>{}
));
using SmemLayoutOScaleA = decltype(cutlass::detail::Sm1xxBlockScaledConfig<32>::deduce_smem_layoutSFA(
    TiledMMA_O{}, Shape<Int<H_Q*2>, Int<256>, _128>{}
));
using SmemLayoutOScaleB = decltype(cutlass::detail::Sm1xxBlockScaledConfig<32>::deduce_smem_layoutSFB(
    TiledMMA_O{}, Shape<Int<H_Q*2>, Int<256>, _128>{}
));

static_assert(cosize_v<SmemLayoutPScaleA> <= 8192);
static_assert(cosize_v<SmemLayoutPScaleB> <= 8192);
static_assert(cosize_v<SmemLayoutOScaleA> <= 8192);
static_assert(cosize_v<SmemLayoutOScaleB> <= 8192);
static_assert(tmem_cols::Q_scale + 16 <= tmem_cols::K_scale);
static_assert(tmem_cols::K_scale + 8 <= tmem_cols::S_scale);
static_assert(tmem_cols::S_scale + 4 <= tmem_cols::V_scale);
static_assert(tmem_cols::V_scale + 4 <= tmem_cols::P);

struct barrier_ids {
    static constexpr int WG0_SYNC = 0;
    static constexpr int WG2_SYNC = 1;
    static constexpr int WG2_WARP02_SYNC = 2;
    static constexpr int WG2_WARP13_SYNC = 3;
};

static __device__ void
sparse_attn_fwd_kernel_devfunc(const ArgT &params, const TmaParams &tma_params);

static void run(const ArgT& params);

};

}
