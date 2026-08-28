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
    SparseAttnDualMxfp8DecodeParams,
    MxFp8SparseAttnFwdParams
>;
using fp8_e4m3 = cutlass::float_e4m3_t;
using fp8_e8m0 = cutlass::float_ue8m0_t;

struct TmaParamsForPrefill {
    CUtensorMap tensor_map_q;
    CUtensorMap tensor_map_kv;
    CUtensorMap tensor_map_k_scale;
    CUtensorMap tensor_map_o;
};

struct TmaParamsForDecode {
    CUtensorMap tensor_map_q;
    CUtensorMap tensor_map_o;
    CUtensorMap tensor_map_o_accum;
    CUtensorMap tensor_map_kv;
    CUtensorMap tensor_map_extra_kv;
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

// Avoid -inf - (-inf) when the first tile contains no valid logits.
static constexpr float MAX_INIT_VAL = -1e30;

// The 2-SM MMA still has 128 logical rows, but each CTA contributes all 64
// heads from a different query token in the adjacent-token pair.
static constexpr int TOKEN_H_Q = 64;
static constexpr int H_Q = 2*TOKEN_H_Q;
static constexpr int B_TOPK = 64;
static constexpr int NUM_THREADS = 128*4;
static constexpr int NUM_WORKER_THREADS = IS_PREFILL
    ? (128 + 4 + (B_TOPK/8) + 64 + 128)*2 + 1
    : (128 + 128 + 1 + 32 + 2 + 128)*2;

static constexpr int NUM_K_BUFS = IS_DECODE ? 3 : 4;
static constexpr int NUM_INDEX_BUFS = 4;

static constexpr int TMA_K_STRIDE_FOR_DECODING = D_QK;
static constexpr int Q_QUANT_GROUP_SIZE = 64;
static constexpr int Q_SCALE_BYTES = D_Q / Q_QUANT_GROUP_SIZE;
static constexpr int Q_SCALE_SLOT_BYTES = 16;
static constexpr int K_QUANT_GROUP_SIZE = 64;
static constexpr int K_SCALE_BYTES = D_K / K_QUANT_GROUP_SIZE;
static constexpr int K_SCALE_SLOT_BYTES = 16;
static constexpr int Q_BYTES_PER_HEAD = D_Q + Q_SCALE_SLOT_BYTES;
static constexpr int KV_BYTES_PER_TOKEN = D_K + K_SCALE_SLOT_BYTES;
static_assert(Q_SCALE_BYTES == 8);
static_assert(Q_SCALE_SLOT_BYTES == 16);
static_assert(K_SCALE_BYTES == 8);
static_assert(K_SCALE_SLOT_BYTES == 2 * K_SCALE_BYTES);

static constexpr int B_EPI = 64;
static constexpr int B_EPI_SPLITKV = 32;
static constexpr int NUM_EPI_SPLITKV_BUFS = 4;
static_assert(
    (H_Q/2)*D_Q*sizeof(bf16)
        >= NUM_EPI_SPLITKV_BUFS*(H_Q/2)*(B_EPI_SPLITKV*2)*sizeof(float)
);

// Tensor memory columns
struct tmem_cols {
    //   0 ~ 256: Output accumulator
    // 256 ~ 384: The two M-tiled P fragments (at 256 and 320)
    // 384 ~ 392: Double-buffered S scale factors
    // 448 ~ 512: Q/K/V scale factors
    static constexpr int O = 0;
    static constexpr int P = 256;
    static constexpr int S_scale = 384;
    static constexpr int S_scale_stride = 4;
    static constexpr int Q_scale = 448;
    static constexpr int K_scale = 480;
    static constexpr int V_scale = 500;
    static constexpr int V_scale_n_stride = 4;
};

struct SharedMemoryPlan {
    array_aligned<fp8_e4m3, (H_Q/2)*D_Q*sizeof(bf16)> Q;
    array_aligned<fp8_e4m3, B_TOPK*(D_K/2)> K[NUM_K_BUFS];
    array_aligned<fp8_e4m3, (H_Q/2)*B_TOPK> S[2];
    CUTE_ALIGNAS(16) float v_token_scale[NUM_K_BUFS][B_TOPK];
    // One S scale per softmax row and per K=32 atom in the 64-token tile.
    // The two bytes are packed into one TMEM scale word before O MMA.
    CUTE_ALIGNAS(16) uint8_t s_scale_exp[NUM_K_BUFS][H_Q/2][2];
    // Final 64-row x 16B post-transpose source consumed by 2x64 UTCCP.
    array_aligned<fp8_e8m0, B_TOPK * 16> k_scale_mma[NUM_K_BUFS];
    float P_exchange[4][(H_Q/2/2)*(B_TOPK/2)];
    float rowwise_max_buf[128], rowwise_li_buf[128];

    CUTE_ALIGNAS(16) char is_k_valid[NUM_INDEX_BUFS][B_TOPK/8];
    CUTE_ALIGNAS(16) int tma_coord[NUM_INDEX_BUFS][B_TOPK];
    CUTE_ALIGNAS(16) int64_t k_scale_offset[NUM_INDEX_BUFS][B_TOPK];

    transac_bar_t bar_sQ_full;
    transac_bar_t bar_Q_scale_ready;
    transac_bar_t bar_tQ_empty, bar_tQ_full;
    transac_bar_t bar_tOut_full, bar_tOut_empty;
    transac_bar_t bar_KV_full[NUM_K_BUFS], bar_KV_empty[NUM_K_BUFS];
    transac_bar_t bar_K_scale_copy_ready[NUM_K_BUFS];
    transac_bar_t bar_v_scale_full[NUM_K_BUFS], bar_v_scale_empty[NUM_K_BUFS];
    transac_bar_t bar_P_empty;
    transac_bar_t bar_QK_done[NUM_K_BUFS];
    transac_bar_t bar_S_empty[2], bar_S_O_full[2];
    transac_bar_t bar_li_full, bar_li_empty;

    transac_bar_t bar_clc_full, bar_clc_empty;
    transac_bar_t bar_valid_coord_scales_full[NUM_INDEX_BUFS];
    transac_bar_t bar_valid_coord_scales_empty[NUM_INDEX_BUFS];

    ku::CLCResponseObj clc_response_obj;
    array_aligned<uint32_t, 1> tmem_start_addr;
};

using TiledMMA_P = decltype(make_tiled_mma(
    SM100_MMA_MXF8F6F4_2x1SM_SS_NOELECT<
        fp8_e4m3, fp8_e4m3, float, fp8_e8m0,
        H_Q, B_TOPK*2, UMMA::Major::K, UMMA::Major::K
    >{}
));

using TiledMMA_O = decltype(make_tiled_mma(
    SM100_MMA_MXF8F6F4_2x1SM_SS_NOELECT<
        fp8_e4m3, fp8_e4m3, float, fp8_e8m0,
        H_Q, 256, UMMA::Major::K, UMMA::Major::MN
    >{},
    Layout<Shape<_1, _1, _1>>{},
    // CTA0 consumes V[:, 0:256], while CTA1 consumes V[:, 256:512].
    Tile<
        Int<128>,
        Layout<Shape<_128, _2, _2>, Stride<_1, _256, _128>>,
        _16
    >{}
));

// Undo the K=32 atom permutation introduced by the dual-64 TMA packing:
// [0,8,1,9,4,12,5,13,2,10,3,11,6,14,7,15].
using SmemLayoutQPhysical = decltype(
    ku::make_umma_canonical_k_major_layout<H_Q/2, D_Q, 128, fp8_e4m3>()
);
using QGlobalToPhysical = Layout<
    Shape<Shape<Int<H_Q/2>, _2, _2>, Shape<_32, _2, _2, _2>>,
    Stride<
        Stride<_1, _0, Int<(H_Q/2)*32>>,
        Stride<
            Int<H_Q/2>,
            Int<(H_Q/2)*32*2>,
            Int<(H_Q/2)*32*8>,
            Int<(H_Q/2)*32*4>
        >
    >
>;
using SmemLayoutQ = decltype(composition(
    SmemLayoutQPhysical{}, QGlobalToPhysical{}
));
using SmemLayoutK = decltype(
    ku::make_umma_canonical_k_major_layout<B_TOPK, D_K/2, 128, fp8_e4m3>()
);
using SmemLayoutS = decltype(
    ku::make_umma_canonical_k_major_layout<H_Q/2, B_TOPK, 0, fp8_e4m3>()
);
using SmemLayoutV = decltype(
    ku::make_umma_canonical_mn_major_layout<D_V/2, B_TOPK, 128, fp8_e4m3>()
);

using SmemLayoutPScaleA = decltype(
    cutlass::detail::Sm1xxBlockScaledConfig<32>::deduce_smem_layoutSFA(
        TiledMMA_P{}, Shape<Int<H_Q*2>, Int<B_TOPK*2>, Int<D_Q>>{}
    )
);
using SmemLayoutPScaleB = decltype(
    cutlass::detail::Sm1xxBlockScaledConfig<32>::deduce_smem_layoutSFB(
        TiledMMA_P{}, Shape<Int<H_Q*2>, Int<B_TOPK*2>, Int<D_K>>{}
    )
);

struct barrier_ids {
    static constexpr int WG0_SYNC = 0;
    static constexpr int WG2_SYNC = 1;
    static constexpr int WG2_WARP02_SYNC = 2;
    static constexpr int WG2_WARP13_SYNC = 3;
    static constexpr int S_SCALE_SYNC = 4;
};

static __device__ void
sparse_attn_fwd_kernel_devfunc(const ArgT &params, const TmaParams &tma_params);

static void run(const ArgT& params);

};

}
