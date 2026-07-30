#pragma once

#include <cutlass/numeric_types.h>
#include <cute/config.hpp>
#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include "defines.h"

namespace sm100::mxfp8_fwd::head64 {

using namespace cute;

#ifndef MXFP8_PREFILL_DEBUG_VALUES
#define MXFP8_PREFILL_DEBUG_VALUES 0
#endif

using e4m3 = cutlass::float_e4m3_t;
using e8m0 = cutlass::float_ue8m0_t;

template<
    typename Shape_O, typename TMA_O,
    typename Shape_Q, typename TMA_Q
>
struct TmaParams {
    Shape_O shape_O; TMA_O tma_O;
    Shape_Q shape_Q; TMA_Q tma_Q;
    CUtensorMap tensor_map_kv;
};

constexpr int D = 512;
constexpr int D_Q = D;
constexpr int D_K = D;
constexpr int D_V = D;
constexpr int MXFP8_SCALE_VEC_SIZE = 32;
constexpr int SCALE_GROUPS_PER_TMEM_BLOCK = 4;
constexpr int TMEM_SCALE_K128_STRIDE = SCALE_GROUPS_PER_TMEM_BLOCK;
constexpr int QK_SCALE_TMEM_COLS = (D_K / 128) * SCALE_GROUPS_PER_TMEM_BLOCK;
constexpr int SV_SCALE_TMEM_COLS = SCALE_GROUPS_PER_TMEM_BLOCK;
constexpr int Q_QUANT_GROUP_SIZE = 32;
constexpr int K_QUANT_GROUP_SIZE = 64;
constexpr int Q_SCALE_BYTES = D_Q / Q_QUANT_GROUP_SIZE;
constexpr int K_SCALE_BYTES = D_K / K_QUANT_GROUP_SIZE;
constexpr int K_SCALE_DUP = K_QUANT_GROUP_SIZE / MXFP8_SCALE_VEC_SIZE;
constexpr int KV_SCALE_ANCHOR = 0;
constexpr int Q_BYTES_PER_TOKEN = D_Q + Q_SCALE_BYTES;
constexpr int KV_BYTES_PER_TOKEN = D_K + K_SCALE_BYTES;
constexpr int TMA_K_CHUNK_BYTES = 128;
constexpr int TMA_K_CHUNK_ELEMS = TMA_K_CHUNK_BYTES / sizeof(uint64_t);
constexpr int Q_TMA_K = 128;

constexpr int B_H = 64;
constexpr int B_TOPK = 64;
constexpr int QK_M = B_H * 2;
constexpr int QK_N = B_TOPK;
constexpr int QK_K = D_K;
constexpr int SV_M = 128;
constexpr int SV_SCALE_K = 128;
constexpr int NUM_SV_TMEM_BLOCKS = D_V / SV_M;
constexpr int V_SCALE_TMEM_COLS = NUM_SV_TMEM_BLOCKS * SV_SCALE_TMEM_COLS;
constexpr int NUM_BUFS = 3;
constexpr int NUM_P_BUFS = 2;
constexpr int NUM_KV_PRODUCER_WARPS = 4;
constexpr int NUM_THREADS = 128 + 128 + 128;
constexpr int B_H_TMEM = B_H;
constexpr float MAX_INIT_VAL = -1e30f;
constexpr float FP8_MAX = 448.0f;

static_assert(Q_BYTES_PER_TOKEN == 528);
static_assert(KV_BYTES_PER_TOKEN == 520);
static_assert(K_SCALE_DUP == 2);
static_assert(D_K % TMA_K_CHUNK_BYTES == 0);
static_assert(D_V % SV_M == 0);
static_assert(NUM_SV_TMEM_BLOCKS > 1);

// Tensor memory columns
namespace tmem_cols {
    //   0 ~ 256: output
    // 256 ~ 308: Q/K/S/V scale-factor columns
    // 384 ~ 448: P stage 0
    // 448 ~ 512: P stage 1
    constexpr int O = 0;
    constexpr int Q_Scale = 256;
    constexpr int K_Scale = Q_Scale + QK_SCALE_TMEM_COLS;
    constexpr int S_Scale = K_Scale + QK_SCALE_TMEM_COLS;
    constexpr int V_Scale = S_Scale + SV_SCALE_TMEM_COLS;
    constexpr int P0 = 384;
    constexpr int P1 = P0 + QK_N;
}

using SmemLayoutQ = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<e4m3>{},
    Shape<Int<B_H>, Int<D_Q>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutQBlock = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<e4m3>{},
    Shape<Int<B_H>, Int<Q_TMA_K>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutQDuplicated = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<e4m3>{},
    Shape<Int<QK_M>, Int<D_Q>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutOTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutO = SmemLayoutOTiles<8>;
using SmemLayoutOBuf_TMA = SmemLayoutOTiles<1>;

template<int NUM_TILES>
using SmemLayoutKTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<e4m3>{},
    Shape<Int<B_TOPK>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutK = SmemLayoutKTiles<8>;

using SmemLayoutV = decltype(coalesce(
    composition(
        SmemLayoutK{},
        Layout<Shape<Int<D_V>, Int<B_TOPK>>, Stride<Int<B_TOPK>, _1>>{}
    )
, Shape<_1, _1>{}));

using SmemLayoutK_TiledMMA = SmemLayoutK;

using SmemLayoutS = decltype(coalesce(tile_to_shape(
  UMMA::Layout_K_INTER_Atom<e4m3>{},
  Shape<Int<B_H>, Int<B_TOPK>>{},
  Step<_1, _2>{}
), Shape<_1, _1>{}));

using TiledMMA_P = decltype(make_tiled_mma( // make the type name shorter
    SM100_MMA_MXF8F6F4_SS_NOELECT<e4m3, e4m3, float, e8m0, QK_M, QK_N, UMMA::Major::K, UMMA::Major::K>{}
));

using TiledMMA_O = decltype(make_tiled_mma(
    SM100_MMA_MXF8F6F4_SS_NOELECT<e4m3, e4m3, float, e8m0, SV_M, B_H, UMMA::Major::MN, UMMA::Major::K>{}
));

using SmemLayoutPScaleAAtom = decltype(cutlass::detail::Sm1xxBlockScaledConfig<MXFP8_SCALE_VEC_SIZE>::deduce_smem_layoutSFA(
    TiledMMA_P{},
    Shape<Int<QK_M>, Int<QK_N>, Int<QK_K>>{}
));
using SmemLayoutPScaleBAtom = decltype(cutlass::detail::Sm1xxBlockScaledConfig<MXFP8_SCALE_VEC_SIZE>::deduce_smem_layoutSFB(
    TiledMMA_P{},
    Shape<Int<QK_M>, Int<QK_N>, Int<QK_K>>{}
));
using SmemLayoutPScaleABlockAtom = decltype(cutlass::detail::Sm1xxBlockScaledConfig<MXFP8_SCALE_VEC_SIZE>::deduce_smem_layoutSFA(
    TiledMMA_P{},
    Shape<Int<QK_M>, Int<QK_N>, Int<128>>{}
));
using SmemLayoutPScaleBBlockAtom = decltype(cutlass::detail::Sm1xxBlockScaledConfig<MXFP8_SCALE_VEC_SIZE>::deduce_smem_layoutSFB(
    TiledMMA_P{},
    Shape<Int<QK_M>, Int<QK_N>, Int<128>>{}
));
// UTCCP moves scale factors in indivisible 128-wide blocks. Pad the
// scale layout even though each SV data tile has only 64 K elements.
using SmemLayoutOScaleBAtom = decltype(cutlass::detail::Sm1xxBlockScaledConfig<MXFP8_SCALE_VEC_SIZE>::deduce_smem_layoutSFB(
    TiledMMA_O{},
    Shape<Int<SV_M>, Int<B_H>, Int<SV_SCALE_K>>{}
));
using SmemLayoutOScaleAAtom = decltype(cutlass::detail::Sm1xxBlockScaledConfig<MXFP8_SCALE_VEC_SIZE>::deduce_smem_layoutSFA(
    TiledMMA_O{},
    Shape<Int<SV_M>, Int<B_H>, Int<SV_SCALE_K>>{}
));

static_assert(cosize_v<SmemLayoutOScaleAAtom> <= cosize_v<SmemLayoutOScaleBAtom>);

#if defined(MXFP8_FWD_BARRIER_TIMING)
constexpr int MXFP8_TIMING_MAX_TILES = 8;

struct MxFp8Wg0Timing {
    uint64_t tile_start_ns;
    uint64_t tile_sync_wait_ns;
    uint64_t qk_wait_ns;
    uint64_t valid_wait_ns;
    uint64_t waits_done_ns;
    uint64_t p_released_ns;
    uint64_t p_prepared_ns;
    uint64_t rowmax_wait_ns;
    uint64_t rowmax_ready_ns;
    uint64_t s_quantized_ns;
    uint64_t li_wait_ns;
    uint64_t softmax_ready_ns;
    uint64_t sv_wait_ns;
    uint64_t head_updated_ns;
    uint64_t pre_rescale_wait_ns;
    uint64_t pre_rescale_ready_ns;
    uint64_t rescale_done_ns;
    uint64_t rescale_wait_ns;
    uint64_t s_arrived_ns;
};

struct MxFp8KvProducerTiming {
    uint64_t tile_start_ns;
    uint64_t sv_free_wait_ns;
    uint64_t indices_ready_ns;
    uint64_t indices_sync_wait_ns;
    uint64_t transaction_ready_ns;
    uint64_t transaction_sync_wait_ns;
    uint64_t tma_issued_ns;
};

struct MxFp8MmaTiming {
    uint64_t iter_start_ns;
    uint64_t p_free_wait_ns;
    uint64_t kv_scale_wait_ns;
    uint64_t k_scale_ready_ns;
    uint64_t kv_wait_ns;
    uint64_t kv_ready_ns;
    uint64_t qk_committed_ns;
    uint64_t s_ready_wait_ns;
    uint64_t s_ready_ns;
    uint64_t s_scale_ready_ns;
    uint64_t sv_committed_ns;
};

struct MxFp8SimpleProducerTiming {
    uint64_t tile_start_ns;
    uint64_t buffer_free_wait_ns;
    uint64_t arrived_ns;
};

struct MxFp8EpilogueTiming {
    uint64_t final_sv_wait_ns;
    uint64_t final_sv_ready_ns;
    uint64_t stats_stored_ns;
    uint64_t scale_sync_wait_ns;
    uint64_t scale_ready_ns;
    uint64_t o_staged_ns;
    uint64_t o_sync_wait_ns;
    uint64_t store_issued_ns;
};

struct MxFp8BarrierTiming {
    uint64_t origin_ns;
    uint64_t branch_end_ns[12];
    uint64_t q_tma_wait_ns;
    uint64_t q_scale_sync_wait_ns;
    uint64_t q_scale_tmem_committed_ns;
    uint64_t v_scale_tmem_committed_ns;
    MxFp8Wg0Timing wg0[4][MXFP8_TIMING_MAX_TILES];
    MxFp8KvProducerTiming kv[4][MXFP8_TIMING_MAX_TILES];
    MxFp8MmaTiming mma[MXFP8_TIMING_MAX_TILES + 1];
    MxFp8SimpleProducerTiming mask[MXFP8_TIMING_MAX_TILES];
    MxFp8SimpleProducerTiming scale[2][MXFP8_TIMING_MAX_TILES];
    MxFp8EpilogueTiming epilogue[4];
};
#endif

struct SharedMemoryPlan {
    array_aligned<e4m3, QK_M*D_Q> q;
    union {
        struct {
            array_aligned<e4m3, B_TOPK*D_K> kv[NUM_BUFS];
            array_aligned<e8m0, cosize_v<SmemLayoutPScaleBAtom>> kv_scale[NUM_BUFS];
        } kv;
        array_aligned<bf16, cosize_v<SmemLayoutO>> o;
    } kvo;
    // Keep the historical union footprint/alignment: the neighboring SMEM
    // tensor descriptors rely on this placement even though Q scales now
    // bypass this staging storage.
    union {
        e4m3 s[B_H*B_TOPK];
        struct {
            array_aligned<e8m0, B_H * (D / MXFP8_SCALE_VEC_SIZE)> compact;
            array_aligned<e8m0, cosize_v<SmemLayoutPScaleAAtom>> mma;
        } q_scale;
    } s_q_scale;
    array_aligned<e8m0, cosize_v<SmemLayoutOScaleBAtom>> s_scale;
    float head_scale[B_H], head_mi[B_H], head_li[B_H], head_real_mi[B_H];
    char is_k_valid[NUM_BUFS][B_TOPK/8];
    char kv_warp_has_valid[NUM_BUFS][NUM_KV_PRODUCER_WARPS];
    char kv_skip_tma[NUM_BUFS];
    float kv_u_scale[NUM_BUFS][B_TOPK];
    transac_bar_t bar_prologue_q, bar_prologue_q_scale;
    transac_bar_t bar_qk_done[NUM_P_BUFS];  // Pi = QKi^T done
    // Each early barrier covers one 128-DV SV MMA stripe.  The final stripe
    // uses bar_sv_done so existing KV-buffer reuse remains unchanged.
    transac_bar_t bar_sv_block_done[NUM_BUFS][NUM_SV_TMEM_BLOCKS - 1];
    transac_bar_t bar_sv_done[NUM_BUFS];    // O += SiVi done (i.e. O, Si and Vi are free)
    transac_bar_t bar_kv_ready[NUM_BUFS], bar_kv_scale_ready[NUM_BUFS];
    transac_bar_t bar_p_free[NUM_P_BUFS];
    transac_bar_t bar_so_ready;   // S and O are ready
    transac_bar_t bar_k_valid_ready[NUM_BUFS], bar_k_valid_free[NUM_BUFS];
    array_aligned<uint32_t, 1> tmem_start_addr;
    float rowwise_max_buf[128], rowwise_li_buf[128];
#if defined(MXFP8_FWD_BARRIER_TIMING)
    MxFp8BarrierTiming barrier_timing;
#endif
};

static_assert(cosize_v<SmemLayoutQDuplicated> == 2 * cosize_v<SmemLayoutQ>);
static_assert(cosize_v<SmemLayoutQ> == (D_Q / Q_TMA_K) * cosize_v<SmemLayoutQBlock>);
static_assert(cosize_v<SmemLayoutK> == cosize_v<SmemLayoutK_TiledMMA>);
static_assert(cosize_v<SmemLayoutPScaleAAtom> == 4 * cosize_v<SmemLayoutPScaleABlockAtom>);
static_assert(cosize_v<SmemLayoutPScaleBAtom> == 4 * cosize_v<SmemLayoutPScaleBBlockAtom>);
static_assert(
    cosize_v<SmemLayoutOTiles<NUM_SV_TMEM_BLOCKS>> * sizeof(bf16)
        <= B_TOPK * D_K * sizeof(e4m3)
);
static_assert(tmem_cols::V_Scale + V_SCALE_TMEM_COLS <= tmem_cols::P0);
static_assert(sizeof(SharedMemoryPlan) < 227 * 1024, "MXFP8 prefill shared memory exceeds the SM100 limit");

enum NamedBarriers : int {
    wg0_sync = 0,
    wg1_tma_sync = 1,
    q_scale_sync = 2,
};

} // namespace sm100::mxfp8_fwd::head64
