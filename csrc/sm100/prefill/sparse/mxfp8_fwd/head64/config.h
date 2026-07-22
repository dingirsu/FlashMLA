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
    typename Shape_Q, typename TMA_Q,
    typename Shape_Q_Scale, typename TMA_Q_Scale
>
struct TmaParams {
    Shape_O shape_O; TMA_O tma_O;
    Shape_Q shape_Q; TMA_Q tma_Q;
    Shape_Q_Scale shape_Q_scale; TMA_Q_Scale tma_Q_scale;
    CUtensorMap tensor_map_kv;
};

constexpr int D = 512;
constexpr int D_Q = D;
constexpr int D_K = D;
constexpr int D_V = D;
constexpr int MXFP8_SCALE_VEC_SIZE = 32;
constexpr int SCALE_GROUPS_PER_TMEM_BLOCK = 4;
constexpr int TMEM_SCALE_K128_STRIDE = SCALE_GROUPS_PER_TMEM_BLOCK;
constexpr int QK_SCALE_TMEM_COLS = 2 * SCALE_GROUPS_PER_TMEM_BLOCK;
constexpr int SV_SCALE_TMEM_COLS = SCALE_GROUPS_PER_TMEM_BLOCK;
constexpr int Q_QUANT_GROUP_SIZE = 32;
constexpr int K_QUANT_GROUP_SIZE = 64;
constexpr int Q_SCALE_BYTES = D_Q / Q_QUANT_GROUP_SIZE;
constexpr int K_SCALE_BYTES = D_K / K_QUANT_GROUP_SIZE;
constexpr int K_SCALE_DUP = K_QUANT_GROUP_SIZE / MXFP8_SCALE_VEC_SIZE;
constexpr int KV_SCALE_ANCHOR = 0;
constexpr uint8_t UE8M0_ONE_BITS = 0x7f;
constexpr int Q_BYTES_PER_TOKEN = D_Q + Q_SCALE_BYTES;
constexpr int KV_BYTES_PER_TOKEN = D_K + K_SCALE_BYTES;
constexpr int TMA_K_CHUNK_BYTES = 128;
constexpr int TMA_K_CHUNK_ELEMS = TMA_K_CHUNK_BYTES / sizeof(uint64_t);

constexpr int B_H = 64;
constexpr int B_TOPK = 64;
constexpr int QK_M = B_TOPK * 2;
constexpr int QK_K = D_K / 2;
constexpr int SV_M = 128;
constexpr int SV_SCALE_K = 128;
constexpr int P_T_STRIDE = B_H + 1;
constexpr int NUM_BUFS = 2;
constexpr int NUM_KV_PRODUCER_WARPS = 4;
constexpr int NUM_THREADS = 128 + 128 + 128; // 128 scale & exp threads, 128 TMA threads, 32 UTCMMA threads
constexpr int B_H_TMEM = B_H;
constexpr float MAX_INIT_VAL = -1e30f;
constexpr float FP8_MAX = 448.0f;
constexpr int Q_SCALE_SMEM_ELEMS = B_H * (D / MXFP8_SCALE_VEC_SIZE);
constexpr int K_SCALE_SMEM_ELEMS = QK_M * (QK_K / MXFP8_SCALE_VEC_SIZE);

static_assert(Q_BYTES_PER_TOKEN == 528);
static_assert(KV_BYTES_PER_TOKEN == 520);
static_assert(K_SCALE_DUP == 2);
static_assert(D_K % TMA_K_CHUNK_BYTES == 0);

// Tensor memory columns
namespace tmem_cols {
    //   0 ~ 256: output
    // 256 ~ 288: Q/K/S/V scale-factor columns
    // 384 ~ 448: P from Q view 0
    // 448 ~ 512: P from Q view 1
    constexpr int O = 0;
    constexpr int Q_Scale0 = 256;
    constexpr int Q_Scale1 = Q_Scale0 + QK_SCALE_TMEM_COLS;
    constexpr int K_Scale = Q_Scale1 + QK_SCALE_TMEM_COLS;
    constexpr int S_Scale = K_Scale + QK_SCALE_TMEM_COLS;
    constexpr int V_Scale = S_Scale + SV_SCALE_TMEM_COLS;
    constexpr int P = 384;
    constexpr int P_Part1 = P + B_H;
}

using SmemLayoutQ = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<e4m3>{},
    Shape<Int<B_H>, Int<D_Q>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));


using SmemLayoutQScaleTMA = Layout<
    Shape<Int<B_H>, Int<Q_SCALE_BYTES>>,
    Stride<Int<Q_SCALE_BYTES>, _1>
>;

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

using SmemLayoutQKDual = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<e4m3>{},
    Shape<Int<QK_M>, Int<QK_K>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutV = decltype(coalesce(
    composition(
        SmemLayoutK{},
        Layout<Shape<Int<D_V>, Int<B_TOPK>>, Stride<Int<B_TOPK>, _1>>{}
    )
, Shape<_1, _1>{}));

using SmemLayoutK_TiledMMA = SmemLayoutQKDual;

using SmemLayoutS = decltype(coalesce(tile_to_shape(
  UMMA::Layout_K_INTER_Atom<e4m3>{},
  Shape<Int<B_H>, Int<B_TOPK>>{},
  Step<_1, _2>{}
), Shape<_1, _1>{}));

using TiledMMA_P = decltype(make_tiled_mma( // make the type name shorter
    SM100_MMA_MXF8F6F4_SS_NOELECT<e4m3, e4m3, float, e8m0, QK_M, B_H, UMMA::Major::K, UMMA::Major::K>{}
));

using TiledMMA_O = decltype(make_tiled_mma(
    SM100_MMA_MXF8F6F4_SS_NOELECT<e4m3, e4m3, float, e8m0, SV_M, B_H, UMMA::Major::MN, UMMA::Major::K>{}
));

using SmemLayoutPScaleAAtom = decltype(cutlass::detail::Sm1xxBlockScaledConfig<MXFP8_SCALE_VEC_SIZE>::deduce_smem_layoutSFA(
    TiledMMA_P{},
    Shape<Int<QK_M>, Int<B_H>, Int<QK_K>>{}
));
using SmemLayoutPScaleBAtom = decltype(cutlass::detail::Sm1xxBlockScaledConfig<MXFP8_SCALE_VEC_SIZE>::deduce_smem_layoutSFB(
    TiledMMA_P{},
    Shape<Int<QK_M>, Int<B_H>, Int<QK_K>>{}
));
using SmemLayoutPScaleABlockAtom = decltype(cutlass::detail::Sm1xxBlockScaledConfig<MXFP8_SCALE_VEC_SIZE>::deduce_smem_layoutSFA(
    TiledMMA_P{},
    Shape<Int<QK_M>, Int<B_H>, Int<128>>{}
));
using SmemLayoutPScaleBBlockAtom = decltype(cutlass::detail::Sm1xxBlockScaledConfig<MXFP8_SCALE_VEC_SIZE>::deduce_smem_layoutSFB(
    TiledMMA_P{},
    Shape<Int<QK_M>, Int<B_H>, Int<128>>{}
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

struct SharedMemoryPlan {
    array_aligned<e4m3, B_H*D_Q> q;
    union {
        struct {
            array_aligned<e4m3, B_TOPK*D_K> kv[NUM_BUFS];
            array_aligned<e8m0, K_SCALE_SMEM_ELEMS> kv_scale[NUM_BUFS];
        } kv;
        array_aligned<bf16, cosize_v<SmemLayoutO>> o;
    } kvo;
    union {
        e4m3 s[B_H*B_TOPK];
        struct {
            array_aligned<e8m0, Q_SCALE_SMEM_ELEMS> compact;
            array_aligned<e8m0, cosize_v<SmemLayoutPScaleBAtom>> mma[2];
        } q_scale;
    } s_q_scale;
    array_aligned<e8m0, cosize_v<SmemLayoutOScaleBAtom>> s_scale;
    float head_scale[B_H], head_mi[B_H], head_li[B_H], head_real_mi[B_H];
    char is_k_valid[NUM_BUFS][B_TOPK/8];
    char kv_warp_has_valid[NUM_BUFS][NUM_KV_PRODUCER_WARPS];
    char kv_skip_tma[NUM_BUFS];
    float kv_u_scale[NUM_BUFS][B_TOPK];
    float p_t[B_TOPK*P_T_STRIDE];
    transac_bar_t bar_prologue_q, bar_prologue_q_scale;
    transac_bar_t bar_qk_done[NUM_BUFS];    // Pi = QKi^T (the nope part) done
    transac_bar_t bar_sv_done[NUM_BUFS];    // O += SiVi done (i.e. O, Si and Vi are free)
    transac_bar_t bar_kv_ready[NUM_BUFS], bar_kv_scale_ready[NUM_BUFS];
    transac_bar_t bar_p_free;
    transac_bar_t bar_so_ready;   // S and O are ready
    transac_bar_t bar_k_valid_ready[NUM_BUFS], bar_k_valid_free[NUM_BUFS];
    array_aligned<uint32_t, 1> tmem_start_addr;
    float rowwise_max_buf[128], rowwise_li_buf[128];
};

static_assert(cosize_v<SmemLayoutQ> == cosize_v<SmemLayoutQKDual>);
static_assert(cosize_v<SmemLayoutK> == cosize_v<SmemLayoutK_TiledMMA>);
static_assert(cosize_v<SmemLayoutPScaleAAtom> == 2 * cosize_v<SmemLayoutPScaleABlockAtom>);
static_assert(cosize_v<SmemLayoutPScaleBAtom> == 2 * cosize_v<SmemLayoutPScaleBBlockAtom>);
static_assert(tmem_cols::V_Scale + SV_SCALE_TMEM_COLS <= tmem_cols::P);
static_assert(sizeof(SharedMemoryPlan) < 227 * 1024, "MXFP8 prefill shared memory exceeds the SM100 limit");

enum NamedBarriers : int {
    wg0_sync = 0,
    wg1_tma_sync = 1,
};

} // namespace sm100::mxfp8_fwd::head64
