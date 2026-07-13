#pragma once

#include <cutlass/numeric_types.h>
#include <cute/config.hpp>
#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include "defines.h"

namespace sm100::mxfp8_fwd::head64 {

using namespace cute;

using e4m3 = cutlass::float_e4m3_t;
using e8m0 = cutlass::float_ue8m0_t;

template<
    typename Shape_O, typename TMA_O,
    typename Shape_Q_Scale, typename TMA_Q_Scale
>
struct TmaParams {
    Shape_O shape_O; TMA_O tma_O;
    Shape_Q_Scale shape_Q_scale; TMA_Q_Scale tma_Q_scale;
    CUtensorMap tensor_map_q;
    CUtensorMap tensor_map_kv;
};

constexpr int D = 512;
constexpr int D_Q = D;
constexpr int D_K = D;
constexpr int D_V = D;
constexpr int NUM_SCALES_EACH_TOKEN = 8;
constexpr int SCALE_GROUP_SIZE = 64;
constexpr int MXFP8_SCALE_VEC_SIZE = 32;
constexpr int K_QUANT_GROUP_SIZE = SCALE_GROUP_SIZE;
constexpr int Q_SCALE_BYTES = NUM_SCALES_EACH_TOKEN;
constexpr int K_SCALE_BYTES = NUM_SCALES_EACH_TOKEN;
constexpr int K_SCALE_DUP = K_QUANT_GROUP_SIZE / MXFP8_SCALE_VEC_SIZE; 
// k sclae group is wider than mxfp8 quant group so we need to duplicate
constexpr int BYTES_PER_TOKEN = D + NUM_SCALES_EACH_TOKEN;

constexpr int B_H = 64;
constexpr int B_TOPK = 128;
constexpr int NUM_BUFS = 3;
constexpr int NUM_THREADS = 128 + 128 + 128; // 128 scale & exp threads, 128 TMA threads, 32 UTCMMA threads
constexpr int B_H_TMEM = B_H;
constexpr float MAX_INIT_VAL = -1e30f;
constexpr float FP8_MAX = 448.0f;
constexpr int Q_SCALE_SMEM_ELEMS = B_H * (D / MXFP8_SCALE_VEC_SIZE);
constexpr int K_SCALE_SMEM_ELEMS = B_TOPK * (D / MXFP8_SCALE_VEC_SIZE);

static_assert(BYTES_PER_TOKEN == 520);

// Tensor memory columns
namespace tmem_cols {
    //   0 ~ 256: output
    // 400 ~ 464: P
    constexpr int O = 0;
    constexpr int Q_Scale = 320;
    constexpr int K_Scale = 338;
    constexpr int S_Scale = 356;
    constexpr int P = 400;
}

using SmemLayoutQ = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<e4m3>{},
    Shape<Int<B_H>, Int<D_Q>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutQScale = decltype(cutlass::detail::Sm1xxBlockScaledConfig<MXFP8_SCALE_VEC_SIZE>::tile_atom_to_shape_SFB(
    Shape<Int<B_TOPK>, Int<B_H>, Int<D_Q>>{}
));

using SmemLayoutQScaleTMA = Layout<
    Shape<Int<B_H>, Int<Q_SCALE_BYTES>>,
    Stride<Int<Q_SCALE_BYTES>, _1>
>;

using SmemLayoutKScale = decltype(cutlass::detail::Sm1xxBlockScaledConfig<MXFP8_SCALE_VEC_SIZE>::tile_atom_to_shape_SFA(
    Shape<Int<B_TOPK>, Int<B_H>, Int<D_K>>{}
));

using SmemLayoutSscale = SmemLayoutQScale;

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
    SM100_MMA_MXF8F6F4_SS_NOELECT<e4m3, e4m3, float, e8m0, B_TOPK, B_H, UMMA::Major::K, UMMA::Major::K>{}
));

using TiledMMA_O = decltype(make_tiled_mma(
    SM100_MMA_MXF8F6F4_SS_NOELECT<e4m3, e4m3, float, e8m0, B_TOPK, B_H, UMMA::Major::MN, UMMA::Major::K>{}
));

using SmemLayoutPScaleAAtom = decltype(cutlass::detail::Sm1xxBlockScaledConfig<MXFP8_SCALE_VEC_SIZE>::deduce_smem_layoutSFA(
    TiledMMA_P{},
    Shape<Int<B_TOPK>, Int<B_H>, Int<D>>{}
));
using SmemLayoutPScaleBAtom = decltype(cutlass::detail::Sm1xxBlockScaledConfig<MXFP8_SCALE_VEC_SIZE>::deduce_smem_layoutSFB(
    TiledMMA_P{},
    Shape<Int<B_TOPK>, Int<B_H>, Int<D>>{}
));
using SmemLayoutOScaleBAtom = decltype(cutlass::detail::Sm1xxBlockScaledConfig<MXFP8_SCALE_VEC_SIZE>::deduce_smem_layoutSFB(
    TiledMMA_O{},
    Shape<Int<B_TOPK>, Int<B_H>, Int<B_TOPK>>{}
));
using SmemLayoutOScaleAAtom = decltype(cutlass::detail::Sm1xxBlockScaledConfig<MXFP8_SCALE_VEC_SIZE>::deduce_smem_layoutSFA(
    TiledMMA_O{},
    Shape<Int<B_TOPK>, Int<B_H>, Int<B_TOPK>>{}
));

struct SharedMemoryPlan {
    union {
        array_aligned<e4m3, B_H*D_Q> q;
        struct {
            array_aligned<e4m3, B_TOPK*D_K> kv[NUM_BUFS];
            array_aligned<e8m0, K_SCALE_SMEM_ELEMS> kv_scale[NUM_BUFS];
        } kv;
        array_aligned<bf16, cosize_v<SmemLayoutO>> o;
    } qkvo;
    union {
        e4m3 s[B_H*B_TOPK];
        array_aligned<e8m0, Q_SCALE_SMEM_ELEMS> q_scale;
    } s_q_scale;
    array_aligned<e8m0, cosize_v<SmemLayoutOScaleBAtom>> s_scale;
    float head_scale[B_H], head_mi[B_H], head_li[B_H], head_real_mi[B_H];
    char is_k_valid[NUM_BUFS][B_TOPK/8];
    float p_t[B_TOPK*B_H];
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

enum NamedBarriers : int {
    wg0_sync = 0,
    wg0_warp02_sync = 1,
    wg0_warp13_sync = 2,
    pepi_sync = 3,
};

} // namespace sm100::mxfp8_fwd::head64
