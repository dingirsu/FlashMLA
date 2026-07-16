#pragma once

#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "defines.h"

namespace sm100::fwd::head_small {

using namespace cute;

template<
    typename Shape_Q_nope, typename TMA_Q_nope,
    typename Shape_Q_rope, typename TMA_Q_rope,
    typename Shape_O, typename TMA_O
>
struct TmaParams {
    Shape_Q_nope shape_Q_nope; TMA_Q_nope tma_Q_nope;
    Shape_Q_rope shape_Q_rope; TMA_Q_rope tma_Q_rope;
    Shape_O shape_O; TMA_O tma_O;
    CUtensorMap tensor_map_kv_nope;
};

struct float2x2 {
    float2 lo, hi;
};

template<int D_QK, int H_Q>
struct KernelTemplate {
static_assert(D_QK == 128);
static_assert(H_Q == 8 || H_Q == 16 || H_Q == 32);

static constexpr int D_Q = D_QK;
static constexpr int D_K = D_QK;
static constexpr int D_V = 128;
static constexpr int D_ROPE = D_QK - D_V;
static constexpr float MAX_INIT_VAL = -1e30;

static constexpr int B_H = H_Q;
static constexpr int B_H_TMEM = H_Q;
static constexpr int B_TOPK = 128;
static constexpr int NUM_BUFS = 2;
static constexpr int NUM_THREADS = 128 + 128 + 128;

struct tmem_cols {
    // Output is transposed as two [64, B_H] tiles.
    // 256 ~ 320: Q NoPE
    // 400 ~ 464: P, transposed as [B_TOPK, B_H]
    static constexpr int O = 0;
    static constexpr int P = 400;
};

using SmemLayoutQNoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<D_V>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutQRoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<64>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutOTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutO = SmemLayoutOTiles<D_V/64>;

template<int NUM_TILES>
using SmemLayoutKTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_TOPK>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutKNoPE = SmemLayoutKTiles<D_V/64>;
using SmemLayoutV = decltype(coalesce(
    composition(
        SmemLayoutKNoPE{},
        Layout<Shape<Int<D_V>, Int<B_TOPK>>, Stride<Int<B_TOPK>, _1>>{}
    )
, Shape<_1, _1>{}));

using SmemLayoutKRoPE = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_TOPK>, Int<64>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutS = decltype(coalesce(tile_to_shape(
	UMMA::Layout_K_SW128_Atom<bf16>{},
	Shape<Int<B_H>, Int<B_TOPK>>{},
	Step<_1, _2>{}
), Shape<_1, _1>{}));

struct SharedMemoryPlan {
    union {
        struct {
            array_aligned<bf16, cosize_v<SmemLayoutQNoPE>> q_nope;
        } q_full;
        struct {
            array_aligned<bf16, cosize_v<SmemLayoutKRoPE>> k_rope;
            array_aligned<bf16, cosize_v<SmemLayoutKNoPE>> k_nope[NUM_BUFS];
        } k;
        array_aligned<bf16, cosize_v<SmemLayoutO>> o;
    } u;
    bf16 s[B_H*B_TOPK];
    array_aligned<bf16, cosize_v<SmemLayoutQRoPE>> q_rope;
    float p_t[B_TOPK*B_H];
    float head_scale[B_H], head_mi[B_H], head_li[B_H], head_real_mi[B_H];
    char is_k_valid[NUM_BUFS][B_TOPK/8];
    transac_bar_t bar_prologue_q_nope, bar_prologue_q_rope, bar_prologue_utccp_nope, bar_prologue_utccp_rope;
    transac_bar_t bar_qk_nope_done[NUM_BUFS], bar_qk_rope_done;
    transac_bar_t bar_sv_done[NUM_BUFS];
    transac_bar_t bar_kv_nope_ready[NUM_BUFS], bar_kv_rope_ready;
    transac_bar_t bar_p_free;
    transac_bar_t bar_so_ready;
    transac_bar_t bar_k_valid_ready[NUM_BUFS], bar_k_valid_free[NUM_BUFS];
    array_aligned<uint32_t, 1> tmem_start_addr;
    float rowwise_max_buf[128], rowwise_li_buf[128];
};

using TiledMMA_P = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_SS_NOELECT<bf16, bf16, float, B_TOPK, B_H, UMMA::Major::K, UMMA::Major::K>{}
)); // May change to WS to spped up?

using TiledMMA_O = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_SS_NOELECT<bf16, bf16, float, 64, B_H, UMMA::Major::MN, UMMA::Major::K>{}
));

enum NamedBarriers : int {
    wg0_sync = 0,
    wg0_warp02_sync = 1,
    wg0_warp13_sync = 2,
    pepi_sync = 3,
};

};

}
