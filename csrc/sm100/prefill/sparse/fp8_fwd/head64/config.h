#pragma once

#include <cutlass/numeric_types.h>
#include <cute/config.hpp>
#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include "defines.h"

namespace sm100::fp8_fwd::head64 {

using namespace cute;

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
constexpr int D_V = 512;
constexpr int KV_SCALE_GROUPS = D_V / 64;

constexpr int KV_SCALE_ANCHOR = 0;
constexpr int TMA_K_CHUNK_BYTES = 128;
constexpr int TMA_K_CHUNK_ELEMS = TMA_K_CHUNK_BYTES / sizeof(uint64_t);

constexpr int B_H = 64;
constexpr int B_TOPK = 64;
constexpr int QK_M = B_TOPK * 2;
constexpr int QK_K = D_K / 2;
constexpr int SV_M = 128;

constexpr int NUM_BUFS = 3;
constexpr int NUM_P_BUFS = 3;
constexpr int NUM_KV_PRODUCER_WARPS = 4;
constexpr int NUM_THREADS = 128 + 128 + 128; // 128 scale & exp threads, 128 TMA threads, 32 UTCMMA threads
constexpr int B_H_TMEM = B_H;
constexpr float MAX_INIT_VAL = -1e30f;
constexpr float FP8_MAX = 448.0f;

namespace tmem_cols {
    constexpr int O = 0;
    constexpr int Q = 256;
    constexpr int P = 320;
}

using SmemLayoutQ = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<e4m3>{},
    Shape<Int<B_H>, Int<D_Q>>{},
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

using SmemLayoutK_TiledMMA = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<e4m3>{},
    Shape<Int<B_TOPK*2>, Int<D_V/2>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutS = decltype(coalesce(tile_to_shape(
	UMMA::Layout_K_INTER_Atom<e4m3>{},
	Shape<Int<B_H>, Int<B_TOPK>>{},
	Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutV = decltype(coalesce(
    composition(
        SmemLayoutK{},
        Layout<Shape<Int<D_V>, Int<B_TOPK>>, Stride<Int<B_TOPK>, _1>>{}
    )
, Shape<_1, _1>{}));

struct SharedMemoryPlan {
    union {
        struct {
            array_aligned<e4m3, cosize_v<SmemLayoutK>> _kv[2]; // to align with kv[2]
            array_aligned<e4m3, cosize_v<SmemLayoutQ>> q;
        } q;
        array_aligned<e4m3, cosize_v<SmemLayoutK>> kv[NUM_BUFS];
        array_aligned<bf16, cosize_v<SmemLayoutO>> o;
    } qkvo;
    float p_exchange_buf[4][32 * (B_TOPK/2)];
    array_aligned<e4m3, cosize_v<SmemLayoutS>> s;
    float kv_dim_scale[NUM_BUFS][B_TOPK];
    float q_head_scale[B_H];
    float kv_w_scale[KV_SCALE_GROUPS];
    char is_k_valid[NUM_BUFS][B_TOPK/8];
    transac_bar_t bar_prologue, bar_prologue_utccp, bar_qw_scale_ready;
    transac_bar_t bar_qk_done[NUM_BUFS];    // Pi = QKi^T (the nope part) done
    transac_bar_t bar_sv_done[NUM_BUFS];    // O += SiVi done (i.e. O, Si and Vi are free)
    transac_bar_t bar_kv_ready[NUM_BUFS][2];
    transac_bar_t bar_kv_scale_ready[NUM_BUFS];
    transac_bar_t bar_p_free[NUM_P_BUFS];
    transac_bar_t bar_so_ready;   // S and O are ready
    transac_bar_t bar_k_valid_ready[NUM_BUFS], bar_k_valid_free[NUM_BUFS];
    array_aligned<uint32_t, 1> tmem_start_addr;
    float rowwise_max_buf[128], rowwise_li_buf[128];
};

// may change to bf16 accumulator for better speed
using TiledMMA_P = decltype(make_tiled_mma(
    SM100_MMA_F8F6F4_WS_TS_NOELECT<e4m3, e4m3, float, B_H, 128, UMMA::Major::K, UMMA::Major::K>{}
)); // maybe p output can be bf16 and use bf16 add to make one fp32

using TiledMMA_O = decltype(make_tiled_mma(
    SM100_MMA_F8F6F4_WS_SS_NOELECT<e4m3, e4m3, float, B_H, 256, UMMA::Major::K, UMMA::Major::MN>{}
));

enum NamedBarriers : int {
    wg0_sync = 0,
    wg0_warp02_sync = 1,
    wg0_warp13_sync = 2,
    pepi_sync = 3,
};

}
