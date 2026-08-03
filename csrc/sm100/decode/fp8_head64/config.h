#pragma once

#include "kernel.h"

#include <cstdint>

#include <cutlass/barrier.h>
#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>

#include <kerutils/kerutils.cuh>

#include "defines.h"
#include "params.h"

namespace sm100::decode::fp8_head64 {

using cutlass::arch::fence_view_async_shared;
using cutlass::arch::NamedBarrier;
using e4m3 = cutlass::float_e4m3_t;
using namespace cute;

enum NamedBarriers : uint32_t {
    mainloop_sync = 0,
    wg0_sync = 1,
    everyone_sync = 2,
};

// The first FP8 decode implementation intentionally targets only Model1's
// 512-wide latent.  Keeping these dimensions fixed lets the Q and O TMEM
// allocations exactly fill the layout described below.
static constexpr int D = 512;
static constexpr int D_Q = D;
static constexpr int D_K = D;
static constexpr int D_V = D;
static constexpr int TMA_K_STRIDE = D_K;
static constexpr int TMA_K_CHUNK_BYTES = 128;
static constexpr int TMA_K_CHUNK_ELEMS = TMA_K_CHUNK_BYTES / sizeof(uint64_t);

static constexpr int B_H = 64;
static constexpr int B_TOPK = 128;
static constexpr int SV_M = 128;
static constexpr int NUM_MAIN_BUFS = 3;
static constexpr int NUM_BUFS = NUM_MAIN_BUFS;
static constexpr int NUM_INDEX_BUFS = NUM_MAIN_BUFS;
static constexpr int NUM_THREADS = 3 * 128;
static constexpr int KV_SCALE_GROUPS = D_V / 64;
static constexpr float MAX_INIT_VAL = -1e30f;
static constexpr float FP8_MAX = 448.0f;
static constexpr float S_FP8_SCALE = 1.0f / FP8_MAX;

template<typename Shape_Q, typename TMA_Q, typename Shape_O, typename TMA_O>
struct TmaParams {
    Shape_Q shape_Q;
    TMA_Q tma_Q;
    Shape_O shape_O;
    TMA_O tma_O;
    CUtensorMap tensor_map_kv;
    CUtensorMap tensor_map_extra_kv;
};

// TMEM is a 128 x 512 FP32 matrix.  O occupies columns [0, 256), Q occupies
// [256, 384), and one complete 64 x 128 P tile occupies [384, 448).
struct tmem_cols {
    static constexpr int O = 0;
    static constexpr int Q = 256;
    static constexpr int P = 384;
};
static_assert(tmem_cols::Q + B_H * D_Q / 2 / 128 == tmem_cols::P);
static_assert(tmem_cols::P + B_TOPK / 2 <= 512);

template<int NUM_TILES>
using SmemLayoutQTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<e4m3>{},
    Shape<Int<B_H>, Int<NUM_TILES * 64>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutQ_SW128 = SmemLayoutQTiles<D_Q / 64>;

using SmemLayoutOBuf = decltype(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<D_V>>{}
));

using SmemLayoutOBuf_TMA = decltype(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<64>>{}
));

using SmemLayoutOAccumBuf = Layout<
    Shape<Int<B_H>, Int<D_V>>,
    Stride<Int<520>, _1>
>;

using SmemLayoutS = decltype(tile_to_shape(
    UMMA::Layout_K_INTER_Atom<e4m3>{},
    Shape<Int<B_H>, Int<B_TOPK>>{},
    Step<_1, _2>{}
));

template<int NUM_TILES>
using SmemLayoutKTiles_SW128 = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<e4m3>{},
    Shape<Int<B_TOPK>, Int<64 * NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutKTilesTransposed_SW128 = decltype(composition(
    SmemLayoutKTiles_SW128<NUM_TILES>{},
    Layout<
        Shape<Int<64 * NUM_TILES>, Int<B_TOPK>>,
        Stride<Int<B_TOPK>, _1>
    >{}
));

struct SharedMemoryPlan {
    // Q is copied to TMEM before the three KV stages reuse this storage.
    union {
        array_aligned<e4m3, cosize_v<SmemLayoutQ_SW128>> q;
        union {
            array_aligned<bf16, cosize_v<SmemLayoutOBuf>> o_buf;
            array_aligned<float, cosize_v<SmemLayoutOAccumBuf>> o_accum_buf;
        } o;
        array_aligned<e4m3, cosize_v<SmemLayoutKTiles_SW128<D_K / 64>>> kv[NUM_BUFS];
    } qkvo;
    array_aligned<e4m3, cosize_v<SmemLayoutS>> s;
    CUTE_ALIGNAS(16) float rowwise_max_buf[128];
    char is_token_valid[NUM_INDEX_BUFS][B_TOPK / 8];
    int tma_coord[NUM_INDEX_BUFS][B_TOPK];

    // Reserved for the scaled-FP8 follow-up.  The raw E4M3 first version uses
    // unit scales and therefore leaves these arrays untouched.
    float kv_token_scale[NUM_MAIN_BUFS][B_TOPK];
    float q_head_scale[B_H];
    float kv_dim_scale[KV_SCALE_GROUPS];

    array_aligned<uint32_t, 1> tmem_start_addr;
    transac_bar_t bar_last_store_done;
    transac_bar_t bar_q_tma;
    transac_bar_t bar_q_utccp;
    transac_bar_t bar_kv_ready[NUM_BUFS];
    transac_bar_t bar_valid_coord_scale_ready[NUM_INDEX_BUFS];
    transac_bar_t bar_valid_coord_scale_free[NUM_INDEX_BUFS];
    transac_bar_t bar_qk_done[NUM_BUFS];
    transac_bar_t bar_so_ready[NUM_BUFS];
    transac_bar_t bar_sv_done[NUM_BUFS];
};

using TiledMMA_P = decltype(make_tiled_mma(
    SM100_MMA_F8F6F4_WS_TS_NOELECT<
        e4m3, e4m3, float, B_H, B_TOPK,
        UMMA::Major::K, UMMA::Major::K
    >{}
));

using TiledMMA_O = decltype(make_tiled_mma(
    SM100_MMA_F8F6F4_WS_SS_NOELECT<
        e4m3, e4m3, float, B_H, SV_M,
        UMMA::Major::K, UMMA::Major::MN
    >{}
));

static_assert(sizeof(SharedMemoryPlan) < 227 * 1024,
    "FP8 decode shared memory exceeds the SM100 limit");

} // namespace sm100::decode::fp8_head64
