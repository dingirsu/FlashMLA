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
constexpr int KV_SCALE_SLOT_BYTES = 16;
constexpr int KV_BYTES_PER_TOKEN = D_K + KV_SCALE_SLOT_BYTES;

constexpr int KV_SCALE_ANCHOR = 0;
constexpr int TMA_K_CHUNK_BYTES = 128;
constexpr int TMA_K_CHUNK_ELEMS = TMA_K_CHUNK_BYTES / sizeof(uint64_t);
static_assert(KV_BYTES_PER_TOKEN % 16 == 0);

constexpr int B_H = 64;
constexpr int B_TOPK = 128;
constexpr int QK_M = B_H;
constexpr int QK_K = D_K;
constexpr int SV_M = 128;
constexpr int NUM_SV_TMEM_BLOCKS = D_V / SV_M;
// One logical 128-DV stripe occupies 64 TMEM columns in the dual-GEMM view.
constexpr int SV_TMEM_COLS_PER_BLOCK = SV_M / 2;
static_assert(NUM_SV_TMEM_BLOCKS == 4);
static_assert(NUM_SV_TMEM_BLOCKS * SV_TMEM_COLS_PER_BLOCK == D_V / 2);

constexpr int NUM_BUFS = 3;
constexpr int NUM_S_BUFS = 2;
// The direct TS Q fragment reserves [256, 384).  A complete 64x128 FP32 P
// occupies 64 columns, so two P stages fit after it.
constexpr int NUM_P_BUFS = 2;
constexpr int NUM_KV_PRODUCER_WARPS = 4;
constexpr int NUM_THREADS = 128 + 128 + 128; // 128 scale & exp threads, 128 TMA threads, 32 UTCMMA threads
constexpr int B_H_TMEM = B_H;
constexpr float MAX_INIT_VAL = -1e30f;
constexpr float FP8_MAX = 448.0f;
static_assert(SV_TMEM_COLS_PER_BLOCK == B_H_TMEM);

namespace tmem_cols {
    constexpr int O = 0;
    constexpr int Q = 256;
    constexpr int P0 = 384;
    constexpr int P1 = P0 + B_TOPK / 2;
}
static_assert(tmem_cols::P1 + B_TOPK / 2 <= 512);

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

using SmemLayoutK_TiledMMA = SmemLayoutK;

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

#if defined(FP8_FWD_BARRIER_TIMING)
constexpr int FP8_TIMING_MAX_TILES = 8;

struct Fp8Wg0Timing {
    uint64_t tile_start_ns;
    uint64_t pair_wait_ns;
    uint64_t qk_wait_ns;
    uint64_t valid_wait_ns;
    uint64_t scale_wait_ns;
    uint64_t rowmax_wait_ns;
    uint64_t smax_wait_ns;
    uint64_t waits_done_ns;
    uint64_t p_released_ns;
    uint64_t p_scaled_ns;
    uint64_t rowmax_ready_ns;
    uint64_t softmax_exp_ready_ns;
    uint64_t softmax_ready_ns;
    uint64_t sv_wait_ns;
    uint64_t s_stored_ns;
    uint64_t o_rescaled_ns;
    uint64_t s_arrived_ns;
};

// Keep the O-rescale handoff split by SV stripe; the aggregate WG0 fields
// above remain unchanged for existing trace consumers.
struct Fp8ORescaleStripeTiming {
    uint64_t stripe_start_ns;
    uint64_t sv_wait_ns;
    uint64_t sv_ready_ns;
    uint64_t tmem_load_done_ns;
    uint64_t fp32_mul_done_ns;
    uint64_t tmem_store_done_ns;
    uint64_t wg0_sync_done_ns;
    uint64_t warp_rescale_active;
};

struct Fp8KvProducerTiming {
    uint64_t tile_start_ns;
    uint64_t indices_ready_ns;
    uint64_t q_reuse_wait_ns;
    uint64_t sv_free_wait_ns;
    uint64_t tma_part0_issued_ns;
    uint64_t tma_part1_issued_ns;
};

struct Fp8MmaTiming {
    uint64_t iter_start_ns;
    uint64_t p_free_wait_ns;
    uint64_t q_copy_wait_ns;
    uint64_t kv_wait_ns[2];
    uint64_t kv_ready_ns[2];
    uint64_t qk_issued_ns[2];
    uint64_t qk_committed_ns;
    uint64_t s_ready_wait_ns;
    uint64_t s_ready_ns;
    uint64_t sv_committed_ns;
};

struct Fp8SimpleProducerTiming {
    uint64_t tile_start_ns;
    uint64_t buffer_free_wait_ns;
    uint64_t arrived_ns;
};

struct Fp8BarrierTiming {
    uint64_t origin_ns;
    uint64_t branch_end_ns[12];
    uint64_t qw_scale_wait_ns[4];
    uint64_t qw_scale_arrived_ns[2];
    uint64_t final_sv_wait_ns[4];
    uint64_t final_sv_ready_ns[4];
    uint64_t q_tma_wait_ns;
    uint64_t q_tmem_committed_ns;
    Fp8Wg0Timing wg0[4][FP8_TIMING_MAX_TILES];
    Fp8ORescaleStripeTiming o_rescale[4][FP8_TIMING_MAX_TILES]
                                          [NUM_SV_TMEM_BLOCKS];
    Fp8KvProducerTiming kv[4][FP8_TIMING_MAX_TILES];
    Fp8MmaTiming mma[FP8_TIMING_MAX_TILES + 1];
    Fp8SimpleProducerTiming mask[FP8_TIMING_MAX_TILES];
    Fp8SimpleProducerTiming scale[2][FP8_TIMING_MAX_TILES];
};
#endif

struct SharedMemoryPlan {
    union {
        struct {
            array_aligned<e4m3, cosize_v<SmemLayoutK>> _kv[NUM_BUFS - 1];
            array_aligned<e4m3, cosize_v<SmemLayoutQ>> q;
        } q;
        array_aligned<e4m3, cosize_v<SmemLayoutK>> kv[NUM_BUFS];
        array_aligned<bf16, cosize_v<SmemLayoutO>> o;
    } qkvo;
    array_aligned<e4m3, cosize_v<SmemLayoutS>> s[NUM_S_BUFS];
    float kv_token_scale[NUM_BUFS][B_TOPK];
    float q_head_scale[B_H];
    float kv_dim_scale[KV_SCALE_GROUPS];
    char is_k_valid[NUM_BUFS][B_TOPK/8];
    transac_bar_t bar_prologue, bar_prologue_utccp, bar_qw_scale_ready;
    transac_bar_t bar_qk_done[NUM_P_BUFS];  // Pi = QKi^T (the nope part) done
    transac_bar_t bar_sv_block_done[NUM_BUFS][NUM_SV_TMEM_BLOCKS - 1];
    transac_bar_t bar_sv_done[NUM_BUFS];    // Final SV stripe is committed.
    // A stripe may accept the next SV accumulation only after WG0 has
    // rescaled the preceding tile's value in the same TMEM columns.
    transac_bar_t bar_o_rescale_done[NUM_SV_TMEM_BLOCKS];
    transac_bar_t bar_kv_ready[NUM_BUFS][2];
    transac_bar_t bar_kv_scale_ready[NUM_BUFS];
    transac_bar_t bar_p_free[NUM_P_BUFS];
    transac_bar_t bar_so_ready;   // Current S buffer is ready.
    transac_bar_t bar_k_valid_ready[NUM_BUFS], bar_k_valid_free[NUM_BUFS];
    array_aligned<uint32_t, 1> tmem_start_addr;
    float rowwise_max_buf[128], rowwise_li_buf[128];
#if defined(FP8_FWD_BARRIER_TIMING)
    Fp8BarrierTiming barrier_timing;
#endif
};

// may change to bf16 accumulator for better speed
using TiledMMA_P = decltype(make_tiled_mma(
    SM100_MMA_F8F6F4_WS_TS_NOELECT<e4m3, e4m3, float, B_H, B_TOPK, UMMA::Major::K, UMMA::Major::K>{}
)); // maybe p output can be bf16 and use bf16 add to make one fp32

using TiledMMA_O = decltype(make_tiled_mma(
    SM100_MMA_F8F6F4_WS_SS_NOELECT<e4m3, e4m3, float, B_H, SV_M, UMMA::Major::K, UMMA::Major::MN>{}
));

enum NamedBarriers : int {
    wg0_sync = 0,
    wg0_warp02_sync = 1,
    wg0_warp13_sync = 2,
    pepi_sync = 3,
};

}
