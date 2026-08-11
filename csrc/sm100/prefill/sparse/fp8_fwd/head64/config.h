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
    typename Shape_Q, typename TMA_Q,
    typename Shape_Q_Tail, typename TMA_Q_Tail
>
struct TmaParams {
    Shape_O shape_O; TMA_O tma_O;
    Shape_Q shape_Q; TMA_Q tma_Q;
    Shape_Q_Tail shape_Q_tail; TMA_Q_Tail tma_Q_tail;
    CUtensorMap tensor_map_kv;
    CUtensorMap tensor_map_kv_tail;
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
constexpr int TMA_K_TAIL_BYTES = 64;
constexpr int TMA_K_TAIL_ELEMS = TMA_K_TAIL_BYTES / sizeof(uint64_t);
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
#if defined(FP8_FWD_QK576)
// The 576-D path keeps the full 128-token tile.  Reducing the P/S stages and
// reusing the tail-Q storage for tail-K leaves enough SMEM for three KV
// stages.
constexpr int NUM_MAIN_BUFS_K576 = 3;
constexpr int NUM_QK_TAIL_BUFS = 3;
constexpr int NUM_S_BUFS = 1;
constexpr int NUM_P_BUFS = 1;
#else
constexpr int NUM_MAIN_BUFS_K576 = 3;
constexpr int NUM_QK_TAIL_BUFS = 2;
constexpr int NUM_S_BUFS = 2;
constexpr int NUM_P_BUFS = 2;
#endif
constexpr int NUM_KV_PRODUCER_WARPS = 4;
constexpr int NUM_THREADS = 128 + 128 + 128 + 128; // 128 scale & exp threads, 128 TMA threads, 32 UTCMMA threads, 128 rescale threads
constexpr int B_H_TMEM = B_H;
constexpr float MAX_INIT_VAL = -1e30f;
constexpr float FP8_MAX = 448.0f;
static_assert(SV_TMEM_COLS_PER_BLOCK == B_H_TMEM);

namespace tmem_cols {
    constexpr int O = 0;
    constexpr int Q = 256;
    constexpr int P0 = 384;
    constexpr int P1 = P0 + B_TOPK / 2;
    // P1 is unused by the 576-D specialization; its columns hold the 64-D
    // Q tail after the SMEM-to-TMEM copy.
    constexpr int Q_TAIL = P1;
}
static_assert(tmem_cols::P1 + B_TOPK / 2 <= 512);

using SmemLayoutQ = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<e4m3>{},
    Shape<Int<B_H>, Int<D_Q>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

// A 64-byte row needs the 64B swizzle atom.  The tail is kept separate from
// the 128B-swizzled main Q/K tensors so its TMA descriptor has matching
// swizzle and box size.
using SmemLayoutQTail = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<e4m3>{},
    Shape<Int<B_H>, Int<64>>{},
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

using SmemLayoutKTail = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<e4m3>{},
    Shape<Int<B_TOPK>, Int<64>>{},
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

struct QKTailStorage {
    // The Q tail is copied to TMEM during the prologue, then this storage is
    // reused by the tail-K pipeline.  The union avoids paying for both.
    union {
        array_aligned<e4m3, cosize_v<SmemLayoutQTail>> q;
        array_aligned<e4m3, cosize_v<SmemLayoutKTail>> kv[NUM_QK_TAIL_BUFS];
    };
};

struct EmptyQKTailStorage {};
template<bool HAVE_QK_TAIL>
struct SharedMemoryPlanT {
    // The 576-D specialization uses three main KV stages.  Its Q tail and
    // tail-K stages are stored in the separate union below.
    static constexpr int NUM_MAIN_BUFS = HAVE_QK_TAIL
        ? NUM_MAIN_BUFS_K576
        : NUM_BUFS;
    static_assert(NUM_MAIN_BUFS >= NUM_P_BUFS);

    union {
        struct {
            array_aligned<e4m3, cosize_v<SmemLayoutK>> _kv[NUM_MAIN_BUFS - 1];
            array_aligned<e4m3, cosize_v<SmemLayoutQ>> q;
        } q;
        array_aligned<e4m3, cosize_v<SmemLayoutK>> kv[NUM_MAIN_BUFS];
        struct{
            array_aligned<e4m3, cosize_v<SmemLayoutK>> _kv[NUM_MAIN_BUFS - 2]; // avoid overlap with q
            array_aligned<bf16, cosize_v<SmemLayoutO>> o; // bf16 o = 64 * 512 * 2 / 128 * 512 = 1
        } o;
        
    } qkvo;
    // For 576-D, Q tail is copied to TMEM and this storage is subsequently
    // reused for tail K.
    std::conditional_t<HAVE_QK_TAIL, QKTailStorage, EmptyQKTailStorage> qk_tail;
    array_aligned<e4m3, cosize_v<SmemLayoutS>> s[NUM_S_BUFS];
    float kv_token_scale[NUM_MAIN_BUFS][B_TOPK];
#if !defined(FP8_FWD_QK576)
    // Keep the established 512-D scale relay intact.  The 576-D instance
    // loads these values directly in WG0 to recover the SMEM needed by its
    // third main/tail-K stage.
    float q_head_scale[B_H];
    float kv_dim_scale[KV_SCALE_GROUPS];
#endif
    // Scale warps publish one 32-token validity word alongside each scale
    // group.  Keeping it word-addressable avoids a separate mask producer.
    uint32_t is_k_valid[NUM_MAIN_BUFS][B_TOPK/32];
#if defined(FP8_FWD_QK576)
    transac_bar_t bar_prologue, bar_prologue_utccp;
#else
    transac_bar_t bar_prologue, bar_prologue_utccp, bar_qw_scale_ready;
#endif
    // Main QK and (for D_QK=576) tail QK use separate commit points.  The
    // final barrier is consumed by WG0; tail-QK stages are reused by phase.
    transac_bar_t bar_qk_part_done[NUM_MAIN_BUFS];
    transac_bar_t bar_qk_done[NUM_MAIN_BUFS];  // Complete QK (including the tail)
    transac_bar_t bar_sv_block_done[NUM_MAIN_BUFS][NUM_SV_TMEM_BLOCKS - 1];
    transac_bar_t bar_sv_done[NUM_MAIN_BUFS];    // Final SV stripe is committed.
    // Each of the four producer warps owns one 16 KiB transaction.  They
    // arrive on a single full-KV barrier before issuing their gathers; the
    // MMA warp only waits once for the complete 64 KiB tile.
    transac_bar_t bar_kv_ready[NUM_MAIN_BUFS];
    transac_bar_t bar_kv_tail_ready[NUM_QK_TAIL_BUFS];
    transac_bar_t bar_kv_scale_ready[NUM_MAIN_BUFS];
    transac_bar_t bar_p_free[NUM_P_BUFS];
    // WG0 publishes S and its per-warp O-rescale decision independently.
    // Only warps that need a rescale publish new_max in
    // rowwise_max_buf[0:B_H]. The consumed handoff prevents WG0's next
    // row-max reduction from overwriting the decision or max too early.
    transac_bar_t bar_s_ready[NUM_S_BUFS];
    transac_bar_t bar_o_rescale_decision_ready;
    transac_bar_t bar_o_rescale_decision_consumed;
    transac_bar_t bar_o_ready[NUM_MAIN_BUFS];
    uint32_t o_rescale_warp_needed[4];
    array_aligned<uint32_t, 1> tmem_start_addr;
    float rowwise_max_buf[128], rowwise_li_buf[128];
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

using SharedMemoryPlan = SharedMemoryPlanT<false>;

}
