#pragma once

#include "kernel.h"

#include <cuda_fp8.h>
#include <cutlass/barrier.h>
#include <cute/tensor.hpp>

#include <kerutils/kerutils.cuh>

#include "defines.h"
#include "params.h"

namespace sm100::decode::mxfp8_head64 {

using cutlass::arch::fence_view_async_shared;
using cutlass::arch::NamedBarrier;
using e8m0 = __nv_fp8_e8m0;
using e4m3 = cutlass::float_e4m3_t;
using namespace cute;

enum NamedBarriers : uint32_t {
    main_loop_sync = 0,
    wg0_sync = 1,
    wg0_warp02_sync = 2,
    wg0_warp13_sync = 3,
    everyone_sync = 4
};

template<ModelType MODEL_TYPE>
struct KernelTemplate {

static constexpr int D_Q = 512;
static constexpr int D_K = D_Q;
static constexpr int D_V = 512;
static constexpr int QUANT_TILE_SIZE = 64;
static constexpr int NUM_SCALES_EACH_TOKEN = 8;   
static constexpr int TMA_K_STRIDE = 512;
static constexpr int B_H = 64;
static constexpr int B_TOPK = 128;
constexpr int MXFP8_SCALE_VEC_SIZE = 32;
constexpr int Q_SCALE_BYTES = NUM_SCALES_EACH_TOKEN;
constexpr int K_SCALE_SMEM_ELEMS = B_TOPK * (D_K / MXFP8_SCALE_VEC_SIZE);
static constexpr int NUM_BUFS = 2;
static constexpr int NUM_INDEX_BUFS = 4;    // Number of buffers for indices (tma_coords) & is_token_valid & scales
static constexpr int NUM_THREADS = 128*3;  // 128 exp + 1/32 utcmma + 1/32 raw KV producer + 1/32 rope producer + 32 index+scale+valid_mask producer + 128 dequant
static constexpr float MAX_INIT_VAL = -1e30f;  // To avoid (-inf) - (-inf) = NaN

template<
    typename Shape_Q, typename TMA_Q,
    typename Shape_O, typename TMA_O
>
struct TmaParams {
    Shape_Q shape_Q; TMA_Q tma_Q;
    Shape_O shape_O; TMA_O tma_O;
    CUtensorMap tensor_map_kv;
    CUtensorMap tensor_map_extra_kv;
};

// Tensor memory columns
struct tmem_cols {
    //   0 ~ 256: output
    // 256 ~ 256 + 64*D_Q/256: Q
    // 400 ~ 464: P
    static constexpr int O = 0;
    static constexpr int QScale = 320;
    static constexpr int KScale = 338;
    static constexpr int SScale = 356;
    static constexpr int P = 400;
};

template<int NUM_TILES>
using SmemLayoutQTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<e4m3>{},
    Shape<Int<B_H>, Int<NUM_TILES*64>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutQ_SW128 = SmemLayoutQTiles<8>;

using SmemLayoutOBuf = decltype(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<D_V>>{}
));

using SmemLayoutOBuf_TMA = decltype(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<64>>{}
)); // A TMA tile

static_assert(D_V == 512);
using SmemLayoutOAccumBuf = Layout<
    Shape<Int<B_H>, Int<D_V>>, // might be D_V, B_H
    Stride<Int<520>, _1>	// We use stride = 520 here to avoid bank conflict
>;

using SmemLayoutS = decltype(tile_to_shape(
    UMMA::Layout_K_INTER_Atom<e4m3>{},
    Shape<Int<B_H>, Int<B_TOPK>>{},
    Step<_1, _2>{}
));

template<int NUM_TILES>
using SmemLayoutKTiles_SW128 = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<e4m3>{},
    Shape<Int<B_H>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutKTilesTransposed_SW128 = decltype(composition(
    SmemLayoutKTiles_SW128<NUM_TILES>{},
    Layout<
        Shape<Int<64*NUM_TILES>, Int<B_TOPK>>,
        Stride<Int<B_TOPK>, _1>
    >{}
));

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

struct SharedMemoryPlan {
    union {
        struct {
            array_aligned<e4m3, cosize_v<SmemLayoutQ_SW128>> q;
            array_aligned<e8m0, Q_SCALE_SMEM_ELEMS> q_scale;
            union {
                array_aligned<bf16, cosize_v<SmemLayoutOBuf>> o_buf;
                array_aligned<float, cosize_v<SmemLayoutOAccumBuf>> o_accum_buf;
            } o;
        } qo;
        struct {
            array_aligned<e8m0, K_SCALE_SMEM_ELEMS> kv_scale[NUM_BUFS];
            array_aligned<e4m3, B_H*D_K> kv[NUM_BUFS];  // Raw (quantized) NoPE part
        } kv;
    } u;
    union {
        float4 p_exchange_buf[4][16 * B_TOPK / 4]; // why this layout?
        array_aligned<e4m3, cosize_v<SmemLayoutS>> s;
    } s_p;
    CUTE_ALIGNAS(16) float rowwise_max_buf[128];
    char is_token_valid[NUM_INDEX_BUFS][B_TOPK/8];
    int tma_coord[NUM_INDEX_BUFS][B_TOPK];
    e8m0 scales[NUM_INDEX_BUFS][B_TOPK][NUM_SCALES_EACH_TOKEN];
    array_aligned<uint32_t, 1> tmem_start_addr;
    transac_bar_t bar_last_store_done;
    transac_bar_t bar_q_tma, bar_q_utccp;
    transac_bar_t bar_q_scale_tma, bar_q_scale_utccp;
    transac_bar_t bar_ready[NUM_BUFS];
    transac_bar_t bar_kv_ready[NUM_BUFS], bar_kv_scale_ready[NUM_BUFS];
    transac_bar_t bar_valid_coord_scale_ready[NUM_INDEX_BUFS], bar_valid_coord_scale_free[NUM_INDEX_BUFS];
    transac_bar_t bar_qk_done[NUM_BUFS], bar_so_ready[NUM_BUFS], bar_sv_done[NUM_BUFS];
};

using TiledMMA_P = decltype(make_tiled_mma( // make the type name shorter
    SM100_MMA_MXF8F6F4_SS_NOELECT<e4m3, e4m3, float, e8m0, B_TOPK, B_H, UMMA::Major::K, UMMA::Major::K>{}
));

using TiledMMA_O = decltype(make_tiled_mma(
    SM100_MMA_MXF8F6F4_SS_NOELECT<e4m3, e4m3, float, e8m0, B_TOPK, B_H, UMMA::Major::MN, UMMA::Major::K>{}
));

template<typename TmaParam>
static __device__ void
flash_fwd_splitkv_mla_mxfp8_sparse_kernel_devfunc(const SparseAttnDecodeParams &params, const TmaParam &tma_params);

static void run(const SparseAttnDecodeParams &params);

};

}