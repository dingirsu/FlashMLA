#pragma once

#include "kernel.h"

#include <cuda_fp8.h>
#include <cutlass/barrier.h>
#include <cute/tensor.hpp>

#include <cutlass/detail/sm100_blockscaled_layout.hpp>

#include <kerutils/kerutils.cuh>

#include "defines.h"
#include "params.h"

namespace sm100::decode::mxfp8_head64 {

using cutlass::arch::fence_view_async_shared;
using cutlass::arch::NamedBarrier;
using e8m0 = cutlass::float_ue8m0_t;
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

// We currently only support the 512 head-dim (MODEL1) case. V32 (576) needs
// a separate config because of the differing K-major length.
static constexpr bool V32_SUPPORTED = false;
static_assert(MODEL_TYPE == ModelType::MODEL1 || V32_SUPPORTED,
    "mxfp8 decode kernel only supports ModelType::MODEL1 (d_qk=512) at this time");

static constexpr int D_Q = 512;
static constexpr int D_K = D_Q;
static constexpr int D_V = 512;
static constexpr int MXFP8_SCALE_VEC_SIZE = 32;
static constexpr int Q_QUANT_GROUP_SIZE = 32;
static constexpr int K_QUANT_GROUP_SIZE = 64;
static constexpr int Q_SCALE_BYTES = D_Q / Q_QUANT_GROUP_SIZE;  // 16
static constexpr int K_SCALE_BYTES = D_K / K_QUANT_GROUP_SIZE;  // 8
static constexpr int K_SCALE_DUP = K_QUANT_GROUP_SIZE / MXFP8_SCALE_VEC_SIZE;  // 2
static constexpr int TMA_K_STRIDE = D_K;  // 512 — pure e4m3, no per-token scales interleaved
static constexpr int B_H = 64;
static constexpr int B_TOPK = 128;
static constexpr int K_SCALE_SMEM_ELEMS = B_TOPK * K_SCALE_BYTES * K_SCALE_DUP;  // 128 * 16 = 2048
static constexpr int Q_SCALE_SMEM_ELEMS = B_H * Q_SCALE_BYTES;                    // 64 * 16 = 1024
static constexpr int S_SCALE_SMEM_ELEMS = B_H * (B_TOPK / MXFP8_SCALE_VEC_SIZE);  // 64 * 4 = 256
static constexpr int NUM_BUFS = 2;
static constexpr int NUM_INDEX_BUFS = 4;    // Number of buffers for indices (tma_coords) & is_token_valid & scales
static constexpr int NUM_THREADS = 128*3;  // 128 exp + 32 utcmma + 32 raw KV producer + 32 kv-scale/idx producer + 128 reserved
static constexpr float MAX_INIT_VAL = -1e30f;  // To avoid (-inf) - (-inf) = NaN
static constexpr float FP8_MAX = 448.0f;  // Max e4m3 value used in mxfp8 quantize

template<
    typename Shape_Q, typename TMA_Q,
    typename Shape_Q_Scale, typename TMA_Q_Scale,
    typename Shape_O, typename TMA_O
>
struct TmaParams {
    Shape_Q shape_Q; TMA_Q tma_Q;
    Shape_Q_Scale shape_Q_scale; TMA_Q_Scale tma_Q_scale;
    Shape_O shape_O; TMA_O tma_O;
    CUtensorMap tensor_map_kv;
    CUtensorMap tensor_map_extra_kv;
};

// Tensor memory columns
struct tmem_cols {
    //   0 ~ 256: O (D_V / 2 cols of 32-bit FP32 per row, fp32 acc)
    // 256 ~ 320: unused
    // 320 ~ 338: Q scale (e8m0)        (Q_Scale: 16 cols)
    // 338 ~ 356: K/V scale (e8m0)      (K_Scale: 16 cols, reused for V after QK^T)
    // 356 ~ 376: S scale (e8m0)        (S_Scale: 16 cols + 4 cols padding)
    // 400 ~ 464: P (fp32, (B_TOPK, B_H) = 128x64)  (P: 64 cols)
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
    Shape<Int<B_TOPK>, Int<64*NUM_TILES>>{},
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

// Tiled MMAs
using TiledMMA_P = decltype(make_tiled_mma(
    SM100_MMA_MXF8F6F4_SS_NOELECT<e4m3, e4m3, float, e8m0, B_TOPK, B_H, UMMA::Major::K, UMMA::Major::K>{}
));

using TiledMMA_O = decltype(make_tiled_mma(
    SM100_MMA_MXF8F6F4_SS_NOELECT<e4m3, e4m3, float, e8m0, B_TOPK, B_H, UMMA::Major::MN, UMMA::Major::K>{}
));

// Q scale smem layout (Q is the B operand of the KQ GEMM, so its scales are SFB)
using SmemLayoutQScale = decltype(cutlass::detail::Sm1xxBlockScaledConfig<MXFP8_SCALE_VEC_SIZE>::tile_atom_to_shape_SFB(
    Shape<Int<B_TOPK>, Int<B_H>, Int<D_Q>>{}
));

// K scale smem layout (K is the A operand of the KQ GEMM, so its scales are SFA)
using SmemLayoutKScale = decltype(cutlass::detail::Sm1xxBlockScaledConfig<MXFP8_SCALE_VEC_SIZE>::tile_atom_to_shape_SFA(
    Shape<Int<B_TOPK>, Int<B_H>, Int<D_K>>{}
));

// Atom layouts derived from the MMA tiles — needed for the UTCCP src layout
// of the S scales and V scales (tV_scale uses the O atom's SFA even though the
// data is physically written by the KQ MMA's SFA UTCCP).
using SmemLayoutPScaleAAtom = decltype(cutlass::detail::Sm1xxBlockScaledConfig<MXFP8_SCALE_VEC_SIZE>::deduce_smem_layoutSFA(
    TiledMMA_P{}, Shape<Int<B_TOPK>, Int<B_H>, Int<D_K>>{}
));
using SmemLayoutPScaleBAtom = decltype(cutlass::detail::Sm1xxBlockScaledConfig<MXFP8_SCALE_VEC_SIZE>::deduce_smem_layoutSFB(
    TiledMMA_P{}, Shape<Int<B_TOPK>, Int<B_H>, Int<D_K>>{}
));
using SmemLayoutOScaleAAtom = decltype(cutlass::detail::Sm1xxBlockScaledConfig<MXFP8_SCALE_VEC_SIZE>::deduce_smem_layoutSFA(
    TiledMMA_O{}, Shape<Int<B_TOPK>, Int<B_H>, Int<B_TOPK>>{}
));
using SmemLayoutOScaleBAtom = decltype(cutlass::detail::Sm1xxBlockScaledConfig<MXFP8_SCALE_VEC_SIZE>::deduce_smem_layoutSFB(
    TiledMMA_O{}, Shape<Int<B_TOPK>, Int<B_H>, Int<B_TOPK>>{}
));

// S scale smem layout (S is the B operand of the VS GEMM, so its scales are SFB)
using SmemLayoutSscale = SmemLayoutOScaleBAtom;

// Q scale smem layout for TMA (a simple (B_H, Q_SCALE_BYTES) layout)
using SmemLayoutQScaleTMA = Layout<
    Shape<Int<B_H>, Int<Q_SCALE_BYTES>>,
    Stride<Int<Q_SCALE_BYTES>, _1>
>;

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
            array_aligned<e4m3, cosize_v<SmemLayoutKTiles_SW128<D_K/64>>> kv[NUM_BUFS];  // Raw (quantized) K data
        } kv;
    } u;
    union {
        float4 p_exchange_buf[4][16 * B_TOPK / 4]; // why this layout?
        array_aligned<e4m3, cosize_v<SmemLayoutS>> s;
    } s_p;
    array_aligned<e8m0, cosize_v<SmemLayoutSscale>> s_scale;
    CUTE_ALIGNAS(16) float rowwise_max_buf[128];
    char is_token_valid[NUM_INDEX_BUFS][B_TOPK/8];
    int tma_coord[NUM_INDEX_BUFS][B_TOPK];
    e8m0 scales[NUM_INDEX_BUFS][B_TOPK][K_SCALE_BYTES];
    array_aligned<uint32_t, 1> tmem_start_addr;
    transac_bar_t bar_last_store_done;
    transac_bar_t bar_q_tma, bar_q_utccp;
    transac_bar_t bar_q_scale_tma, bar_q_scale_utccp;
    transac_bar_t bar_kv_ready[NUM_BUFS];
    transac_bar_t bar_kv_scale_ready[NUM_BUFS];
    transac_bar_t bar_valid_coord_scale_ready[NUM_INDEX_BUFS], bar_valid_coord_scale_free[NUM_INDEX_BUFS];
    transac_bar_t bar_qk_done[NUM_BUFS], bar_so_ready[NUM_BUFS], bar_sv_done[NUM_BUFS];
};

template<typename TmaParam>
static __device__ void
flash_fwd_splitkv_mla_mxfp8_sparse_kernel_devfunc(const SparseAttnMxfp8DecodeParams &params, const TmaParam &tma_params);

static void run(const SparseAttnMxfp8DecodeParams &params);

};

}
