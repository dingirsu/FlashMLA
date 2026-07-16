#pragma once

#include <type_traits>

#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "defines.h"

namespace sm100::bwd::head_small {

using namespace cute;

template<
    typename ShapeQ, typename TmaQ,
    typename ShapeO, typename TmaO,
    typename ShapeDO, typename TmaDO
>
struct TmaParams {
    ShapeQ shape_q;
    TmaQ tma_q;
    ShapeO shape_o;
    TmaO tma_o;
    ShapeDO shape_do;
    TmaDO tma_do;
    CUtensorMap tensor_map_kv;
};

template<int D_QK, int H_Q>
struct KernelTemplate {
    static_assert(D_QK == 128, "The first SM100 sparse backward supports D_QK=128 only");
    static_assert(H_Q == 8 || H_Q == 16 || H_Q == 32);

    static constexpr int D_Q = D_QK;
    static constexpr int D_K = D_QK;
    static constexpr int D_V = 128;
    static constexpr int B_H = H_Q;
    static constexpr int B_H_MMA = ((H_Q + 15) / 16) * 16;
    static constexpr int B_H_TMEM = H_Q;
    static constexpr int B_TOPK = 64;
    static constexpr int NUM_BUFS = 2;
    static constexpr int NUM_THREADS = 3 * 128;

    struct tmem_cols {
        static constexpr int DV = 0;
        static constexpr int DK = 128;
        static constexpr int SCORE = 256;
        static constexpr int DQ = 320;
        static constexpr int END = DQ + 2 * B_H_TMEM;
    };
    static_assert(tmem_cols::END <= 512, "TMEM column budget overflow");

    using SmemLayoutHeadDim = decltype(coalesce(tile_to_shape(
        UMMA::Layout_K_SW128_Atom<bf16>{},
        Shape<Int<B_H>, Int<D_Q>>{},
        Step<_1, _2>{}
    ), Shape<_1, _1>{}));

    using SmemLayoutDimHead = decltype(coalesce(tile_to_shape(
        UMMA::Layout_MN_SW128_Atom<bf16>{},
        Shape<Int<D_Q>, Int<B_H_MMA>>{},
        Step<_2, _1>{}
    ), Shape<_1, _1>{}));

    using SmemLayoutK = decltype(coalesce(tile_to_shape(
        UMMA::Layout_K_SW128_Atom<bf16>{},
        Shape<Int<B_TOPK>, Int<D_K>>{},
        Step<_1, _2>{}
    ), Shape<_1, _1>{}));

    using SmemLayoutKT = decltype(coalesce(
        composition(
            SmemLayoutK{},
            Layout<Shape<Int<D_K>, Int<B_TOPK>>, Stride<Int<B_TOPK>, _1>>{}
        ),
        Shape<_1, _1>{}
    ));

    using SmemLayoutProbAtom = std::conditional_t<
        B_H_MMA == 16,
        UMMA::Layout_K_SW32_Atom<bf16>,
        UMMA::Layout_K_SW64_Atom<bf16>
    >;

    using SmemLayoutProb = decltype(coalesce(tile_to_shape(
        SmemLayoutProbAtom{},
        Shape<Int<B_TOPK>, Int<B_H_MMA>>{},
        Step<_1, _2>{}
    ), Shape<_1, _1>{}));

    using SmemLayoutDSForDQ = decltype(coalesce(tile_to_shape(
        UMMA::Layout_K_SW128_Atom<bf16>{},
        Shape<Int<B_H>, Int<B_TOPK>>{},
        Step<_1, _2>{}
    ), Shape<_1, _1>{}));

    struct SharedMemoryPlan {
        array_aligned<bf16, cosize_v<SmemLayoutHeadDim>> q;
        array_aligned<bf16, cosize_v<SmemLayoutHeadDim>> d_o;
        array_aligned<bf16, cosize_v<SmemLayoutHeadDim>> o;
        array_aligned<bf16, cosize_v<SmemLayoutDimHead>> q_t;
        array_aligned<bf16, cosize_v<SmemLayoutDimHead>> d_o_t;
        array_aligned<bf16, cosize_v<SmemLayoutK>> kv[NUM_BUFS];
        array_aligned<bf16, cosize_v<SmemLayoutProb>> prob;
        array_aligned<bf16, cosize_v<SmemLayoutProb>> ds_for_dk;
        array_aligned<bf16, cosize_v<SmemLayoutDSForDQ>> ds_for_dq;

        float p[B_TOPK * B_H];
        float dp[B_TOPK * B_H];
        float sum_odo[B_H];
        float lse[B_H];
        float token_mass[B_H];
        int indices[NUM_BUFS][B_TOPK];
        uint8_t valid[NUM_BUFS][B_TOPK];
        int valid_length;
        int have_valid_token;

        transac_bar_t bar_q_ready;
        transac_bar_t bar_o_ready;
        transac_bar_t bar_do_ready;
        transac_bar_t bar_operands_ready;
        transac_bar_t bar_kv_ready[NUM_BUFS];
        transac_bar_t bar_kv_free[NUM_BUFS];
        transac_bar_t bar_score_ready;
        transac_bar_t bar_score_free;
        transac_bar_t bar_ds_ready;
        transac_bar_t bar_dkv_ready[NUM_BUFS];
        transac_bar_t bar_dkv_free;
        array_aligned<uint32_t, 1> tmem_start_addr;
    };

    using TiledMMAQK = decltype(make_tiled_mma(
        SM100_MMA_F16BF16_SS_NOELECT<
            bf16, bf16, float, B_TOPK, B_H, UMMA::Major::K, UMMA::Major::K
        >{}
    ));

    using TiledMMADKV = decltype(make_tiled_mma(
        SM100_MMA_F16BF16_SS_NOELECT<
            bf16, bf16, float, B_TOPK, 64, UMMA::Major::K, UMMA::Major::MN
        >{}
    ));

    using TiledMMADQ = decltype(make_tiled_mma(
        SM100_MMA_F16BF16_SS_NOELECT<
            bf16, bf16, float, 64, B_H, UMMA::Major::MN, UMMA::Major::K
        >{}
    ));

    enum NamedBarriers : int {
        WG0_SYNC = 0,
        WG1_SYNC = 1,
    };
};

} // namespace sm100::bwd::head_small
