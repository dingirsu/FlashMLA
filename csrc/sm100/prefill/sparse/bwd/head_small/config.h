#pragma once

#include <cute/tensor.hpp>
#include <kerutils/kerutils.cuh>

#include "defines.h"

namespace sm100::bwd::head_small {

using namespace cute;

template<
    typename Shape_Q_nope,    typename TMA_Q_nope,
    typename Shape_Q_rope,    typename TMA_Q_rope,
    typename Shape_O,         typename TMA_O,
    typename Shape_DO,        typename TMA_DO,
    typename Shape_DO_t,      typename TMA_DO_t
>
struct TmaParams {
    Shape_Q_nope shape_Q_nope; TMA_Q_nope tma_Q_nope;
    Shape_Q_rope shape_Q_rope; TMA_Q_rope tma_Q_rope;
    Shape_O      shape_O;      TMA_O      tma_O;
    Shape_DO     shape_DO;     TMA_DO     tma_DO;
    Shape_DO_t   shape_DO_t;   TMA_DO_t   tma_DO_t;
    CUtensorMap tensor_map_kv_nope;
};

template<int D_QK, int H_Q>
struct KernelTemplate {
    static_assert(D_QK == 128 || D_QK == 192);
    static_assert(H_Q == 8 || H_Q == 16 || H_Q == 24 || H_Q == 32);

    static constexpr int D_Q = D_QK;
    static constexpr int D_K = D_QK;
    static constexpr int D_V = 128;
    static constexpr int D_ROPE = D_QK - D_V;

    static constexpr int B_H = H_Q;
    static constexpr int B_H_TMEM = H_Q == 24 ? 32 : H_Q;
    static constexpr int B_TOPK = 64;
    static constexpr int NUM_BUFS = 2;
    static constexpr int NUM_THREADS = 128 + 128 + 128;

    // ---- TMEM column allocation (from plan §"TMEM Allocation", B_TOPK=64) ----
    struct tmem_cols {
        // dV [B_TOPK, D_V]      : 2 tiles of 64 cols each = 128 cols
        static constexpr int DV         = 0;
        static constexpr int DV_WIDTH   = 128;
        // dK_nope [B_TOPK, 128] : 2 tiles of 64 cols each = 128 cols
        static constexpr int DK         = 128;
        static constexpr int DK_WIDTH   = 128;
        // Reused for P_t then dP_t, shape [B_TOPK, B_H], col-width = B_H_TMEM
        static constexpr int SCORE      = 256;
        static constexpr int SCORE_WIDTH = B_H_TMEM;
        // dQ_nope_t [128, B_H]  : 2 tiles of B_H_TMEM cols each
        static constexpr int DQ         = 320;
        static constexpr int DQ_WIDTH   = 2 * B_H_TMEM;
        // dK_rope [B_TOPK, 64]  : only when D_QK == 192
        static constexpr int DK_ROPE        = 384;
        static constexpr int DK_ROPE_WIDTH  = D_QK == 192 ? 64 : 0;
        // dQ_rope_t [64, B_H]   : only when D_QK == 192
        static constexpr int DQ_ROPE        = 448;
        static constexpr int DQ_ROPE_WIDTH  = D_QK == 192 ? B_H_TMEM : 0;
        static constexpr int TOTAL_RESERVED = 480;
    };
    static_assert(tmem_cols::TOTAL_RESERVED <= 512, "TMEM budget overflow");

    // ---- SMEM layouts ----
    // K/V NoPE (persistent across both K and V views): [B_TOPK, 128]
    template<int NUM_TILES>
    using SmemLayoutKVNopETiles = decltype(coalesce(tile_to_shape(
        UMMA::Layout_K_SW128_Atom<bf16>{},
        Shape<Int<B_TOPK>, Int<64 * NUM_TILES>>{},
        Step<_1, _2>{}
    ), Shape<_1, _1>{}));

    using SmemLayoutKNopE = SmemLayoutKVNopETiles<D_V / 64>;

    // V is a re-view of the same buffer: [D_V, B_TOPK] in MN-major view.
    using SmemLayoutV = decltype(coalesce(
        composition(
            SmemLayoutKNopE{},
            Layout<Shape<Int<D_V>, Int<B_TOPK>>, Stride<Int<B_TOPK>, _1>>{}
        ),
        Shape<_1, _1>{}
    ));

    // Q NoPE / Q RoPE: forward's layout, used unchanged.
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

    // K RoPE: [B_TOPK, 64]
    using SmemLayoutKRoPE = decltype(coalesce(tile_to_shape(
        UMMA::Layout_K_SW128_Atom<bf16>{},
        Shape<Int<B_TOPK>, Int<64>>{},
        Step<_1, _2>{}
    ), Shape<_1, _1>{}));

    // dO in forward's natural [B_H, D_V] layout: used as Q operand of dP_t = V @ dO^T.
    // TODO(large-inner): when B_H < 16 the SW128_Atom swizzle may be invalid; review
    // and switch to a smaller swizzle or non-swizzled layout if compilation fails.
    using SmemLayoutDO = decltype(coalesce(tile_to_shape(
        UMMA::Layout_K_SW128_Atom<bf16>{},
        Shape<Int<B_H>, Int<D_V>>{},
        Step<_1, _2>{}
    ), Shape<_1, _1>{}));

    // dO transposed [D_V, B_H]: used as B operand of dV = Prob @ dO.
    // TODO(major): MN-major on (D_V, B_H) requires careful swizzle choice; verify the
    // chosen CuTe composition round-trips to a valid swizzled layout at instantiation.
    using SmemLayoutDOTransposed = decltype(coalesce(
        composition(
            SmemLayoutDO{},
            Layout<Shape<Int<D_V>, Int<B_H>>, Stride<Int<B_H>, _1>>{}
        ),
        Shape<_1, _1>{}
    ));

    // O kept in forward layout for sum_odo computation.
    template<int NUM_TILES>
    using SmemLayoutOTiles = decltype(coalesce(tile_to_shape(
        UMMA::Layout_K_SW128_Atom<bf16>{},
        Shape<Int<B_H>, Int<64 * NUM_TILES>>{},
        Step<_1, _2>{}
    ), Shape<_1, _1>{}));
    using SmemLayoutO = SmemLayoutOTiles<D_V / 64>;

    // prob_t / draw_t: [B_H, B_TOPK] in UMMA-friendly layout (same as forward's S).
    using SmemLayoutS = decltype(coalesce(tile_to_shape(
        UMMA::Layout_K_SW128_Atom<bf16>{},
        Shape<Int<B_H>, Int<B_TOPK>>{},
        Step<_1, _2>{}
    ), Shape<_1, _1>{}));

    // ---- Shared memory plan ----
    // Persistent: q_nope, q_rope, dO (natural), dO_t (transposed), o, scalars.
    // Double-buffered: k_nope[2]/v[2] (shared buffer), indices[2], valid[2], k_rope.
    // Sequential: p_t (reused as dp_t) — not aliased in v1 to keep it simple.
    struct SharedMemoryPlan {
        union {
            // Q persistent
            struct { array_aligned<bf16, cosize_v<SmemLayoutQNoPE>> q_nope; } q_full;
            // KV NoPE/V double-buffered, K RoPE single-buffered
            struct {
                array_aligned<bf16, cosize_v<SmemLayoutKRoPE>> k_rope;
                array_aligned<bf16, cosize_v<SmemLayoutKNopE>> k_nope[NUM_BUFS];
            } k;
            // dO in natural layout
            struct {
                array_aligned<bf16, cosize_v<SmemLayoutDO>> do_natural;
            } do_pack;
            // dO transposed
            struct {
                array_aligned<bf16, cosize_v<SmemLayoutDOTransposed>> do_t;
            } do_t_pack;
            // O (used only during prologue for sum_odo)
            struct { array_aligned<bf16, cosize_v<SmemLayoutO>> o; } o_pack;
        } u;

        // score/prob/derivative staging (sequential, do NOT alias with q/k SMEM)
        float p_t[B_TOPK * B_H];           // P_t in float (B_TOPK rows × B_H cols)
        float dp_t[B_TOPK * B_H];          // dP_t in float
        bf16  prob_t[B_H * B_TOPK];        // used as dV/dK/dQ MMA operand
        bf16  draw_t[B_H * B_TOPK];        // used as dK/dQ MMA operand (dRaw_t)

        // sum_odo reduction (per-CTA, 128 partials → B_H finals)
        float sum_odo_partial[128];
        float sum_odo[B_H];

        // per-head scalars
        float lse[B_H];
        float neg_lse[B_H];
        float attn_sink_log2[B_H];         // attn_sink[h] * CUDART_L2E_F, for sink_prob = exp2(.)
        float sink_prob[B_H];

        // indices / valid / k_validness (double-buffered)
        int   indices[NUM_BUFS][B_TOPK];
        char  is_k_valid[NUM_BUFS][B_TOPK / 8];
        int*  gIndices_smem_ptr[NUM_BUFS];  // pointer cached for epilogue

        // dQ direct-store staging (optional: could also write directly from TMEM
        // without this buffer). Per-thread temp is enough; this is a scratch row.
        // TODO(dq-store): decide between direct TMEM→global or via this buffer.
        // bf16  dq_smem[B_H * D_QK];

        // Barriers (see plan §"Barrier Plan")
        transac_bar_t bar_prologue_q_nope,  bar_prologue_q_rope;
        transac_bar_t bar_prologue_dO,      bar_prologue_dO_t;
        transac_bar_t bar_prologue_o;
        transac_bar_t bar_kv_ready[NUM_BUFS];
        transac_bar_t bar_valid_ready[NUM_BUFS];
        transac_bar_t bar_score_free;
        transac_bar_t bar_qk_done, bar_dp_done;
        transac_bar_t bar_ds_ready;            // prob_t/draw_t ready in SMEM
        transac_bar_t bar_dv_done[NUM_BUFS];
        transac_bar_t bar_dk_done[NUM_BUFS];
        transac_bar_t bar_dq_done[2];          // no_rope / rope (rope slot unused when D_QK=128)
        transac_bar_t bar_store_done[NUM_BUFS];
        transac_bar_t bar_kv_rope_ready;
        // inner score rows (P_t, dP_t) are kept inside WG0 in this v1, so
        // we add a 128-wide named barrier; see wg0_sync below.
        array_aligned<uint32_t, 1> tmem_start_addr;
    };

    // ---- MMA atoms (per plan §"MMA Major Selection") ----
    // P_t = K @ Q^T, dP_t = V @ dO^T — both [B_TOPK, B_H]
    using TiledMMA_QK = decltype(make_tiled_mma(
        SM100_MMA_F16BF16_SS_NOELECT<bf16, bf16, float, B_TOPK, B_H, UMMA::Major::K, UMMA::Major::K>{}
    ));
    // dV = Prob @ dO  — [B_TOPK, 64]  (per 64-wide D_V tile)
    using TiledMMA_DV = decltype(make_tiled_mma(
        SM100_MMA_F16BF16_SS_NOELECT<bf16, bf16, float, B_TOPK, 64, UMMA::Major::K, UMMA::Major::MN>{}
    ));
    // dK = draw_t @ Q  — [B_TOPK, 64]  (per 64-wide D tile)
    using TiledMMA_DK = decltype(make_tiled_mma(
        SM100_MMA_F16BF16_SS_NOELECT<bf16, bf16, float, B_TOPK, 64, UMMA::Major::K, UMMA::Major::MN>{}
    ));
    // dQ_t = K^T @ draw_t — [64, B_H] (per 64-wide D tile)
    using TiledMMA_DQ = decltype(make_tiled_mma(
        SM100_MMA_F16BF16_SS_NOELECT<bf16, bf16, float, 64, B_H, UMMA::Major::MN, UMMA::Major::MN>{}
    ));

    // Named barriers (for wg0_sync, used to keep Score TMEM ownership inside WG0 in v1)
    enum NamedBarriers : int {
        wg0_sync = 0,
        wg0_warp02_sync = 1,
        wg0_warp13_sync = 2,
    };
};

}
