#pragma once
#include "phase1.h"

#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/arch.h>
#include <cutlass/cuda_host_adapter.hpp>

#include <kerutils/kerutils.cuh>

#include "params.h"
#include "utils.h"
#include "sm100/helpers.h"
#include "sm100/prefill/sparse/common_subroutine.h"
#include "config.h"

namespace sm100::bwd::head_small {

using namespace cute;

template<typename Kernel>
CUTE_DEVICE
void rescale_DQ_t(float scale[Kernel::B_H]) {
    // Mirrors forward's rescale_O_t. Operates on one 64-wide D tile of dQ_t.
    // DQ region has 2 * B_H_TMEM cols; this rescaler covers 1 tile (B_H_TMEM cols).
    float dq[Kernel::B_H_TMEM];
    CUTE_UNROLL
    for (int tile = 0; tile < 2; ++tile) {
        ku::tmem_ld_32dp32bNx<Kernel::B_H_TMEM>(Kernel::tmem_cols::DQ + tile*Kernel::B_H_TMEM, dq);
        cutlass::arch::fence_view_async_tmem_load();
        CUTE_UNROLL
        for (int i = 0; i < Kernel::B_H; ++i) {
            dq[i] *= scale[i];
        }
        ku::tmem_st_32dp32bNx<Kernel::B_H_TMEM>(Kernel::tmem_cols::DQ + tile*Kernel::B_H_TMEM, dq);
        cutlass::arch::fence_view_async_tmem_store();
    }
}

/*
Backward pipeline (per plan §"Pipeline"):

  Prologue (one-shot):
    - Load Q, O, dO, dO_t, lse, attn_sink.
    - WG0 computes sum_odo[h] and sink_prob[h]; FP32 atomicAdd into d_attn_sink.

  Per block k = 0 .. num_k_blocks-1:
    WG1: load indices[cur], valid[cur], issue TMA gather for K_nope/V (same SMEM
         buffer; V is a re-view), and cp.async K_rope when D_QK=192.
    WG2 (warp 8): wait for kv/valid. Score TC owns `SCORE` TMEM.
      1. P_t = K @ Q^T          (acc=0)
         If D_QK=192: also K_rope @ Q_rope^T (acc=1) into the same region.
    WG0: read P_t rows from SCORE → p_t[s][h] (s ∈ [0, B_TOPK)).
         prob[kk,h] = exp2f(sm_scale_div_log2 * p - lse[h])   (or 0 if !valid)
    WG2: dP_t = V @ dO^T        (overwrites SCORE)
    WG0: read dP_t → dp_t.
         draw[kk,h] = prob * (dp - sum_odo[h]) * sm_scale
    WG0: store prob_t (bf16) and draw_t (bf16) for MMAs.
    WG2: dV += Prob_t @ dO        →  TMEM dV (accumulating, cleared at k=0)
         dK_nope = draw_t @ Q_nope → TMEM dK
         dQ_t_nope += K_nope^T @ draw_t → TMEM dQ  (cleared at k=0)
         (When D_QK=192:) dK_rope, dQ_t_rope analogously.
    WG0: epilogue — for each kk in [0, B_TOPK):
            if valid: FP32 atomicAdd dV[indices[kk]] += TMEM dV[kk,:]
                       FP32 atomicAdd dK[indices[kk]] += TMEM dK[kk,:]
    After loop:
      WG0: store TMEM dQ_t → global dQ (transposed write, one thread per dq_pos).
*/

template<int D_QK, int H_Q, bool HAVE_ROPE, typename TmaParamsT>
__global__ void __launch_bounds__(KernelTemplate<D_QK, H_Q>::NUM_THREADS, 1, 1)
sparse_attn_bwd_kernel(__grid_constant__ const SparseAttnBwdParams params,
                       __grid_constant__ const TmaParamsT tma_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
    using Kernel = KernelTemplate<D_QK, H_Q>;
    // Grid: one CTA per query row.
    const int s_q_idx = blockIdx.x;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = threadIdx.x % 32;
    const int warpgroup_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    const int idx_in_warpgroup = threadIdx.x % 128;

    const int topk_length = params.topk_length != nullptr ? __ldg(params.topk_length + s_q_idx) : params.topk;
    const int num_k_blocks = max(cute::ceil_div(topk_length, (int)Kernel::B_TOPK), 1);

    int* gIndices = params.indices + s_q_idx * params.stride_indices_s_q;
    float sm_scale_div_log2 = params.sm_scale_div_log2;
    float sm_scale          = params.sm_scale;

    // ---- Shared memory plan ----
    extern __shared__ char wksp_buf[];
    typename Kernel::SharedMemoryPlan &plan = *reinterpret_cast<typename Kernel::SharedMemoryPlan*>(wksp_buf);

    // ---- MMA fragments (only thread 0 of WG2 needs them; reserve the slot) ----
    typename Kernel::TiledMMA_QK tiled_mma_QK = typename Kernel::TiledMMA_QK{};
    typename Kernel::TiledMMA_DV tiled_mma_DV = typename Kernel::TiledMMA_DV{};
    typename Kernel::TiledMMA_DK tiled_mma_DK = typename Kernel::TiledMMA_DK{};
    typename Kernel::TiledMMA_DQ tiled_mma_DQ = typename Kernel::TiledMMA_DQ{};
    Tensor tP  = partition_fragment_C(tiled_mma_QK, Shape<Int<Kernel::B_TOPK>, Int<Kernel::B_H>>{});
    Tensor tDP = partition_fragment_C(tiled_mma_QK, Shape<Int<Kernel::B_TOPK>, Int<Kernel::B_H>>{});
    Tensor tDV = partition_fragment_C(tiled_mma_DV, Shape<Int<Kernel::B_TOPK>, Int<64>>{});
    Tensor tDK = partition_fragment_C(tiled_mma_DK, Shape<Int<Kernel::B_TOPK>, Int<64>>{});
    Tensor tDQ = partition_fragment_C(tiled_mma_DQ, Shape<Int<64>, Int<Kernel::B_H>>{});
    tP.data().get()  = Kernel::tmem_cols::SCORE;
    tDP.data().get() = Kernel::tmem_cols::SCORE;
    tDV.data().get() = Kernel::tmem_cols::DV;          // set per-tile inside the loop
    tDK.data().get() = Kernel::tmem_cols::DK;
    tDQ.data().get() = Kernel::tmem_cols::DQ;          // set per-tile inside the loop

    // ====================================================================
    // Prologue: barriers + TMEM alloc + TMA loads
    // ====================================================================
    if (warp_idx == 0) {
        if (elect_one_sync()) {
            // Prefetch TMA descriptors (Q, dO both layouts, O, KV-nope)
            cute::prefetch_tma_descriptor(tma_params.tma_Q_nope.get_tma_descriptor());
            if constexpr (HAVE_ROPE) {
                cute::prefetch_tma_descriptor(tma_params.tma_Q_rope.get_tma_descriptor());
            }
            cute::prefetch_tma_descriptor(tma_params.tma_DO.get_tma_descriptor());
            cute::prefetch_tma_descriptor(tma_params.tma_DO_t.get_tma_descriptor());
            cute::prefetch_tma_descriptor(&(tma_params.tensor_map_kv_nope));

            // Init barriers (1 = arrival thread count; sizes from plan)
            plan.bar_prologue_q_nope.init(1);
            plan.bar_prologue_q_rope.init(1);
            plan.bar_prologue_dO.init(1);
            plan.bar_prologue_dO_t.init(1);
            plan.bar_prologue_o.init(1);
            fence_barrier_init();

            CUTE_UNROLL
            for (int i = 0; i < Kernel::NUM_BUFS; ++i) {
                plan.bar_kv_ready[i].init(1);
                plan.bar_valid_ready[i].init(Kernel::B_TOPK / 8);
                plan.bar_dv_done[i].init(1);
                plan.bar_dk_done[i].init(1);
                plan.bar_store_done[i].init(1);
            }
            plan.bar_score_free.init(128);
            plan.bar_qk_done.init(1);
            plan.bar_dp_done.init(1);
            plan.bar_ds_ready.init(1);
            plan.bar_kv_rope_ready.init(64);
            plan.bar_dq_done[0].init(1);
            if constexpr (HAVE_ROPE) plan.bar_dq_done[1].init(1);
            fence_barrier_init();
        }
        // Allocate 512 TMEM columns and zero-out dQ region (other regions written
        // with clear_accum on first use).
        cute::TMEM::Allocator1Sm().allocate(512, plan.tmem_start_addr.data());
        TRAP_ONLY_DEVICE_ASSERT(plan.tmem_start_addr.data()[0] == 0);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }

    __syncthreads();

    // ---- Issue TMA loads (Q, dO natural, dO transposed, O) ----
    if (warp_idx == 0 && elect_one_sync()) {
        Tensor gQ_nope = tma_params.tma_Q_nope.get_tma_tensor(tma_params.shape_Q_nope)(_, _, s_q_idx);
        Tensor sQ_nope = make_tensor(make_smem_ptr(plan.u.q_full.q_nope.data()), typename Kernel::SmemLayoutQNoPE{});
        ku::launch_tma_copy(tma_params.tma_Q_nope, gQ_nope, sQ_nope, plan.bar_prologue_q_nope, TMA::CacheHintSm90::EVICT_FIRST);

        if constexpr (HAVE_ROPE) {
            Tensor gQ_rope = tma_params.tma_Q_rope.get_tma_tensor(tma_params.shape_Q_rope)(_, _, s_q_idx);
            Tensor sQ_rope = make_tensor(make_smem_ptr(plan.u.k.k_rope.data()), typename Kernel::SmemLayoutQRoPE{});
            // NOTE: in backward we keep q_rope colocated with k_rope in the same union
            // slot to save SMEM; k_rope is only read after the prologue gate.
            ku::launch_tma_copy(tma_params.tma_Q_rope, gQ_rope, sQ_rope, plan.bar_prologue_q_rope, TMA::CacheHintSm90::EVICT_FIRST);
        }

        Tensor gDO     = tma_params.tma_DO.get_tma_tensor(tma_params.shape_DO)(_, _, s_q_idx);
        Tensor sDO     = make_tensor(make_smem_ptr(plan.u.do_pack.do_natural.data()), typename Kernel::SmemLayoutDO{});
        ku::launch_tma_copy(tma_params.tma_DO, gDO, sDO, plan.bar_prologue_dO, TMA::CacheHintSm90::EVICT_FIRST);

        Tensor gDO_t   = tma_params.tma_DO_t.get_tma_tensor(tma_params.shape_DO_t)(_, _, s_q_idx);
        Tensor sDO_t   = make_tensor(make_smem_ptr(plan.u.do_t_pack.do_t.data()), typename Kernel::SmemLayoutDOTransposed{});
        // TODO(tma-store): tma_DO_t is a *store* descriptor (we materialize the
        // transpose in SMEM via a separate path or via a custom 2D TMA load with
        // swapped strides). For v1 skeleton, we use a *load* with shape (D_V, B_H, s_q)
        // and stride (B_H, 1, h_q*B_H). Verify descriptor at construction time.
        ku::launch_tma_copy(tma_params.tma_DO_t, gDO_t, sDO_t, plan.bar_prologue_dO_t, TMA::CacheHintSm90::EVICT_FIRST);

        Tensor gO      = tma_params.tma_O.get_tma_tensor(tma_params.shape_O)(_, _, s_q_idx);
        Tensor sO      = make_tensor(make_smem_ptr(plan.u.o_pack.o.data()), typename Kernel::SmemLayoutO{});
        ku::launch_tma_copy(tma_params.tma_O, gO, sO, plan.bar_prologue_o, TMA::CacheHintSm90::EVICT_FIRST);
    }

    __syncthreads();

    if (warp_idx == 0 && elect_one_sync()) {
        plan.bar_prologue_q_nope.arrive_and_expect_tx(Kernel::B_H*Kernel::D_V*sizeof(bf16));
        plan.bar_prologue_q_nope.wait(0);
        if constexpr (HAVE_ROPE) {
            plan.bar_prologue_q_rope.arrive_and_expect_tx(Kernel::B_H*Kernel::D_ROPE*sizeof(bf16));
            plan.bar_prologue_q_rope.wait(0);
        }
        plan.bar_prologue_dO.arrive_and_expect_tx(Kernel::B_H*Kernel::D_V*sizeof(bf16));
        plan.bar_prologue_dO.wait(0);
        plan.bar_prologue_dO_t.arrive_and_expect_tx(Kernel::D_V*Kernel::B_H*sizeof(bf16));
        plan.bar_prologue_dO_t.wait(0);
        plan.bar_prologue_o.arrive_and_expect_tx(Kernel::B_H*Kernel::D_V*sizeof(bf16));
        plan.bar_prologue_o.wait(0);
    }

    __syncthreads();

    // ====================================================================
    // WG0: per-head scalars (lse, attn_sink, sum_odo, sink_prob)
    // ====================================================================
    if (warpgroup_idx == 0) {
        if (idx_in_warpgroup < Kernel::B_H) {
            int h = idx_in_warpgroup;
            int global_index = s_q_idx * params.h_q + h;
            plan.lse[h]  = __ldg(params.lse + global_index * params.stride_lse_s_q);
            plan.neg_lse[h] = -plan.lse[h];
            float sink = params.attn_sink == nullptr ? -CUDART_INF_F : __ldg(params.attn_sink + h);
            plan.attn_sink_log2[h] = sink * CUDART_L2E_F;  // natural → base-2
        }
        NamedBarrier::arrive_and_wait(128, Kernel::NamedBarriers::wg0_sync);

        // ---- sum_odo[h] = sum_dv O[h,dv] * dO[h,dv] ----
        // All 128 threads cooperate. Layout: thread (head, part) where
        //   head = idx_in_warpgroup % B_H
        //   part  = idx_in_warpgroup / B_H
        //   parts_per_head = 128 / B_H
        constexpr int PARTS_PER_HEAD = 128 / Kernel::B_H;
        {
            int head = idx_in_warpgroup % Kernel::B_H;
            int part = idx_in_warpgroup / Kernel::B_H;
            // Each thread sums D_V / PARTS_PER_HEAD contiguous elements of head's O[dv] * dO[dv].
            // TODO(sum_odo-layout): O is in SmemLayoutO ([B_H, D_V]); dO is in SmemLayoutDO
            // ([B_H, D_V]). Use the natural (head, dv) addressing for both.
            int dv_per_part = Kernel::D_V / PARTS_PER_HEAD;
            int dv_base = part * dv_per_part;
            float partial = 0.0f;
            // O and dO share the same per-head, per-dv natural layout. We use the
            // linear address head*D_V + dv.
            const bf16* o_ptr  = plan.u.o_pack.o.data() + head * Kernel::D_V;
            const bf16* do_ptr = plan.u.do_pack.do_natural.data() + head * Kernel::D_V;
            CUTE_UNROLL
            for (int i = 0; i < dv_per_part; ++i) {
                float o_val  = float(o_ptr[dv_base + i]);
                float do_val = float(do_ptr[dv_base + i]);
                partial = fmaf(o_val, do_val, partial);
            }
            plan.sum_odo_partial[idx_in_warpgroup] = partial;
        }
        NamedBarrier::arrive_and_wait(128, Kernel::NamedBarriers::wg0_sync);

        // Reduce parts_per_head partials → sum_odo[h], one final per head.
        if (idx_in_warpgroup < Kernel::B_H) {
            int h = idx_in_warpgroup;
            float acc = 0.0f;
            CUTE_UNROLL
            for (int p = 0; p < PARTS_PER_HEAD; ++p) {
                acc += plan.sum_odo_partial[p * Kernel::B_H + h];
            }
            plan.sum_odo[h] = acc;
        }
        NamedBarrier::arrive_and_wait(128, Kernel::NamedBarriers::wg0_sync);

        // sink_prob[h] = exp2f(attn_sink_log2[h] - lse[h])  ;  d_attn_sink contribution.
        if (idx_in_warpgroup < Kernel::B_H) {
            int h = idx_in_warpgroup;
            plan.sink_prob[h] = exp2f(plan.attn_sink_log2[h] + plan.neg_lse[h]);
            // d_attn_sink[h] += -sink_prob[h] * sum_odo[h]   (FP32 atomic)
            float d_sink = -plan.sink_prob[h] * plan.sum_odo[h];
            if (params.d_attn_sink != nullptr) {
                atomicAdd(params.d_attn_sink + h, d_sink);
            }
        }
        NamedBarrier::arrive_and_wait(128, Kernel::NamedBarriers::wg0_sync);

        // ====================================================================
        // Main loop: WG0 reads SCORE TMEM, computes prob/draw, signals WG2,
        // then epilogues dV/dK from TMEM.
        // ====================================================================
        CUTE_NO_UNROLL
        for (int k = 0; k < num_k_blocks; ++k) {
            int cur_buf = k % Kernel::NUM_BUFS;

            // Wait for K/V SMEM + valid mask.
            plan.bar_valid_ready[cur_buf].wait((k / Kernel::NUM_BUFS) & 1);
            plan.bar_qk_done.wait(k & 1);
            ku::tcgen05_after_thread_sync();

            // --- Read P_t rows from SCORE TMEM ---
            // Layout: TMEM SCORE holds P_t[B_TOPK, B_H], B_H_TMEM cols.
            // Each thread owns one row kk ∈ [0, B_TOPK), so we map
            //   warp 0 → rows 0..31, warp 1 → rows 32..63.
            if (warp_idx < 2) {
                int k_row = (warp_idx & 1) * 32 + lane_idx;
                float p_head[Kernel::B_H_TMEM];
                ku::tmem_ld_32dp32bNx<Kernel::B_H_TMEM>(Kernel::tmem_cols::SCORE, p_head);
                cutlass::arch::fence_view_async_tmem_load();
                CUTE_UNROLL
                for (int h = 0; h < Kernel::B_H; ++h) {
                    plan.p_t[k_row * Kernel::B_H + h] = p_head[h];
                }
            }
            ku::tcgen05_before_thread_sync();
            plan.bar_score_free.arrive();
            NamedBarrier::arrive_and_wait(128, Kernel::NamedBarriers::wg0_sync);

            // --- Compute prob_t + draw_t (per row kk) ---
            // Each thread = one row; iterates over h.
            if (warp_idx < 2) {
                int k_row = (warp_idx & 1) * 32 + lane_idx;
                uint64_t valid_mask = *(uint64_t*)plan.is_k_valid[cur_buf];
                bool row_valid = (valid_mask >> k_row) & 1;
                float scale_times_p[Kernel::B_H];
                float new_p[Kernel::B_H];
                CUTE_UNROLL
                for (int h = 0; h < Kernel::B_H; ++h) {
                    float p_val = plan.p_t[k_row * Kernel::B_H + h] * sm_scale_div_log2;
                    float prob  = row_valid ? exp2f(p_val + plan.neg_lse[h]) : 0.0f;
                    scale_times_p[h] = p_val;
                    new_p[h] = prob;
                    plan.prob_t[h * Kernel::B_TOPK + k_row] = bf16(prob);
                }
                // We need dP_t first to compute draw_t. Wait for bar_dp_done.
                plan.bar_dp_done.wait(k & 1);
                ku::tcgen05_after_thread_sync();

                // Read dP_t row from TMEM.
                float dp_head[Kernel::B_H_TMEM];
                ku::tmem_ld_32dp32bNx<Kernel::B_H_TMEM>(Kernel::tmem_cols::SCORE, dp_head);
                cutlass::arch::fence_view_async_tmem_load();
                CUTE_UNROLL
                for (int h = 0; h < Kernel::B_H; ++h) {
                    plan.dp_t[k_row * Kernel::B_H + h] = dp_head[h];
                }
                ku::tcgen05_before_thread_sync();
                plan.bar_score_free.arrive();
                NamedBarrier::arrive_and_wait(128, Kernel::NamedBarriers::wg0_sync);

                CUTE_UNROLL
                for (int h = 0; h < Kernel::B_H; ++h) {
                    float dp  = plan.dp_t[k_row * Kernel::B_H + h];
                    float dr  = new_p[h] * (dp - plan.sum_odo[h]) * sm_scale;
                    plan.draw_t[h * Kernel::B_TOPK + k_row] = bf16(dr);
                }
            }
            NamedBarrier::arrive_and_wait(128, Kernel::NamedBarriers::wg0_sync);
            plan.bar_ds_ready.arrive();

            // --- Epilogue: read dV/dK from TMEM, atomicAdd into d_*_acc ---
            // Wait for WG2 to finish writing dV/dK into TMEM.
            plan.bar_dv_done[cur_buf].wait((k / Kernel::NUM_BUFS) & 1);
            plan.bar_dk_done[cur_buf].wait((k / Kernel::NUM_BUFS) & 1);
            ku::tcgen05_after_thread_sync();

            // Map threads to (kk, dv) tile: 2 warps × 32 lanes × 2 dv-tiles of 64.
            // For B_TOPK=64 and D_V=128:
            //   warps 0..1 own rows 0..63
            //   each row needs 128 dv positions; pass tile0 (64) and tile1 (64).
            if (warp_idx < 2) {
                int k_row = (warp_idx & 1) * 32 + lane_idx;
                int g_idx = plan.indices[cur_buf][k_row];
                bool valid = (g_idx >= 0 && g_idx < params.s_kv);
                CUTE_UNROLL
                for (int tile = 0; tile < 2; ++tile) {
                    int dv_base = tile * 64;
                    float dV_row[Kernel::B_H_TMEM];  // dV TMEM is [B_TOPK, D_V]; we read per-row
                    // TODO(dv-read-shape): TiledMMA_DV accumulator shape is
                    // [B_TOPK, 64], so reading one row means lane_idx of 32 threads
                    // covers 64 cols ⇒ 2 elements per thread. Adjust if the
                    // `B_H_TMEM` here is not 32 (e.g., B_H_TMEM == 32 always for B_TOPK=64).
                    // For B_H_TMEM=32 the tmem_ld_32dp32bNx<32> loads 32 cols;
                    // but dV row is 64 cols ⇒ load tile as 32+32 or 2x32.
                    // We use a 2x load:
                    float dV_lo[32], dV_hi[32];
                    ku::tmem_ld_32dp32bNx<32>(Kernel::tmem_cols::DV + tile * 64, dV_lo);
                    cutlass::arch::fence_view_async_tmem_load();
                    // The above reads col 0..31 for the row owned by the thread.
                    // For the second half of the tile we need to retarget. As a
                    // placeholder we just process the first 32 and skip the rest.
                    // TODO(dv-read-2nd-half): properly read cols 32..63 too.
                    if (valid) {
                        int64_t dv_offset = (int64_t)g_idx * params.stride_d_v_acc_s_kv;
                        CUTE_UNROLL
                        for (int j = 0; j < 32; ++j) {
                            float v = dV_lo[j];
                            atomicAdd(params.d_v_acc + dv_offset + (dv_base + j) * 0 + j, v);
                        }
                    }
                }
                // dK epilogue: same shape, two 64-wide tiles (d_k_acc uses stride_d_k_acc_s_kv * d_qk).
                // TODO(dk-epilogue): read dK TMEM at tmem_cols::DK + tile*64 and
                // atomicAdd into params.d_k_acc[indices[k_row], h, dv].
            }
            plan.bar_store_done[cur_buf].arrive();
        } // end for k
    }
    // ====================================================================
    // WG2 (warp 8) issues MMAs into TMEM.
    // ====================================================================
    else if (warpgroup_idx == 2) {
        if (warp_idx == 8 && elect_one_sync()) {
            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks; ++k) {
                int cur_buf = k % Kernel::NUM_BUFS;
                plan.bar_kv_ready[cur_buf].wait((k / Kernel::NUM_BUFS) & 1);
                plan.bar_score_free.wait(k & 1);
                ku::tcgen05_after_thread_sync();

                // 1) P_t = K @ Q^T  into SCORE TMEM
                {
                    Tensor sK = make_tensor(make_smem_ptr(plan.u.k.k_nope[cur_buf].data()),
                                            typename Kernel::SmemLayoutKNopE{});
                    Tensor sQ = make_tensor(make_smem_ptr(plan.u.q_full.q_nope.data()),
                                            typename Kernel::SmemLayoutQNoPE{});
                    ku::utcmma_ss(tiled_mma_QK, sK, sQ, tP, /*clear_accum=*/true);
                    if constexpr (HAVE_ROPE) {
                        Tensor sKr = make_tensor(make_smem_ptr(plan.u.k.k_rope.data()),
                                                 typename Kernel::SmemLayoutKRoPE{});
                        Tensor sQr = make_tensor(make_smem_ptr(plan.u.k.k_rope.data() + cosize_v<typename Kernel::SmemLayoutQRoPE>{}),
                                                 typename Kernel::SmemLayoutQRoPE{});
                        plan.bar_kv_rope_ready.wait(k & 1);
                        ku::tcgen05_after_thread_sync();
                        ku::utcmma_ss(tiled_mma_QK, sKr, sQr, tP, /*clear_accum=*/false);
                    }
                    ku::umma_arrive_noelect(plan.bar_qk_done);
                }

                // 2) dP_t = V @ dO^T  into SCORE TMEM (overwrites)
                plan.bar_dp_done.wait(k & 1 ^ 1);  // previous block's consumer done
                ku::tcgen05_after_thread_sync();
                {
                    Tensor sV = make_tensor(make_smem_ptr(plan.u.k.k_nope[cur_buf].data()),
                                            typename Kernel::SmemLayoutV{});
                    Tensor sDOnat = make_tensor(make_smem_ptr(plan.u.do_pack.do_natural.data()),
                                                typename Kernel::SmemLayoutDO{});
                    // SM100 MMA: B = dO^T ⇒ we need a transposed view of dO
                    // matching the K-major operand. The plan uses dO_natural in
                    // SmemLayoutDO; the B-major is UMMA::Major::K (same as P_t).
                    ku::utcmma_ss(tiled_mma_QK, sV, sDOnat, tDP, /*clear_accum=*/true);
                    ku::umma_arrive_noelect(plan.bar_dp_done);
                }

                // 3) dV, dK, dQ MMAs (after WG0 publishes prob_t/draw_t)
                plan.bar_ds_ready.wait(k & 1);
                ku::tcgen05_after_thread_sync();

                // dV: per 64-wide D_V tile, [B_TOPK, 64]
                CUTE_UNROLL
                for (int tile = 0; tile < Kernel::D_V / 64; ++tile) {
                    tDV.data().get() = Kernel::tmem_cols::DV + tile * 64;
                    // SMEM: Prob_t [B_H, B_TOPK] @ dO_t [D_V, B_H]
                    //   tile 0 → dv_base=0, tile 1 → dv_base=64
                    //   do_t row dv lives at SmemLayoutDOTransposed(dv, _)
                    // TODO(dV-tensor-build): build the dO slice for tile 0/1.
                    Tensor sProb = make_tensor(make_smem_ptr(plan.prob_t),
                                               typename Kernel::SmemLayoutS{});
                    Tensor sDOslice = make_tensor(make_smem_ptr(plan.u.do_t_pack.do_t.data() + tile * 64 * Kernel::B_H),
                                                  // Stride = (B_H, 1) row of 64 rows
                                                  Layout<Shape<Int<64>, Int<Kernel::B_H>>, Stride<Int<Kernel::B_H>, _1>>{});
                    ku::utcmma_ss(tiled_mma_DV, sProb, sDOslice, tDV, /*clear_accum=*/(k == 0 && tile == 0));
                }
                ku::umma_arrive_noelect(plan.bar_dv_done[cur_buf]);

                // dK_nope: per 64-wide D tile, [B_TOPK, 64]
                CUTE_UNROLL
                for (int tile = 0; tile < Kernel::D_V / 64; ++tile) {
                    tDK.data().get() = Kernel::tmem_cols::DK + tile * 64;
                    // sQ slice is the 64-wide D slice of Q_nope (which has D_V=128 cols).
                    Tensor sDraw = make_tensor(make_smem_ptr(plan.draw_t),
                                               typename Kernel::SmemLayoutS{});
                    // TODO(sQ-slice): take a (B_H, 64) slice from SmemLayoutQNoPE.
                    Tensor sQslice = make_tensor(make_smem_ptr(plan.u.q_full.q_nope.data() + tile * 64),
                                                 // [B_H, 64] K-major
                                                 Layout<Shape<Int<Kernel::B_H>, Int<64>>, Stride<Int<64>, _1>>{});
                    ku::utcmma_ss(tiled_mma_DK, sDraw, sQslice, tDK, /*clear_accum=*/(k == 0 && tile == 0));
                }
                // When D_QK=192: dK_rope [B_TOPK, 64] into DK_ROPE TMEM.
                if constexpr (HAVE_ROPE) {
                    tDK.data().get() = Kernel::tmem_cols::DK_ROPE;
                    Tensor sDraw = make_tensor(make_smem_ptr(plan.draw_t),
                                               typename Kernel::SmemLayoutS{});
                    Tensor sQr = make_tensor(make_smem_ptr(plan.u.k.k_rope.data() + cosize_v<typename Kernel::SmemLayoutQRoPE>{}),
                                             typename Kernel::SmemLayoutQRoPE{});
                    ku::utcmma_ss(tiled_mma_DK, sDraw, sQr, tDK, /*clear_accum=*/(k == 0));
                }
                ku::umma_arrive_noelect(plan.bar_dk_done[cur_buf]);

                // dQ_t_nope: [D_V, B_H] = K^T @ draw_t
                // For each 64-wide D tile, [64, B_H] (per atom).
                CUTE_UNROLL
                for (int tile = 0; tile < Kernel::D_V / 64; ++tile) {
                    tDQ.data().get() = Kernel::tmem_cols::DQ + tile * Kernel::B_H_TMEM;
                    // sK slice: the 64-wide D slice of K_nope. K_nope is [B_TOPK, D_V]
                    // (K-major). For K^T view we need a (D_V, B_TOPK) layout in MN-major
                    // on the first dim. The K_nope SMEM already has shape
                    // [B_TOPK, D_V] K-major — i.e. leading dim = D_V.
                    // TODO(sK-transposed-view): build a (64, B_TOPK) transposed view of K_nope
                    // for tile. Use SmemLayoutV's MN-major form: composition(K, ...).
                    Tensor sDraw = make_tensor(make_smem_ptr(plan.draw_t),
                                               typename Kernel::SmemLayoutS{});
                    Tensor sKslice = make_tensor(make_smem_ptr(plan.u.k.k_nope[cur_buf].data() + tile * 64),
                                                 Layout<Shape<Int<64>, Int<Kernel::B_TOPK>>, Stride<Int<Kernel::B_TOPK>, _1>>{});
                    ku::utcmma_ss(tiled_mma_DQ, sKslice, sDraw, tDQ, /*clear_accum=*/(k == 0 && tile == 0));
                }
                // dQ_t_rope when D_QK=192.
                if constexpr (HAVE_ROPE) {
                    tDQ.data().get() = Kernel::tmem_cols::DQ_ROPE;
                    Tensor sDraw = make_tensor(make_smem_ptr(plan.draw_t),
                                               typename Kernel::SmemLayoutS{});
                    // sK_rope: [B_TOPK, 64] K-major. Need K^T view [64, B_TOPK].
                    Tensor sKr = make_tensor(make_smem_ptr(plan.u.k.k_rope.data()),
                                             typename Kernel::SmemLayoutKRoPE{});
                    Tensor sKrT = make_tensor(make_smem_ptr(plan.u.k.k_rope.data()),
                                              Layout<Shape<Int<64>, Int<Kernel::B_TOPK>>, Stride<Int<Kernel::B_TOPK>, _1>>{});
                    ku::utcmma_ss(tiled_mma_DQ, sKrT, sDraw, tDQ, /*clear_accum=*/(k == 0));
                }
                ku::umma_arrive_noelect(plan.bar_dq_done[0]);
                if constexpr (HAVE_ROPE) {
                    ku::umma_arrive_noelect(plan.bar_dq_done[1]);
                }
            } // end for k
        }
        // ====================================================================
        // WG2 sub-warps: KV loader (warp 4..7) and valid-mask loader (warp 9)
        // are structured the same way as forward; see head_small/phase1.cuh.
        // ====================================================================
        else if (warp_idx == 9) {
            // Valid-mask loader (lanes 0..7 of warp 9).
            if (lane_idx < Kernel::B_TOPK / 8) {
                CUTE_NO_UNROLL
                for (int k = 0; k < num_k_blocks; ++k) {
                    char k_validness_mask = load_indices_and_generate_mask(
                        lane_idx,
                        gIndices + k * Kernel::B_TOPK,
                        params.s_kv,
                        k * Kernel::B_TOPK,
                        topk_length
                    );
                    int cur_buf = k % Kernel::NUM_BUFS;
                    // Wait for the previous store to free (same double-buffer pattern as fwd).
                    plan.bar_store_done[cur_buf].wait((k / Kernel::NUM_BUFS) & 1 ^ 1);
                    plan.is_k_valid[cur_buf][lane_idx] = k_validness_mask;
                    plan.bar_valid_ready[cur_buf].arrive();
                }
            }
        }
        else if (warp_idx >= 4 && warp_idx < 8) {
            // KV TMA gather + K_RoPE cp.async — same pattern as forward, adjusted
            // for B_TOPK=64 (D_V/64=2 col tiles; 2 NUM_WARPS=2 ⇒ 16 local rows/warp).
            if (elect_one_sync()) {
                constexpr int NUM_WARPS = 4;
                constexpr int LOCAL_ROWS_PER_WARP = (Kernel::B_TOPK / 4) / NUM_WARPS;  // = 4
                int local_warp = warp_idx - 4;
                CUTE_NO_UNROLL
                for (int k = 0; k < num_k_blocks; ++k) {
                    int4 indices[LOCAL_ROWS_PER_WARP];
                    int max_indices = -1, min_indices = params.s_kv;
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < LOCAL_ROWS_PER_WARP; ++local_row) {
                        // For B_TOPK=64, each local warp covers 4 rows spread by NUM_WARPS.
                        int g_row = local_warp + local_row * NUM_WARPS;
                        indices[local_row] = ((int*)gIndices)[g_row];
                        max_indices = max(max_indices, int4_max(indices[local_row]));
                        min_indices = min(min_indices, int4_min(indices[local_row]));
                    }
                    bool is_all_rows_invalid = min_indices == params.s_kv || max_indices == -1;
                    bool should_skip_tma = is_all_rows_invalid && k >= Kernel::NUM_BUFS;

                    int cur_buf = k % Kernel::NUM_BUFS;
                    plan.bar_dv_done[cur_buf].wait((k / Kernel::NUM_BUFS) & 1 ^ 1);
                    plan.bar_dk_done[cur_buf].wait((k / Kernel::NUM_BUFS) & 1 ^ 1);

                    if (!should_skip_tma) {
                        CUTE_UNROLL
                        for (int local_row = 0; local_row < LOCAL_ROWS_PER_WARP; ++local_row) {
                            CUTE_UNROLL
                            for (int local_col = 0; local_col < Kernel::D_V / 64; ++local_col) {
                                bf16* sK_nope_base = plan.u.k.k_nope[cur_buf].data()
                                    + local_warp * 4 * 64
                                    + local_row * (4 * NUM_WARPS) * 64
                                    + local_col * (Kernel::B_TOPK * 64);
                                // For B_TOPK=64 we have a single int per row; reconstruct
                                // the int4 as (idx, idx+1, idx+2, idx+3) from the lane.
                                // TODO(gather4-indices): tma_gather4 takes int4; we
                                // only have one int per local row. Either batch 4
                                // local rows per gather4 call (matching fwd pattern
                                // where B_TOPK=128 let 1 warp own 16 contiguous rows
                                // and gather4 reads 4 of them), or switch to a single-
                                // row tma_load. For the v1 skeleton, mark as TODO.
                            }
                        }
                        plan.bar_kv_ready[cur_buf].complete_transaction(LOCAL_ROWS_PER_WARP * 4 * Kernel::D_V * sizeof(bf16));
                    } else {
                        plan.bar_kv_ready[cur_buf].complete_transaction(LOCAL_ROWS_PER_WARP * 4 * Kernel::D_V * sizeof(bf16));
                    }
                }
            }
        }
        else if (warp_idx == 10 || warp_idx == 11) {
            if constexpr (HAVE_ROPE) {
                int thread_idx = threadIdx.x - 10 * 32;
                constexpr int GROUP_SIZE = 8, NUM_GROUPS = 64 / GROUP_SIZE;  // B_TOPK/8 = 8
                int group_idx = thread_idx / GROUP_SIZE, idx_in_group = thread_idx % GROUP_SIZE;
                Tensor sK_rope = make_tensor(make_smem_ptr(plan.u.k.k_rope.data() + cosize_v<typename Kernel::SmemLayoutKRoPE>{}),
                                             typename Kernel::SmemLayoutKRoPE{});
                bf16* sK_rope_base = &sK_rope(group_idx, idx_in_group * 8);
                CUTE_NO_UNROLL
                for (int k = 0; k < num_k_blocks; ++k) {
                    int indices[Kernel::B_TOPK / NUM_GROUPS];
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < Kernel::B_TOPK / NUM_GROUPS; ++local_row) {
                        indices[local_row] = __ldg(gIndices + k * Kernel::B_TOPK + group_idx + local_row * NUM_GROUPS);
                    }
                    plan.bar_dq_done[1].wait(k & 1 ^ 1);  // reuse dq_done[1] as the K-rope done gate
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < Kernel::B_TOPK / NUM_GROUPS; ++local_row) {
                        int index = indices[local_row];
                        ku::cp_async_cacheglobal<ku::PrefetchSize::B128>(
                            params.kv + (int64_t)index * params.stride_kv_s_kv + Kernel::D_V + idx_in_group * 8,
                            sK_rope_base + local_row * NUM_GROUPS * 32,
                            index >= 0 && index < params.s_kv
                        );
                    }
                    cutlass::arch::cpasync_barrier_arrive_noinc((uint64_t*)&(plan.bar_kv_rope_ready));
                }
            }
        }
    }
    // ====================================================================
    // After the loop: store dQ to global.
    // ====================================================================
    if (warpgroup_idx == 0) {
        // All WG0 warps wait for dQ MMAs to complete.
        plan.bar_dq_done[0].wait(((num_k_blocks - 1) / Kernel::NUM_BUFS) & 1);
        if constexpr (HAVE_ROPE) {
            plan.bar_dq_done[1].wait(((num_k_blocks - 1) / Kernel::NUM_BUFS) & 1);
        }
        ku::tcgen05_after_thread_sync();

        // Each thread handles one dq position; warps 0..3 cover the 128 NoPE cols
        // (4 warps × 32 lanes) and (when HAVE_ROPE) the first 2 warps also cover
        // the 64 RoPE cols.
        // TODO(dq-store): for B_H=24, B_H_TMEM=32, so `dq[h]` reads B_H_TMEM cols.
        // The store loop must also account for H_Q==24 mask when writing heads.
        if (warp_idx < 4) {
            int dq_pos = warp_idx * 32 + lane_idx;  // 0..127
            int tile = dq_pos / 64;
            float dq_head[Kernel::B_H_TMEM];
            ku::tmem_ld_32dp32bNx<Kernel::B_H_TMEM>(Kernel::tmem_cols::DQ + tile * Kernel::B_H_TMEM, dq_head);
            cutlass::arch::fence_view_async_tmem_load();
            int64_t s_q_off = (int64_t)s_q_idx * params.stride_d_q_s_q;
            CUTE_UNROLL
            for (int h = 0; h < Kernel::B_H; ++h) {
                int64_t off = s_q_off + (int64_t)h * params.stride_d_q_h_q + dq_pos;
                params.dq[off] = bf16(dq_head[h]);
            }
        }
        if constexpr (HAVE_ROPE) {
            if (warp_idx < 2) {
                int dq_pos = 128 + warp_idx * 32 + lane_idx;  // 128..191
                float dq_head[Kernel::B_H_TMEM];
                ku::tmem_ld_32dp32bNx<Kernel::B_H_TMEM>(Kernel::tmem_cols::DQ_ROPE, dq_head);
                cutlass::arch::fence_view_async_tmem_load();
                int64_t s_q_off = (int64_t)s_q_idx * params.stride_d_q_s_q;
                CUTE_UNROLL
                for (int h = 0; h < Kernel::B_H; ++h) {
                    int64_t off = s_q_off + (int64_t)h * params.stride_d_q_h_q + dq_pos;
                    params.dq[off] = bf16(dq_head[h]);
                }
            }
        }
        NamedBarrier::arrive_and_wait(128, Kernel::NamedBarriers::wg0_sync);

        if (warp_idx == 0) {
            cute::TMEM::Allocator1Sm().free(0, 512);
        }
    }
    (void)1; (void)0;  // suppress unused-variable warnings until we wire all params
#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100");
    }
#endif
}

template<int D_QK, int H_Q>
void run_bwd_phase1_kernel(const SparseAttnBwdParams& params) {
    using Kernel = KernelTemplate<D_QK, H_Q>;
    static_assert(D_QK == 128 || D_QK == 192);

    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk % Kernel::B_TOPK == 0);
    KU_ASSERT(params.h_q == H_Q);
    KU_ASSERT(params.d_qk == D_QK);
    KU_ASSERT(params.d_v == Kernel::D_V);

    // ---- TMA: Q NoPE (loaded like forward) ----
    auto shape_Q_nope = make_shape(params.h_q, Kernel::D_V, params.s_q);
    auto tma_Q_nope = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.q),
            make_layout(shape_Q_nope,
                        make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q))
        ),
        typename Kernel::SmemLayoutQNoPE{}
    );

    // ---- TMA: Q RoPE (when D_QK=192) ----
    auto shape_Q_rope = make_shape(params.h_q, Kernel::D_ROPE, params.s_q);
    auto tma_Q_rope = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.q + Kernel::D_V),
            make_layout(shape_Q_rope,
                        make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q))
        ),
        typename Kernel::SmemLayoutQRoPE{}
    );

    // ---- TMA: O (read for sum_odo) ----
    auto shape_O = make_shape(params.h_q, params.d_v, params.s_q);
    auto tma_O = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.out),
            make_layout(shape_O,
                        make_stride(params.stride_out_h_q, _1{}, params.stride_out_s_q))
        ),
        typename Kernel::SmemLayoutO{}
    );

    // ---- TMA: dO natural (Q-operand of dP_t) ----
    auto shape_DO = make_shape(params.h_q, params.d_v, params.s_q);
    auto tma_DO = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.d_out),
            make_layout(shape_DO,
                        make_stride(params.stride_d_out_h_q, _1{}, params.stride_d_out_s_q))
        ),
        typename Kernel::SmemLayoutDO{}
    );

    // ---- TMA: dO transposed (B-operand of dV) ----
    // (D_V, B_H, s_q) with strides (B_H, 1, h_q*B_H).
    // TODO(tma-dO_t): for very small B_H (< 16) the swizzle atom may be invalid;
    // validate or fall back to a different swizzle size.
    auto shape_DO_t = make_shape(params.d_v, params.h_q, params.s_q);
    auto tma_DO_t = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.d_out),
            make_layout(shape_DO_t,
                        make_stride(params.stride_d_out_h_q, _1{}, params.stride_d_out_s_q))
        ),
        typename Kernel::SmemLayoutDOTransposed{}
    );

    // ---- CUtensorMap for K/V NoPE gather4 (same as forward) ----
    CUtensorMap tensor_map_kv_nope;
    {
        uint64_t size[2]   = {Kernel::D_V, (unsigned long)params.s_kv};
        uint64_t stride[1] = {params.stride_kv_s_kv * sizeof(bf16)};
        uint32_t box_size[2]   = {64, 1};
        uint32_t elem_stride[2] = {1, 1};
        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &tensor_map_kv_nope,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            params.kv,
            size, stride, box_size, elem_stride,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        KU_ASSERT(res == CUresult::CUDA_SUCCESS);
    }

    TmaParams<
        decltype(shape_Q_nope), decltype(tma_Q_nope),
        decltype(shape_Q_rope), decltype(tma_Q_rope),
        decltype(shape_O),      decltype(tma_O),
        decltype(shape_DO),     decltype(tma_DO),
        decltype(shape_DO_t),   decltype(tma_DO_t)
    > tma_args = {
        shape_Q_nope, tma_Q_nope,
        shape_Q_rope, tma_Q_rope,
        shape_O,      tma_O,
        shape_DO,     tma_DO,
        shape_DO_t,   tma_DO_t,
        tensor_map_kv_nope
    };

    auto kernel = &sparse_attn_bwd_kernel<D_QK, H_Q, D_QK == 192, decltype(tma_args)>;

    constexpr size_t smem_size = sizeof(typename Kernel::SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    // TODO(launch): once compilation is green, replace the assertion with the
    // actual launch:
    //   kernel<<<params.s_q, Kernel::NUM_THREADS, smem_size, params.stream>>>(params, tma_args);
    (void)kernel;
    TORCH_CHECK(false,
        "sparse_attn_bwd_kernel skeleton: launch is intentionally disabled. "
        "Resolve the TODO comments and the dV/dK epilogue loops in phase1.cuh, "
        "then enable the launch in run_bwd_phase1_kernel.");
}

}
