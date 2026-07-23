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

namespace sm100::fp8_fwd::head64 {

using namespace cute;

CUTE_DEVICE
float ue8m0_bits_to_float(uint8_t bits) {
    TRAP_ONLY_DEVICE_ASSERT(bits != 0xff);
    if (bits == 0) {
        return __uint_as_float(0x00400000u);
    }
    return __uint_as_float(static_cast<uint32_t>(bits) << 23);
}

template<typename TmaParams>
__global__ void __launch_bounds__(NUM_THREADS, 1, 1)
sprase_fp8_attn_fwd_kernel(__grid_constant__ const Head64Fp8SparseAttnFwdParams params, __grid_constant__ const TmaParams tma_params) {

    const int s_q_idx = blockIdx.x;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = threadIdx.x % 32;
    const int warpgroup_idx = __shfl_xor_sync(0xffffffff, threadIdx.x / 128, 0);
    const int idx_in_warpgroup = threadIdx.x % 128;
    const int requested_topk_length = params.topk_length != nullptr ? __ldg(params.topk_length + s_q_idx) : params.topk;
    const int topk_length = max(0, min(requested_topk_length, params.topk));
    const int num_k_blocks = max(cute::ceil_div(topk_length, (int)B_TOPK), 1);  // num_k_blocks always >= 1

    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
    if (warp_idx == 0 && elect_one_sync()) {
        cute::prefetch_tma_descriptor(tma_params.tma_O.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(&tma_params.tensor_map_kv);
    }

    int* gIndices = params.indices + s_q_idx*params.stride_indices_s_q; // [topk]

    TiledMMA tiled_mma_P = TiledMMA_P{};
    TiledMMA tiled_mma_O = TiledMMA_O{};

    Tensor tQ_part0 = tiled_mma_P.get_slice(_0{}).make_fragment_A(
        partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<(D_V/2)/2>>{})
    );
    Tensor tQ_part1 = tiled_mma_P.get_slice(_0{}).make_fragment_A(
        partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<(D_V/2)/2>>{})
    );

    Tensor tP0 = partition_fragment_C(tiled_mma_P, Shape<Int<B_H>, _128>{});
    Tensor tP1 = partition_fragment_C(tiled_mma_P, Shape<Int<B_H>, _128>{});
    Tensor tP2 = partition_fragment_C(tiled_mma_P, Shape<Int<B_H>, _128>{});
    Tensor tO = partition_fragment_C(tiled_mma_O, Shape<Int<B_H>, Int<D_V>>{});
    tP0.data().get() = tmem_cols::P;
    tP1.data().get() = tmem_cols::P + 64;
    tP2.data().get() = tmem_cols::P + 128;
    tQ_part0.data().get() = tmem_cols::Q;
    tQ_part1.data().get() = tmem_cols::Q + 32;
    tO.data().get() = tmem_cols::O;

        if (warp_idx == 0) {
        if (elect_one_sync()) {
            // Copy Q
            cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());

            plan.bar_prologue.init(1);
            fence_barrier_init();

            Tensor gQ = tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx);
            Tensor sQ = make_tensor(make_smem_ptr(plan.qkvo.q.q.data()), SmemLayoutQ{});
            ku::launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_prologue, TMA::CacheHintSm90::EVICT_FIRST);

            cute::prefetch_tma_descriptor(tma_params.tma_O.get_tma_descriptor());
            cute::prefetch_tma_descriptor(&(tma_params.tensor_map_kv));
            
            // Initialize other barriers
            plan.bar_prologue_utccp.init(1);
            plan.bar_qw_scale_ready.init(2);
            CUTE_UNROLL
            for (int i = 0; i < NUM_BUFS; ++i) {
                plan.bar_qk_done[i].init(1);
                plan.bar_sv_done[i].init(1);
                plan.bar_kv_ready[i][0].init(1);
                plan.bar_kv_ready[i][1].init(1);
                plan.bar_kv_scale_ready[i].init(2);
                plan.bar_k_valid_ready[i].init(B_TOPK/8);
                plan.bar_k_valid_free[i].init(128);
            }
            for (int i = 0; i < NUM_P_BUFS; ++i) {
                plan.bar_p_free[i].init(128); // warp group 0 touch this
            }
            plan.bar_so_ready.init(128);
            fence_barrier_init();
        }

        // Initialize TMEM
        cute::TMEM::Allocator1Sm().allocate(512, plan.tmem_start_addr.data());
        TRAP_ONLY_DEVICE_ASSERT(plan.tmem_start_addr.data()[0] == 0);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }

    __syncthreads();

    if (warpgroup_idx == 0) {
        plan.bar_qw_scale_ready.wait(0);

        float mi = MAX_INIT_VAL;
        float li = 0.0f;
        float real_mi = -CUDART_INF_F;
        float s_scale_for_o = 1.0f;

        const int h = idx_in_warpgroup % B_H;
        const int token_group = idx_in_warpgroup / B_H;
        const int token_base = token_group * (B_TOPK / 2);
        const float q_scale = plan.q_head_scale[h];
        Tensor sS = make_tensor(make_smem_ptr(plan.s.data()), SmemLayoutS{});
        static constexpr int NUM_ELEMS_PER_THREAD = B_TOPK / 2;

        CUTE_NO_UNROLL
        for (int k = 0; k < num_k_blocks; ++k) {
            // Wait for P
            NamedBarrier::arrive_and_wait(64, NamedBarriers::wg0_warp02_sync+(warp_idx&1));
            const int cur_buf = k % NUM_BUFS;
            const int p_idx = k % NUM_P_BUFS;
            plan.bar_qk_done[p_idx].wait((k / NUM_P_BUFS) & 1);
            plan.bar_k_valid_ready[cur_buf].wait((k / NUM_BUFS) & 1);
            plan.bar_kv_scale_ready[cur_buf].wait((k / NUM_BUFS) & 1);
            ku::tcgen05_after_thread_sync();

            // One scale per lane stays in registers; shuffles expose the 32
            // token scales needed by every head in this half tile.
            const float lane_kv_scale =
                plan.kv_dim_scale[cur_buf][token_base + lane_idx];

            // Load P
            float p[NUM_ELEMS_PER_THREAD];
            auto release_p = [&]() { plan.bar_p_free[p_idx].arrive(); };
            if (p_idx == 0) {
                retrieve_mask_and_reduce_p<
                    NUM_ELEMS_PER_THREAD,
                    tmem_cols::P,
                    NamedBarriers::wg0_warp02_sync,
                    NamedBarriers::wg0_warp13_sync,
                    false
                >(
                    plan.is_k_valid[cur_buf], warp_idx, lane_idx,
                    release_p, plan.p_exchange_buf, p
                );
            } else if (p_idx == 1) {
                retrieve_mask_and_reduce_p<
                    NUM_ELEMS_PER_THREAD,
                    tmem_cols::P + 64,
                    NamedBarriers::wg0_warp02_sync,
                    NamedBarriers::wg0_warp13_sync,
                    false
                >(
                    plan.is_k_valid[cur_buf], warp_idx, lane_idx,
                    release_p, plan.p_exchange_buf, p
                );
            } else {
                retrieve_mask_and_reduce_p<
                    NUM_ELEMS_PER_THREAD,
                    tmem_cols::P + 128,
                    NamedBarriers::wg0_warp02_sync,
                    NamedBarriers::wg0_warp13_sync,
                    false
                >(
                    plan.is_k_valid[cur_buf], warp_idx, lane_idx,
                    release_p, plan.p_exchange_buf, p
                );
            }
            plan.bar_k_valid_free[cur_buf].arrive();

            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD; ++i) {
                const float kv_scale = __shfl_sync(
                    0xffffffff, lane_kv_scale, i
                );
                p[i] *= q_scale * kv_scale * params.sm_scale_div_log2;
            }
            
            // Get rowwise max of Pi
            float cur_pi_max = get_max<NUM_ELEMS_PER_THREAD>(p);

            plan.rowwise_max_buf[idx_in_warpgroup] = cur_pi_max;
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
            cur_pi_max = max(cur_pi_max, plan.rowwise_max_buf[idx_in_warpgroup^64]);
            real_mi = max(real_mi, cur_pi_max);
            bool should_scale_o = __any_sync(0xffffffff, cur_pi_max - mi > 6.0f);

            float new_max, scale_for_old;
            if (!should_scale_o) {
                // Don't scale O
                scale_for_old = 1.0f;
                new_max = mi;
            } else {
                new_max = max(cur_pi_max, mi);
                scale_for_old = exp2f(mi - new_max);
            }
            mi = new_max;   // mi is still identical within each row

            // Absorb the token factor into S, then quantize each head row.
            e4m3 s[NUM_ELEMS_PER_THREAD];
            float cur_sum = 0.0f;
            float local_s_max = 0.0f;
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD; ++i) {
                const float softmax_s = exp2f(p[i] - new_max);
                cur_sum += softmax_s;
                const float kv_scale = __shfl_sync(
                    0xffffffff, lane_kv_scale, i
                );
                p[i] = softmax_s * kv_scale;
                local_s_max = max(local_s_max, fabsf(p[i]));
            }

            plan.rowwise_li_buf[idx_in_warpgroup] = local_s_max;
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
            const float s_max = max(
                local_s_max,
                plan.rowwise_li_buf[idx_in_warpgroup ^ B_H]
            );
            const e8m0 s_scale_e8m0(
                s_max > 0.0f ? s_max / FP8_MAX : 1.0f
            );
            const float current_s_scale = float(s_scale_e8m0);
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD; ++i) {
                s[i] = e4m3(p[i] / current_s_scale);
            }
            li = fma(li, scale_for_old, cur_sum);

            // Wait for last SV gemm, write S
            if (k > 0) {
                plan.bar_sv_done[(k-1)%NUM_BUFS].wait(((k-1)/NUM_BUFS)&1);
            }
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD; ++i) {
                sS(h, token_base + i) = s[i];
            }

            // O is kept in units of the current S scale. The fixed W(g)
            // factor is restored once in the epilogue.
            if (k > 0) {
                const float o_rescale = scale_for_old
                    * s_scale_for_o / current_s_scale;
                ku::tcgen05_after_thread_sync();
                rescale_O<D_V, 32, tmem_cols::O>(o_rescale);
                ku::tcgen05_before_thread_sync();
            }
            s_scale_for_o = current_s_scale;
            
            fence_view_async_shared();
            plan.bar_so_ready.arrive();
        }

        NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);

    if (real_mi == -CUDART_INF_F) {
            // real_mi == -CUDART_INF_F <=> No valid TopK indices
            // We set li to 0 to fit the definition that li := exp(x[i] - mi)
            li = 0.0f;
            mi = -CUDART_INF_F;
        }
        
        // Exchange li
        plan.rowwise_li_buf[idx_in_warpgroup] = li;
        NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
        li += plan.rowwise_li_buf[idx_in_warpgroup^64];

        // Store mi and li
        if (idx_in_warpgroup < 64) {
            int global_index = s_q_idx*params.h_q + idx_in_warpgroup;
            float cur_lse = fmaf(mi, CUDART_LN2_F, logf(li));
            cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;
            params.max_logits[global_index] = real_mi*CUDART_LN2_F;
            params.lse[global_index] = cur_lse;
        }

        // Wait for the last GEMM
        plan.bar_sv_done[(num_k_blocks-1)%NUM_BUFS].wait(((num_k_blocks-1)/NUM_BUFS)&1);
        ku::tcgen05_after_thread_sync();

        // Fetch dO if necessary

        // Store O
        float attn_sink = params.attn_sink == nullptr ? -CUDART_INF_F : __ldg(params.attn_sink + (idx_in_warpgroup%64))*CUDART_L2E_F;
        float output_scale = __fdividef(1.0f, li + exp2f(attn_sink - mi));
        Tensor sO = make_tensor(make_smem_ptr(plan.qkvo.o.data()), SmemLayoutO{});
        constexpr int B_EPI = 64;
        Tensor tma_gO = flat_divide(
            tma_params.tma_O.get_tma_tensor(tma_params.shape_O)(_, _, s_q_idx),
            Shape<Int<B_H>, Int<B_EPI>>{}
        )(_, _, _0{}, _);
        Tensor sO_divided = flat_divide(
            sO,
            Shape<Int<B_H>, Int<B_EPI>>{}
        )(_, _, _0{}, _);
        auto thr_tma = tma_params.tma_O.get_slice(_0{});

        float2 o[B_EPI/2];
        bool have_valid_indices = __any_sync(0xffffffff, li != 0);  // Prevent some threads' li == 0 and some threads' li != 0 which lead to deadlock during ku::tmem_ld
        if (!have_valid_indices) {
            // If there are no valid indices, we set o[i] to 0 and don't load from TMEM
            CUTE_UNROLL
            for (int i = 0; i < B_EPI/2; ++i)
                o[i].x = o[i].y = 0.0f;
            output_scale = 1.0f;
        }

        bf16* sO_addrs[8];
        CUTE_UNROLL
        for (int i = 0; i < B_EPI/8; ++i) {
            sO_addrs[i] = &sO(idx_in_warpgroup%64, i*8);
        }

        CUTE_UNROLL
        for (int c = 0; c < 2; ++c) {
            // Each tile: 64 x 256
            CUTE_UNROLL
            for (int k = 0; k < (D_V/4)/B_EPI; ++k) {
                // Load O from tO
                if (have_valid_indices) {
                    ku::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::O + c*128 + k*B_EPI, o);
                    cutlass::arch::fence_view_async_tmem_load();
                }

                const int d_group = c * 4
                    + (idx_in_warpgroup / B_H) * 2 + k;
                const float output_dequant_scale = output_scale
                    * s_scale_for_o * plan.kv_w_scale[d_group];
                const float2 output_scale_float2 = make_float2(
                    output_dequant_scale, output_dequant_scale
                );

                // Convert and store
                CUTE_UNROLL
                for (int i = 0; i < B_EPI/8; ++i) {
                    nv_bfloat162 o_bf16[4];
                    CUTE_UNROLL
                    for (int j = 0; j < 4; ++j) {
                        o[i*4+j] = ku::float2_mul(o[i*4+j], output_scale_float2);
                        o_bf16[j] = __float22bfloat162_rn(o[i*4+j]);
                    }
                    *(uint128_t*)(sO_addrs[i] + (c*(D_V/2) + (idx_in_warpgroup/64)*(D_V/4) + k*B_EPI)*64) = *(uint128_t*)(o_bf16);
                }

                // Sync
                fence_view_async_shared();
                NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
                
                if (warp_idx == 0 && elect_one_sync()) {
                    int epi_chunk_idx = c*(D_V/2/B_EPI) + k;
                    cute::copy(
                        tma_params.tma_O,
                        thr_tma.partition_S(sO_divided(_, _, epi_chunk_idx)),
                        thr_tma.partition_D(tma_gO(_, _, epi_chunk_idx))
                    );
                }
                if (warp_idx == 1 && elect_one_sync()) {
                    int epi_chunk_idx = c*(D_V/2/B_EPI) + (D_V/B_EPI/4) + k;
                    cute::copy(
                        tma_params.tma_O,
                        thr_tma.partition_S(sO_divided(_, _, epi_chunk_idx)),
                        thr_tma.partition_D(tma_gO(_, _, epi_chunk_idx))
                    );
                }
            }
        }
        if (warp_idx == 0) {
            cute::TMEM::Allocator1Sm().free(0, 512);
        }
} else if (warpgroup_idx == 1) {

    // Producer warp for KV
        int warp_idx = cutlass::canonical_warp_idx_sync() - 4;
        constexpr int NUM_WARPS = 4, NUM_LOCAL_ROWS_PER_WARP = (B_TOPK/4)/NUM_WARPS;
        if (elect_one_sync()) {
            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks; ++k) {
                int4 indices[NUM_LOCAL_ROWS_PER_WARP];
                int max_indices = -1, min_indices = params.s_kv;
                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                    indices[local_row] = __ldg((int4*)(gIndices + k*B_TOPK) + local_row*NUM_WARPS + warp_idx);
                    max_indices = max(max_indices, int4_max(indices[local_row]));
                    min_indices = min(min_indices, int4_min(indices[local_row]));
                }
                bool is_all_rows_invalid = min_indices == params.s_kv || max_indices == -1;
                bool should_skip_tma = is_all_rows_invalid && k >= NUM_BUFS;

                if (k == 2) {
                    plan.bar_prologue_utccp.wait(0);   // Since q coincidences with k[2]
                }

                // Copy NoPE
                int cur_buf = k%NUM_BUFS;
                plan.bar_sv_done[cur_buf].wait((k/NUM_BUFS)&1^1);
                e4m3* sK_base = plan.qkvo.kv[cur_buf].data() + warp_idx*4*64;

                auto load_kv_part = [&](int part_idx) {
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                        CUTE_UNROLL
                        for (int local_col = part_idx*(D_V/2/64); local_col < (part_idx+1)*(D_V/2/64); ++local_col) {
                            ku::tma_gather4(
                                &(tma_params.tensor_map_kv),
                                plan.bar_kv_ready[cur_buf][part_idx],
                                sK_base + local_row*(4*NUM_WARPS)*64 + local_col*(B_TOPK*64),
                                local_col*64,
                                indices[local_row],
                                (int64_t)TMA::CacheHintSm90::EVICT_LAST
                            );
                        }
                    }
                };

                if (!should_skip_tma) {
                    load_kv_part(0);
                    load_kv_part(1);
                } else {
                    // NOTE See head128/phase1.cuh for this TMA skipping technique
                    CUTE_UNROLL
                    for (int part_idx = 0; part_idx < 2; ++part_idx)
                        plan.bar_kv_ready[cur_buf][part_idx].complete_transaction(NUM_LOCAL_ROWS_PER_WARP*4*D_V/2*sizeof(e4m3));
                }
            }
        }

} else {
    if (warp_idx == 8 && elect_one_sync()) {
        UMMA::SmemDescriptor sQ_desc = UMMA::make_umma_desc<UMMA::Major::K>(
                make_tensor(
                    make_smem_ptr(plan.qkvo.q.q.data()),
                    tile_to_shape(
                        UMMA::Layout_K_SW128_Atom<e4m3>{},
                        Shape<Int<B_H*2>, Int<128>>{}    // We use this shape for dual gemm (TODO Link)
                    )
                )
            );
        
        plan.bar_prologue.arrive_and_expect_tx(B_H*D_V*sizeof(e4m3));
        plan.bar_prologue.wait(0);
        ku::tcgen05_after_thread_sync();
        CUTE_UNROLL
        for (int tile_idx = 0; tile_idx < D_V/256; ++tile_idx) {
            // A tile is 128 rows * 128 e4m3 values, or 64 rows * 256 values in our dual-GEMM view.
            CUTE_UNROLL
            for (int subtile_idx = 0; subtile_idx < 4; ++subtile_idx) {
                // A subtile is 128 rows * 16 cols (256b, 32B) (in UTCCP's view), or 64 rows * 16 cols * 2 (in our view)
                SM100_UTCCP_128dp256bit_1cta::copy(
                    sQ_desc + (tile_idx*(B_H*128*2) + subtile_idx*32) / 16, 
                    tmem_cols::Q + tile_idx*32 + subtile_idx*8
                );
            }
        }
        ku::umma_arrive_noelect(plan.bar_prologue_utccp);

        CUTE_NO_UNROLL
        for (int k = 0; k < num_k_blocks+1; ++k) {
            if (k < num_k_blocks) {
                int cur_buf = k%NUM_BUFS;
                int p_stage = k%NUM_P_BUFS;
                Tensor sK = make_tensor(make_smem_ptr(plan.qkvo.kv[cur_buf].data()), SmemLayoutK_TiledMMA{});
                plan.bar_p_free[p_stage].wait(((k/NUM_P_BUFS)&1)^1);
                ku::tcgen05_after_thread_sync();
                if (k == 0) {
                    plan.bar_prologue_utccp.wait(0);
                }
                Tensor sK_divided = flat_divide(sK, Tile<Int<B_TOPK*2>, Int<D_V/4>>{})(_, _, _0{}, _);
                CUTE_UNROLL
                for (int kv_part_idx = 0; kv_part_idx < 2; ++kv_part_idx) {
                    plan.bar_kv_ready[cur_buf][kv_part_idx].arrive_and_expect_tx(B_TOPK*D_V/2*sizeof(e4m3));
                    plan.bar_kv_ready[cur_buf][kv_part_idx].wait((k/NUM_BUFS)&1);
                    ku::tcgen05_after_thread_sync();

                    // P += Q(nope) @ K(nope)^T
                    bool clear_accum = kv_part_idx == 0;
                    if (p_stage == 0) {
                        ku::utcmma_ts(tiled_mma_P, kv_part_idx ? tQ_part1 : tQ_part0, sK_divided(_, _, kv_part_idx), tP0, clear_accum);
                    } else if (p_stage == 1) {
                        ku::utcmma_ts(tiled_mma_P, kv_part_idx ? tQ_part1 : tQ_part0, sK_divided(_, _, kv_part_idx), tP1, clear_accum);
                    } else {
                        ku::utcmma_ts(tiled_mma_P, kv_part_idx ? tQ_part1 : tQ_part0, sK_divided(_, _, kv_part_idx), tP2, clear_accum);
                    }
                }
                ku::umma_arrive_noelect(plan.bar_qk_done[p_stage]);
            }

            if (k > 0) {
                    // O += S(i-1)V(i-1)
                    int cur_buf = (k-1)%NUM_BUFS;

                    Tensor sS = make_tensor(make_smem_ptr(plan.s.data()), SmemLayoutS{});
                    Tensor sV = make_tensor(make_smem_ptr(plan.qkvo.kv[cur_buf].data()), SmemLayoutV{});

                    // Wait for S(i-1) and O to be scaled
                    plan.bar_so_ready.wait((k-1)&1);
                    ku::tcgen05_after_thread_sync();

                    // O += sS @ sV
                    ku::utcmma_ss(tiled_mma_O, sS, sV, tO, k == 1);
                    ku::umma_arrive_noelect(plan.bar_sv_done[cur_buf]);
                }
        }
    } else if (warp_idx == 9) {
        if (lane_idx < B_TOPK/8) {
            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks; ++k) {
                char k_validness_mask = load_indices_and_generate_mask(
                    lane_idx,
                    gIndices + k*B_TOPK,
                    params.s_kv,
                    k*B_TOPK,
                    topk_length
                );

                int cur_buf = k%NUM_BUFS;
                plan.bar_k_valid_free[cur_buf].wait((k/NUM_BUFS)&1^1);
                plan.is_k_valid[cur_buf][lane_idx] = k_validness_mask;
                plan.bar_k_valid_ready[cur_buf].arrive();
            }
        }
    } else if (warp_idx == 10 || warp_idx == 11) {
        const int scale_warp_idx = warp_idx - 10;
        const int scale_row = scale_warp_idx * 32 + lane_idx;

        const uint8_t* q_scale_base =
            reinterpret_cast<const uint8_t*>(params.q)
            + static_cast<int64_t>(s_q_idx) * params.stride_q_s_q
            + B_H * D_Q;
        plan.q_head_scale[scale_row] = ue8m0_bits_to_float(
            __ldg(q_scale_base + scale_row)
        );
        if (scale_warp_idx == 0 && lane_idx < KV_SCALE_GROUPS) {
            plan.kv_w_scale[lane_idx] = ue8m0_bits_to_float(
                __ldg(params.kv_scale_w + lane_idx)
            );
        }
        fence_view_async_shared();
        if (elect_one_sync()) {
            plan.bar_qw_scale_ready.arrive();
        }

        CUTE_NO_UNROLL
        for (int k = 0; k < num_k_blocks; ++k) {
            const int cur_buf = k % NUM_BUFS;
            plan.bar_sv_done[cur_buf].wait(
                ((k / NUM_BUFS) & 1) ^ 1
            );

            const int src_idx = __ldg(
                gIndices + k * B_TOPK + scale_row
            );
            const bool is_valid = src_idx >= 0
                && src_idx < params.s_kv
                && k * B_TOPK + scale_row < topk_length;
            uint8_t scale_bits = 0x7f;
            if (is_valid) {
                const uint8_t* kv_scale_ptr =
                    reinterpret_cast<const uint8_t*>(params.kv)
                    + static_cast<int64_t>(src_idx)
                        * params.stride_kv_s_kv
                    + D_K;
                scale_bits = __ldg(kv_scale_ptr);
            }
            plan.kv_dim_scale[cur_buf][scale_row] =
                ue8m0_bits_to_float(scale_bits);
            fence_view_async_shared();
            if (elect_one_sync()) {
                plan.bar_kv_scale_ready[cur_buf].arrive();
            }
        }
    }
}

}

template<int D_QK>
void run_fp8_fwd_phase1_kernel(const Head64Fp8SparseAttnFwdParams& params) {

    auto shape_O = make_shape(B_H, D_V, params.s_q);
    auto tma_O = cute::make_tma_copy(
        SM90_TMA_STORE{},
        make_tensor(
            make_gmem_ptr((bf16*)params.out),
            make_layout(
                shape_O,
                make_stride(D_V, _1{}, params.h_q*params.d_v)
            )
        ),
        SmemLayoutOBuf_TMA{}
    );

    auto shape_Q = make_shape(B_H, D_Q, params.s_q);
    auto tma_Q = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr(reinterpret_cast<e4m3*>(params.q)),
            make_layout(
                shape_Q,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q)
            )
        ),
        SmemLayoutQ{}
    );

    CUtensorMap tensor_map_kv = ku::make_tensor_map(
            {D_K / 8, static_cast<uint64_t>(params.s_kv)},
            {D_K},
            {TMA_K_CHUNK_ELEMS, 1},
            params.kv,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT64,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B
        );

    TmaParams<
        decltype(shape_O), decltype(tma_O),
        decltype(shape_Q), decltype(tma_Q)
    > tma_params = {
        shape_O, tma_O,
        shape_Q, tma_Q,
        tensor_map_kv
    };

    auto kernel = &sprase_fp8_attn_fwd_kernel<decltype(tma_params)>;

    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    kernel<<<params.s_q, NUM_THREADS, smem_size, params.stream>>>(params, tma_params);
    KU_CHECK_KERNEL_LAUNCH();

}


}
