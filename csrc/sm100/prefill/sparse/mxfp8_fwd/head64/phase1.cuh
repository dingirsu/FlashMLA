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

namespace sm100::mxfp8_fwd::head64 {

using namespace cute;

template<int B_H, int B_H_TMEM, int TMEM_COL_START, int D_V>
CUTE_DEVICE
void rescale_O_t(float scale[B_H]) {
    float o[B_H_TMEM];
    CUTE_UNROLL
    for (int tile = 0; tile < D_V/64; ++tile) {
        ku::tmem_ld_32dp32bNx<B_H_TMEM>(TMEM_COL_START + tile*B_H_TMEM, o);
        cutlass::arch::fence_view_async_tmem_load();
        CUTE_UNROLL
        for (int i = 0; i < B_H; ++i) {
            o[i] *= scale[i];
        }
        ku::tmem_st_32dp32bNx<B_H_TMEM>(TMEM_COL_START + tile*B_H_TMEM, o);
        cutlass::arch::fence_view_async_tmem_store();
    }
}

using FwdMode = SparseAttnFwdMode;

template<bool HAVE_ROPE, typename TmaParams>
__global__ void __launch_bounds__(NUM_THREADS, 1, 1)
sparse_attn_fwd_kernel(__grid_constant__ const MxFp8SparseAttnFwdParams params, __grid_constant__ const TmaParams tma_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))

    const int s_q_idx = blockIdx.x;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = threadIdx.x % 32;
    const int warpgroup_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    const int idx_in_warpgroup = threadIdx.x % 128;
    const int topk_length = params.topk_length != nullptr ? __ldg(params.topk_length + s_q_idx) : params.topk;
    const int num_k_blocks = max(cute::ceil_div(topk_length, (int)B_TOPK), 1);  // num_k_blocks always >= 1

    // Define shared tensors
    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);

    if (warp_idx == 0 && elect_one_sync()) {
        cute::prefetch_tma_descriptor(tma_params.tma_O.get_tma_descriptor());
        cute::prefetch_tma_descriptor(&tma_params.tensor_map_q_nope);
        cute::prefetch_tma_descriptor(tma_params.tma_Q_scale.get_tma_descriptor());
        cute::prefetch_tma_descriptor(&tma_params.tensor_map_q_rope);
        cute::prefetch_tma_descriptor(&tma_params.tensor_map_kv_nope);
        cute::prefetch_tma_descriptor(&tma_params.tensor_map_kv_rope);
    }

    int* gIndices = params.indices + s_q_idx*params.stride_indices_s_q; // [topk]

    TiledMMA tiled_mma_P = TiledMMA_P{};
    TiledMMA tiled_mma_O = TiledMMA_O{};
    TiledMMA tiled_mma_P_rope = TiledMMA_P_RoPE{};

    Tensor tP = partition_fragment_C(tiled_mma_P, Shape<B_TOPK, Int<B_H>>{});
    Tensor tO = partition_fragment_C(tiled_mma_O, Shape<Int<D_V>, Int<B_H>>{});

    tP.data().get() = tmem_cols::P;
    tO.data().get() = tmem_cols::O;
    tQ_scale.data().get() = tmem_cols::Q_Scale;
    tK_scale.data().get() = tmem_cols::K_Scale;

    if (warp_idx == 0) {
        if(elect_one_sync()) {
            plan.bar_prologue_q_nope.init(1);
            plan.bar_prologue_q_rope.init(1);
            plan.bar_prologue_q_scale.init(1);
            fence_barrier_init();

            // Q is stored as: e4m3 NoPE data, e8m0 block scales, then bf16 RoPE.
            cute::SM90_TMA_LOAD_3D::copy(
                &tma_params.tensor_map_q_nope,
                (uint64_t*)&plan.bar_prologue_q_nope,
                (uint64_t)TMA::CacheHintSm90::EVICT_FIRST,
                plan.qkvo.q.q_nope.data(),
                0, 0, s_q_idx
            );
            Tensor gQ_scale = tma_params.tma_Q_scale.get_tma_tensor(tma_params.shape_Q_scale)(_, _, s_q_idx);
            Tensor sQ_scale = make_tensor(make_smem_ptr(plan.s_q_rope.q_tail.q_scale.data()), SmemLayoutQScale{});
            ku::launch_tma_copy(tma_params.tma_Q_scale, gQ_scale, sQ_scale, plan.bar_prologue_q_scale, TMA::CacheHintSm90::EVICT_FIRST);
            if constexpr (HAVE_ROPE) {
                cute::SM90_TMA_LOAD_3D::copy(
                    &tma_params.tensor_map_q_rope,
                    (uint64_t*)&plan.bar_prologue_q_rope,
                    (uint64_t)TMA::CacheHintSm90::EVICT_FIRST,
                    plan.s_q_rope.q_tail.q_rope.data(),
                    0, 0, s_q_idx
                );
            }
        plan.bar_prologue_utccp_rope.init(1);
        plan.bar_prologue_utccp_nope.init(1);
        CUTE_UNROLL
        for (int i = 0; i < NUM_BUFS; ++i) {
            plan.bar_qk_nope_done[i].init(1);
            plan.bar_sv_done[i].init(1);
            plan.bar_kv_nope_ready[i].init(1);
            plan.bar_kv_scale_ready[i].init(1);
            plan.bar_k_valid_ready[i].init(B_TOPK/8); // TODO: Check the NUmber here 
            plan.bar_k_valid_free[i].init(128);
        }
        plan.bar_p_free.init(128);
        plan.bar_so_ready.init(128);
        plan.bar_qk_rope_done.init(1);
        plan.bar_kv_rope_ready.init(64);
        fence_barrier_init();
        }
        // Initialize TMEM
        cute::TMEM::Allocator1Sm().allocate(512, plan.tmem_start_addr.data());
        TRAP_ONLY_DEVICE_ASSERT(plan.tmem_start_addr.data()[0] == 0);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }

    __syncthreads();

    if (warpgroup_idx == 0) {
        // math instructions 
        if (idx_in_warpgroup < B_H) {
            plan.head_mi[idx_in_warpgroup] = MAX_INIT_VAL;
            plan.head_li[idx_in_warpgroup] = 0.0f;
            plan.head_real_mi[idx_in_warpgroup] = -CUDART_INF_F;
        }
        NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);

        // e4m3* sS_base = plan.s_q_rope.s + lane_idx*8 + (warp_idx&1)*(B_H/2)*8 + (warp_idx/2)*B_H*(B_TOPK/2);
        static constexpr int NUM_ELEMS_PER_THREAD = B_H;

        CUTE_NO_UNROLL
        for (int k = 0; k < num_k_blocks; ++k) {
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
            plan.bar_qk_nope_done[k%NUM_BUFS].wait((k/NUM_BUFS)&1);
            plan.bar_k_valid_ready[k%NUM_BUFS].wait((k/NUM_BUFS)&1);    // Put the barrier wait here for more code reordering space
            ku::tcgen05_after_thread_sync();

            // load P
            float p[NUM_ELEMS_PER_THREAD];
            ku::tmem_ld_32dp32bNx<NUM_ELEMS_PER_THREAD>(tmem_cols::P + warp_idx * NUM_ELEMS_PER_THREAD, p);
            cutlass::arch::fence_view_async_tmem_load();
            ku::tcgen05_before_thread_sync();
            slot_bar_P_empty_arrival();
            int k_row = lane_idx + (warp_idx&3);
            for (int h = 0; h < B_H; ++h) {
                plan.p_t[h + B_H * k_row] = p[h]; // How to improve the transpose efficiency here?
            }
            plan.bar_k_valid_free[k%NUM_BUFS].arrive();
        }

        if (idx_in_warpgroup < B_H) {
            int h = idx_in_warpgroup;
            Tensor sS_out = make_tensor(make_smem_ptr(plan.s), typename SmemLayoutS{});
            uint64_t valid_mask = *(uint64_t*)plan.is_k_valid[k%NUM_BUFS];
            float cur_pi_max = -CUDART_INF_F;
            int NUM_QUANT_GROUPS = B_TOPK/MXFP8_SCALE_VEC_SIZE;
            float absmax_p[NUM_QUANT_GROUPS];
            CUTE_UNROLL
            for (int g = 0; g < NUM_QUANT_GROUPS; g++) {
                absmax_p[g] = -CUDART_INF_F;
                for (int k = 0; k < 32; ++k) {
                    int kk = k + 32 * g;
                    float p_val = (valid_mask >> kk) & 1 ? plan.p_t[kk*B_H + h] : -CUDART_INF_F;
                    p_val *= params.sm_scale_div_log2;
                    absmax_p[g] = max(absmax_p[g], p_val);
                    cur_pi_max = max(cur_pi_max, p_val);
                    plan.p_t[kk*B_H + h] = p_val;
                }
            }
            
            float old_mi = plan.head_mi[h];
            float new_max = max(cur_pi_max, old_mi);
            CUTE_UNROLL
            for (int g = 0; g < NUM_QUANT_GROUPS; g++) {
                absmax_p[g] = exp2f(absmax_p[g] - new_max);
            }
            float scale_for_old = exp2f(old_mi - new_max);
            bool should_scale_o = cur_pi_max - old_mi > 6.0f;
            plan.head_scale[h] = should_scale_o ? scale_for_old : 1.0f;
            plan.head_mi[h] = new_max;
            plan.head_real_mi[h] = max(plan.head_real_mi[h], cur_pi_max);

            e8m0* s_scale[NUM_QUANT_GROUPS];
            e4m3* s_fp8[B_TOPK];

            float cur_sum = 0.0f;
            CUTE_UNROLL
            for (int g = 0; g < NUM_QUANT_GROUPS; ++g) {
                e8m0 scale_g = e8m0(absmax_p[g] / FP8_MAX); //TODO: change here to vectorized type conversion
                s_scale[g] = scale_g;
                for (int k = 0; k < 32; ++k) {
                    kk = k + g * 32;
                    float s_val = exp2f(plan.p_t[kk*B_H + h] - new_max); //TODO: change here to vectorzied load from shmem
                    cur_sum += s_val;
                    sS_out(h, kk) = e4m3(s_val / float(scale_g)); //TODO: change here to vectorized store and type conversion
                }

            }
            plan.head_li[h] = fma(plan.head_li[h], scale_for_old, cur_sum);
            if (k > 0) {
                plan.bar_sv_done[(k-1)%Kernel::NUM_BUFS].wait(((k-1)/Kernel::NUM_BUFS)&1);
            }

            if (k > 0) {
                ku::tcgen05_after_thread_sync();
                rescale_O_t<Kernel>(plan.head_scale);
                ku::tcgen05_before_thread_sync();
            }

            fence_view_async_shared();
            plan.bar_so_ready.arrive();
        }
        NamedBarrier::arrive_and_wait(128, Kernel::NamedBarriers::wg0_sync);
        if (idx_in_warpgroup < Kernel::B_H) {
            float mi = plan.head_mi[idx_in_warpgroup];
            float li = plan.head_li[idx_in_warpgroup];
            float real_mi = plan.head_real_mi[idx_in_warpgroup];
            if (real_mi == -CUDART_INF_F) {
                li = 0.0f;
                mi = -CUDART_INF_F;
            }
            int global_index = s_q_idx*params.h_q + idx_in_warpgroup;
            float cur_lse = fmaf(mi, CUDART_LN2_F, logf(li));
            cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;
            params.max_logits[global_index] = real_mi*CUDART_LN2_F;
            params.lse[global_index] = cur_lse;
        }

        plan.bar_sv_done[(num_k_blocks-1)%Kernel::NUM_BUFS].wait(((num_k_blocks-1)/Kernel::NUM_BUFS)&1);
        ku::tcgen05_after_thread_sync();

        if (idx_in_warpgroup < Kernel::B_H) {
            int h = idx_in_warpgroup;
            float attn_sink = params.attn_sink == nullptr ? -CUDART_INF_F : __ldg(params.attn_sink + h)*CUDART_L2E_F;
            float output_scale = __fdividef(1.0f, plan.head_li[h] + exp2f(attn_sink - plan.head_mi[h]));
            plan.head_scale[h] = output_scale;
        }
        NamedBarrier::arrive_and_wait(128, Kernel::NamedBarriers::wg0_sync);

        if (warp_idx < 4) {
            float o_head[Kernel::B_H_TMEM];
            int dv = warp_idx*32 + lane_idx;
            int tile = dv / 64;
            ku::tmem_ld_32dp32bNx<Kernel::B_H_TMEM>(Kernel::tmem_cols::O + tile*Kernel::B_H_TMEM, o_head);
            cutlass::arch::fence_view_async_tmem_load();
            CUTE_UNROLL
            for (int h = 0; h < Kernel::B_H; ++h) {
                plan.u.o.data()[h*Kernel::D_V + dv] = bf16(o_head[h] * plan.head_scale[h]);
            }
        }
        NamedBarrier::arrive_and_wait(128, Kernel::NamedBarriers::wg0_sync);

        // Store O using TMA
        constexpr int B_EPI = 64;
        Tensor sO = make_tensor(make_smem_ptr(plan.u.o.data()), typename Kernel::SmemLayoutO{});
        Tensor tma_gO = flat_divide(
            tma_params.tma_O.get_tma_tensor(tma_params.shape_O)(_, _, s_q_idx),
            Shape<Int<Kernel::B_H>, Int<B_EPI>>{}
        )(_, _, _0{}, _);
        Tensor sO_divided = flat_divide(
            sO,
            Shape<Int<Kernel::B_H>, Int<B_EPI>>{}
        )(_, _, _0{}, _);
        auto thr_tma = tma_params.tma_O.get_slice(_0{});

        CUTE_UNROLL
        for (int k = 0; k < Kernel::D_V/B_EPI; ++k) {
            if (warp_idx == 0 && elect_one_sync()) {
                cute::copy(
                    tma_params.tma_O,
                    thr_tma.partition_S(sO_divided(_, _, k)),
                    thr_tma.partition_D(tma_gO(_, _, k))
                );
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
                    plan.bar_prologue_utccp_nope.wait(0);   // Since q_nope coincidences with k[2]
                }

                // Copy NoPE data with gather4. Scale factors are scattered into the
                // SM100 block-scale SFA shared layout expected by tcgen05 block_scale MMA.
                int cur_buf = k%NUM_BUFS;
                plan.bar_sv_done[cur_buf].wait((k/NUM_BUFS)&1^1);

                Tensor sK_nope = make_tensor(make_smem_ptr(plan.qkvo.kv.kv_nope[cur_buf].data()), SmemLayoutKNoPE{});
                Tensor sK_scale = make_tensor(make_smem_ptr(plan.qkvo.kv.kv_nope_scale[cur_buf].data()), SmemLayoutKScale{});
                e4m3* sK_nope_base = &sK_nope(warp_idx*4, _0{});

                auto load_kv_nope = [&]() {
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                        CUTE_UNROLL
                        for (int local_col = 0; local_col < D_NOPE/64; ++local_col) {
                            ku::tma_gather4(
                                &(tma_params.tensor_map_kv_nope),
                                plan.bar_kv_nope_ready[cur_buf],
                                sK_nope_base + local_row*(4*NUM_WARPS)*64 + local_col*(B_TOPK*64),
                                local_col*64,
                                indices[local_row],
                                (int64_t)TMA::CacheHintSm90::EVICT_LAST
                            );
                        }

                        CUTE_UNROLL
                        for (int i = 0; i < 4; ++i) {
                            int src_idx = (&indices[local_row].x)[i];
                            int row = local_row*(4*NUM_WARPS) + warp_idx*4 + i;
                            e8m0 scale[K_SCALE_BYTES];
                            uint8_t* src_scale = (uint8_t*)params.kv + (int64_t)src_idx*params.stride_kv_s_kv + D_NOPE;
                            if (src_idx >= 0 && src_idx < params.s_kv) {
                                *reinterpret_cast<uint64_t*>(scale) = *reinterpret_cast<uint64_t*>(src_scale);
                            } else {
                                *reinterpret_cast<uint64_t*>(scale) = 0;
                            }
                            CUTE_UNROLL
                            for (int src_sf_idx = 0; src_sf_idx < K_SCALE_BYTES; ++src_sf_idx) {
                                CUTE_UNROLL
                                for (int dup = 0; dup < K_SCALE_DUP; ++dup) {
                                    int dst_sf_idx = src_sf_idx*K_SCALE_DUP + dup;
                                    sK_scale(row, dst_sf_idx*MXFP8_SCALE_VEC_SIZE, _0{}) = scale[src_sf_idx];
                                }
                            }
                        }
                    }
                    fence_view_async_shared();
                    plan.bar_kv_scale_ready[cur_buf].arrive();
                };

                if (!should_skip_tma) {
                    load_kv_nope();
                } else {
                    plan.bar_kv_nope_ready[cur_buf].complete_transaction(B_TOPK*D_NOPE*sizeof(e4m3));
                    plan.bar_kv_scale_ready[cur_buf].arrive();
                }
            }
        }
    } else {
        if (warp_idx == 8 && elect_one_sync()) {
            if constexpr (HAVE_ROPE) {
                Tensor sQ_rope = make_tensor(make_smem_ptr(...));
                // Copy the RoPE tile: 128 rows * 32 cols (64B) (in UTCCP's view), or 64 rows * 64 cols (in our view)
                plan.bar_prologue_q_rope.arrive_and_expect_tx(B_H*(D_Q-D_V)*sizeof(bf16));
                plan.bar_prologue_q_rope.wait(0);
            }

            Tensor sQ_nope = make_tensor(make_smem_ptr(...));
            plan.bar_prologue_q_nope.arrive_and_expect_tx(B_H*D_V*sizeof(e4m3));
            plan.bar_prologue_q_nope.wait(0);

            plan.bar_prologue_q_scale.arrive_and_expect_tx(B_H*(D_V/32)*sizeof(e4m3));
            plan.bar_prologue_q_scale.wait(0);

            Tensor sQ_scale = make_tensor(
                make_smem_ptr(plan.s_q_rope.q_tail.q_scale.data()),
                SmemLayoutQScale{}
            );

            Tensor tQ_scale = make_tensor<typename TiledMMA_P::FrgTypeSFB>(
                shape(SmemLayoutQScale{})
            );

            auto sQ_compact = make_tensor(sQ_scale.data(), filter_zeros(sQ_scale.layout()));
            auto tQ_compact = make_tensor(tQ_scale.data(), filter_zeros(tQ_scale.layout()));

            auto copy_Q_scale = make_utccp_copy(SM100_UTCCP_4x32dp128bit_1cta{}, tQ_compact);

            auto thr_Q = copy_Q_scale.get_slice(0);

            auto src_Q = get_utccp_smem_desc_tensor<SM100_UTCCP_4x32dp128bit_1cta>(
                thr_Q.partition_S(sQ_compact)
            );
            auto dst_Q = thr_Q.partition_D(tQ_compact);

            cute::copy(copy_Q_scale, src_Q, dst_Q);
            
            ku::umma_arrive_noelect(plan.bar_prologue_q_scale);

            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks+1; ++k) {
                if (k < num_k_blocks) {
                    // Pi = QKi^T
                    int cur_buf = k%NUM_BUFS;
                    Tensor sK_nope = make_tensor(make_smem_ptr(plan.u.k.k_nope[cur_buf].data()), SmemLayoutKNoPE_TiledMMA{});
                    Tensor sK_rope = make_tensor(make_smem_ptr(plan.u.k.k_rope.data()), SmemLayoutKRoPE_TiledMMA{});

                    plan.bar_p_free.wait(k&1^1);
                    ku::tcgen05_after_thread_sync();
                    
                    // Wait for K (RoPE)
                    // P = Q(rope) @ K(rope)^T
                    if constexpr (HAVE_ROPE) {
                        plan.bar_kv_rope_ready.wait(k&1);
                        ku::tcgen05_after_thread_sync();
                        ku::utcmma_ss(tiled_mma_P_rope, sK_rope, sQ_rope, tP, true);
                        ku::umma_arrive_noelect(plan.bar_qk_rope_done);
                    }

                    //TODO: How to wait for K scale ready in smem?
                    Tensor sK_scale = make_tensor(
                        make_smem_ptr(plan.qkvo.kv.kv_nope_scale[cur_buf].data()),
                        SmemLayoutKScale{}
                    );
                    Tensor tK_scale = make_tensor<typename TiledMMA_P::FrgTypeSFA>(
                        shape(SmemLayoutKScale{})
                    );
                    tK_scale.data().get() = tmem_cols::K_Scale;
                    auto sK_compact = make_tensor(sK_scale.data(), filter_zeros(sK_scale.layout()));
                    auto tK_compact = make_tensor(tK_scale.data(), filter_zeros(tK_scale.layout()));
                    auto copy_K_scale = make_utccp_copy(SM100_UTCCP_4x32dp128bit_1cta{}, tK_compact);
                    auto thr_K = copy_K_scale.get_slice(0);
                    auto src_K = get_utccp_smem_desc_tensor<SM100_UTCCP_4x32dp128bit_1cta>(
                        thr_K.partition_S(sK_compact)
                    );
                    auto dst_K = thr_K.partition_D(tK_compact);
                    cute::copy(copy_K_scale, src_K, dst_K);
                    ku::umma_arrive_noelect(plan.bar_prologue_k_scale);

                    plan.bar_kv_nope_ready[cur_buf].arrive_and_expect_tx(B_TOPK*D_V*sizeof(e4m3));
                    plan.bar_kv_nope_ready[cur_buf].wait((k/NUM_BUFS)&1);
                    ku::tcgen05_after_thread_sync();

                    // P += Q(nope) @ K(nope)^T
                    bool clear_accum = !HAVE_ROPE;
                    ku::utcmma_blockscaled_ss(
                        tiled_mma_P, sK_nope, sQ_nope, tK_scale, tQ_scale,
                        tP, clear_accum
                    );
                    
                    ku::umma_arrive_noelect(plan.bar_qk_nope_done[cur_buf]);
                }

                if (k > 0) {
                    // O += S(i-1)V(i-1)
                    int cur_buf = (k-1)%NUM_BUFS;

                    Tensor sS = make_tensor(make_smem_ptr(plan.s_q_rope.s), SmemLayoutS{});
                    Tensor sV = make_tensor(make_smem_ptr(plan.u.k.k_nope[cur_buf].data()), SmemLayoutV{});

                    // Wait for S(i-1) and O to be scaled
                    plan.bar_so_ready.wait((k-1)&1);
                    ku::tcgen05_after_thread_sync();

                    // O += sS @ sV
                    ku::utcmma_blockscaled_ss(
                        tiled_mma_O, sV, sS, tS_scale, tK_scale, tO, k == 1
                    );  //TODO: We need to produce tS_scale in tmem in warpgroup idx 0
                    ku::umma_arrive_noelect(plan.bar_sv_done[cur_buf]);
                }

            }

        } else if (warp_idx == 9) {
            // KV valid loading warp
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
            // load k rope
            if constexpr (HAVE_ROPE) {
                int thread_idx = threadIdx.x - 10*32;
                constexpr int GROUP_SIZE = 8, NUM_GROUPS = 64/GROUP_SIZE, ROWS_PER_THREAD = B_TOPK/NUM_GROUPS;
                int group_idx = thread_idx / GROUP_SIZE, idx_in_group = thread_idx % GROUP_SIZE;
                Tensor sK_rope = make_tensor(make_smem_ptr(plan.u.k.k_rope.data()), SmemLayoutKRoPE{});
                bf16* sK_rope_base = &sK_rope(group_idx, idx_in_group*8);
                CUTE_NO_UNROLL
                for (int k = 0; k < num_k_blocks; ++k) {
                    int indices[ROWS_PER_THREAD];
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < ROWS_PER_THREAD; ++local_row)
                        indices[local_row] = __ldg(gIndices + k*B_TOPK + group_idx + local_row*NUM_GROUPS);
                    plan.bar_qk_rope_done.wait(k&1^1);
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < ROWS_PER_THREAD; ++local_row) {
                        int index = indices[local_row];
                        ku::cp_async_cacheglobal<ku::PrefetchSize::B128>(
                            params.kv + (int64_t)index*params.stride_kv_s_kv + 512 + idx_in_group*8,
                            sK_rope_base + local_row*NUM_GROUPS*32,
                            index >= 0 && index < params.s_kv
                        );  // NOTE Using cp.async instead of TMA is faster here
                        // NOTE Here we only consider the range of `index` instead of also checking against topk_length, as it's noted that under this scenario (i.e. there exists a valid index among indices[topk_length: ] that points to a token who has NaN inside)
                    }
                    cutlass::arch::cpasync_barrier_arrive_noinc((uint64_t*)&(plan.bar_kv_rope_ready));
                }
            }
        }
    }


#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100");
    }
#endif
}


template<int D_QK>
void run_fwd_phase1_kernel(const MxFp8SparseAttnFwdParams& params) {
    KU_ASSERT(params.topk % B_TOPK == 0, "topk (%d) mod B_TOPK (%d) must be 0", params.topk, B_TOPK);
    KU_ASSERT(params.h_q == B_H);
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.d_qk == D_Q);
    KU_ASSERT(params.d_v == D_V);
    static_assert(D_QK == 576 || D_QK == 512);

    auto shape_O = make_shape(B_H, D_V, params.s_q, params.b);
    auto tma_O = cute::make_tma_copy(
        SM90_TMA_STORE{},
        make_tensor(
            make_gmem_ptr((bf16*)params.out),
            make_layout(
                shape_O,
                make_stride(params.stride_o_h_q, _1{}, params.stride_o_s_q, params.stride_o_b)
            )
        ),
        SmemLayoutOBuf_TMA{}
    );

    CUtensorMap tensor_map_q_nope = ku::make_tensor_map(
            {D_NOPE, (uint64_t)params.h_q, (uint64_t)params.s_q},
            ku::make_stride_helper(std::vector<int64_t>{params.stride_q_h_q, params.stride_q_s_q}, sizeof(uint8_t)),
            {D_NOPE, B_H, 1},
            params.q,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B
        );

    auto shape_Q_scale = make_shape(B_H, D_NOPE / 32, params.s_q);
    auto tma_Q_scale = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((e8m0*)((uint8_t*)params.q + D_NOPE)),
            make_layout(
                shape_Q_scale,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q)
            )
        ),
        SmemLayoutQScale{}
    );

    CUtensorMap tensor_map_q_rope = ku::make_tensor_map(
            {D_ROPE, (uint64_t)params.h_q, (uint64_t)params.s_q},
            ku::make_stride_helper(std::vector<int64_t>{params.stride_q_h_q, params.stride_q_s_q}, sizeof(uint8_t)),
            {D_ROPE, B_H, 1},
            (uint8_t*)params.q + D_NOPE + Q_SCALE_BYTES,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B
        );

    CUtensorMap tensor_map_kv_rope = ku::make_tensor_map(
            {D_ROPE, (uint64_t)params.h_kv, D_ROPE / 32, (uint64_t)params.kv_s_kv},
            ku::make_stride_helper(std::vector<int64_t>{params.stride_kv_h_kv, (int64_t)32, params.stride_kv_s_kv}, sizeof(bf16)),
            {32, D_ROPE / 32, 1},
            (uint8_t*)params.kv + D_NOPE + K_SCALE_BYTES, // K NoPE uses one 8-bit scale per 64 elements
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B
        );

    CUtensorMap tensor_map_kv_nope = ku::make_tensor_map(
            {D_NOPE / 8, (uint64_t)params.h_kv, (uint64_t)params.kv_s_kv},
            ku::make_stride_helper(std::vector<int64_t>{params.stride_kv_h_kv, params.stride_kv_s_kv}, sizeof(uint64_t)),
            {D_NOPE / 8, (uint64_t)params.h_kv, 1},
            params.kv,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT64,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B
        );

}

}