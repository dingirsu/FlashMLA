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

namespace sm100::fwd::head_small {

using namespace cute;

template<typename Kernel>
CUTE_DEVICE
void rescale_O_t(float scale[Kernel::B_H]) {
    float o[Kernel::B_H_TMEM];
    CUTE_UNROLL
    for (int tile = 0; tile < Kernel::D_V/64; ++tile) {
        ku::tmem_ld_32dp32bNx<Kernel::B_H_TMEM>(Kernel::tmem_cols::O + tile*Kernel::B_H_TMEM, o);
        cutlass::arch::fence_view_async_tmem_load();
        CUTE_UNROLL
        for (int i = 0; i < Kernel::B_H; ++i) {
            o[i] *= scale[i];
        }
        ku::tmem_st_32dp32bNx<Kernel::B_H_TMEM>(Kernel::tmem_cols::O + tile*Kernel::B_H_TMEM, o);
        cutlass::arch::fence_view_async_tmem_store();
    }
}

/*
Pipeline Overview:

| Copy |    MMA    |   Scale & Exp   |

KV0
KV1
KV2
        P0 = QK0^T
                    S0 = exp(P0)
                    scale(O) w.r.t P0
        P1 = QK1^T
                    S1 = exp(P1)
        O += S0V0
KV3                 scale(O) w.r.t P1
        P2 = QK2^T
                    S2 = exp(P2)
        O += S1V1
KV4                 scale(O) w.r.t P2
        P3 = QK3^T
                    S3 = exp(P3)
        O += S2V2
KV5                 scale(O) w.r.t P3

...

        O += S(n-3)V(n-3)
                    scale(O) w.r.t P(n-2)
        P(n-1) = QK(n-1)^T
                   S(n-1) = exp(P(n-1))
        O += S(n-2)V(n-2)
                   scale(O) w.r.t P(n-1)
        O += S(n-1)V(n-1)
*/

using FwdMode = SparseAttnFwdMode;

template<int D_QK, int H_Q, bool HAVE_ROPE, typename TmaParams>
__global__ void __launch_bounds__(KernelTemplate<D_QK, H_Q>::NUM_THREADS, 1, 1)
sparse_attn_fwd_kernel(__grid_constant__ const SparseAttnFwdParams params, __grid_constant__ const TmaParams tma_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
    using Kernel = KernelTemplate<D_QK, H_Q>;
    // Grid shape: [s_q, 1, 1]

    const int s_q_idx = blockIdx.x;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = threadIdx.x % 32;
    const int warpgroup_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    const int idx_in_warpgroup = threadIdx.x % 128;
    const int topk_length = params.topk_length != nullptr ? __ldg(params.topk_length + s_q_idx) : params.topk;
    const int num_k_blocks = max(cute::ceil_div(topk_length, (int)Kernel::B_TOPK), 1);  // num_k_blocks always >= 1

    // Define shared tensors
    extern __shared__ char wksp_buf[];
    typename Kernel::SharedMemoryPlan &plan = *reinterpret_cast<typename Kernel::SharedMemoryPlan*>(wksp_buf);

    int* gIndices = params.indices + s_q_idx*params.stride_indices_s_q; // [topk]

    // Allocate tmem tensors
    typename Kernel::TiledMMA_P tiled_mma_P = typename Kernel::TiledMMA_P{};
    typename Kernel::TiledMMA_O tiled_mma_O = typename Kernel::TiledMMA_O{};
    Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<Kernel::B_TOPK>, Int<Kernel::B_H>>{});
    Tensor tO = partition_fragment_C(tiled_mma_O, Shape<Int<64>, Int<Kernel::B_H>>{});
    tP.data().get() = Kernel::tmem_cols::P;
    tO.data().get() = Kernel::tmem_cols::O;

    if (warp_idx == 0) {
        if (elect_one_sync()) {
            plan.bar_prologue_q_nope.init(1);
            plan.bar_prologue_q_rope.init(1);
            fence_barrier_init();

            cute::prefetch_tma_descriptor(tma_params.tma_O.get_tma_descriptor());
            cute::prefetch_tma_descriptor(&(tma_params.tensor_map_kv_nope));
            
            // Initialize other barriers
            plan.bar_prologue_utccp_rope.init(1);
            plan.bar_prologue_utccp_nope.init(1);
            CUTE_UNROLL
            for (int i = 0; i < Kernel::NUM_BUFS; ++i) {
                plan.bar_qk_nope_done[i].init(1);
                plan.bar_sv_done[i].init(1);
                plan.bar_kv_nope_ready[i].init(1);
                plan.bar_k_valid_ready[i].init(Kernel::B_TOPK/8);
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

    {
        Tensor sQ_nope = make_tensor(make_smem_ptr(plan.u.q_full.q_nope.data()), typename Kernel::SmemLayoutQNoPE{});
        for (int i = threadIdx.x; i < Kernel::B_H*Kernel::D_V; i += Kernel::NUM_THREADS) {
            int h = i / Kernel::D_V;
            int d = i % Kernel::D_V;
            sQ_nope(h, d) = params.q[(int64_t)s_q_idx*params.stride_q_s_q + h*params.stride_q_h_q + d];
        }
        if constexpr (HAVE_ROPE) {
            Tensor sQ_rope = make_tensor(make_smem_ptr(plan.q_rope.data()), typename Kernel::SmemLayoutQRoPE{});
            for (int i = threadIdx.x; i < Kernel::B_H*(Kernel::D_Q-Kernel::D_V); i += Kernel::NUM_THREADS) {
                int h = i / (Kernel::D_Q-Kernel::D_V);
                int d = i % (Kernel::D_Q-Kernel::D_V);
                sQ_rope(h, d) = params.q[(int64_t)s_q_idx*params.stride_q_s_q + h*params.stride_q_h_q + Kernel::D_V + d];
            }
        }
    }

    __syncthreads();

    if (warpgroup_idx == 0) {
        // Scale & Exp warps for transposed P/S: P_t and S_t are [topk, head].
        static_assert(Kernel::B_TOPK == 64);
        if (idx_in_warpgroup < Kernel::B_H) {
            plan.head_mi[idx_in_warpgroup] = Kernel::MAX_INIT_VAL;
            plan.head_li[idx_in_warpgroup] = 0.0f;
            plan.head_real_mi[idx_in_warpgroup] = -CUDART_INF_F;
        }
        NamedBarrier::arrive_and_wait(128, Kernel::NamedBarriers::wg0_sync);

        CUTE_NO_UNROLL
        for (int k = 0; k < num_k_blocks; ++k) {
            plan.bar_qk_nope_done[k%Kernel::NUM_BUFS].wait((k/Kernel::NUM_BUFS)&1);
            plan.bar_k_valid_ready[k%Kernel::NUM_BUFS].wait((k/Kernel::NUM_BUFS)&1);
            ku::tcgen05_after_thread_sync();

            if (warp_idx < 2) {
                float p_head[Kernel::B_H_TMEM];
                ku::tmem_ld_32dp32bNx<Kernel::B_H_TMEM>(Kernel::tmem_cols::P, p_head);
                cutlass::arch::fence_view_async_tmem_load();
                int k_row = (warp_idx&1)*32 + lane_idx;
                CUTE_UNROLL
                for (int h = 0; h < Kernel::B_H; ++h) {
                    plan.p_t[k_row*Kernel::B_H + h] = p_head[h];
                }
            }
            ku::tcgen05_before_thread_sync();
            plan.bar_p_free.arrive();
            NamedBarrier::arrive_and_wait(128, Kernel::NamedBarriers::wg0_sync);
            plan.bar_k_valid_free[k%Kernel::NUM_BUFS].arrive();

            if (idx_in_warpgroup < Kernel::B_H) {
                int h = idx_in_warpgroup;
                Tensor sS_out = make_tensor(make_smem_ptr(plan.s), typename Kernel::SmemLayoutS{});
                uint64_t valid_mask = *(uint64_t*)plan.is_k_valid[k%Kernel::NUM_BUFS];
                float cur_pi_max = -CUDART_INF_F;
                CUTE_UNROLL
                for (int kk = 0; kk < Kernel::B_TOPK; ++kk) {
                    float p_val = (valid_mask >> kk) & 1 ? plan.p_t[kk*Kernel::B_H + h] : -CUDART_INF_F;
                    p_val *= params.sm_scale_div_log2;
                    cur_pi_max = max(cur_pi_max, p_val);
                    plan.p_t[kk*Kernel::B_H + h] = p_val;
                }

                float old_mi = plan.head_mi[h];
                float new_max = max(cur_pi_max, old_mi);
                float scale_for_old = exp2f(old_mi - new_max);
                bool should_scale_o = cur_pi_max - old_mi > 6.0f;
                plan.head_scale[h] = should_scale_o ? scale_for_old : 1.0f;
                plan.head_mi[h] = new_max;
                plan.head_real_mi[h] = max(plan.head_real_mi[h], cur_pi_max);

                float cur_sum = 0.0f;
                CUTE_UNROLL
                for (int kk = 0; kk < Kernel::B_TOPK; ++kk) {
                    float s_val = exp2f(plan.p_t[kk*Kernel::B_H + h] - new_max);
                    cur_sum += s_val;
                    sS_out(h, kk) = bf16(s_val);
                }
                plan.head_li[h] = fma(plan.head_li[h], scale_for_old, cur_sum);
            }
            NamedBarrier::arrive_and_wait(128, Kernel::NamedBarriers::wg0_sync);

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
        for (int i = idx_in_warpgroup; i < Kernel::B_H*Kernel::D_V; i += 128) {
            int h = i / Kernel::D_V;
            int dv = i % Kernel::D_V;
            params.out[(int64_t)s_q_idx*Kernel::B_H*Kernel::D_V + h*Kernel::D_V + dv] = plan.u.o.data()[h*Kernel::D_V + dv];
        }

        if (warp_idx == 0) {
            cute::TMEM::Allocator1Sm().free(0, 512);
        }
    } else if (warpgroup_idx == 1) {
        // Producer warp for KV
        int warp_idx = cutlass::canonical_warp_idx_sync() - 4;
        constexpr int NUM_WARPS = 4, NUM_LOCAL_ROWS_PER_WARP = (Kernel::B_TOPK/4)/NUM_WARPS;
        if (elect_one_sync()) {
            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks; ++k) {
                int4 indices[NUM_LOCAL_ROWS_PER_WARP];
                int max_indices = -1, min_indices = params.s_kv;
                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                    indices[local_row] = __ldg((int4*)(gIndices + k*Kernel::B_TOPK) + local_row*NUM_WARPS + warp_idx);
                    max_indices = max(max_indices, int4_max(indices[local_row]));
                    min_indices = min(min_indices, int4_min(indices[local_row]));
                }
                bool is_all_rows_invalid = min_indices == params.s_kv || max_indices == -1;
                bool should_skip_tma = is_all_rows_invalid && k >= Kernel::NUM_BUFS;

                if (k == 2) {
                    plan.bar_prologue_utccp_nope.wait(0);   // Since q_nope coincidences with k[2]
                }

                // Copy NoPE
                int cur_buf = k%Kernel::NUM_BUFS;
                plan.bar_sv_done[cur_buf].wait((k/Kernel::NUM_BUFS)&1^1);
                bf16* sK_nope_base = plan.u.k.k_nope[cur_buf].data() + warp_idx*4*64;

                if (!should_skip_tma) {
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                        CUTE_UNROLL
                        for (int local_col = 0; local_col < Kernel::D_V/64; ++local_col) {
                            ku::tma_gather4(
                                &(tma_params.tensor_map_kv_nope),
                                plan.bar_kv_nope_ready[cur_buf],
                                sK_nope_base + local_row*(4*NUM_WARPS)*64 + local_col*(Kernel::B_TOPK*64),
                                local_col*64,
                                indices[local_row],
                                (int64_t)TMA::CacheHintSm90::EVICT_LAST
                            );
                        }
                    }
                } else {
                    // NOTE See head128/phase1.cuh for this TMA skipping technique
                    plan.bar_kv_nope_ready[cur_buf].complete_transaction(NUM_LOCAL_ROWS_PER_WARP*4*Kernel::D_V*sizeof(bf16));
                }
            }
        }
    } else {
        // MMA warp
        if (warp_idx == 8 && elect_one_sync()) {
            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks+1; ++k) {
                if (k < num_k_blocks) {
                    // P_t = K @ Q^T, stored as [topk, head].
                    int cur_buf = k%Kernel::NUM_BUFS;
                    Tensor sK_nope = make_tensor(make_smem_ptr(plan.u.k.k_nope[cur_buf].data()), typename Kernel::SmemLayoutKNoPE{});
                    Tensor sQ_nope = make_tensor(make_smem_ptr(plan.u.q_full.q_nope.data()), typename Kernel::SmemLayoutQNoPE{});
                    Tensor sK_rope = make_tensor(make_smem_ptr(plan.u.k.k_rope.data()), typename Kernel::SmemLayoutKRoPE{});
                    Tensor sQ_rope = make_tensor(make_smem_ptr(plan.q_rope.data()), typename Kernel::SmemLayoutQRoPE{});

                    plan.bar_p_free.wait(k&1^1);
                    ku::tcgen05_after_thread_sync();

                    if constexpr (HAVE_ROPE) {
                        plan.bar_kv_rope_ready.wait(k&1);
                        ku::tcgen05_after_thread_sync();
                        ku::utcmma_ss(tiled_mma_P, sK_rope, sQ_rope, tP, true);
                        ku::umma_arrive_noelect(plan.bar_qk_rope_done);
                    }

                    plan.bar_kv_nope_ready[cur_buf].arrive_and_expect_tx(Kernel::B_TOPK*Kernel::D_V*sizeof(bf16));
                    plan.bar_kv_nope_ready[cur_buf].wait((k/Kernel::NUM_BUFS)&1);
                    ku::tcgen05_after_thread_sync();

                    ku::utcmma_ss(tiled_mma_P, sK_nope, sQ_nope, tP, !HAVE_ROPE);
                    ku::umma_arrive_noelect(plan.bar_qk_nope_done[cur_buf]);
                }
                if (k > 0) {
                    // O_t += V^T @ S_t, stored as [d_v, head].
                    int cur_buf = (k-1)%Kernel::NUM_BUFS;

                    Tensor sS = make_tensor(make_smem_ptr(plan.s), typename Kernel::SmemLayoutS{});
                    Tensor sV = make_tensor(make_smem_ptr(plan.u.k.k_nope[cur_buf].data()), typename Kernel::SmemLayoutV{});
                    Tensor sV_divided = flat_divide(sV, Tile<Int<64>, Int<Kernel::B_TOPK>>{})(_, _, _0{}, _);

                    plan.bar_so_ready.wait((k-1)&1);
                    ku::tcgen05_after_thread_sync();

                    CUTE_UNROLL
                    for (int tile = 0; tile < Kernel::D_V/64; ++tile) {
                        tO.data().get() = Kernel::tmem_cols::O + tile*Kernel::B_H_TMEM;
                        ku::utcmma_ss(tiled_mma_O, sV_divided(_, _, tile), sS, tO, k == 1);
                    }
                    ku::umma_arrive_noelect(plan.bar_sv_done[cur_buf]);
                }
            }
        } else if (warp_idx == 9) {
            // KV valid loading warp
            if (lane_idx < Kernel::B_TOPK/8) {
                CUTE_NO_UNROLL
                for (int k = 0; k < num_k_blocks; ++k) {
                    char k_validness_mask = load_indices_and_generate_mask(
                        lane_idx,
                        gIndices + k*Kernel::B_TOPK,
                        params.s_kv,
                        k*Kernel::B_TOPK,
                        topk_length
                    );

                    int cur_buf = k%Kernel::NUM_BUFS;
                    plan.bar_k_valid_free[cur_buf].wait((k/Kernel::NUM_BUFS)&1^1);
                    plan.is_k_valid[cur_buf][lane_idx] = k_validness_mask;
                    plan.bar_k_valid_ready[cur_buf].arrive();
                }
            }
        } else if (warp_idx == 10 || warp_idx == 11) {
            if constexpr (HAVE_ROPE) {
                int thread_idx = threadIdx.x - 10*32;
                constexpr int GROUP_SIZE = 8, NUM_GROUPS = 64/GROUP_SIZE, ROWS_PER_THREAD = Kernel::B_TOPK/NUM_GROUPS;
                int group_idx = thread_idx / GROUP_SIZE, idx_in_group = thread_idx % GROUP_SIZE;
                Tensor sK_rope = make_tensor(make_smem_ptr(plan.u.k.k_rope.data()), typename Kernel::SmemLayoutKRoPE{});
                bf16* sK_rope_base = &sK_rope(group_idx, idx_in_group*8);
                CUTE_NO_UNROLL
                for (int k = 0; k < num_k_blocks; ++k) {
                    int indices[ROWS_PER_THREAD];
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < ROWS_PER_THREAD; ++local_row)
                        indices[local_row] = __ldg(gIndices + k*Kernel::B_TOPK + group_idx + local_row*NUM_GROUPS);
                    plan.bar_qk_rope_done.wait(k&1^1);
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < ROWS_PER_THREAD; ++local_row) {
                        int index = indices[local_row];
                        ku::cp_async_cacheglobal<ku::PrefetchSize::B128>(
                            params.kv + (int64_t)index*params.stride_kv_s_kv + Kernel::D_V + idx_in_group*8,
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

template<int D_QK, int H_Q>
void run_fwd_phase1_kernel(const SparseAttnFwdParams& params) {
    using Kernel = KernelTemplate<D_QK, H_Q>;
    static_assert(D_QK == 128 || D_QK == 192);

    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk % Kernel::B_TOPK == 0);   // To save some boundry checkings
    KU_ASSERT(params.h_q == H_Q);
    KU_ASSERT(params.d_qk == D_QK);
    KU_ASSERT(params.d_v == Kernel::D_V);

    auto shape_O = make_shape(params.h_q, params.d_v, params.s_q);
    auto tma_O = cute::make_tma_copy(
        SM90_TMA_STORE{},
        make_tensor(
            make_gmem_ptr((bf16*)params.out),
            make_layout(
                shape_O,
                make_stride(params.d_v, _1{}, params.h_q*params.d_v)
            )
        ),
        typename Kernel::template SmemLayoutOTiles<1>{}
    );


    CUtensorMap tensor_map_kv_nope;
    {
        uint64_t size[2] = {Kernel::D_V, (unsigned long)params.s_kv};
        uint64_t stride[1] = {params.stride_kv_s_kv*sizeof(bf16)};
        uint32_t box_size[2] = {64, 1};
        uint32_t elem_stride[2] = {1, 1};
        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &tensor_map_kv_nope,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            params.kv,
            size,
            stride,
            box_size,
            elem_stride,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        KU_ASSERT(res == CUresult::CUDA_SUCCESS);
    }

    TmaParams<
        decltype(shape_O), decltype(tma_O)
    > tma_params = {
        shape_O, tma_O,
        tensor_map_kv_nope
    };
    auto kernel = &sparse_attn_fwd_kernel<D_QK, H_Q, D_QK == 192, decltype(tma_params)>;

    constexpr size_t smem_size = sizeof(typename Kernel::SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    kernel<<<params.s_q, Kernel::NUM_THREADS, smem_size, params.stream>>>(params, tma_params);
    KU_CHECK_KERNEL_LAUNCH();
}

}
