#pragma once

#include "phase1.h"

#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cuda_host_adapter.hpp>
#include <kerutils/kerutils.cuh>

#include "config.h"
#include "params.h"
#include "sm100/helpers.h"
#include "utils.h"

namespace sm100::bwd::head_small {

using namespace cute;

CUTE_DEVICE
float token_probability_mass(float lse, float sink, bool have_valid_token) {
    if (!have_valid_token || sink == CUDART_INF_F) {
        return 0.0f;
    }
    if (sink == -CUDART_INF_F) {
        return 1.0f;
    }
    float delta = sink - lse;
    if (delta >= 0.0f) {
        float z = expf(-delta);
        return z / (1.0f + z);
    }
    return 1.0f / (1.0f + expf(delta));
}

template<int D_QK, int H_Q, typename TmaParamsT>
__global__ void __launch_bounds__(KernelTemplate<D_QK, H_Q>::NUM_THREADS, 1, 1)
sparse_attn_bwd_kernel(
    __grid_constant__ const SparseAttnBwdParams params,
    __grid_constant__ const TmaParamsT tma_params
) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200)) || defined(__CLION_IDE__) || defined(__VSCODE_IDE__)
    using Kernel = KernelTemplate<D_QK, H_Q>;
    static_assert(D_QK == 128);

    const int s_q_idx = blockIdx.x;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = threadIdx.x % 32;
    const int warpgroup_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    const int idx_in_warpgroup = threadIdx.x % 128;
    const int requested_topk = params.topk_length == nullptr
        ? params.topk
        : __ldg(params.topk_length + s_q_idx);
    const int valid_length = max(0, min(requested_topk, params.topk));
    const int num_k_blocks = max(
        cute::ceil_div(valid_length, static_cast<int>(Kernel::B_TOPK)), 1
    );
    int* g_indices = params.indices + s_q_idx * params.stride_indices_s_q;

    extern __shared__ char workspace[];
    typename Kernel::SharedMemoryPlan& plan =
        *reinterpret_cast<typename Kernel::SharedMemoryPlan*>(workspace);

    typename Kernel::TiledMMAQK tiled_mma_qk{};
    typename Kernel::TiledMMADKV tiled_mma_dkv{};
    typename Kernel::TiledMMADQ tiled_mma_dq{};
    Tensor t_score = partition_fragment_C(
        tiled_mma_qk, Shape<Int<Kernel::B_TOPK>, Int<Kernel::B_H>>{}
    );
    Tensor t_dv = partition_fragment_C(
        tiled_mma_dkv, Shape<Int<Kernel::B_TOPK>, Int<64>>{}
    );
    Tensor t_dk = partition_fragment_C(
        tiled_mma_dkv, Shape<Int<Kernel::B_TOPK>, Int<64>>{}
    );
    Tensor t_dq = partition_fragment_C(
        tiled_mma_dq, Shape<Int<64>, Int<Kernel::B_H>>{}
    );
    t_score.data().get() = Kernel::tmem_cols::SCORE;
    t_dv.data().get() = Kernel::tmem_cols::DV;
    t_dk.data().get() = Kernel::tmem_cols::DK;
    t_dq.data().get() = Kernel::tmem_cols::DQ;

    if (warp_idx == 0) {
        if (elect_one_sync()) {
            cute::prefetch_tma_descriptor(tma_params.tma_q.get_tma_descriptor());
            cute::prefetch_tma_descriptor(tma_params.tma_o.get_tma_descriptor());
            cute::prefetch_tma_descriptor(tma_params.tma_do.get_tma_descriptor());
            cute::prefetch_tma_descriptor(&tma_params.tensor_map_kv);

            plan.bar_q_ready.init(1);
            plan.bar_o_ready.init(1);
            plan.bar_do_ready.init(1);
            plan.bar_operands_ready.init(128);
            CUTE_UNROLL
            for (int stage = 0; stage < Kernel::NUM_BUFS; ++stage) {
                plan.bar_kv_ready[stage].init(1);
                plan.bar_kv_free[stage].init(1);
                plan.bar_dkv_ready[stage].init(1);
            }
            plan.bar_score_ready.init(1);
            plan.bar_score_free.init(1);
            plan.bar_ds_ready.init(1);
            plan.bar_dkv_free.init(1);
            fence_barrier_init();
        }

        cute::TMEM::Allocator1Sm().allocate(512, plan.tmem_start_addr.data());
        TRAP_ONLY_DEVICE_ASSERT(plan.tmem_start_addr.data()[0] == 0);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }
    __syncthreads();

    if (warp_idx == 0 && elect_one_sync()) {
        Tensor g_q = tma_params.tma_q.get_tma_tensor(tma_params.shape_q)(_, _, s_q_idx);
        Tensor s_q = make_tensor(make_smem_ptr(plan.q.data()), typename Kernel::SmemLayoutHeadDim{});
        plan.bar_q_ready.arrive_and_expect_tx(Kernel::B_H * Kernel::D_Q * sizeof(bf16));
        ku::launch_tma_copy(
            tma_params.tma_q, g_q, s_q, plan.bar_q_ready, TMA::CacheHintSm90::EVICT_FIRST
        );

        Tensor g_o = tma_params.tma_o.get_tma_tensor(tma_params.shape_o)(_, _, s_q_idx);
        Tensor s_o = make_tensor(make_smem_ptr(plan.o.data()), typename Kernel::SmemLayoutHeadDim{});
        plan.bar_o_ready.arrive_and_expect_tx(Kernel::B_H * Kernel::D_V * sizeof(bf16));
        ku::launch_tma_copy(
            tma_params.tma_o, g_o, s_o, plan.bar_o_ready, TMA::CacheHintSm90::EVICT_FIRST
        );

        Tensor g_do = tma_params.tma_do.get_tma_tensor(tma_params.shape_do)(_, _, s_q_idx);
        Tensor s_do = make_tensor(make_smem_ptr(plan.d_o.data()), typename Kernel::SmemLayoutHeadDim{});
        plan.bar_do_ready.arrive_and_expect_tx(Kernel::B_H * Kernel::D_V * sizeof(bf16));
        ku::launch_tma_copy(
            tma_params.tma_do, g_do, s_do, plan.bar_do_ready, TMA::CacheHintSm90::EVICT_FIRST
        );

        plan.bar_q_ready.wait(0);
        plan.bar_o_ready.wait(0);
        plan.bar_do_ready.wait(0);
    }
    __syncthreads();

    if (warpgroup_idx == 0) {
        Tensor s_q = make_tensor(make_smem_ptr(plan.q.data()), typename Kernel::SmemLayoutHeadDim{});
        Tensor s_o = make_tensor(make_smem_ptr(plan.o.data()), typename Kernel::SmemLayoutHeadDim{});
        Tensor s_do = make_tensor(make_smem_ptr(plan.d_o.data()), typename Kernel::SmemLayoutHeadDim{});
        Tensor s_qt = make_tensor(make_smem_ptr(plan.q_t.data()), typename Kernel::SmemLayoutDimHead{});
        Tensor s_dot = make_tensor(make_smem_ptr(plan.d_o_t.data()), typename Kernel::SmemLayoutDimHead{});

        for (int i = idx_in_warpgroup; i < cosize_v<typename Kernel::SmemLayoutDimHead>; i += 128) {
            plan.q_t.data()[i] = bf16(0.0f);
            plan.d_o_t.data()[i] = bf16(0.0f);
        }
        if (idx_in_warpgroup < Kernel::B_H) {
            plan.sum_odo[idx_in_warpgroup] = 0.0f;
            plan.lse[idx_in_warpgroup] = __ldg(
                params.lse + s_q_idx * params.stride_lse_s_q + idx_in_warpgroup
            );
        }
        if (idx_in_warpgroup == 0) {
            plan.valid_length = valid_length;
            int any_valid = 0;
            for (int k = 0; k < valid_length; ++k) {
                int index = __ldg(g_indices + k);
                any_valid |= index >= 0 && index < params.s_kv;
            }
            plan.have_valid_token = any_valid;
        }
        NamedBarrier::arrive_and_wait(128, Kernel::NamedBarriers::WG0_SYNC);

        for (int linear = idx_in_warpgroup;
             linear < Kernel::B_H * Kernel::D_Q;
             linear += 128) {
            int h = linear / Kernel::D_Q;
            int d = linear % Kernel::D_Q;
            s_qt(d, h) = s_q(h, d);
            s_dot(d, h) = s_do(h, d);
            atomicAdd(
                plan.sum_odo + h,
                static_cast<float>(s_o(h, d)) * static_cast<float>(s_do(h, d))
            );
        }
        NamedBarrier::arrive_and_wait(128, Kernel::NamedBarriers::WG0_SYNC);

        if (idx_in_warpgroup < Kernel::B_H) {
            int h = idx_in_warpgroup;
            float sink = params.attn_sink == nullptr
                ? -CUDART_INF_F
                : __ldg(params.attn_sink + h);
            float mass = token_probability_mass(
                plan.lse[h], sink, plan.have_valid_token != 0
            );
            plan.token_mass[h] = mass;
            if (params.d_attn_sink != nullptr && plan.have_valid_token) {
                atomicAdd(params.d_attn_sink + h, -(1.0f - mass) * plan.sum_odo[h]);
            }
        }
        fence_view_async_shared();
        plan.bar_operands_ready.arrive();

        CUTE_NO_UNROLL
        for (int k = 0; k < num_k_blocks; ++k) {
            int stage = k % Kernel::NUM_BUFS;

            plan.bar_score_ready.wait((2 * k) & 1);
            ku::tcgen05_after_thread_sync();
            if (warp_idx < 2) {
                int row = warp_idx * 32 + lane_idx;
                float values[Kernel::B_H_TMEM];
                ku::tmem_ld_32dp32bNx<Kernel::B_H_TMEM>(Kernel::tmem_cols::SCORE, values);
                cutlass::arch::fence_view_async_tmem_load();
                CUTE_UNROLL
                for (int h = 0; h < Kernel::B_H; ++h) {
                    plan.p[row * Kernel::B_H + h] = values[h];
                }
            }
            NamedBarrier::arrive_and_wait(128, Kernel::NamedBarriers::WG0_SYNC);
            ku::tcgen05_before_thread_sync();
            if (idx_in_warpgroup == 0) {
                plan.bar_score_free.arrive();
            }

            plan.bar_score_ready.wait((2 * k + 1) & 1);
            ku::tcgen05_after_thread_sync();
            if (warp_idx < 2) {
                int row = warp_idx * 32 + lane_idx;
                float values[Kernel::B_H_TMEM];
                ku::tmem_ld_32dp32bNx<Kernel::B_H_TMEM>(Kernel::tmem_cols::SCORE, values);
                cutlass::arch::fence_view_async_tmem_load();
                CUTE_UNROLL
                for (int h = 0; h < Kernel::B_H; ++h) {
                    plan.dp[row * Kernel::B_H + h] = values[h];
                }
            }
            NamedBarrier::arrive_and_wait(128, Kernel::NamedBarriers::WG0_SYNC);
            ku::tcgen05_before_thread_sync();
            if (idx_in_warpgroup == 0) {
                plan.bar_score_free.arrive();
            }

            Tensor s_prob = make_tensor(
                make_smem_ptr(plan.prob.data()), typename Kernel::SmemLayoutProb{}
            );
            Tensor s_ds_dk = make_tensor(
                make_smem_ptr(plan.ds_for_dk.data()), typename Kernel::SmemLayoutProb{}
            );
            Tensor s_ds_dq = make_tensor(
                make_smem_ptr(plan.ds_for_dq.data()), typename Kernel::SmemLayoutDSForDQ{}
            );
            if (idx_in_warpgroup < Kernel::B_TOPK) {
                int row = idx_in_warpgroup;
                bool valid = plan.valid[stage][row] != 0;
                CUTE_UNROLL
                for (int h = 0; h < Kernel::B_H_MMA; ++h) {
                    float probability = 0.0f;
                    float ds = 0.0f;
                    if (valid && h < Kernel::B_H) {
                        probability = expf(
                            plan.p[row * Kernel::B_H + h] * params.sm_scale - plan.lse[h]
                        ) * plan.token_mass[h];
                        ds = probability *
                            (plan.dp[row * Kernel::B_H + h] - plan.sum_odo[h]) *
                            params.sm_scale;
                    }
                    s_prob(row, h) = bf16(probability);
                    s_ds_dk(row, h) = bf16(ds);
                    if (h < Kernel::B_H) {
                        s_ds_dq(h, row) = bf16(ds);
                    }
                }
            }
            NamedBarrier::arrive_and_wait(128, Kernel::NamedBarriers::WG0_SYNC);
            fence_view_async_shared();
            if (idx_in_warpgroup == 0) {
                plan.bar_ds_ready.arrive();
            }

            plan.bar_dkv_ready[stage].wait((k / Kernel::NUM_BUFS) & 1);
            ku::tcgen05_after_thread_sync();
            if (warp_idx < 2) {
                int row = warp_idx * 32 + lane_idx;
                int kv_index = plan.indices[stage][row];
                bool valid = plan.valid[stage][row] != 0;
                CUTE_UNROLL
                for (int tile = 0; tile < 2; ++tile) {
                    CUTE_UNROLL
                    for (int half = 0; half < 2; ++half) {
                        float values[32];
                        int col = tile * 64 + half * 32;
                        ku::tmem_ld_32dp32bNx<32>(Kernel::tmem_cols::DV + col, values);
                        cutlass::arch::fence_view_async_tmem_load();
                        if (valid) {
                            int64_t base = static_cast<int64_t>(kv_index) *
                                params.stride_d_v_acc_s_kv;
                            CUTE_UNROLL
                            for (int j = 0; j < 32; ++j) {
                                atomicAdd(params.d_v_acc + base + col + j, values[j]);
                            }
                        }
                        ku::tmem_ld_32dp32bNx<32>(Kernel::tmem_cols::DK + col, values);
                        cutlass::arch::fence_view_async_tmem_load();
                        if (valid) {
                            int64_t base = static_cast<int64_t>(kv_index) *
                                params.stride_d_k_acc_s_kv;
                            CUTE_UNROLL
                            for (int j = 0; j < 32; ++j) {
                                atomicAdd(params.d_k_acc + base + col + j, values[j]);
                            }
                        }
                    }
                }
            }
            NamedBarrier::arrive_and_wait(128, Kernel::NamedBarriers::WG0_SYNC);
            ku::tcgen05_before_thread_sync();
            if (idx_in_warpgroup == 0) {
                plan.bar_dkv_free.arrive();
                plan.bar_kv_free[stage].arrive();
            }
        }

        if (warp_idx < 2) {
            int row = warp_idx * 32 + lane_idx;
            CUTE_UNROLL
            for (int tile = 0; tile < 2; ++tile) {
                float values[Kernel::B_H_TMEM];
                ku::tmem_ld_32dp32bNx<Kernel::B_H_TMEM>(
                    Kernel::tmem_cols::DQ + tile * Kernel::B_H_TMEM, values
                );
                cutlass::arch::fence_view_async_tmem_load();
                CUTE_UNROLL
                for (int h = 0; h < Kernel::B_H; ++h) {
                    int64_t offset = static_cast<int64_t>(s_q_idx) * params.stride_d_q_s_q +
                        static_cast<int64_t>(h) * params.stride_d_q_h_q + tile * 64 + row;
                    params.dq[offset] = bf16(values[h]);
                }
            }
        }
        NamedBarrier::arrive_and_wait(128, Kernel::NamedBarriers::WG0_SYNC);
        if (warp_idx == 0) {
            cute::TMEM::Allocator1Sm().free(0, 512);
        }
    } else if (warpgroup_idx == 1) {
        int local_warp = warp_idx - 4;
        constexpr int NUM_WARPS = 4;
        constexpr int ROW_GROUPS_PER_WARP = Kernel::B_TOPK / (4 * NUM_WARPS);

        CUTE_NO_UNROLL
        for (int k = 0; k < num_k_blocks; ++k) {
            int stage = k % Kernel::NUM_BUFS;
            plan.bar_kv_free[stage].wait(((k / Kernel::NUM_BUFS) & 1) ^ 1);

            if (idx_in_warpgroup < Kernel::B_TOPK) {
                int row = idx_in_warpgroup;
                int global_row = k * Kernel::B_TOPK + row;
                int index = __ldg(g_indices + global_row);
                plan.indices[stage][row] = index;
                plan.valid[stage][row] = static_cast<uint8_t>(
                    global_row < valid_length && index >= 0 && index < params.s_kv
                );
            }
            NamedBarrier::arrive_and_wait(128, Kernel::NamedBarriers::WG1_SYNC);

            if (local_warp == 0 && elect_one_sync()) {
                plan.bar_kv_ready[stage].arrive_and_expect_tx(
                    Kernel::B_TOPK * Kernel::D_K * sizeof(bf16)
                );
            }
            NamedBarrier::arrive_and_wait(128, Kernel::NamedBarriers::WG1_SYNC);

            if (elect_one_sync()) {
                CUTE_UNROLL
                for (int group = 0; group < ROW_GROUPS_PER_WARP; ++group) {
                    int first_row = (group * NUM_WARPS + local_warp) * 4;
                    int4 coordinates;
                    coordinates.x = plan.valid[stage][first_row + 0]
                        ? plan.indices[stage][first_row + 0] : 0;
                    coordinates.y = plan.valid[stage][first_row + 1]
                        ? plan.indices[stage][first_row + 1] : 0;
                    coordinates.z = plan.valid[stage][first_row + 2]
                        ? plan.indices[stage][first_row + 2] : 0;
                    coordinates.w = plan.valid[stage][first_row + 3]
                        ? plan.indices[stage][first_row + 3] : 0;
                    CUTE_UNROLL
                    for (int col_tile = 0; col_tile < Kernel::D_K / 64; ++col_tile) {
                        bf16* destination = plan.kv[stage].data()
                            + local_warp * 4 * 64
                            + group * (4 * NUM_WARPS) * 64
                            + col_tile * (Kernel::B_TOPK * 64);
                        ku::tma_gather4(
                            &tma_params.tensor_map_kv,
                            plan.bar_kv_ready[stage],
                            destination,
                            col_tile * 64,
                            coordinates,
                            static_cast<int64_t>(TMA::CacheHintSm90::EVICT_LAST)
                        );
                    }
                }
            }
        }
    } else {
        if (warp_idx == 8 && elect_one_sync()) {
            Tensor s_q = make_tensor(make_smem_ptr(plan.q.data()), typename Kernel::SmemLayoutHeadDim{});
            Tensor s_do = make_tensor(make_smem_ptr(plan.d_o.data()), typename Kernel::SmemLayoutHeadDim{});
            Tensor s_qt = make_tensor(make_smem_ptr(plan.q_t.data()), typename Kernel::SmemLayoutDimHead{});
            Tensor s_dot = make_tensor(make_smem_ptr(plan.d_o_t.data()), typename Kernel::SmemLayoutDimHead{});
            Tensor s_qt_tiles = flat_divide(
                s_qt, Tile<Int<64>, Int<Kernel::B_H_MMA>>{}
            )(_, _, _0{}, _);
            Tensor s_dot_tiles = flat_divide(
                s_dot, Tile<Int<64>, Int<Kernel::B_H_MMA>>{}
            )(_, _, _0{}, _);

            plan.bar_operands_ready.wait(0);
            ku::tcgen05_after_thread_sync();

            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks; ++k) {
                int stage = k % Kernel::NUM_BUFS;
                Tensor s_k = make_tensor(
                    make_smem_ptr(plan.kv[stage].data()), typename Kernel::SmemLayoutK{}
                );

                plan.bar_score_free.wait((((2 * k) & 1) ^ 1));
                plan.bar_kv_ready[stage].wait((k / Kernel::NUM_BUFS) & 1);
                ku::tcgen05_after_thread_sync();
                ku::utcmma_ss(tiled_mma_qk, s_k, s_q, t_score, true);
                ku::umma_arrive_noelect(plan.bar_score_ready);

                plan.bar_score_free.wait((((2 * k + 1) & 1) ^ 1));
                ku::tcgen05_after_thread_sync();
                ku::utcmma_ss(tiled_mma_qk, s_k, s_do, t_score, true);
                ku::umma_arrive_noelect(plan.bar_score_ready);

                plan.bar_ds_ready.wait(k & 1);
                plan.bar_dkv_free.wait((k & 1) ^ 1);
                ku::tcgen05_after_thread_sync();

                Tensor s_prob = make_tensor(
                    make_smem_ptr(plan.prob.data()), typename Kernel::SmemLayoutProb{}
                );
                Tensor s_ds_dk = make_tensor(
                    make_smem_ptr(plan.ds_for_dk.data()), typename Kernel::SmemLayoutProb{}
                );
                Tensor s_ds_dq = make_tensor(
                    make_smem_ptr(plan.ds_for_dq.data()), typename Kernel::SmemLayoutDSForDQ{}
                );
                Tensor s_kt = make_tensor(
                    make_smem_ptr(plan.kv[stage].data()), typename Kernel::SmemLayoutKT{}
                );
                Tensor s_kt_tiles = flat_divide(
                    s_kt, Tile<Int<64>, Int<Kernel::B_TOPK>>{}
                )(_, _, _0{}, _);

                CUTE_UNROLL
                for (int tile = 0; tile < 2; ++tile) {
                    t_dv.data().get() = Kernel::tmem_cols::DV + tile * 64;
                    ku::utcmma_ss(
                        tiled_mma_dkv, s_prob, s_dot_tiles(_, _, tile), t_dv, true
                    );

                    t_dk.data().get() = Kernel::tmem_cols::DK + tile * 64;
                    ku::utcmma_ss(
                        tiled_mma_dkv, s_ds_dk, s_qt_tiles(_, _, tile), t_dk, true
                    );

                    t_dq.data().get() =
                        Kernel::tmem_cols::DQ + tile * Kernel::B_H_TMEM;
                    ku::utcmma_ss(
                        tiled_mma_dq, s_kt_tiles(_, _, tile), s_ds_dq, t_dq, k == 0
                    );
                }
                ku::umma_arrive_noelect(plan.bar_dkv_ready[stage]);
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
void run_bwd_phase1_kernel(const SparseAttnBwdParams& params) {
    using Kernel = KernelTemplate<D_QK, H_Q>;
    static_assert(D_QK == 128);
    static_assert(H_Q == 8 || H_Q == 16 || H_Q == 32);

    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.h_q == H_Q);
    KU_ASSERT(params.d_qk == D_QK);
    KU_ASSERT(params.d_v == Kernel::D_V);
    KU_ASSERT(params.topk % Kernel::B_TOPK == 0);

    auto shape_q = make_shape(params.h_q, params.d_qk, params.s_q);
    auto tma_q = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.q),
            make_layout(
                shape_q,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q)
            )
        ),
        typename Kernel::SmemLayoutHeadDim{}
    );

    auto shape_o = make_shape(params.h_q, params.d_v, params.s_q);
    auto tma_o = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.out),
            make_layout(
                shape_o,
                make_stride(params.stride_out_h_q, _1{}, params.stride_out_s_q)
            )
        ),
        typename Kernel::SmemLayoutHeadDim{}
    );

    auto shape_do = make_shape(params.h_q, params.d_v, params.s_q);
    auto tma_do = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.d_out),
            make_layout(
                shape_do,
                make_stride(
                    params.stride_d_out_h_q, _1{}, params.stride_d_out_s_q
                )
            )
        ),
        typename Kernel::SmemLayoutHeadDim{}
    );

    CUtensorMap tensor_map_kv;
    uint64_t size[2] = {Kernel::D_K, static_cast<uint64_t>(params.s_kv)};
    uint64_t stride[1] = {
        static_cast<uint64_t>(params.stride_kv_s_kv) * sizeof(bf16)
    };
    uint32_t box_size[2] = {64, 1};
    uint32_t element_stride[2] = {1, 1};
    CUresult result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
        &tensor_map_kv,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        params.kv,
        size,
        stride,
        box_size,
        element_stride,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    KU_ASSERT(result == CUresult::CUDA_SUCCESS);

    TmaParams<
        decltype(shape_q), decltype(tma_q),
        decltype(shape_o), decltype(tma_o),
        decltype(shape_do), decltype(tma_do)
    > tma_params = {
        shape_q, tma_q,
        shape_o, tma_o,
        shape_do, tma_do,
        tensor_map_kv
    };

    auto kernel = &sparse_attn_bwd_kernel<D_QK, H_Q, decltype(tma_params)>;
    constexpr size_t smem_size = sizeof(typename Kernel::SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size
    ));
    kernel<<<params.s_q, Kernel::NUM_THREADS, smem_size, params.stream>>>(
        params, tma_params
    );
    KU_CHECK_KERNEL_LAUNCH();
}

} // namespace sm100::bwd::head_small
