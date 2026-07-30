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

#if defined(FP8_FWD_DEBUG_MARKERS)
#define FP8_MARK_WARP(...)                                                      \
    do {                                                                        \
        if (s_q_idx == 0 && elect_one_sync()) {                                 \
            cute::print(__VA_ARGS__);                                           \
            cute::print("\n");                                                 \
        }                                                                       \
    } while (0)
#define FP8_MARK_ONE(...)                                                       \
    do {                                                                        \
        if (s_q_idx == 0) {                                                     \
            cute::print(__VA_ARGS__);                                           \
            cute::print("\n");                                                 \
        }                                                                       \
    } while (0)
#else
#define FP8_MARK_WARP(...) do { } while (0)
#define FP8_MARK_ONE(...) do { } while (0)
#endif

#if defined(FP8_FWD_BARRIER_TIMING)
CUTE_DEVICE
uint64_t fp8_timing_now_ns() {
    uint64_t timestamp;
    asm volatile(
        "mov.u64 %0, %%globaltimer;"
        : "=l"(timestamp)
        :
        : "memory"
    );
    return timestamp;
}

#define FP8_TIMED_WAIT(sample, destination, wait_expression)                    \
    do {                                                                        \
        uint64_t fp8_wait_begin_ns = 0;                                          \
        if (sample) {                                                           \
            fp8_wait_begin_ns = fp8_timing_now_ns();                            \
        }                                                                       \
        wait_expression;                                                        \
        if (sample) {                                                           \
            destination = fp8_timing_now_ns() - fp8_wait_begin_ns;              \
        }                                                                       \
    } while (0)
#define FP8_TIMEPOINT(sample, destination)                                      \
    do {                                                                        \
        if (sample) {                                                           \
            destination = fp8_timing_now_ns()                                   \
                - plan.barrier_timing.origin_ns;                                \
        }                                                                       \
    } while (0)

CUTE_DEVICE
void print_fp8_barrier_timing(
    const SharedMemoryPlan& plan,
    int num_k_blocks
) {
    const auto& timing = plan.barrier_timing;
    const int traced_tiles = min(num_k_blocks, FP8_TIMING_MAX_TILES);
    cute::print(
        "FP8_TIME header unit=ns tiles=%d traced=%d origin=%llu\n",
        num_k_blocks,
        traced_tiles,
        static_cast<unsigned long long>(timing.origin_ns)
    );
    CUTE_UNROLL
    for (int warp = 0; warp < 12; ++warp) {
        cute::print(
            "FP8_TIME branch warp=%d end=%llu\n",
            warp,
            static_cast<unsigned long long>(timing.branch_end_ns[warp])
        );
    }
    CUTE_UNROLL
    for (int warp = 0; warp < 4; ++warp) {
        cute::print(
            "FP8_TIME wg0_init warp=%d qw_wait=%llu final_sv_wait=%llu "
            "final_sv_ready=%llu\n",
            warp,
            static_cast<unsigned long long>(timing.qw_scale_wait_ns[warp]),
            static_cast<unsigned long long>(timing.final_sv_wait_ns[warp]),
            static_cast<unsigned long long>(timing.final_sv_ready_ns[warp])
        );
        CUTE_UNROLL
        for (int dv_block = 0;
             dv_block < NUM_SV_TMEM_BLOCKS;
             ++dv_block) {
            cute::print(
                "FP8_TIME tail_block warp=%d block=%d "
                "wait=%llu ready=%llu drained=%llu\n",
                warp,
                dv_block,
                static_cast<unsigned long long>(
                    timing.final_o_block_wait_ns[warp][dv_block]
                ),
                static_cast<unsigned long long>(
                    timing.final_o_block_ready_ns[warp][dv_block]
                ),
                static_cast<unsigned long long>(
                    timing.final_o_block_drained_ns[warp][dv_block]
                )
            );
        }
    }
    cute::print(
        "FP8_TIME q_pipeline tma_wait=%llu tmem_commit=%llu\n",
        static_cast<unsigned long long>(timing.q_tma_wait_ns),
        static_cast<unsigned long long>(timing.q_tmem_committed_ns)
    );
    CUTE_UNROLL
    for (int warp = 0; warp < 2; ++warp) {
        cute::print(
            "FP8_TIME scale_init warp=%d arrived=%llu\n",
            warp + 10,
            static_cast<unsigned long long>(timing.qw_scale_arrived_ns[warp])
        );
    }
    for (int tile = 0; tile < traced_tiles; ++tile) {
        CUTE_UNROLL
        for (int warp = 0; warp < 4; ++warp) {
            const auto& value = timing.wg0[warp][tile];
            cute::print(
                "FP8_TIME wg0 warp=%d tile=%d start=%llu pair_wait=%llu "
                "qk_wait=%llu valid_wait=%llu scale_wait=%llu rowmax_wait=%llu "
                "waits_done=%llu p_released=%llu "
                "p_scaled=%llu rowmax_ready=%llu softmax_exp_ready=%llu "
                "softmax_ready=%llu sv_wait=%llu s_stored=%llu "
                "o_rescaled=%llu s_arrived=%llu\n",
                warp,
                tile,
                static_cast<unsigned long long>(value.tile_start_ns),
                static_cast<unsigned long long>(value.pair_wait_ns),
                static_cast<unsigned long long>(value.qk_wait_ns),
                static_cast<unsigned long long>(value.valid_wait_ns),
                static_cast<unsigned long long>(value.scale_wait_ns),
                static_cast<unsigned long long>(value.rowmax_wait_ns),
                static_cast<unsigned long long>(value.waits_done_ns),
                static_cast<unsigned long long>(value.p_released_ns),
                static_cast<unsigned long long>(value.p_scaled_ns),
                static_cast<unsigned long long>(value.rowmax_ready_ns),
                static_cast<unsigned long long>(value.softmax_exp_ready_ns),
                static_cast<unsigned long long>(value.softmax_ready_ns),
                static_cast<unsigned long long>(value.sv_wait_ns),
                static_cast<unsigned long long>(value.s_stored_ns),
                static_cast<unsigned long long>(value.o_rescaled_ns),
                static_cast<unsigned long long>(value.s_arrived_ns)
            );
            CUTE_UNROLL
            for (int dv_block = 0;
                 dv_block < NUM_SV_TMEM_BLOCKS;
                 ++dv_block) {
                cute::print(
                    "FP8_TIME wg0_o_block warp=%d tile=%d block=%d "
                    "ready=%llu rescaled=%llu\n",
                    warp,
                    tile,
                    dv_block,
                    static_cast<unsigned long long>(
                        value.o_block_ready_ns[dv_block]
                    ),
                    static_cast<unsigned long long>(
                        value.o_block_rescaled_ns[dv_block]
                    )
                );
            }
        }
        CUTE_UNROLL
        for (int warp = 0; warp < 4; ++warp) {
            const auto& value = timing.kv[warp][tile];
            cute::print(
                "FP8_TIME kv warp=%d tile=%d start=%llu indices_ready=%llu "
                "q_reuse_wait=%llu sv_free_wait=%llu tma0_issued=%llu "
                "tma1_issued=%llu\n",
                warp + 4,
                tile,
                static_cast<unsigned long long>(value.tile_start_ns),
                static_cast<unsigned long long>(value.indices_ready_ns),
                static_cast<unsigned long long>(value.q_reuse_wait_ns),
                static_cast<unsigned long long>(value.sv_free_wait_ns),
                static_cast<unsigned long long>(value.tma_part0_issued_ns),
                static_cast<unsigned long long>(value.tma_part1_issued_ns)
            );
        }
        const auto& mask = timing.mask[tile];
        cute::print(
            "FP8_TIME mask warp=9 tile=%d start=%llu free_wait=%llu arrived=%llu\n",
            tile,
            static_cast<unsigned long long>(mask.tile_start_ns),
            static_cast<unsigned long long>(mask.buffer_free_wait_ns),
            static_cast<unsigned long long>(mask.arrived_ns)
        );
        CUTE_UNROLL
        for (int warp = 0; warp < 2; ++warp) {
            const auto& scale = timing.scale[warp][tile];
            cute::print(
                "FP8_TIME scale warp=%d tile=%d start=%llu free_wait=%llu "
                "arrived=%llu\n",
                warp + 10,
                tile,
                static_cast<unsigned long long>(scale.tile_start_ns),
                static_cast<unsigned long long>(scale.buffer_free_wait_ns),
                static_cast<unsigned long long>(scale.arrived_ns)
            );
        }
    }
    for (int iter = 0; iter <= traced_tiles; ++iter) {
        const auto& value = timing.mma[iter];
        cute::print(
            "FP8_TIME mma warp=8 iter=%d start=%llu p_free_wait=%llu "
            "q_copy_wait=%llu kv0_wait=%llu kv0_ready=%llu qk0_issued=%llu "
            "kv1_wait=%llu kv1_ready=%llu qk1_issued=%llu qk_committed=%llu "
            "s_ready_wait=%llu s_ready=%llu s_scale_ready=%llu "
            "sv_committed=%llu\n",
            iter,
            static_cast<unsigned long long>(value.iter_start_ns),
            static_cast<unsigned long long>(value.p_free_wait_ns),
            static_cast<unsigned long long>(value.q_copy_wait_ns),
            static_cast<unsigned long long>(value.kv_wait_ns[0]),
            static_cast<unsigned long long>(value.kv_ready_ns[0]),
            static_cast<unsigned long long>(value.qk_issued_ns[0]),
            static_cast<unsigned long long>(value.kv_wait_ns[1]),
            static_cast<unsigned long long>(value.kv_ready_ns[1]),
            static_cast<unsigned long long>(value.qk_issued_ns[1]),
            static_cast<unsigned long long>(value.qk_committed_ns),
            static_cast<unsigned long long>(value.s_ready_wait_ns),
            static_cast<unsigned long long>(value.s_ready_ns),
            static_cast<unsigned long long>(value.s_scale_ready_ns),
            static_cast<unsigned long long>(value.sv_committed_ns)
        );
        CUTE_UNROLL
        for (int dv_block = 0;
             dv_block < NUM_SV_TMEM_BLOCKS;
             ++dv_block) {
            cute::print(
                "FP8_TIME mma_o_block iter=%d block=%d committed=%llu\n",
                iter,
                dv_block,
                static_cast<unsigned long long>(
                    value.o_block_committed_ns[dv_block]
                )
            );
        }
    }
}
#else
#define FP8_TIMED_WAIT(sample, destination, wait_expression)                    \
    do {                                                                        \
        wait_expression;                                                        \
    } while (0)
#define FP8_TIMEPOINT(sample, destination) do { } while (0)
#endif

CUTE_DEVICE
float ue8m0_bits_to_float(uint8_t bits) {
    TRAP_ONLY_DEVICE_ASSERT(bits != 0xff);
    if (bits == 0) {
        return __uint_as_float(0x00400000u);
    }
    return __uint_as_float(static_cast<uint32_t>(bits) << 23);
}

CUTE_DEVICE
void transpose_bf16_8x8(uint32_t values[4], int lane_idx) {
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) {
        const uint32_t peer = __shfl_xor_sync(
            0xffffffffu, values[i], 1, 8
        );
        values[i] = (lane_idx & 1) == 0
            ? (values[i] & 0x0000ffffu) | (peer << 16)
            : (values[i] & 0xffff0000u) | (peer >> 16);
    }

    const uint32_t peer0 = __shfl_xor_sync(0xffffffffu, values[0], 2, 8);
    const uint32_t peer1 = __shfl_xor_sync(0xffffffffu, values[1], 2, 8);
    const uint32_t peer2 = __shfl_xor_sync(0xffffffffu, values[2], 2, 8);
    const uint32_t peer3 = __shfl_xor_sync(0xffffffffu, values[3], 2, 8);
    if ((lane_idx & 2) == 0) {
        values[1] = peer0;
        values[3] = peer2;
    } else {
        values[0] = peer1;
        values[2] = peer3;
    }

    const uint32_t peer02 = __shfl_xor_sync(0xffffffffu, values[0], 4, 8);
    const uint32_t peer12 = __shfl_xor_sync(0xffffffffu, values[1], 4, 8);
    const uint32_t peer22 = __shfl_xor_sync(0xffffffffu, values[2], 4, 8);
    const uint32_t peer32 = __shfl_xor_sync(0xffffffffu, values[3], 4, 8);
    if ((lane_idx & 4) == 0) {
        values[2] = peer02;
        values[3] = peer12;
    } else {
        values[0] = peer22;
        values[1] = peer32;
    }
}

template<int B_H_, int B_H_TMEM_, int TMEM_COL_START>
CUTE_DEVICE
void rescale_o_block(float scale[B_H_], int dv_block) {
    float o[B_H_TMEM_];
    ku::tmem_ld_32dp32bNx<B_H_TMEM_>(
        TMEM_COL_START + dv_block * B_H_TMEM_, o
    );
    cutlass::arch::fence_view_async_tmem_load();
    CUTE_UNROLL
    for (int i = 0; i < B_H_; ++i) {
        o[i] *= scale[i];
    }
    ku::tmem_st_32dp32bNx<B_H_TMEM_>(
        TMEM_COL_START + dv_block * B_H_TMEM_, o
    );
    cutlass::arch::fence_view_async_tmem_store();
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
    Tensor tV_scale = make_tensor<typename TiledMMA_O::FrgTypeSFA>(
        shape(SmemLayoutOScaleAAtom{})
    );
    Tensor tS_scale = make_tensor<typename TiledMMA_O::FrgTypeSFB>(
        shape(SmemLayoutOScaleBAtom{})
    );
    tP0.data().get() = tmem_cols::P0;
    tP1.data().get() = tmem_cols::P1;
    tQ_part0.data().get() = tmem_cols::Q;
    tQ_part1.data().get() = tmem_cols::Q + 32;
    tV_scale.data().get() = tmem_cols::V_Scale;
    tS_scale.data().get() = tmem_cols::S_Scale;

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
                CUTE_UNROLL
                for (int dv_block = 0;
                     dv_block < NUM_SV_TMEM_BLOCKS - 1;
                     ++dv_block) {
                    plan.bar_sv_block_done[i][dv_block].init(1);
                }
                plan.bar_sv_done[i].init(1);
                plan.bar_kv_ready[i][0].init(1);
                plan.bar_kv_ready[i][1].init(1);
                plan.bar_kv_scale_ready[i].init(2);
                plan.bar_k_valid_ready[i].init(B_TOPK/8);
                plan.bar_k_valid_free[i].init(128);
            }
            for (int i = 0; i < NUM_P_BUFS; ++i) {
                plan.bar_qk_done[i].init(1);
                plan.bar_p_free[i].init(128); // warp group 0 touch this
            }
            plan.bar_so_ready.init(1);
            fence_barrier_init();
        }

        // Initialize TMEM
        cute::TMEM::Allocator1Sm().allocate(512, plan.tmem_start_addr.data());
        TRAP_ONLY_DEVICE_ASSERT(plan.tmem_start_addr.data()[0] == 0);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }

    __syncthreads();
#if defined(FP8_FWD_BARRIER_TIMING)
    if (s_q_idx == 0) {
        static_assert(sizeof(Fp8BarrierTiming) % sizeof(uint64_t) == 0);
        uint64_t* timing_words = reinterpret_cast<uint64_t*>(
            &plan.barrier_timing
        );
        CUTE_UNROLL
        for (
            int i = threadIdx.x;
            i < sizeof(Fp8BarrierTiming) / sizeof(uint64_t);
            i += NUM_THREADS
        ) {
            timing_words[i] = 0;
        }
    }
    __syncthreads();
    if (s_q_idx == 0 && threadIdx.x == 0) {
        plan.barrier_timing.origin_ns = fp8_timing_now_ns();
    }
    __syncthreads();
#endif
    FP8_MARK_WARP(
        "FP8_MARK 00 post_init warp=%d wg=%d",
        warp_idx,
        warpgroup_idx
    );

    if (warpgroup_idx == 0) {
        FP8_MARK_WARP("FP8_MARK 10 wg0_qscale_wait_before warp=%d", warp_idx);
        FP8_TIMED_WAIT(
            s_q_idx == 0 && lane_idx == 0,
            plan.barrier_timing.qw_scale_wait_ns[warp_idx],
            plan.bar_qw_scale_ready.wait(0)
        );
        FP8_MARK_WARP("FP8_MARK 11 wg0_qscale_wait_after warp=%d", warp_idx);

        // All four WG0 warps write their physical TMEM-row replicas.  W(g)
        // is constant over K and changes once per 64 output dimensions.
        uint32_t v_scale_words[V_SCALE_TMEM_COLS];
        CUTE_UNROLL
        for (int dv_block = 0; dv_block < NUM_SV_TMEM_BLOCKS; ++dv_block) {
            const uint32_t w_lo = static_cast<uint32_t>(
                plan.kv_w_scale_bits[2 * dv_block]
            ) * 0x01010101u;
            const uint32_t w_hi = static_cast<uint32_t>(
                plan.kv_w_scale_bits[2 * dv_block + 1]
            ) * 0x01010101u;
            const int col = dv_block * SV_SCALE_TMEM_COLS;
            v_scale_words[col + 0] = w_lo;
            v_scale_words[col + 1] = w_lo;
            v_scale_words[col + 2] = w_hi;
            v_scale_words[col + 3] = w_hi;
        }
        ku::tmem_st_32dp32bNx<V_SCALE_TMEM_COLS>(
            tmem_cols::V_Scale, v_scale_words
        );
        cutlass::arch::fence_view_async_tmem_store();

        float mi = MAX_INIT_VAL;
        float li = 0.0f;
        float real_mi = -CUDART_INF_F;

        const int h = idx_in_warpgroup % B_H;
        const int token_group = idx_in_warpgroup / B_H;
        const int token_base = token_group * (B_TOPK / 2);
        const float q_scale = plan.q_head_scale[h];
        Tensor sS = make_tensor(make_smem_ptr(plan.s.data()), SmemLayoutS{});
        Tensor sS_scale = make_tensor(
            make_smem_ptr(plan.s_scale.data()), SmemLayoutOScaleBAtom{}
        );
        static constexpr int NUM_ELEMS_PER_THREAD = B_TOPK / 2;

        CUTE_NO_UNROLL
        for (int k = 0; k < num_k_blocks; ++k) {
#if defined(FP8_FWD_BARRIER_TIMING)
            const bool trace_wg0_tile = s_q_idx == 0
                && lane_idx == 0 && k < FP8_TIMING_MAX_TILES;
#endif
            FP8_TIMEPOINT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].tile_start_ns
            );
            // Wait for P
            FP8_MARK_WARP(
                "FP8_MARK 12 wg0_pair_sync_before warp=%d tile=%d",
                warp_idx,
                k
            );
            FP8_TIMED_WAIT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].pair_wait_ns,
                (NamedBarrier::arrive_and_wait(
                    64,
                    NamedBarriers::wg0_warp02_sync + (warp_idx & 1)
                ))
            );
            FP8_MARK_WARP(
                "FP8_MARK 13 wg0_pair_sync_after warp=%d tile=%d",
                warp_idx,
                k
            );
            const int cur_buf = k % NUM_BUFS;
            const int p_idx = k % NUM_P_BUFS;
            FP8_TIMED_WAIT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].qk_wait_ns,
                plan.bar_qk_done[p_idx].wait((k / NUM_P_BUFS) & 1)
            );
            FP8_MARK_WARP(
                "FP8_MARK 14 wg0_qk_done_after warp=%d tile=%d pstage=%d",
                warp_idx,
                k,
                p_idx
            );
            FP8_TIMED_WAIT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].valid_wait_ns,
                plan.bar_k_valid_ready[cur_buf].wait((k / NUM_BUFS) & 1)
            );
            FP8_MARK_WARP(
                "FP8_MARK 15 wg0_valid_after warp=%d tile=%d kvstage=%d",
                warp_idx,
                k,
                cur_buf
            );
            FP8_TIMED_WAIT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].scale_wait_ns,
                plan.bar_kv_scale_ready[cur_buf].wait((k / NUM_BUFS) & 1)
            );
            FP8_TIMEPOINT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].waits_done_ns
            );
            FP8_MARK_WARP(
                "FP8_MARK 16 wg0_kv_scale_after warp=%d tile=%d kvstage=%d",
                warp_idx,
                k,
                cur_buf
            );
            ku::tcgen05_after_thread_sync();

            // One scale per lane stays in registers; shuffles expose the 32
            // token scales needed by every head in this half tile.
            const float lane_kv_scale =
                plan.kv_dim_scale[cur_buf][token_base + lane_idx];

            float p[NUM_ELEMS_PER_THREAD];
            auto release_p = [&]() { plan.bar_p_free[p_idx].arrive(); };
            if (p_idx == 0) {
                retrieve_mask_and_reduce_p<
                    NUM_ELEMS_PER_THREAD,
                    tmem_cols::P0,
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
                    tmem_cols::P1,
                    NamedBarriers::wg0_warp02_sync,
                    NamedBarriers::wg0_warp13_sync,
                    false
                >(
                    plan.is_k_valid[cur_buf], warp_idx, lane_idx,
                    release_p, plan.p_exchange_buf, p
                );
            }
            plan.bar_k_valid_free[cur_buf].arrive();
            FP8_TIMEPOINT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].p_released_ns
            );
            FP8_MARK_WARP(
                "FP8_MARK 17 wg0_p_retrieved warp=%d tile=%d",
                warp_idx,
                k
            );

            float kv_scale[NUM_ELEMS_PER_THREAD];
            const float q_sm_scale = q_scale * params.sm_scale_div_log2;
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD; ++i) {
                kv_scale[i] = __shfl_sync(
                    0xffffffff, lane_kv_scale, i
                );
                p[i] *= q_sm_scale * kv_scale[i];
            }
            FP8_TIMEPOINT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].p_scaled_ns
            );
            
            float cur_pi_max = get_max<NUM_ELEMS_PER_THREAD>(p);

            plan.rowwise_max_buf[idx_in_warpgroup] = cur_pi_max;
            FP8_TIMED_WAIT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].rowmax_wait_ns,
                (NamedBarrier::arrive_and_wait(
                    128, NamedBarriers::wg0_sync
                ))
            );
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
            FP8_TIMEPOINT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].rowmax_ready_ns
            );

            float cur_sum = 0.0f;
            float local_s_max = 0.0f;
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD; ++i) {
                const float softmax_s = exp2f(p[i] - new_max);
                cur_sum += softmax_s;
                p[i] = softmax_s * kv_scale[i];
                local_s_max = max(local_s_max, fabsf(p[i]));
            }
            FP8_TIMEPOINT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].softmax_exp_ready_ns
            );

            const e8m0 s_scale_e8m0(
                local_s_max > 0.0f ? local_s_max / FP8_MAX : 1.0f
            );
            const float current_s_scale = float(s_scale_e8m0);
            sS_scale(
                h,
                _0{},
                make_coord(token_group, _0{})
            ) = s_scale_e8m0;

            uint32_t s_words[NUM_ELEMS_PER_THREAD / 4];
            const float quant_multiplier = 1.0f / current_s_scale;
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD; i += 4) {
                const uint16_t s01 = ku::float2_to_e4m3x2_bits(
                    make_float2(
                        p[i] * quant_multiplier,
                        p[i + 1] * quant_multiplier
                    )
                );
                const uint16_t s23 = ku::float2_to_e4m3x2_bits(
                    make_float2(
                        p[i + 2] * quant_multiplier,
                        p[i + 3] * quant_multiplier
                    )
                );
                s_words[i / 4] = static_cast<uint32_t>(s01)
                    | (static_cast<uint32_t>(s23) << 16);
            }
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD; i += 16) {
                *reinterpret_cast<uint4*>(&sS(h, token_base + i)) = make_uint4(
                    s_words[i / 4 + 0],
                    s_words[i / 4 + 1],
                    s_words[i / 4 + 2],
                    s_words[i / 4 + 3]
                );
            }

            li = fma(li, scale_for_old, cur_sum);
            plan.head_scale[h] = scale_for_old;
            FP8_TIMEPOINT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].softmax_ready_ns
            );
            FP8_TIMEPOINT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].s_stored_ns
            );

            fence_view_async_shared();
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);

            if (k > 0) {
                const int prev_buf = (k - 1) % NUM_BUFS;
                const int prev_phase = ((k - 1) / NUM_BUFS) & 1;
                CUTE_UNROLL
                for (int dv_block = 0;
                     dv_block < NUM_SV_TMEM_BLOCKS;
                     ++dv_block) {
                    if (dv_block == 0) {
                        FP8_TIMED_WAIT(
                            trace_wg0_tile,
                            plan.barrier_timing.wg0[warp_idx][k].sv_wait_ns,
                            plan.bar_sv_block_done[prev_buf][dv_block].wait(
                                prev_phase
                            )
                        );
                    } else if (dv_block + 1 < NUM_SV_TMEM_BLOCKS) {
                        plan.bar_sv_block_done[prev_buf][dv_block].wait(
                            prev_phase
                        );
                    } else {
                        plan.bar_sv_done[prev_buf].wait(prev_phase);
                    }
                    FP8_TIMEPOINT(
                        trace_wg0_tile,
                        plan.barrier_timing.wg0[warp_idx][k]
                            .o_block_ready_ns[dv_block]
                    );

                    ku::tcgen05_after_thread_sync();
                    rescale_o_block<B_H, B_H_TMEM, tmem_cols::O>(
                        plan.head_scale, dv_block
                    );
                    ku::tcgen05_before_thread_sync();
                    FP8_TIMEPOINT(
                        trace_wg0_tile,
                        plan.barrier_timing.wg0[warp_idx][k]
                            .o_block_rescaled_ns[dv_block]
                    );
                    NamedBarrier::arrive_and_wait(
                        128, NamedBarriers::wg0_sync
                    );
                }
            }
            FP8_TIMEPOINT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].o_rescaled_ns
            );

            if (idx_in_warpgroup == 0) {
                plan.bar_so_ready.arrive();
            }
            FP8_TIMEPOINT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].s_arrived_ns
            );
            FP8_MARK_WARP(
                "FP8_MARK 1a wg0_s_ready warp=%d tile=%d",
                warp_idx,
                k
            );
        }

        NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
        if (real_mi == -CUDART_INF_F) {
            li = 0.0f;
            mi = -CUDART_INF_F;
        }

        plan.rowwise_li_buf[idx_in_warpgroup] = li;
        NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
        li += plan.rowwise_li_buf[idx_in_warpgroup ^ B_H];

        if (idx_in_warpgroup < B_H) {
            const int global_index = s_q_idx * params.h_q + idx_in_warpgroup;
            float cur_lse = fmaf(mi, CUDART_LN2_F, logf(li));
            cur_lse = cur_lse == -CUDART_INF_F ? CUDART_INF_F : cur_lse;
            params.max_logits[global_index] = real_mi * CUDART_LN2_F;
            params.lse[global_index] = cur_lse;

            const float attn_sink = params.attn_sink == nullptr
                ? -CUDART_INF_F
                : __ldg(params.attn_sink + idx_in_warpgroup) * CUDART_L2E_F;
            plan.head_scale[idx_in_warpgroup] = li == 0.0f
                ? 0.0f
                : __fdividef(1.0f, li + exp2f(attn_sink - mi));
        }
        fence_view_async_shared();
        NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);

        constexpr int B_EPI = 64;
        const int final_buf = (num_k_blocks - 1) % NUM_BUFS;
        const int final_phase = ((num_k_blocks - 1) / NUM_BUFS) & 1;
        const int scratch_lo_buf = (final_buf + 1) % NUM_BUFS;
        const int scratch_hi_buf = (final_buf + 2) % NUM_BUFS;
        Tensor tma_gO = flat_divide(
            tma_params.tma_O.get_tma_tensor(tma_params.shape_O)(
                _, _, s_q_idx
            ),
            Shape<Int<B_H>, Int<B_EPI>>{}
        )(_, _, _0{}, _);
        Tensor sO_lo = make_tensor(
            make_smem_ptr(reinterpret_cast<bf16*>(
                plan.qkvo.kv[scratch_lo_buf].data()
            )),
            SmemLayoutOTiles<NUM_SV_TMEM_BLOCKS>{}
        );
        Tensor sO_hi = make_tensor(
            make_smem_ptr(reinterpret_cast<bf16*>(
                plan.qkvo.kv[scratch_hi_buf].data()
            )),
            SmemLayoutOTiles<NUM_SV_TMEM_BLOCKS>{}
        );
        Tensor sO_lo_divided = flat_divide(
            sO_lo, Shape<Int<B_H>, Int<B_EPI>>{}
        )(_, _, _0{}, _);
        Tensor sO_hi_divided = flat_divide(
            sO_hi, Shape<Int<B_H>, Int<B_EPI>>{}
        )(_, _, _0{}, _);
        auto thr_tma = tma_params.tma_O.get_slice(_0{});
        const bool have_valid_indices = __any_sync(0xffffffff, li != 0.0f);

        float o_head[B_H_TMEM];
        CUTE_UNROLL
        for (int dv_block = 0;
             dv_block < NUM_SV_TMEM_BLOCKS;
            ++dv_block) {
            if (dv_block == 0) {
                FP8_TIMED_WAIT(
                    s_q_idx == 0 && lane_idx == 0,
                    plan.barrier_timing
                        .final_o_block_wait_ns[warp_idx][dv_block],
                    plan.bar_sv_block_done[final_buf][dv_block].wait(
                        final_phase
                    )
                );
            } else if (dv_block + 1 < NUM_SV_TMEM_BLOCKS) {
                FP8_TIMED_WAIT(
                    s_q_idx == 0 && lane_idx == 0,
                    plan.barrier_timing
                        .final_o_block_wait_ns[warp_idx][dv_block],
                    plan.bar_sv_block_done[final_buf][dv_block].wait(
                        final_phase
                    )
                );
            } else {
                FP8_TIMED_WAIT(
                    s_q_idx == 0 && lane_idx == 0,
                    plan.barrier_timing
                        .final_o_block_wait_ns[warp_idx][dv_block],
                    plan.bar_sv_done[final_buf].wait(final_phase)
                );
            }
            FP8_TIMEPOINT(
                s_q_idx == 0 && lane_idx == 0,
                plan.barrier_timing
                    .final_o_block_ready_ns[warp_idx][dv_block]
            );
#if defined(FP8_FWD_BARRIER_TIMING)
            if (s_q_idx == 0 && lane_idx == 0 && dv_block == 0) {
                plan.barrier_timing.final_sv_wait_ns[warp_idx] =
                    plan.barrier_timing
                        .final_o_block_wait_ns[warp_idx][dv_block];
                plan.barrier_timing.final_sv_ready_ns[warp_idx] =
                    plan.barrier_timing
                        .final_o_block_ready_ns[warp_idx][dv_block];
            }
#endif
            ku::tcgen05_after_thread_sync();

            if (have_valid_indices) {
                ku::tmem_ld_32dp32bNx<B_H_TMEM>(
                    tmem_cols::O + dv_block * B_H_TMEM,
                    o_head
                );
                cutlass::arch::fence_view_async_tmem_load();
            } else {
                CUTE_UNROLL
                for (int i = 0; i < B_H_TMEM; ++i) {
                    o_head[i] = 0.0f;
                }
            }

            const int d_vec = (warp_idx & 1) * 32 + (lane_idx & ~7);
            CUTE_UNROLL
            for (int h_base = 0; h_base < B_H; h_base += 8) {
                uint32_t values[4];
                CUTE_UNROLL
                for (int i = 0; i < 4; ++i) {
                    const int h0 = h_base + 2 * i;
                    const int h1 = h0 + 1;
                    const bf16 v0 = bf16(
                        o_head[h0] * plan.head_scale[h0]
                    );
                    const bf16 v1 = bf16(
                        o_head[h1] * plan.head_scale[h1]
                    );
                    values[i] = static_cast<uint32_t>(v0.storage)
                        | (static_cast<uint32_t>(v1.storage) << 16);
                }
                transpose_bf16_8x8(values, lane_idx);

                const uint4 vector = make_uint4(
                    values[0], values[1], values[2], values[3]
                );
                const int out_h = h_base + (lane_idx & 7);
                const int out_d = dv_block * B_EPI + d_vec;
                if (warp_idx < 2) {
                    *reinterpret_cast<uint4*>(&sO_lo(out_h, out_d)) = vector;
                } else {
                    *reinterpret_cast<uint4*>(&sO_hi(out_h, out_d)) = vector;
                }
            }

            fence_view_async_shared();
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
            FP8_TIMEPOINT(
                s_q_idx == 0 && lane_idx == 0,
                plan.barrier_timing
                    .final_o_block_drained_ns[warp_idx][dv_block]
            );

            const int epi_chunk = dv_block * (SV_M / B_EPI);
            if (warp_idx == 0 && elect_one_sync()) {
                cute::copy(
                    tma_params.tma_O,
                    thr_tma.partition_S(sO_lo_divided(_, _, dv_block)),
                    thr_tma.partition_D(tma_gO(_, _, epi_chunk))
                );
            }
            if (warp_idx == 1 && elect_one_sync()) {
                cute::copy(
                    tma_params.tma_O,
                    thr_tma.partition_S(sO_hi_divided(_, _, dv_block)),
                    thr_tma.partition_D(tma_gO(_, _, epi_chunk + 1))
                );
            }
        }
        if (warp_idx == 0) {
            cute::TMEM::Allocator1Sm().free(0, 512);
        }
        FP8_MARK_WARP("FP8_MARK 1d wg0_epilogue_done warp=%d", warp_idx);
} else if (warpgroup_idx == 1) {

    // Producer warp for KV
        int warp_idx = cutlass::canonical_warp_idx_sync() - 4;
        FP8_MARK_WARP("FP8_MARK 20 kv_producer_start local_warp=%d", warp_idx);
        constexpr int NUM_WARPS = 4, NUM_LOCAL_ROWS_PER_WARP = (B_TOPK/4)/NUM_WARPS;
        if (elect_one_sync()) {
            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks; ++k) {
#if defined(FP8_FWD_BARRIER_TIMING)
                const bool trace_kv_tile = s_q_idx == 0
                    && lane_idx == 0 && k < FP8_TIMING_MAX_TILES;
#endif
                FP8_TIMEPOINT(
                    trace_kv_tile,
                    plan.barrier_timing.kv[warp_idx][k].tile_start_ns
                );
                int4 indices[NUM_LOCAL_ROWS_PER_WARP];
                int max_indices = -1, min_indices = params.s_kv;
                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                    indices[local_row] = __ldg((int4*)(gIndices + k*B_TOPK) + local_row*NUM_WARPS + warp_idx);
                    max_indices = max(max_indices, int4_max(indices[local_row]));
                    min_indices = min(min_indices, int4_min(indices[local_row]));
                }
                FP8_TIMEPOINT(
                    trace_kv_tile,
                    plan.barrier_timing.kv[warp_idx][k].indices_ready_ns
                );
                bool is_all_rows_invalid = min_indices == params.s_kv || max_indices == -1;
                bool should_skip_tma = is_all_rows_invalid && k >= NUM_BUFS;

                if (k == 2) {
                    FP8_TIMED_WAIT(
                        trace_kv_tile,
                        plan.barrier_timing.kv[warp_idx][k].q_reuse_wait_ns,
                        plan.bar_prologue_utccp.wait(0)
                    );  // Q shares storage with K stage 2.
                }

                // Copy NoPE
                int cur_buf = k%NUM_BUFS;
                FP8_MARK_ONE(
                    "FP8_MARK 21 kv_free_wait_before local_warp=%d tile=%d stage=%d",
                    warp_idx,
                    k,
                    cur_buf
                );
                FP8_TIMED_WAIT(
                    trace_kv_tile,
                    plan.barrier_timing.kv[warp_idx][k].sv_free_wait_ns,
                    plan.bar_sv_done[cur_buf].wait(
                        ((k / NUM_BUFS) & 1) ^ 1
                    )
                );
                FP8_MARK_ONE(
                    "FP8_MARK 22 kv_free_wait_after local_warp=%d tile=%d stage=%d",
                    warp_idx,
                    k,
                    cur_buf
                );
                e4m3* sK_base = plan.qkvo.kv[cur_buf].data()
                    + warp_idx * 4 * TMA_K_CHUNK_BYTES;

                auto load_kv_part = [&](int part_idx) {
                    constexpr int CHUNKS_PER_PART =
                        (D_V / 2) / TMA_K_CHUNK_BYTES;
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                        CUTE_UNROLL
                        for (
                            int local_col = part_idx * CHUNKS_PER_PART;
                            local_col < (part_idx + 1) * CHUNKS_PER_PART;
                            ++local_col
                        ) {
                            ku::tma_gather4(
                                &(tma_params.tensor_map_kv),
                                plan.bar_kv_ready[cur_buf][part_idx],
                                sK_base
                                    + local_row * (4 * NUM_WARPS)
                                        * TMA_K_CHUNK_BYTES
                                    + local_col * B_TOPK
                                        * TMA_K_CHUNK_BYTES,
                                local_col * TMA_K_CHUNK_ELEMS,
                                indices[local_row],
                                (int64_t)TMA::CacheHintSm90::EVICT_LAST
                            );
                        }
                    }
                };

                if (!should_skip_tma) {
                    load_kv_part(0);
                    FP8_TIMEPOINT(
                        trace_kv_tile,
                        plan.barrier_timing.kv[warp_idx][k].tma_part0_issued_ns
                    );
                    FP8_MARK_ONE(
                        "FP8_MARK 23 kv_tma_part0_issued local_warp=%d tile=%d",
                        warp_idx,
                        k
                    );
                    load_kv_part(1);
                    FP8_TIMEPOINT(
                        trace_kv_tile,
                        plan.barrier_timing.kv[warp_idx][k].tma_part1_issued_ns
                    );
                    FP8_MARK_ONE(
                        "FP8_MARK 24 kv_tma_part1_issued local_warp=%d tile=%d",
                        warp_idx,
                        k
                    );
                } else {
                    // NOTE See head128/phase1.cuh for this TMA skipping technique
                    CUTE_UNROLL
                    for (int part_idx = 0; part_idx < 2; ++part_idx)
                        plan.bar_kv_ready[cur_buf][part_idx].complete_transaction(NUM_LOCAL_ROWS_PER_WARP*4*D_V/2*sizeof(e4m3));
                }
            }
        }
        FP8_MARK_WARP("FP8_MARK 25 kv_producer_done local_warp=%d", warp_idx);

} else {
    if (warp_idx == 8 && elect_one_sync()) {
        FP8_MARK_ONE("FP8_MARK 30 warp8_q_wait_before");
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
        FP8_TIMED_WAIT(
            s_q_idx == 0 && lane_idx == 0,
            plan.barrier_timing.q_tma_wait_ns,
            plan.bar_prologue.wait(0)
        );
        FP8_MARK_ONE("FP8_MARK 31 warp8_q_wait_after");
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
        FP8_TIMEPOINT(
            s_q_idx == 0 && lane_idx == 0,
            plan.barrier_timing.q_tmem_committed_ns
        );
        FP8_MARK_ONE("FP8_MARK 32 warp8_q_utccp_committed");

        CUTE_NO_UNROLL
        for (int k = 0; k < num_k_blocks+1; ++k) {
#if defined(FP8_FWD_BARRIER_TIMING)
            const bool trace_mma_iter = s_q_idx == 0
                && lane_idx == 0 && k <= FP8_TIMING_MAX_TILES;
#endif
            FP8_TIMEPOINT(
                trace_mma_iter,
                plan.barrier_timing.mma[k].iter_start_ns
            );
            if (k < num_k_blocks) {
                int cur_buf = k%NUM_BUFS;
                int p_stage = k%NUM_P_BUFS;
                Tensor sK = make_tensor(make_smem_ptr(plan.qkvo.kv[cur_buf].data()), SmemLayoutK_TiledMMA{});
                FP8_MARK_ONE(
                    "FP8_MARK 33 warp8_p_free_wait_before tile=%d pstage=%d",
                    k,
                    p_stage
                );
                FP8_TIMED_WAIT(
                    trace_mma_iter,
                    plan.barrier_timing.mma[k].p_free_wait_ns,
                    plan.bar_p_free[p_stage].wait(
                        ((k / NUM_P_BUFS) & 1) ^ 1
                    )
                );
                FP8_MARK_ONE(
                    "FP8_MARK 34 warp8_p_free_wait_after tile=%d pstage=%d",
                    k,
                    p_stage
                );
                ku::tcgen05_after_thread_sync();
                if (k == 0) {
                    FP8_TIMED_WAIT(
                        trace_mma_iter,
                        plan.barrier_timing.mma[k].q_copy_wait_ns,
                        plan.bar_prologue_utccp.wait(0)
                    );
                    FP8_MARK_ONE("FP8_MARK 35 warp8_q_utccp_wait_after");
                }
                Tensor sK_divided = flat_divide(sK, Tile<Int<B_TOPK*2>, Int<D_V/4>>{})(_, _, _0{}, _);
                CUTE_UNROLL
                for (int kv_part_idx = 0; kv_part_idx < 2; ++kv_part_idx) {
                    plan.bar_kv_ready[cur_buf][kv_part_idx].arrive_and_expect_tx(B_TOPK*D_V/2*sizeof(e4m3));
                    FP8_TIMED_WAIT(
                        trace_mma_iter,
                        plan.barrier_timing.mma[k].kv_wait_ns[kv_part_idx],
                        plan.bar_kv_ready[cur_buf][kv_part_idx].wait(
                            (k / NUM_BUFS) & 1
                        )
                    );
                    FP8_TIMEPOINT(
                        trace_mma_iter,
                        plan.barrier_timing.mma[k].kv_ready_ns[kv_part_idx]
                    );
                    FP8_MARK_ONE(
                        "FP8_MARK 36 warp8_kv_ready_after tile=%d part=%d stage=%d",
                        k,
                        kv_part_idx,
                        cur_buf
                    );
                    ku::tcgen05_after_thread_sync();

                    // P += Q(nope) @ K(nope)^T
                    bool clear_accum = kv_part_idx == 0;
                    if (p_stage == 0) {
                        ku::utcmma_ts(tiled_mma_P, kv_part_idx ? tQ_part1 : tQ_part0, sK_divided(_, _, kv_part_idx), tP0, clear_accum);
                    } else {
                        ku::utcmma_ts(tiled_mma_P, kv_part_idx ? tQ_part1 : tQ_part0, sK_divided(_, _, kv_part_idx), tP1, clear_accum);
                    }
                    FP8_TIMEPOINT(
                        trace_mma_iter,
                        plan.barrier_timing.mma[k].qk_issued_ns[kv_part_idx]
                    );
                    FP8_MARK_ONE(
                        "FP8_MARK 37 warp8_qk_mma_issued tile=%d part=%d",
                        k,
                        kv_part_idx
                    );
                }
                ku::umma_arrive_noelect(plan.bar_qk_done[p_stage]);
                FP8_TIMEPOINT(
                    trace_mma_iter,
                    plan.barrier_timing.mma[k].qk_committed_ns
                );
                FP8_MARK_ONE(
                    "FP8_MARK 38 warp8_qk_committed tile=%d pstage=%d",
                    k,
                    p_stage
                );
            }

            if (k > 0) {
                const int cur_buf = (k - 1) % NUM_BUFS;
                Tensor sS = make_tensor(
                    make_smem_ptr(plan.s.data()), SmemLayoutS{}
                );
                Tensor sS_scale = make_tensor(
                    make_smem_ptr(plan.s_scale.data()),
                    SmemLayoutOScaleBAtom{}
                );
                Tensor sV = make_tensor(
                    make_smem_ptr(plan.qkvo.kv[cur_buf].data()),
                    SmemLayoutV{}
                );

                FP8_TIMED_WAIT(
                    trace_mma_iter,
                    plan.barrier_timing.mma[k].s_ready_wait_ns,
                    plan.bar_so_ready.wait((k - 1) & 1)
                );
                FP8_TIMEPOINT(
                    trace_mma_iter,
                    plan.barrier_timing.mma[k].s_ready_ns
                );
                ku::tcgen05_after_thread_sync();

                auto sS_compact = make_tensor(
                    sS_scale.data(), filter_zeros(sS_scale.layout())
                );
                auto tS_compact = make_tensor(
                    tS_scale.data(), filter_zeros(tS_scale.layout())
                );
                auto copy_S_scale = make_utccp_copy(
                    SM100_UTCCP_4x32dp128bit_1cta{}, tS_compact
                );
                auto thr_S = copy_S_scale.get_slice(0);
                auto src_S = get_utccp_smem_desc_tensor<
                    SM100_UTCCP_4x32dp128bit_1cta
                >(thr_S.partition_S(sS_compact));
                auto dst_S = thr_S.partition_D(tS_compact);
                cute::copy(copy_S_scale, src_S, dst_S);
                ku::tcgen05_after_thread_sync();
                FP8_TIMEPOINT(
                    trace_mma_iter,
                    plan.barrier_timing.mma[k].s_scale_ready_ns
                );

                CUTE_UNROLL
                for (int dv_block = 0;
                     dv_block < NUM_SV_TMEM_BLOCKS;
                     ++dv_block) {
                    Tensor tO_block = partition_fragment_C(
                        tiled_mma_O, Shape<Int<SV_M>, Int<B_H>>{}
                    );
                    tO_block.data().get() = tmem_cols::O
                        + dv_block * B_H_TMEM;
                    tV_scale.data().get() = tmem_cols::V_Scale
                        + dv_block * SV_SCALE_TMEM_COLS;
                    ku::utcmma_blockscaled_ss(
                        tiled_mma_O,
                        sV(make_coord(_, dv_block), _),
                        sS,
                        tV_scale,
                        tS_scale,
                        tO_block,
                        k == 1
                    );
                    if (dv_block + 1 < NUM_SV_TMEM_BLOCKS) {
                        ku::umma_arrive_noelect(
                            plan.bar_sv_block_done[cur_buf][dv_block]
                        );
                    } else {
                        ku::umma_arrive_noelect(plan.bar_sv_done[cur_buf]);
                    }
                    FP8_TIMEPOINT(
                        trace_mma_iter,
                        plan.barrier_timing.mma[k]
                            .o_block_committed_ns[dv_block]
                    );
                }
                FP8_TIMEPOINT(
                    trace_mma_iter,
                    plan.barrier_timing.mma[k].sv_committed_ns
                );
                FP8_MARK_ONE(
                    "FP8_MARK 3c warp8_mxfp8_vs_committed sv_tile=%d stage=%d",
                    k - 1,
                    cur_buf
                );
            }
        }
        FP8_MARK_ONE("FP8_MARK 3d warp8_done");
    } else if (warp_idx == 9) {
        FP8_MARK_WARP("FP8_MARK 40 warp9_mask_start");
        if (lane_idx < B_TOPK/8) {
            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks; ++k) {
#if defined(FP8_FWD_BARRIER_TIMING)
                const bool trace_mask_tile = s_q_idx == 0
                    && lane_idx == 0 && k < FP8_TIMING_MAX_TILES;
#endif
                FP8_TIMEPOINT(
                    trace_mask_tile,
                    plan.barrier_timing.mask[k].tile_start_ns
                );
                char k_validness_mask = load_indices_and_generate_mask(
                    lane_idx,
                    gIndices + k*B_TOPK,
                    params.s_kv,
                    k*B_TOPK,
                    topk_length
                );

                int cur_buf = k%NUM_BUFS;
                FP8_TIMED_WAIT(
                    trace_mask_tile,
                    plan.barrier_timing.mask[k].buffer_free_wait_ns,
                    plan.bar_k_valid_free[cur_buf].wait(
                        ((k / NUM_BUFS) & 1) ^ 1
                    )
                );
                plan.is_k_valid[cur_buf][lane_idx] = k_validness_mask;
                plan.bar_k_valid_ready[cur_buf].arrive();
                FP8_TIMEPOINT(
                    trace_mask_tile,
                    plan.barrier_timing.mask[k].arrived_ns
                );
            }
        }
        FP8_MARK_WARP("FP8_MARK 41 warp9_mask_done");
    } else if (warp_idx == 10 || warp_idx == 11) {
        FP8_MARK_WARP("FP8_MARK 50 scale_loader_start warp=%d", warp_idx);
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
            const uint8_t w_scale_bits = __ldg(
                params.kv_scale_w + lane_idx
            );
            plan.kv_w_scale_bits[lane_idx] = w_scale_bits;
            plan.kv_w_scale[lane_idx] = ue8m0_bits_to_float(w_scale_bits);
        }
        fence_view_async_shared();
        if (elect_one_sync()) {
            plan.bar_qw_scale_ready.arrive();
        }
        FP8_TIMEPOINT(
            s_q_idx == 0 && lane_idx == 0,
            plan.barrier_timing.qw_scale_arrived_ns[scale_warp_idx]
        );
        FP8_MARK_WARP("FP8_MARK 51 qw_scale_published warp=%d", warp_idx);

        CUTE_NO_UNROLL
        for (int k = 0; k < num_k_blocks; ++k) {
            const int cur_buf = k % NUM_BUFS;
#if defined(FP8_FWD_BARRIER_TIMING)
            const bool trace_scale_tile = s_q_idx == 0
                && lane_idx == 0 && k < FP8_TIMING_MAX_TILES;
#endif
            FP8_TIMEPOINT(
                trace_scale_tile,
                plan.barrier_timing.scale[scale_warp_idx][k].tile_start_ns
            );
            FP8_MARK_WARP(
                "FP8_MARK 52 kv_scale_free_wait_before warp=%d tile=%d stage=%d",
                warp_idx,
                k,
                cur_buf
            );
            FP8_TIMED_WAIT(
                trace_scale_tile,
                plan.barrier_timing.scale[scale_warp_idx][k].buffer_free_wait_ns,
                plan.bar_sv_done[cur_buf].wait(
                    ((k / NUM_BUFS) & 1) ^ 1
                )
            );
            FP8_MARK_WARP(
                "FP8_MARK 53 kv_scale_free_wait_after warp=%d tile=%d stage=%d",
                warp_idx,
                k,
                cur_buf
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
            FP8_TIMEPOINT(
                trace_scale_tile,
                plan.barrier_timing.scale[scale_warp_idx][k].arrived_ns
            );
            FP8_MARK_WARP(
                "FP8_MARK 54 kv_scale_published warp=%d tile=%d stage=%d",
                warp_idx,
                k,
                cur_buf
            );
        }
        FP8_MARK_WARP("FP8_MARK 55 scale_loader_done warp=%d", warp_idx);
    }
}

#if defined(FP8_FWD_BARRIER_TIMING)
    if (s_q_idx == 0 && lane_idx == 0) {
        plan.barrier_timing.branch_end_ns[warp_idx] = fp8_timing_now_ns()
            - plan.barrier_timing.origin_ns;
    }
    __syncthreads();
    if (s_q_idx == 0 && threadIdx.x == 0) {
        print_fp8_barrier_timing(plan, num_k_blocks);
    }
#endif
}

#undef FP8_MARK_WARP
#undef FP8_MARK_ONE
#undef FP8_TIMED_WAIT
#undef FP8_TIMEPOINT

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
            {static_cast<uint64_t>(params.stride_kv_s_kv)},
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
