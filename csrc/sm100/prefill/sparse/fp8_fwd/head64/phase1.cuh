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
        cute::print(
            "FP8_TIME tail_stage warp=%d "
            "stage0_wait=%llu stage0_ready=%llu stage0_drained=%llu "
            "stage1_wait=%llu stage1_ready=%llu stage1_drained=%llu\n",
            warp,
            static_cast<unsigned long long>(
                timing.final_o_stage_wait_ns[warp][0]
            ),
            static_cast<unsigned long long>(
                timing.final_o_stage_ready_ns[warp][0]
            ),
            static_cast<unsigned long long>(
                timing.final_o_stage_drained_ns[warp][0]
            ),
            static_cast<unsigned long long>(
                timing.final_o_stage_wait_ns[warp][1]
            ),
            static_cast<unsigned long long>(
                timing.final_o_stage_ready_ns[warp][1]
            ),
            static_cast<unsigned long long>(
                timing.final_o_stage_drained_ns[warp][1]
            )
        );
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
                "smax_wait=%llu waits_done=%llu p_released=%llu "
                "p_scaled=%llu rowmax_ready=%llu softmax_exp_ready=%llu "
                "softmax_ready=%llu sv_wait=%llu s_stored=%llu "
                "o_stage0_arrived=%llu o_stage1_arrived=%llu "
                "o_rescaled=%llu s_arrived=%llu\n",
                warp,
                tile,
                static_cast<unsigned long long>(value.tile_start_ns),
                static_cast<unsigned long long>(value.pair_wait_ns),
                static_cast<unsigned long long>(value.qk_wait_ns),
                static_cast<unsigned long long>(value.valid_wait_ns),
                static_cast<unsigned long long>(value.scale_wait_ns),
                static_cast<unsigned long long>(value.rowmax_wait_ns),
                static_cast<unsigned long long>(value.smax_wait_ns),
                static_cast<unsigned long long>(value.waits_done_ns),
                static_cast<unsigned long long>(value.p_released_ns),
                static_cast<unsigned long long>(value.p_scaled_ns),
                static_cast<unsigned long long>(value.rowmax_ready_ns),
                static_cast<unsigned long long>(value.softmax_exp_ready_ns),
                static_cast<unsigned long long>(value.softmax_ready_ns),
                static_cast<unsigned long long>(value.sv_wait_ns),
                static_cast<unsigned long long>(value.s_stored_ns),
                static_cast<unsigned long long>(value.o_stage_arrived_ns[0]),
                static_cast<unsigned long long>(value.o_stage_arrived_ns[1]),
                static_cast<unsigned long long>(value.o_rescaled_ns),
                static_cast<unsigned long long>(value.s_arrived_ns)
            );
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
            "s_ready_wait=%llu s_ready=%llu "
            "o_stage0_wait=%llu o_stage0_ready=%llu o_stage0_committed=%llu "
            "o_stage1_wait=%llu o_stage1_ready=%llu o_stage1_committed=%llu "
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
            static_cast<unsigned long long>(value.o_stage_wait_ns[0]),
            static_cast<unsigned long long>(value.o_stage_ready_ns[0]),
            static_cast<unsigned long long>(value.o_stage_committed_ns[0]),
            static_cast<unsigned long long>(value.o_stage_wait_ns[1]),
            static_cast<unsigned long long>(value.o_stage_ready_ns[1]),
            static_cast<unsigned long long>(value.o_stage_committed_ns[1]),
            static_cast<unsigned long long>(value.sv_committed_ns)
        );
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

template<
    int NUM_ELEMS_PER_THREAD,
    int TMEM_COL_START,
    int BARRIER_WARP02_SYNC_ID,
    int BARRIER_WARP13_SYNC_ID
>
CUTE_DEVICE
void retrieve_mask_and_reduce_p_f16(
    char* k_validness_base,
    int local_warp_idx,
    int lane_idx,
    auto slot_bar_p_empty_arrival,
    float p_exchange_buf[4][32 * (B_TOPK / 2)],
    __half2 p[NUM_ELEMS_PER_THREAD / 2]
) {
    static_assert(NUM_ELEMS_PER_THREAD == 32);
    static_assert(BARRIER_WARP13_SYNC_ID == BARRIER_WARP02_SYNC_ID + 1);

    __half2 p_peer[NUM_ELEMS_PER_THREAD / 2];
    if (local_warp_idx < 2) {
        ku::tmem_ld_32dp32bNx_pack16<NUM_ELEMS_PER_THREAD>(TMEM_COL_START, p);
        ku::tmem_ld_32dp32bNx_pack16<NUM_ELEMS_PER_THREAD>(
            TMEM_COL_START + NUM_ELEMS_PER_THREAD,
            p_peer
        );
    } else {
        ku::tmem_ld_32dp32bNx_pack16<NUM_ELEMS_PER_THREAD>(TMEM_COL_START, p_peer);
        ku::tmem_ld_32dp32bNx_pack16<NUM_ELEMS_PER_THREAD>(
            TMEM_COL_START + NUM_ELEMS_PER_THREAD,
            p
        );
    }
    cutlass::arch::fence_view_async_tmem_load();
    ku::tcgen05_before_thread_sync();
    slot_bar_p_empty_arrival();

    const uint32_t is_k_valid = *reinterpret_cast<uint32_t*>(
        k_validness_base
        + (local_warp_idx >= 2 ? NUM_ELEMS_PER_THREAD / 8 : 0)
    );
    constexpr uint32_t NEG_INF_F16X2 = 0xfc00fc00u;
    CUTE_UNROLL
    for (int i = 0; i < NUM_ELEMS_PER_THREAD / 2; ++i) {
        const uint32_t valid_pair = (is_k_valid >> (i * 2)) & 0x3u;
        const uint32_t valid_mask =
            (valid_pair & 0x1u ? 0x0000ffffu : 0u)
            | (valid_pair & 0x2u ? 0xffff0000u : 0u);
        uint32_t& p_bits = reinterpret_cast<uint32_t&>(p[i]);
        p_bits = (p_bits & valid_mask) | (NEG_INF_F16X2 & ~valid_mask);
    }

    uint32_t* exchange_write = reinterpret_cast<uint32_t*>(
        p_exchange_buf[local_warp_idx ^ 2]
    );
    CUTE_UNROLL
    for (int i = 0; i < NUM_ELEMS_PER_THREAD / 8; ++i) {
        ku::st_shared(
            &exchange_write[i * 32 * 4 + lane_idx * 4],
            *reinterpret_cast<__int128_t*>(p_peer + i * 4)
        );
    }
    NamedBarrier::arrive_and_wait(
        64,
        BARRIER_WARP02_SYNC_ID + (local_warp_idx & 1)
    );

    uint32_t* exchange_read = reinterpret_cast<uint32_t*>(
        p_exchange_buf[local_warp_idx]
    );
    CUTE_UNROLL
    for (int i = 0; i < NUM_ELEMS_PER_THREAD / 8; ++i) {
        uint4 peer_values = *reinterpret_cast<uint4*>(
            &exchange_read[i * 32 * 4 + lane_idx * 4]
        );
        __half2* peer_half2 = reinterpret_cast<__half2*>(&peer_values);
        CUTE_UNROLL
        for (int j = 0; j < 4; ++j) {
            p[i * 4 + j] = ku::fp16x2_add(p[i * 4 + j], peer_half2[j]);
        }
    }
}

template<int D_V_, int CHUNK_SIZE>
CUTE_DEVICE
void rescale_o_f16(float scale_factor, uint32_t tmem_col_start) {
    const __half2 scale = __float2half2_rn(scale_factor);
    __half2 o[CHUNK_SIZE / 2];

    CUTE_UNROLL
    for (int chunk_idx = 0; chunk_idx < (D_V_ / 2) / CHUNK_SIZE; ++chunk_idx) {
        const int tmem_col = tmem_col_start + chunk_idx * CHUNK_SIZE;
        ku::tmem_ld_32dp32bNx_pack16<CHUNK_SIZE>(tmem_col, o);
        cutlass::arch::fence_view_async_tmem_load();
        CUTE_UNROLL
        for (int i = 0; i < CHUNK_SIZE / 2; ++i) {
            o[i] = ku::fp16x2_mul(o[i], scale);
        }
        ku::tmem_st_32dp32bNx_unpack16<CHUNK_SIZE>(tmem_col, o);
        cutlass::arch::fence_view_async_tmem_store();
    }
}

CUTE_DEVICE
void wait_all_o_mma_stages(
    transac_bar_t (&barriers)[NUM_O_MMA_STAGES],
    int phase
) {
    CUTE_UNROLL
    for (int stage = 0; stage < NUM_O_MMA_STAGES; ++stage) {
        barriers[stage].wait(phase);
    }
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
    Tensor tO = partition_fragment_C(
        tiled_mma_O, Shape<Int<B_H>, Int<O_MMA_STAGE_SIZE>>{}
    );
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
                CUTE_UNROLL
                for (int stage = 0; stage < NUM_O_MMA_STAGES; ++stage) {
                    plan.bar_sv_done[i][stage].init(1);
                }
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
            CUTE_UNROLL
            for (int stage = 0; stage < NUM_O_MMA_STAGES; ++stage) {
                plan.bar_so_ready[stage].init(128);
            }
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

            // FP16 accumulators are packed two-at-a-time when read from TMEM.
            __half2 p[NUM_ELEMS_PER_THREAD / 2];
            auto release_p = [&]() { plan.bar_p_free[p_idx].arrive(); };
            if (p_idx == 0) {
                retrieve_mask_and_reduce_p_f16<
                    NUM_ELEMS_PER_THREAD,
                    tmem_cols::P,
                    NamedBarriers::wg0_warp02_sync,
                    NamedBarriers::wg0_warp13_sync
                >(
                    plan.is_k_valid[cur_buf], warp_idx, lane_idx,
                    release_p, plan.p_exchange_buf, p
                );
            } else if (p_idx == 1) {
                retrieve_mask_and_reduce_p_f16<
                    NUM_ELEMS_PER_THREAD,
                    tmem_cols::P + 64,
                    NamedBarriers::wg0_warp02_sync,
                    NamedBarriers::wg0_warp13_sync
                >(
                    plan.is_k_valid[cur_buf], warp_idx, lane_idx,
                    release_p, plan.p_exchange_buf, p
                );
            } else {
                retrieve_mask_and_reduce_p_f16<
                    NUM_ELEMS_PER_THREAD,
                    tmem_cols::P + 128,
                    NamedBarriers::wg0_warp02_sync,
                    NamedBarriers::wg0_warp13_sync
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

            __half2 kv_scale_h2[NUM_ELEMS_PER_THREAD / 2];
            const float q_sm_scale = q_scale * params.sm_scale_div_log2;
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD / 2; ++i) {
                const float2 kv_scale_pair = make_float2(
                    __shfl_sync(0xffffffff, lane_kv_scale, i * 2),
                    __shfl_sync(0xffffffff, lane_kv_scale, i * 2 + 1)
                );
                kv_scale_h2[i] = __float22half2_rn(kv_scale_pair);
                const float2 p_scale_pair = ku::float2_mul(
                    kv_scale_pair,
                    make_float2(q_sm_scale, q_sm_scale)
                );
                p[i] = ku::fp16x2_mul(
                    p[i],
                    __float22half2_rn(p_scale_pair)
                );
            }
            FP8_TIMEPOINT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].p_scaled_ns
            );
            
            // Get rowwise max of Pi
            const uint32_t neg_inf_f16x2 = 0xfc00fc00u;
            __half2 cur_pi_max_h2 = reinterpret_cast<const __half2&>(
                neg_inf_f16x2
            );
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD / 2; ++i) {
                cur_pi_max_h2 = ku::fp16x2_max(cur_pi_max_h2, p[i]);
            }
            const float2 cur_pi_max_pair = __half22float2(cur_pi_max_h2);
            float cur_pi_max = max(cur_pi_max_pair.x, cur_pi_max_pair.y);

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

            // Absorb the token factor into S, then quantize each head row.
            uint32_t s[NUM_ELEMS_PER_THREAD / 4];
            __half2 sum_h2[4] = {
                __float2half2_rn(0.0f), __float2half2_rn(0.0f),
                __float2half2_rn(0.0f), __float2half2_rn(0.0f)
            };
            __half2 local_s_max_h2 = __float2half2_rn(0.0f);
            const __half2 new_max_h2 = __float2half2_rn(
                max(new_max, -65504.0f)
            );
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD / 2; ++i) {
                const __half2 softmax_s = ku::fp16x2_exp2(
                    ku::fp16x2_sub(p[i], new_max_h2)
                );
                sum_h2[i & 3] = ku::fp16x2_add(sum_h2[i & 3], softmax_s);
                p[i] = ku::fp16x2_mul(softmax_s, kv_scale_h2[i]);
                local_s_max_h2 = ku::fp16x2_max(
                    local_s_max_h2,
                    ku::fp16x2_abs(p[i])
                );
            }
            sum_h2[0] = ku::fp16x2_add(sum_h2[0], sum_h2[1]);
            sum_h2[2] = ku::fp16x2_add(sum_h2[2], sum_h2[3]);
            const float2 cur_sum_pair = __half22float2(
                ku::fp16x2_add(sum_h2[0], sum_h2[2])
            );
            const float cur_sum = cur_sum_pair.x + cur_sum_pair.y;
            const float2 local_s_max_pair = __half22float2(local_s_max_h2);
            const float local_s_max = max(
                local_s_max_pair.x,
                local_s_max_pair.y
            );
            FP8_TIMEPOINT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].softmax_exp_ready_ns
            );

            plan.rowwise_li_buf[idx_in_warpgroup] = local_s_max;
            FP8_TIMED_WAIT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].smax_wait_ns,
                (NamedBarrier::arrive_and_wait(
                    128, NamedBarriers::wg0_sync
                ))
            );
            const float s_max = max(
                local_s_max,
                plan.rowwise_li_buf[idx_in_warpgroup ^ B_H]
            );
            const e8m0 s_scale_e8m0(
                s_max > 0.0f
                    ? s_max * S_ACCUM_HEADROOM / FP8_MAX
                    : 1.0f
            );
            const float current_s_scale = float(s_scale_e8m0);
            const __half2 quant_multiplier = __float2half2_rn(
                1.0f / current_s_scale
            );
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD / 2; i += 2) {
                const uint16_t s01 = ku::fp16x2_to_e4m3x2_bits(
                    ku::fp16x2_mul(p[i], quant_multiplier)
                );
                const uint16_t s23 = ku::fp16x2_to_e4m3x2_bits(
                    ku::fp16x2_mul(p[i + 1], quant_multiplier)
                );
                s[i / 2] = static_cast<uint32_t>(s01)
                    | (static_cast<uint32_t>(s23) << 16);
            }
            li = fma(li, scale_for_old, cur_sum);
            FP8_TIMEPOINT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].softmax_ready_ns
            );

            // Wait for last SV gemm, write S
            if (k > 0) {
                FP8_MARK_WARP(
                    "FP8_MARK 18 wg0_prev_sv_wait_before warp=%d tile=%d",
                    warp_idx,
                    k
                );
                FP8_TIMED_WAIT(
                    trace_wg0_tile,
                    plan.barrier_timing.wg0[warp_idx][k].sv_wait_ns,
                    wait_all_o_mma_stages(
                        plan.bar_sv_done[(k - 1) % NUM_BUFS],
                        ((k - 1) / NUM_BUFS) & 1
                    )
                );
                FP8_MARK_WARP(
                    "FP8_MARK 19 wg0_prev_sv_wait_after warp=%d tile=%d",
                    warp_idx,
                    k
                );
            }
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD; i += 16) {
                *reinterpret_cast<uint4*>(&sS(h, token_base + i)) = make_uint4(
                    s[i / 4],
                    s[i / 4 + 1],
                    s[i / 4 + 2],
                    s[i / 4 + 3]
                );
            }
            FP8_TIMEPOINT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].s_stored_ns
            );

            // Publish each N=256 stage before rescaling the next stage.
            const float o_rescale = scale_for_old
                * s_scale_for_o / current_s_scale;
            fence_view_async_shared();
            for_each(make_seq<NUM_O_MMA_STAGES>{}, [&](auto stage) {
                for_each(make_seq<O_CHUNKS_PER_MMA_STAGE>{}, [&](auto local_chunk) {
                    constexpr int chunk = decltype(stage)::value
                        * O_CHUNKS_PER_MMA_STAGE
                        + decltype(local_chunk)::value;
                    if (k > 0) {
                        ku::tcgen05_after_thread_sync();
                        rescale_o_f16<O_CHUNK_SIZE, 32>(
                            o_rescale,
                            tmem_cols::O + chunk * O_TMEM_COLS_PER_CHUNK
                        );
                        ku::tcgen05_before_thread_sync();
                    }
                });
                plan.bar_so_ready[stage].arrive();
                FP8_TIMEPOINT(
                    trace_wg0_tile,
                    plan.barrier_timing.wg0[warp_idx][k]
                        .o_stage_arrived_ns[stage]
                );
            });
            FP8_TIMEPOINT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].o_rescaled_ns
            );
            s_scale_for_o = current_s_scale;
            
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

        __half2 o[B_EPI / 2];
        bool have_valid_indices = __any_sync(0xffffffff, li != 0);  // Prevent some threads' li == 0 and some threads' li != 0 which lead to deadlock during ku::tmem_ld
        if (!have_valid_indices) {
            // If there are no valid indices, we set o[i] to 0 and don't load from TMEM
            CUTE_UNROLL
            for (int i = 0; i < B_EPI / 2; ++i) {
                o[i] = __float2half2_rn(0.0f);
            }
            output_scale = 1.0f;
        }

        bf16* sO_addrs[8];
        CUTE_UNROLL
        for (int i = 0; i < B_EPI/8; ++i) {
            sO_addrs[i] = &sO(idx_in_warpgroup%64, i*8);
        }

        for_each(make_seq<2>{}, [&](auto c_value) {
            constexpr int c = decltype(c_value)::value;
            // Each tile: 64 x 256
            for_each(make_seq<(D_V / 4) / B_EPI>{}, [&](auto k_value) {
                constexpr int epi_k = decltype(k_value)::value;
                constexpr int o_chunk = c * ((D_V / 4) / B_EPI) + epi_k;
                constexpr int o_mma_stage =
                    o_chunk / O_CHUNKS_PER_MMA_STAGE;
                if constexpr (o_chunk % O_CHUNKS_PER_MMA_STAGE == 0) {
                    FP8_MARK_WARP(
                        "FP8_MARK 1b wg0_final_o_stage_wait_before warp=%d stage=%d",
                        warp_idx,
                        o_mma_stage
                    );
                    FP8_TIMED_WAIT(
                        s_q_idx == 0 && lane_idx == 0,
                        plan.barrier_timing
                            .final_o_stage_wait_ns[warp_idx][o_mma_stage],
                        plan.bar_sv_done[
                            (num_k_blocks - 1) % NUM_BUFS
                        ][o_mma_stage].wait(
                            ((num_k_blocks - 1) / NUM_BUFS) & 1
                        )
                    );
                    FP8_TIMEPOINT(
                        s_q_idx == 0 && lane_idx == 0,
                        plan.barrier_timing
                            .final_o_stage_ready_ns[warp_idx][o_mma_stage]
                    );
#if defined(FP8_FWD_BARRIER_TIMING)
                    if constexpr (o_mma_stage == 0) {
                        plan.barrier_timing.final_sv_wait_ns[warp_idx] =
                            plan.barrier_timing
                                .final_o_stage_wait_ns[warp_idx][0];
                        plan.barrier_timing.final_sv_ready_ns[warp_idx] =
                            plan.barrier_timing
                                .final_o_stage_ready_ns[warp_idx][0];
                    }
#endif
                    FP8_MARK_WARP(
                        "FP8_MARK 1c wg0_final_o_stage_wait_after warp=%d stage=%d",
                        warp_idx,
                        o_mma_stage
                    );
                    ku::tcgen05_after_thread_sync();
                }

                // Load O from tO
                if (have_valid_indices) {
                    ku::tmem_ld_32dp32bNx_pack16<B_EPI>(
                        tmem_cols::O + c * 128 + epi_k * B_EPI,
                        o
                    );
                    cutlass::arch::fence_view_async_tmem_load();
                }

                const int output_group = c * 4
                    + (idx_in_warpgroup / B_H) * 2 + epi_k;
                const int d_group = output_group;
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
                        const float2 o_scaled = ku::float2_mul(
                            __half22float2(o[i * 4 + j]),
                            output_scale_float2
                        );
                        o_bf16[j] = __float22bfloat162_rn(o_scaled);
                    }
                    *reinterpret_cast<uint128_t*>(
                        sO_addrs[i] + output_group * B_EPI * B_H
                    ) = *reinterpret_cast<uint128_t*>(o_bf16);
                }

                // Sync
                fence_view_async_shared();
                NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
                
                if (warp_idx == 0 && elect_one_sync()) {
                    constexpr int epi_chunk_idx =
                        c * (D_V / 2 / B_EPI) + epi_k;
                    cute::copy(
                        tma_params.tma_O,
                        thr_tma.partition_S(sO_divided(_, _, epi_chunk_idx)),
                        thr_tma.partition_D(tma_gO(_, _, epi_chunk_idx))
                    );
                }
                if (warp_idx == 1 && elect_one_sync()) {
                    constexpr int epi_chunk_idx = c * (D_V / 2 / B_EPI)
                        + (D_V / B_EPI / 4) + epi_k;
                    cute::copy(
                        tma_params.tma_O,
                        thr_tma.partition_S(sO_divided(_, _, epi_chunk_idx)),
                        thr_tma.partition_D(tma_gO(_, _, epi_chunk_idx))
                    );
                }
                if constexpr (
                    o_chunk % O_CHUNKS_PER_MMA_STAGE
                    == O_CHUNKS_PER_MMA_STAGE - 1
                ) {
                    FP8_TIMEPOINT(
                        s_q_idx == 0 && lane_idx == 0,
                        plan.barrier_timing
                            .final_o_stage_drained_ns[warp_idx][o_mma_stage]
                    );
                }
            });
        });
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
                    wait_all_o_mma_stages(
                        plan.bar_sv_done[cur_buf],
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
                    } else if (p_stage == 1) {
                        ku::utcmma_ts(tiled_mma_P, kv_part_idx ? tQ_part1 : tQ_part0, sK_divided(_, _, kv_part_idx), tP1, clear_accum);
                    } else {
                        ku::utcmma_ts(tiled_mma_P, kv_part_idx ? tQ_part1 : tQ_part0, sK_divided(_, _, kv_part_idx), tP2, clear_accum);
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
                    // O += S(i-1)V(i-1)
                    int cur_buf = (k-1)%NUM_BUFS;

                    Tensor sS = make_tensor(make_smem_ptr(plan.s.data()), SmemLayoutS{});
                    Tensor sV = make_tensor(make_smem_ptr(plan.qkvo.kv[cur_buf].data()), SmemLayoutV{});
                    Tensor sV_divided = flat_divide(
                        sV, Tile<Int<O_MMA_STAGE_SIZE>, Int<B_TOPK>>{}
                    )(_, _, _, _0{});

                    for_each(make_seq<NUM_O_MMA_STAGES>{}, [&](auto stage) {
                        constexpr int stage_idx = decltype(stage)::value;
                        FP8_MARK_ONE(
                            "FP8_MARK 39 warp8_o_stage_wait_before sv_tile=%d stage=%d",
                            k - 1,
                            stage_idx
                        );
                        FP8_TIMED_WAIT(
                            trace_mma_iter,
                            plan.barrier_timing.mma[k]
                                .o_stage_wait_ns[stage],
                            plan.bar_so_ready[stage].wait((k - 1) & 1)
                        );
                        FP8_TIMEPOINT(
                            trace_mma_iter,
                            plan.barrier_timing.mma[k]
                                .o_stage_ready_ns[stage]
                        );
#if defined(FP8_FWD_BARRIER_TIMING)
                        if constexpr (stage_idx == 0) {
                            plan.barrier_timing.mma[k].s_ready_wait_ns =
                                plan.barrier_timing.mma[k]
                                    .o_stage_wait_ns[stage];
                            plan.barrier_timing.mma[k].s_ready_ns =
                                plan.barrier_timing.mma[k]
                                    .o_stage_ready_ns[stage];
                        }
#endif
                        FP8_MARK_ONE(
                            "FP8_MARK 3a warp8_o_stage_wait_after sv_tile=%d stage=%d",
                            k - 1,
                            stage_idx
                        );
                        ku::tcgen05_after_thread_sync();

                        tO.data().get() = tmem_cols::O
                            + stage_idx * O_TMEM_COLS_PER_MMA_STAGE;
                        ku::utcmma_ss(
                            tiled_mma_O,
                            sS,
                            sV_divided(_, _, stage),
                            tO,
                            k == 1
                        );
                        ku::umma_arrive_noelect(
                            plan.bar_sv_done[cur_buf][stage]
                        );
                        FP8_TIMEPOINT(
                            trace_mma_iter,
                            plan.barrier_timing.mma[k]
                                .o_stage_committed_ns[stage]
                        );
                        FP8_MARK_ONE(
                            "FP8_MARK 3b warp8_sv_stage_committed sv_tile=%d stage=%d",
                            k - 1,
                            stage_idx
                        );
                    });
                    FP8_TIMEPOINT(
                        trace_mma_iter,
                        plan.barrier_timing.mma[k].sv_committed_ns
                    );
                    FP8_MARK_ONE(
                        "FP8_MARK 3c warp8_sv_committed sv_tile=%d stage=%d",
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
            plan.kv_w_scale[lane_idx] = ue8m0_bits_to_float(
                __ldg(params.kv_scale_w + lane_idx)
            );
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
                wait_all_o_mma_stages(
                    plan.bar_sv_done[cur_buf],
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
