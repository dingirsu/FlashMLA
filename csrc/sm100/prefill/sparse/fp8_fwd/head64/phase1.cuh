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

template<bool HAVE_QK_TAIL>
CUTE_DEVICE
void print_fp8_barrier_timing(
    const SharedMemoryPlanT<HAVE_QK_TAIL>& plan,
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
                static_cast<unsigned long long>(value.o_rescaled_ns),
                static_cast<unsigned long long>(value.s_arrived_ns)
            );
        }
        CUTE_UNROLL
        for (int warp = 0; warp < 4; ++warp) {
            CUTE_UNROLL
            for (int stripe = 0; stripe < NUM_SV_TMEM_BLOCKS; ++stripe) {
                const auto& value = timing.o_rescale[warp][tile][stripe];
                cute::print(
                    "FP8_TIME rescale warp=%d tile=%d stripe=%d active=%llu "
                    "start=%llu sv_wait=%llu sv_ready=%llu tmem_load=%llu "
                    "fp32_mul=%llu tmem_store=%llu wg0_sync=%llu\n",
                    warp,
                    tile,
                    stripe,
                    static_cast<unsigned long long>(value.warp_rescale_active),
                    static_cast<unsigned long long>(value.stripe_start_ns),
                    static_cast<unsigned long long>(value.sv_wait_ns),
                    static_cast<unsigned long long>(value.sv_ready_ns),
                    static_cast<unsigned long long>(value.tmem_load_done_ns),
                    static_cast<unsigned long long>(value.fp32_mul_done_ns),
                    static_cast<unsigned long long>(value.tmem_store_done_ns),
                    static_cast<unsigned long long>(value.wg0_sync_done_ns)
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
            "s_ready_wait=%llu s_ready=%llu sv_committed=%llu\n",
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

CUTE_DEVICE
void rescale_o_tmem_stripe(
    float scale,
    uint32_t tmem_col
#if defined(FP8_FWD_BARRIER_TIMING)
    , uint64_t timing_origin_ns,
    Fp8ORescaleStripeTiming* stripe_timing
#endif
) {
#if !defined(FP8_BENCH_O_RESCALE_SKIP_TMEM) \
    || !defined(FP8_BENCH_O_RESCALE_SKIP_MUL)
    float2 o[SV_TMEM_COLS_PER_BLOCK / 2];
    const float2 scale2 = make_float2(scale, scale);
#endif

#if defined(FP8_BENCH_O_RESCALE_SKIP_TMEM)
#if !defined(FP8_BENCH_O_RESCALE_SKIP_MUL)
    // Keep the FP32x2 instruction stream when measuring TMEM load/store cost.
    CUTE_UNROLL
    for (int i = 0; i < SV_TMEM_COLS_PER_BLOCK / 2; ++i) {
        o[i] = scale2;
    }
#endif
#else
    ku::tmem_ld_32dp32bNx<SV_TMEM_COLS_PER_BLOCK>(tmem_col, o);
    cutlass::arch::fence_view_async_tmem_load();
#endif
#if defined(FP8_FWD_BARRIER_TIMING)
    if (stripe_timing != nullptr) {
        stripe_timing->tmem_load_done_ns = fp8_timing_now_ns()
            - timing_origin_ns;
    }
#endif
#if !defined(FP8_BENCH_O_RESCALE_SKIP_MUL)
    CUTE_UNROLL
    for (int i = 0; i < SV_TMEM_COLS_PER_BLOCK / 2; ++i) {
        o[i] = ku::float2_mul(o[i], scale2);
    }
#endif
#if defined(FP8_FWD_BARRIER_TIMING)
    if (stripe_timing != nullptr) {
        stripe_timing->fp32_mul_done_ns = fp8_timing_now_ns()
            - timing_origin_ns;
    }
#endif
#if !defined(FP8_BENCH_O_RESCALE_SKIP_TMEM)
    ku::tmem_st_32dp32bNx<SV_TMEM_COLS_PER_BLOCK>(tmem_col, o);
    cutlass::arch::fence_view_async_tmem_store();
#endif
#if defined(FP8_FWD_BARRIER_TIMING)
    if (stripe_timing != nullptr) {
        stripe_timing->tmem_store_done_ns = fp8_timing_now_ns()
            - timing_origin_ns;
    }
#endif
}

template<bool HAVE_QK_TAIL, typename TmaParams>
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
    using Plan = SharedMemoryPlanT<HAVE_QK_TAIL>;
    constexpr int NUM_MAIN_BUFS = Plan::NUM_MAIN_BUFS;
    constexpr int QK_DIM = HAVE_QK_TAIL ? D_Q + 64 : D_Q;
    Plan &plan = *reinterpret_cast<Plan*>(wksp_buf);
    if (warp_idx == 0 && elect_one_sync()) {
        cute::prefetch_tma_descriptor(tma_params.tma_O.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(&tma_params.tensor_map_kv);
    }

    int* gIndices = params.indices + s_q_idx*params.stride_indices_s_q; // [topk]

    TiledMMA tiled_mma_P = TiledMMA_P{};
    TiledMMA tiled_mma_O = TiledMMA_O{};

    Tensor tQ = tiled_mma_P.get_slice(_0{}).make_fragment_A(
        partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<D_K>>{})
    );
    Tensor tQ_tail = tiled_mma_P.get_slice(_0{}).make_fragment_A(
        partition_shape_A(tiled_mma_P, Shape<Int<B_H>, Int<64>>{})
    );
    Tensor tP0 = partition_fragment_C(
        tiled_mma_P, Shape<Int<B_H>, Int<B_TOPK>>{}
    );
    Tensor tP1 = partition_fragment_C(
        tiled_mma_P, Shape<Int<B_H>, Int<B_TOPK>>{}
    );
    Tensor tO = partition_fragment_C(
        tiled_mma_O, Shape<Int<B_H>, Int<SV_M>>{}
    );
    tP0.data().get() = tmem_cols::P0;
    tP1.data().get() = tmem_cols::P1;
    tQ.data().get() = tmem_cols::Q;
    tQ_tail.data().get() = tmem_cols::Q_TAIL;
    tO.data().get() = tmem_cols::O;

        if (warp_idx == 0) {
        if (elect_one_sync()) {
            // Copy Q
            cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
            if constexpr (HAVE_QK_TAIL) {
                cute::prefetch_tma_descriptor(
                    tma_params.tma_Q_tail.get_tma_descriptor()
                );
            }

            plan.bar_prologue.init(1);
            fence_barrier_init();

            Tensor gQ = tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx);
            Tensor sQ = make_tensor(make_smem_ptr(plan.qkvo.q.q.data()), SmemLayoutQ{});
            ku::launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_prologue, TMA::CacheHintSm90::EVICT_FIRST);
            if constexpr (HAVE_QK_TAIL) {
                Tensor gQ_tail = tma_params.tma_Q_tail.get_tma_tensor(
                    tma_params.shape_Q_tail
                )(_, _, s_q_idx);
                Tensor sQ_tail = make_tensor(
                    make_smem_ptr(plan.qk_tail.q.data()), SmemLayoutQTail{}
                );
                ku::launch_tma_copy(
                    tma_params.tma_Q_tail,
                    gQ_tail,
                    sQ_tail,
                    plan.bar_prologue,
                    TMA::CacheHintSm90::EVICT_FIRST
                );
            }

            cute::prefetch_tma_descriptor(tma_params.tma_O.get_tma_descriptor());
            cute::prefetch_tma_descriptor(&(tma_params.tensor_map_kv));
            if constexpr (HAVE_QK_TAIL) {
                cute::prefetch_tma_descriptor(&tma_params.tensor_map_kv_tail);
            }
            
            // Initialize other barriers
            plan.bar_prologue_utccp.init(1);
#if !defined(FP8_FWD_QK576)
            plan.bar_qw_scale_ready.init(2);
#endif
            CUTE_UNROLL
            for (int i = 0; i < NUM_MAIN_BUFS; ++i) {
                plan.bar_sv_done[i].init(1);
                plan.bar_kv_ready[i][0].init(1);
                plan.bar_kv_ready[i][1].init(1);
                plan.bar_kv_scale_ready[i].init(2);
                plan.bar_k_valid_ready[i].init(B_TOPK/8);
                plan.bar_k_valid_free[i].init(128);
            }
            for (int i = 0; i < NUM_MAIN_BUFS; ++i) {
                plan.bar_qk_part_done[i].init(1);
                plan.bar_qk_done[i].init(1);
            }
            for (int i = 0; i < NUM_P_BUFS; ++i) {
                plan.bar_p_free[i].init(128); // warp group 0 touch this
            }
            CUTE_UNROLL
            for (int i = 0; i < NUM_QK_TAIL_BUFS; ++i) {
                plan.bar_kv_tail_ready[i].init(1);
            }
            plan.bar_so_ready.init(128);
            CUTE_UNROLL
            for (int dv_block = 0;
                 dv_block < NUM_SV_TMEM_BLOCKS;
                 ++dv_block) {
                plan.bar_o_rescale_done[dv_block].init(128);
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

    if (warpgroup_idx == 0) {
        cutlass::arch::warpgroup_reg_alloc<232>();
#if !defined(FP8_FWD_QK576)
        FP8_TIMED_WAIT(
            s_q_idx == 0 && lane_idx == 0,
            plan.barrier_timing.qw_scale_wait_ns[warp_idx],
            plan.bar_qw_scale_ready.wait(0)
        );
#endif

        float mi = MAX_INIT_VAL;
        float li = 0.0f;
        float real_mi = -CUDART_INF_F;
        float s_scale_for_o = 1.0f;

        const int h = idx_in_warpgroup % B_H;
        const int token_group = idx_in_warpgroup / B_H;
        const int token_base = token_group * (B_TOPK / 2);
#if defined(FP8_FWD_QK576)
        const uint8_t* q_scale_base =
            reinterpret_cast<const uint8_t*>(params.q)
            + static_cast<int64_t>(s_q_idx) * params.stride_q_s_q
            + B_H * QK_DIM;
        const float q_scale = ue8m0_bits_to_float(
            __ldg(q_scale_base + h)
        );
#else
        const float q_scale = plan.q_head_scale[h];
#endif
        const float qk_base_scale = q_scale * params.sm_scale_div_log2;
        const float2 qk_base_scale2 = make_float2(
            qk_base_scale,
            qk_base_scale
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
            const int cur_buf = k % NUM_MAIN_BUFS;
            const int p_idx = k % NUM_P_BUFS;
            const int qk_tail_buf = k % NUM_QK_TAIL_BUFS;
            const int qk_bar_idx = HAVE_QK_TAIL ? qk_tail_buf : p_idx;
            const int qk_bar_phase = HAVE_QK_TAIL
                ? ((k / NUM_QK_TAIL_BUFS) & 1)
                : ((k / NUM_P_BUFS) & 1);
            FP8_TIMED_WAIT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].qk_wait_ns,
                plan.bar_qk_done[qk_bar_idx].wait(qk_bar_phase)
            );
            float p[NUM_ELEMS_PER_THREAD];
            ku::tcgen05_after_thread_sync();
            const uint32_t p_tmem_col = p_idx == 0
                ? tmem_cols::P0
                : tmem_cols::P1;
            ku::tmem_ld_32dp32bNx<NUM_ELEMS_PER_THREAD>(p_tmem_col, p);
            cutlass::arch::fence_view_async_tmem_load();
            ku::tcgen05_before_thread_sync();
            plan.bar_p_free[p_idx].arrive();
            FP8_TIMED_WAIT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].valid_wait_ns,
                plan.bar_k_valid_ready[cur_buf].wait((k / NUM_MAIN_BUFS) & 1)
            );
            FP8_TIMED_WAIT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].scale_wait_ns,
                plan.bar_kv_scale_ready[cur_buf].wait((k / NUM_MAIN_BUFS) & 1)
            );
            FP8_TIMEPOINT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].waits_done_ns
            );

            // Shuffles expose the token scales for this thread's half-tile.
            float lane_kv_scale[NUM_ELEMS_PER_THREAD / 32];
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD / 32; ++i) {
                lane_kv_scale[i] = plan.kv_token_scale[cur_buf][
                    token_base + i * 32 + lane_idx
                ];
            }
            // Keep both the token scale and the complete QK scale in
            // registers.  The latter avoids rebuilding qk_base_scale *
            // kv_scale while walking P, and both paths use the FP32x2
            // instruction.
            float kv_scale[NUM_ELEMS_PER_THREAD];
            float cur_pi_max = -CUDART_INF_F;
            {
                float2 qk_scale[NUM_ELEMS_PER_THREAD / 2];
                CUTE_UNROLL
                for (int i = 0; i < NUM_ELEMS_PER_THREAD / 2; ++i) {
                    const int base = i * 2;
                    const float2 token_scale = make_float2(
                        __shfl_sync(
                            0xffffffff,
                            lane_kv_scale[(base + 0) / 32],
                            (base + 0) % 32
                        ),
                        __shfl_sync(
                            0xffffffff,
                            lane_kv_scale[(base + 1) / 32],
                            (base + 1) % 32
                        )
                    );
                    kv_scale[base + 0] = token_scale.x;
                    kv_scale[base + 1] = token_scale.y;
                    qk_scale[i] = ku::float2_mul(
                        qk_base_scale2,
                        token_scale
                    );
                }

                // The two WG0 thread halves cover disjoint token ranges for
                // each head; max and li are merged across the halves below.
                const uint32_t* valid_masks =
                    reinterpret_cast<const uint32_t*>(
                        plan.is_k_valid[cur_buf] + token_base / 8
                    );
                const uint32_t valid0 = valid_masks[0];
                uint32_t valid1 = 0;
                if constexpr (NUM_ELEMS_PER_THREAD > 32) {
                    valid1 = valid_masks[1];
                }

                CUTE_UNROLL
                for (int i = 0; i < NUM_ELEMS_PER_THREAD; ++i) {
                    const uint32_t mask = i < 32 ? valid0 : valid1;
                    const int bit = i & 31;

                    if (((mask >> bit) & 1u) == 0) {
                        p[i] = -CUDART_INF_F;
                    }
                }

                plan.bar_k_valid_free[cur_buf].arrive();
                FP8_TIMEPOINT(
                    trace_wg0_tile,
                    plan.barrier_timing.wg0[warp_idx][k].p_released_ns
                );

                // P is already resident in registers; one max chain is
                // cheaper than carrying four independent reductions on this
                // path.
                CUTE_UNROLL
                for (int i = 0; i < NUM_ELEMS_PER_THREAD; i += 2) {
                    const float2 scaled_p = ku::float2_mul(
                        make_float2(p[i + 0], p[i + 1]),
                        qk_scale[i / 2]
                    );
                    p[i + 0] = scaled_p.x;
                    p[i + 1] = scaled_p.y;
                    cur_pi_max = max(cur_pi_max, scaled_p.x);
                    cur_pi_max = max(cur_pi_max, scaled_p.y);
                }
            }
            FP8_TIMEPOINT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].p_scaled_ns
            );
            
            plan.rowwise_max_buf[idx_in_warpgroup] = cur_pi_max;
            FP8_TIMED_WAIT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].rowmax_wait_ns,
                (NamedBarrier::arrive_and_wait(
                    128, NamedBarriers::wg0_sync
                ))
            );
            cur_pi_max = max(
                cur_pi_max,
                plan.rowwise_max_buf[idx_in_warpgroup ^ B_H]
            );
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
            float cur_sum = 0.0f;
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD; i += 2) {
                const float2 softmax_s = make_float2(
                    exp2f(p[i + 0] - new_max),
                    exp2f(p[i + 1] - new_max)
                );
                cur_sum += softmax_s.x + softmax_s.y;
                const float2 s_pair = ku::float2_mul(
                    softmax_s,
                    make_float2(
                        kv_scale[i + 0],
                        kv_scale[i + 1]
                    )
                );
                p[i + 0] = s_pair.x;
                p[i + 1] = s_pair.y;
            }
            FP8_TIMEPOINT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].softmax_exp_ready_ns
            );

            // Reuse the otherwise-idle LSE exchange scratch to make the
            // no-rescale decision uniform across WG0.  The named barrier
            // below already orders this four-warp publication.
            if (lane_idx == 0) {
                plan.rowwise_li_buf[warp_idx] = static_cast<float>(
                    should_scale_o
                );
            }
            FP8_TIMED_WAIT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].smax_wait_ns,
                (NamedBarrier::arrive_and_wait(
                    128, NamedBarriers::wg0_sync
                ))
            );
            bool wg0_needs_o_rescale = false;
            if (lane_idx == 0) {
                wg0_needs_o_rescale = plan.rowwise_li_buf[0] != 0.0f
                    || plan.rowwise_li_buf[1] != 0.0f
                    || plan.rowwise_li_buf[2] != 0.0f
                    || plan.rowwise_li_buf[3] != 0.0f;
            }
            wg0_needs_o_rescale = __shfl_sync(
                0xffffffff, wg0_needs_o_rescale, 0
            );
            // const float s_max = max(
            //     local_s_max,
            //     plan.rowwise_li_buf[idx_in_warpgroup ^ B_H]
            // );
            // const e8m0 s_scale_e8m0(
            //     s_max > 0.0f ? s_max / FP8_MAX : 1.0f
            // );

            // const float current_s_scale = float(s_scale_e8m0);
            // const float inv_current_s_scale = __fdividef(
            //     1.0f, current_s_scale
            // );
            constexpr float current_s_scale = 1.0f/448.f;
            constexpr float inv_current_s_scale = 448.f;
            const float2 inv_current_s_scale2 = make_float2(
                inv_current_s_scale,
                inv_current_s_scale
            );
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD; i += 4) {
                const uint16_t s01 = ku::float2_to_e4m3x2_bits(
                    ku::float2_mul(
                        make_float2(p[i], p[i + 1]),
                        inv_current_s_scale2
                    )
                );
                const uint16_t s23 = ku::float2_to_e4m3x2_bits(
                    ku::float2_mul(
                        make_float2(p[i + 2], p[i + 3]),
                        inv_current_s_scale2
                    )
                );
                s[i / 4] = static_cast<uint32_t>(s01)
                    | (static_cast<uint32_t>(s23) << 16);
            }
            li = fma(li, scale_for_old, cur_sum);
            FP8_TIMEPOINT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].softmax_ready_ns
            );

            // Warp 8 can keep consuming S(k-1) while WG0 publishes S(k) into
            // the other stage.  A stage is not reused until the preceding
            // iteration has observed the matching SV completion barrier.
            // With one S stage (the 576-D fit), wait before writing the next
            // tile because there is no alternate buffer to hide this wait.
            if constexpr (NUM_S_BUFS == 1) {
                if (k > 0) {
                    const int prev_buf = (k - 1) % NUM_MAIN_BUFS;
                    const int prev_phase = ((k - 1) / NUM_MAIN_BUFS) & 1;
                    plan.bar_sv_done[prev_buf].wait(prev_phase);
                }
            }
            Tensor sS = make_tensor(
                make_smem_ptr(plan.s[k % NUM_S_BUFS].data()), SmemLayoutS{}
            );
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

            // S can be consumed as soon as its SMEM stores are visible. O
            // reuse is guarded by the whole-O handoff below.
            fence_view_async_shared();
            plan.bar_so_ready.arrive();

            // Wait for the preceding SV tile to finish, rescale all four O
            // stripes, then publish one handoff for the whole O accumulator.
            if (k > 0) {
                const int prev_buf = (k - 1) % NUM_MAIN_BUFS;
                const int prev_phase = ((k - 1) / NUM_MAIN_BUFS) & 1;
                // const float o_rescale = scale_for_old
                //     * s_scale_for_o / current_s_scale;
                // const bool warp_needs_o_rescale = __any_sync(
                //     0xffffffff, o_rescale != 1.0f
                // );
                const float o_rescale = scale_for_old;
                const bool warp_needs_o_rescale = should_scale_o;
#if defined(FP8_FWD_BARRIER_TIMING)
                auto* const first_stripe_timing = trace_wg0_tile
                    ? &plan.barrier_timing.o_rescale[warp_idx][k][0]
                    : nullptr;
                FP8_TIMEPOINT(
                    trace_wg0_tile,
                    first_stripe_timing->stripe_start_ns
                );
                if (trace_wg0_tile) {
                    first_stripe_timing->warp_rescale_active =
                        static_cast<uint64_t>(warp_needs_o_rescale);
                }
#endif
                FP8_TIMED_WAIT(
                    trace_wg0_tile,
                    first_stripe_timing->sv_wait_ns,
                    plan.bar_sv_done[prev_buf].wait(prev_phase)
                );
#if defined(FP8_FWD_BARRIER_TIMING)
                if (trace_wg0_tile) {
                    plan.barrier_timing.wg0[warp_idx][k].sv_wait_ns =
                        first_stripe_timing->sv_wait_ns;
                }
#endif
                CUTE_UNROLL
                for (int dv_block = 0;
                     dv_block < NUM_SV_TMEM_BLOCKS;
                     ++dv_block) {
#if defined(FP8_FWD_BARRIER_TIMING)
                    auto* const stripe_timing = trace_wg0_tile
                        ? &plan.barrier_timing.o_rescale[warp_idx][k][dv_block]
                        : nullptr;
                    FP8_TIMEPOINT(
                        trace_wg0_tile,
                        stripe_timing->stripe_start_ns
                    );
                    if (trace_wg0_tile) {
                        stripe_timing->warp_rescale_active =
                            static_cast<uint64_t>(warp_needs_o_rescale);
                    }
#endif
                    if (warp_needs_o_rescale) {
                        if (dv_block == 0) {
                            ku::tcgen05_after_thread_sync();
                        }
#if defined(FP8_FWD_BARRIER_TIMING)
                        rescale_o_tmem_stripe(
                            o_rescale,
                            tmem_cols::O + dv_block * SV_TMEM_COLS_PER_BLOCK,
                            plan.barrier_timing.origin_ns,
                            stripe_timing
                        );
#else
                        rescale_o_tmem_stripe(
                            o_rescale,
                            tmem_cols::O + dv_block * SV_TMEM_COLS_PER_BLOCK
                        );
#endif
                        if (dv_block + 1 == NUM_SV_TMEM_BLOCKS) {
                            ku::tcgen05_before_thread_sync();
                        }
                    }
                    FP8_TIMEPOINT(
                        trace_wg0_tile,
                        stripe_timing->sv_ready_ns
                    );
                }
                // The named barrier is still needed when any WG0 warp touched
                // TMEM.  With no rescale, the single transaction-barrier
                // arrival below is the fast relay and no named barrier is
                // entered.
                if (wg0_needs_o_rescale) {
                    NamedBarrier::arrive_and_wait(
                        128, NamedBarriers::wg0_sync
                    );
                }
                plan.bar_o_rescale_done[0].arrive();
#if defined(FP8_FWD_BARRIER_TIMING)
                CUTE_UNROLL
                for (int dv_block = 0;
                     dv_block < NUM_SV_TMEM_BLOCKS;
                     ++dv_block) {
                    FP8_TIMEPOINT(
                        trace_wg0_tile,
                        plan.barrier_timing.o_rescale[warp_idx][k][dv_block]
                            .wg0_sync_done_ns
                    );
                }
#endif
            }
            FP8_TIMEPOINT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].o_rescaled_ns
            );
            s_scale_for_o = current_s_scale;
            FP8_TIMEPOINT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].s_arrived_ns
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
        li += plan.rowwise_li_buf[idx_in_warpgroup ^ B_H];

        // Store mi and li
        if (idx_in_warpgroup < 64) {
            int global_index = s_q_idx*params.h_q + idx_in_warpgroup;
            float cur_lse = CUDART_INF_F;
            if (li != 0.0f) {
                cur_lse = fmaf(mi, CUDART_LN2_F, logf(li));
            }
            params.max_logits[global_index] = real_mi * CUDART_LN2_F;
            params.lse[global_index] = cur_lse;
        }

        // Wait for the last GEMM
        
        FP8_TIMED_WAIT(
            s_q_idx == 0 && lane_idx == 0,
            plan.barrier_timing.final_sv_wait_ns[warp_idx],
            plan.bar_sv_done[(num_k_blocks - 1) % NUM_MAIN_BUFS].wait(
                ((num_k_blocks - 1) / NUM_MAIN_BUFS) & 1
            )
        );
        FP8_TIMEPOINT(
            s_q_idx == 0 && lane_idx == 0,
            plan.barrier_timing.final_sv_ready_ns[warp_idx]
        );
        
        ku::tcgen05_after_thread_sync();

        // Fetch dO if necessary

        // Store O
        const bool have_valid_indices = __any_sync(
            0xffffffff, li != 0.0f
        );
        float output_scale = 1.0f;
        if (have_valid_indices) {
            const float attn_sink = params.attn_sink == nullptr
                ? -CUDART_INF_F
                : __ldg(params.attn_sink + (idx_in_warpgroup % 64))
                    * CUDART_L2E_F;
            output_scale = __fdividef(
                1.0f, li + exp2f(attn_sink - mi)
            );
        }
        Tensor sO = make_tensor(make_smem_ptr(plan.qkvo.o.o.data()), SmemLayoutO{});
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
        if (!have_valid_indices) {
            // If there are no valid indices, we set o[i] to 0 and don't load from TMEM
            CUTE_UNROLL
            for (int i = 0; i < B_EPI/2; ++i)
                o[i].x = o[i].y = 0.0f;
        }
        const float output_base_scale = output_scale * s_scale_for_o;

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

                // A 128-D N=128 MMA packs adjacent logical 64-D groups
                // into the two TMEM row halves of its 64-column stripe.
                const int d_group = c * 4 + k * 2
                    + idx_in_warpgroup / B_H;
#if defined(FP8_FWD_QK576)
                const float output_dequant_scale = output_base_scale
                    * ue8m0_bits_to_float(
                        __ldg(params.kv_scale_w + d_group)
                    );
#else
                const float output_dequant_scale = output_base_scale
                    * plan.kv_dim_scale[d_group];
#endif
                const float2 output_dequant_scale2 = make_float2(
                    output_dequant_scale,
                    output_dequant_scale
                );

                // Convert and store
                CUTE_UNROLL
                for (int i = 0; i < B_EPI/8; ++i) {
                    nv_bfloat162 o_bf16[4];
                    CUTE_UNROLL
                    for (int j = 0; j < 4; ++j) {
                        o[i*4+j] = ku::float2_mul(
                            o[i*4+j], output_dequant_scale2
                        );
                        o_bf16[j] = __float22bfloat162_rn(o[i*4+j]);
                    }
                    *(uint128_t*)(sO_addrs[i] + d_group * B_EPI * B_H) = *(uint128_t*)(o_bf16);
                }

                // Sync
                fence_view_async_shared();
                NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
                
                if (warp_idx == 0 && elect_one_sync()) {
                    int epi_chunk_idx = c * 4 + k * 2;
                    cute::copy(
                        tma_params.tma_O,
                        thr_tma.partition_S(sO_divided(_, _, epi_chunk_idx)),
                        thr_tma.partition_D(tma_gO(_, _, epi_chunk_idx))
                    );
                }
                if (warp_idx == 1 && elect_one_sync()) {
                    int epi_chunk_idx = c * 4 + k * 2 + 1;
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
    cutlass::arch::warpgroup_reg_dealloc<80>();

    // Producer warp for KV
        int warp_idx = cutlass::canonical_warp_idx_sync() - 4;

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
                bool should_skip_tma = is_all_rows_invalid && k >= NUM_MAIN_BUFS;

                if (k == NUM_MAIN_BUFS - 1) {
                    FP8_TIMED_WAIT(
                        trace_kv_tile,
                        plan.barrier_timing.kv[warp_idx][k].q_reuse_wait_ns,
                        plan.bar_prologue_utccp.wait(0)
                    );  // Q shares storage with the last main K stage.
                }

                // Copy NoPE
                int cur_buf = k%NUM_MAIN_BUFS;
                const int tail_buf = k % NUM_QK_TAIL_BUFS;
                if constexpr (HAVE_QK_TAIL) {
                    // The tail-K stage is independent of V.  Do not
                    // overwrite it until the tail QK commit for the same
                    // buffer has completed.
                    if (k >= NUM_QK_TAIL_BUFS) {
                        plan.bar_qk_done[tail_buf].wait(
                            ((k / NUM_QK_TAIL_BUFS) & 1) ^ 1
                        );
                    }
                }
                FP8_TIMED_WAIT(
                    trace_kv_tile,
                    plan.barrier_timing.kv[warp_idx][k].sv_free_wait_ns,
                    plan.bar_sv_done[cur_buf].wait(((k / NUM_MAIN_BUFS) & 1) ^ 1)
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

                auto load_kv_tail = [&]() {
                    if constexpr (HAVE_QK_TAIL) {
                        // Buffer 0 initially holds Q tail.  Do not let the
                        // first tail-K gather overwrite it until warp 8 has
                        // copied Q tail to TMEM.
                        if (k == 0) {
                            plan.bar_prologue_utccp.wait(0);
                        }
                        e4m3* sK_tail_base = plan.qk_tail.kv[tail_buf].data()
                            + warp_idx * 4 * TMA_K_TAIL_BYTES;
                        CUTE_UNROLL
                        for (int local_row = 0;
                             local_row < NUM_LOCAL_ROWS_PER_WARP;
                             ++local_row) {
                            ku::tma_gather4(
                                &(tma_params.tensor_map_kv_tail),
                                plan.bar_kv_tail_ready[tail_buf],
                                sK_tail_base
                                    + local_row * (4 * NUM_WARPS)
                                        * TMA_K_TAIL_BYTES,
                                0,
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
                    load_kv_part(1);
                    FP8_TIMEPOINT(
                        trace_kv_tile,
                        plan.barrier_timing.kv[warp_idx][k].tma_part1_issued_ns
                    );
                    load_kv_tail();
                    
                } else {
                    // NOTE See head128/phase1.cuh for this TMA skipping technique
                    CUTE_UNROLL
                    for (int part_idx = 0; part_idx < 2; ++part_idx)
                        plan.bar_kv_ready[cur_buf][part_idx].complete_transaction(NUM_LOCAL_ROWS_PER_WARP*4*D_V/2*sizeof(e4m3));
                    if constexpr (HAVE_QK_TAIL) {
                        plan.bar_kv_tail_ready[tail_buf].complete_transaction(
                            NUM_LOCAL_ROWS_PER_WARP * 4
                                * TMA_K_TAIL_BYTES * sizeof(e4m3)
                        );
                    }
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
                        Shape<Int<B_H>, Int<128>>{}
                    )
                )
            );
        plan.bar_prologue.arrive_and_expect_tx(
            B_H * (D_V + (HAVE_QK_TAIL ? 64 : 0)) * sizeof(e4m3)
        );
        FP8_TIMED_WAIT(
            s_q_idx == 0 && lane_idx == 0,
            plan.barrier_timing.q_tma_wait_ns,
            plan.bar_prologue.wait(0)
        );
        ku::tcgen05_after_thread_sync();
        CUTE_UNROLL
        for (int tile_idx = 0; tile_idx < D_Q / 128; ++tile_idx) {
            CUTE_UNROLL
            for (int subtile_idx = 0; subtile_idx < 8; ++subtile_idx) {
                // The direct M=64 fragment broadcasts each 64-row Q chunk
                // into its two TMEM datapath halves.
                SM100_UTCCP_2x64dp128bitlw0213_1cta::copy(
                    sQ_desc + (tile_idx * B_H * 128 + subtile_idx * 16) / 16,
                    tmem_cols::Q + tile_idx * 32 + subtile_idx * 4
                );
            }
        }
        if constexpr (HAVE_QK_TAIL) {
            // The 64-D Q tail occupies the TMEM columns that would otherwise
            // hold P1.  Four 128-bit copies cover the 64x64 fragment.
            UMMA::SmemDescriptor sQ_tail_desc =
                UMMA::make_umma_desc<UMMA::Major::K>(
                    make_tensor(
                        make_smem_ptr(plan.qk_tail.q.data()),
                        tile_to_shape(
                            UMMA::Layout_K_SW64_Atom<e4m3>{},
                            Shape<Int<B_H>, Int<64>>{}
                        )
                    )
                );
            CUTE_UNROLL
            for (int subtile_idx = 0; subtile_idx < 4; ++subtile_idx) {
                SM100_UTCCP_2x64dp128bitlw0213_1cta::copy(
                    sQ_tail_desc + (subtile_idx * 16) / 16,
                    tmem_cols::Q_TAIL + subtile_idx * 4
                );
            }
        }
        ku::umma_arrive_noelect(plan.bar_prologue_utccp);
        FP8_TIMEPOINT(
            s_q_idx == 0 && lane_idx == 0,
            plan.barrier_timing.q_tmem_committed_ns
        );

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
                int cur_buf = k%NUM_MAIN_BUFS;
                int p_stage = k%NUM_P_BUFS;
                Tensor sK = make_tensor(make_smem_ptr(plan.qkvo.kv[cur_buf].data()), SmemLayoutK_TiledMMA{});

                FP8_TIMED_WAIT(
                    trace_mma_iter,
                    plan.barrier_timing.mma[k].p_free_wait_ns,
                    plan.bar_p_free[p_stage].wait(
                        ((k / NUM_P_BUFS) & 1) ^ 1
                    )
                );

                ku::tcgen05_after_thread_sync();
                if (k == 0) {
                    FP8_TIMED_WAIT(
                        trace_mma_iter,
                        plan.barrier_timing.mma[k].q_copy_wait_ns,
                        plan.bar_prologue_utccp.wait(0)
                    );

                }
                // The producer still signals the two gather halves
                // independently, but the direct GEMM consumes the full K.
                CUTE_UNROLL
                for (int kv_part_idx = 0; kv_part_idx < 2; ++kv_part_idx) {
                    plan.bar_kv_ready[cur_buf][kv_part_idx].arrive_and_expect_tx(B_TOPK*D_V/2*sizeof(e4m3));
                    FP8_TIMED_WAIT(
                        trace_mma_iter,
                        plan.barrier_timing.mma[k].kv_wait_ns[kv_part_idx],
                        plan.bar_kv_ready[cur_buf][kv_part_idx].wait(
                            (k / NUM_MAIN_BUFS) & 1
                        )
                    );
                    FP8_TIMEPOINT(
                        trace_mma_iter,
                        plan.barrier_timing.mma[k].kv_ready_ns[kv_part_idx]
                    );

                }
                ku::tcgen05_after_thread_sync();

                if (p_stage == 0) {
                    ku::utcmma_ts(tiled_mma_P, tQ, sK, tP0, true);
                } else {
                    ku::utcmma_ts(tiled_mma_P, tQ, sK, tP1, true);
                }
                FP8_TIMEPOINT(
                    trace_mma_iter,
                    plan.barrier_timing.mma[k].qk_issued_ns[0]
                );
                FP8_TIMEPOINT(
                    trace_mma_iter,
                    plan.barrier_timing.mma[k].qk_issued_ns[1]
                );

                if constexpr (HAVE_QK_TAIL) {
                    // Commit the 512-dim product first.  The tail TS MMA is
                    // issued only after this commit has completed, so it
                    // accumulates into the same P stage deterministically.
                    const int tail_buf = k % NUM_QK_TAIL_BUFS;
                    ku::umma_arrive_noelect(plan.bar_qk_part_done[cur_buf]);

                    Tensor sK_tail = make_tensor(
                        make_smem_ptr(plan.qk_tail.kv[tail_buf].data()),
                        SmemLayoutKTail{}
                    );
                    plan.bar_kv_tail_ready[tail_buf].arrive_and_expect_tx(
                        B_TOPK * TMA_K_TAIL_BYTES * sizeof(e4m3)
                    );
                    plan.bar_kv_tail_ready[tail_buf].wait(
                        (k / NUM_QK_TAIL_BUFS) & 1
                    );
                    plan.bar_qk_part_done[cur_buf].wait(
                        (k / NUM_MAIN_BUFS) & 1
                    );
                    ku::tcgen05_after_thread_sync();
                    if (p_stage == 0) {
                        ku::utcmma_ts(
                            tiled_mma_P, tQ_tail, sK_tail, tP0, false
                        );
                    } else {
                        ku::utcmma_ts(
                            tiled_mma_P, tQ_tail, sK_tail, tP1, false
                        );
                    }
                    ku::umma_arrive_noelect(plan.bar_qk_done[tail_buf]);
                } else {
                    ku::umma_arrive_noelect(plan.bar_qk_done[p_stage]);
                }
                FP8_TIMEPOINT(
                    trace_mma_iter,
                    plan.barrier_timing.mma[k].qk_committed_ns
                );
            }

            if (k > 0) {
                    // O += S(i-1)V(i-1)
                    int cur_buf = (k-1)%NUM_MAIN_BUFS;

                    Tensor sS = make_tensor(
                        make_smem_ptr(plan.s[(k - 1) % NUM_S_BUFS].data()),
                        SmemLayoutS{}
                    );
                    Tensor sV = make_tensor(
                        make_smem_ptr(plan.qkvo.kv[cur_buf].data()),
                        SmemLayoutV{}
                    );

                    // S(i-1) is ready independently of the O accumulator.
                    FP8_TIMED_WAIT(
                        trace_mma_iter,
                        plan.barrier_timing.mma[k].s_ready_wait_ns,
                        plan.bar_so_ready.wait((k - 1) & 1)
                    );
                    FP8_TIMEPOINT(
                        trace_mma_iter,
                        plan.barrier_timing.mma[k].s_ready_ns
                    );

                    // The entire O accumulator is handed back at once; do
                    // not start any stripe until WG0 has finished the whole
                    // preceding O rescale.
                    if (k > 1) {
                        plan.bar_o_rescale_done[0].wait((k - 2) & 1);
                    }
                    ku::tcgen05_after_thread_sync();

                    Tensor sV_divided = flat_divide(
                        sV, Tile<Int<SV_M>, Int<B_TOPK>>{}
                    )(_, _, _, _0{});

                    // Keep all four O stripes in TMEM.  Only the first
                    // TopK tile clears them; later tiles accumulate in place.
                    CUTE_UNROLL
                    for (int dv_block = 0;
                         dv_block < NUM_SV_TMEM_BLOCKS;
                         ++dv_block) {
                        tO.data().get() = tmem_cols::O
                            + dv_block * SV_TMEM_COLS_PER_BLOCK;
                        ku::utcmma_ss(
                            tiled_mma_O,
                            sS,
                            sV_divided(_, _, dv_block),
                            tO,
                            k == 1
                        );
                        if (dv_block + 1 == NUM_SV_TMEM_BLOCKS) {
                            ku::umma_arrive_noelect(plan.bar_sv_done[cur_buf]);
                        }
                    }
                    FP8_TIMEPOINT(
                        trace_mma_iter,
                        plan.barrier_timing.mma[k].sv_committed_ns
                    );
                }
        }
    } else if (warp_idx == 9) {
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

                int cur_buf = k%NUM_MAIN_BUFS;
                FP8_TIMED_WAIT(
                    trace_mask_tile,
                    plan.barrier_timing.mask[k].buffer_free_wait_ns,
                    plan.bar_k_valid_free[cur_buf].wait(
                        ((k / NUM_MAIN_BUFS) & 1) ^ 1
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
    } else if (warp_idx == 10 || warp_idx == 11) {
        const int scale_warp_idx = warp_idx - 10;
#if !defined(FP8_FWD_QK576)
        const int q_scale_row = scale_warp_idx * 32 + lane_idx;

        const uint8_t* q_scale_base =
            reinterpret_cast<const uint8_t*>(params.q)
            + static_cast<int64_t>(s_q_idx) * params.stride_q_s_q
            + B_H * QK_DIM;
        plan.q_head_scale[q_scale_row] = ue8m0_bits_to_float(
            __ldg(q_scale_base + q_scale_row)
        );
        if (scale_warp_idx == 0 && lane_idx < KV_SCALE_GROUPS) {
            plan.kv_dim_scale[lane_idx] = ue8m0_bits_to_float(
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
#endif

        CUTE_NO_UNROLL
        for (int k = 0; k < num_k_blocks; ++k) {
            const int cur_buf = k % NUM_MAIN_BUFS;
#if defined(FP8_FWD_BARRIER_TIMING)
            const bool trace_scale_tile = s_q_idx == 0
                && lane_idx == 0 && k < FP8_TIMING_MAX_TILES;
#endif
            FP8_TIMEPOINT(
                trace_scale_tile,
                plan.barrier_timing.scale[scale_warp_idx][k].tile_start_ns
            );
            
            FP8_TIMED_WAIT(
                trace_scale_tile,
                plan.barrier_timing.scale[scale_warp_idx][k].buffer_free_wait_ns,
                plan.bar_sv_done[cur_buf].wait(
                    ((k / NUM_MAIN_BUFS) & 1) ^ 1
                )
            );
            

            CUTE_UNROLL
            for (int i = 0; i < B_TOPK / 64; ++i) {
                const int scale_row = scale_warp_idx * (B_TOPK / 2)
                    + i * 32 + lane_idx;
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
                        + QK_DIM;
#if defined(FP8_FWD_VECTOR_KV_SCALE_LOAD)
                    // The trailing scale slot is 16-byte aligned. Keep this
                    // opt-in because random top-k indices can change cache
                    // behavior even when the extracted scale byte is identical.
                    const uint4 packed_scale_slot = __ldg(
                        reinterpret_cast<const uint4*>(kv_scale_ptr)
                    );
                    scale_bits = static_cast<uint8_t>(packed_scale_slot.x);
#else
                    scale_bits = __ldg(kv_scale_ptr);
#endif

                }
                plan.kv_token_scale[cur_buf][scale_row] =
                    ue8m0_bits_to_float(scale_bits);
            }
            fence_view_async_shared();
            if (elect_one_sync()) {
                plan.bar_kv_scale_ready[cur_buf].arrive();
            }
            FP8_TIMEPOINT(
                trace_scale_tile,
                plan.barrier_timing.scale[scale_warp_idx][k].arrived_ns
            );
            
        }
    }
}

#if defined(FP8_FWD_BARRIER_TIMING)
    if (s_q_idx == 0 && lane_idx == 0) {
        plan.barrier_timing.branch_end_ns[warp_idx] = fp8_timing_now_ns()
            - plan.barrier_timing.origin_ns;
    }
    __syncthreads();
    if (s_q_idx == 0 && threadIdx.x == 0) {
        print_fp8_barrier_timing<HAVE_QK_TAIL>(plan, num_k_blocks);
    }
#endif
}

#undef FP8_TIMED_WAIT
#undef FP8_TIMEPOINT

template<int D_QK>
void run_fp8_fwd_phase1_kernel(const Head64Fp8SparseAttnFwdParams& params) {
    static_assert(D_QK == 512 || D_QK == 576);
#if defined(FP8_FWD_QK576)
    static_assert(D_QK == 576 && B_TOPK == 128);
#else
    static_assert(D_QK == 512 && B_TOPK == 128);
#endif
    constexpr bool HAVE_QK_TAIL = D_QK == 576;
    KU_ASSERT(params.d_qk == D_QK);

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

    auto shape_Q_tail = make_shape(B_H, 64, params.s_q);
    auto tma_Q_tail = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr(reinterpret_cast<e4m3*>(params.q) + D_K),
            make_layout(
                shape_Q_tail,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q)
            )
        ),
        SmemLayoutQTail{}
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

    CUtensorMap tensor_map_kv_tail = ku::make_tensor_map(
            {TMA_K_TAIL_ELEMS, static_cast<uint64_t>(params.s_kv)},
            {static_cast<uint64_t>(params.stride_kv_s_kv)},
            {TMA_K_TAIL_ELEMS, 1},
            reinterpret_cast<uint8_t*>(params.kv) + D_K,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT64,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B
        );

    TmaParams<
        decltype(shape_O), decltype(tma_O),
        decltype(shape_Q), decltype(tma_Q),
        decltype(shape_Q_tail), decltype(tma_Q_tail)
    > tma_params = {
        shape_O, tma_O,
        shape_Q, tma_Q,
        shape_Q_tail, tma_Q_tail,
        tensor_map_kv,
        tensor_map_kv_tail
    };

    auto kernel = &sprase_fp8_attn_fwd_kernel<HAVE_QK_TAIL, decltype(tma_params)>;

    constexpr size_t smem_size = sizeof(SharedMemoryPlanT<HAVE_QK_TAIL>);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    kernel<<<params.s_q, NUM_THREADS, smem_size, params.stream>>>(params, tma_params);
    KU_CHECK_KERNEL_LAUNCH();

}


}
