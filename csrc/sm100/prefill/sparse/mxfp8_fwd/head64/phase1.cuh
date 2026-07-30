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

#if defined(MXFP8_FWD_DEBUG_MARKERS)
#define MXFP8_MARK_WARP(...)                                                    \
    do {                                                                        \
        if (s_q_idx == 0 && elect_one_sync()) {                                 \
            cute::print(__VA_ARGS__);                                           \
            cute::print("\n");                                                 \
        }                                                                       \
    } while (0)
#define MXFP8_MARK_ONE(...)                                                     \
    do {                                                                        \
        if (s_q_idx == 0) {                                                     \
            cute::print(__VA_ARGS__);                                           \
            cute::print("\n");                                                 \
        }                                                                       \
    } while (0)
#else
#define MXFP8_MARK_WARP(...) do { } while (0)
#define MXFP8_MARK_ONE(...) do { } while (0)
#endif

#if defined(MXFP8_FWD_BARRIER_TIMING)
CUTE_DEVICE
uint64_t mxfp8_timing_now_ns() {
    uint64_t timestamp;
    asm volatile(
        "mov.u64 %0, %%globaltimer;"
        : "=l"(timestamp)
        :
        : "memory"
    );
    return timestamp;
}

#define MXFP8_TIMED_WAIT(sample, destination, wait_expression)                  \
    do {                                                                        \
        uint64_t mxfp8_wait_begin_ns = 0;                                       \
        if (sample) {                                                           \
            mxfp8_wait_begin_ns = mxfp8_timing_now_ns();                        \
        }                                                                       \
        wait_expression;                                                        \
        if (sample) {                                                           \
            destination = mxfp8_timing_now_ns() - mxfp8_wait_begin_ns;          \
        }                                                                       \
    } while (0)
#define MXFP8_TIMEPOINT(sample, destination)                                    \
    do {                                                                        \
        if (sample) {                                                           \
            destination = mxfp8_timing_now_ns()                                \
                - plan.barrier_timing.origin_ns;                                \
        }                                                                       \
    } while (0)

CUTE_DEVICE
void print_mxfp8_barrier_timing(
    const SharedMemoryPlan& plan,
    int num_k_blocks
) {
    const auto& timing = plan.barrier_timing;
    const int traced_tiles = min(num_k_blocks, MXFP8_TIMING_MAX_TILES);
    cute::print(
        "MXFP8_TIME header unit=ns tiles=%d traced=%d origin=%llu\n",
        num_k_blocks,
        traced_tiles,
        static_cast<unsigned long long>(timing.origin_ns)
    );
    CUTE_UNROLL
    for (int warp = 0; warp < 12; ++warp) {
        cute::print(
            "MXFP8_TIME branch warp=%d end=%llu\n",
            warp,
            static_cast<unsigned long long>(timing.branch_end_ns[warp])
        );
    }
    cute::print(
        "MXFP8_TIME mma_init q_wait=%llu q_scale_sync_wait=%llu "
        "q_scale_tmem=%llu v_scale_tmem=%llu\n",
        static_cast<unsigned long long>(timing.q_tma_wait_ns),
        static_cast<unsigned long long>(timing.q_scale_sync_wait_ns),
        static_cast<unsigned long long>(timing.q_scale_tmem_committed_ns),
        static_cast<unsigned long long>(timing.v_scale_tmem_committed_ns)
    );
    CUTE_UNROLL
    for (int warp = 0; warp < 4; ++warp) {
        const auto& value = timing.epilogue[warp];
        cute::print(
            "MXFP8_TIME epilogue warp=%d final_sv_wait=%llu "
            "final_sv_ready=%llu stats_stored=%llu scale_sync_wait=%llu "
            "scale_ready=%llu o_staged=%llu o_sync_wait=%llu "
            "store_issued=%llu\n",
            warp,
            static_cast<unsigned long long>(value.final_sv_wait_ns),
            static_cast<unsigned long long>(value.final_sv_ready_ns),
            static_cast<unsigned long long>(value.stats_stored_ns),
            static_cast<unsigned long long>(value.scale_sync_wait_ns),
            static_cast<unsigned long long>(value.scale_ready_ns),
            static_cast<unsigned long long>(value.o_staged_ns),
            static_cast<unsigned long long>(value.o_sync_wait_ns),
            static_cast<unsigned long long>(value.store_issued_ns)
        );
    }
    for (int tile = 0; tile < traced_tiles; ++tile) {
        CUTE_UNROLL
        for (int warp = 0; warp < 4; ++warp) {
            const auto& value = timing.wg0[warp][tile];
            cute::print(
                "MXFP8_TIME wg0 warp=%d tile=%d start=%llu tile_sync_wait=%llu "
                "qk_wait=%llu valid_wait=%llu waits_done=%llu "
                "p_released=%llu p_prepared=%llu rowmax_wait=%llu "
                "rowmax_ready=%llu s_quantized=%llu li_wait=%llu "
                "softmax_ready=%llu sv_wait=%llu head_updated=%llu "
                "pre_rescale_wait=%llu pre_rescale_ready=%llu "
                "rescale_done=%llu rescale_wait=%llu s_arrived=%llu\n",
                warp,
                tile,
                static_cast<unsigned long long>(value.tile_start_ns),
                static_cast<unsigned long long>(value.tile_sync_wait_ns),
                static_cast<unsigned long long>(value.qk_wait_ns),
                static_cast<unsigned long long>(value.valid_wait_ns),
                static_cast<unsigned long long>(value.waits_done_ns),
                static_cast<unsigned long long>(value.p_released_ns),
                static_cast<unsigned long long>(value.p_prepared_ns),
                static_cast<unsigned long long>(value.rowmax_wait_ns),
                static_cast<unsigned long long>(value.rowmax_ready_ns),
                static_cast<unsigned long long>(value.s_quantized_ns),
                static_cast<unsigned long long>(value.li_wait_ns),
                static_cast<unsigned long long>(value.softmax_ready_ns),
                static_cast<unsigned long long>(value.sv_wait_ns),
                static_cast<unsigned long long>(value.head_updated_ns),
                static_cast<unsigned long long>(value.pre_rescale_wait_ns),
                static_cast<unsigned long long>(value.pre_rescale_ready_ns),
                static_cast<unsigned long long>(value.rescale_done_ns),
                static_cast<unsigned long long>(value.rescale_wait_ns),
                static_cast<unsigned long long>(value.s_arrived_ns)
            );
        }
        CUTE_UNROLL
        for (int warp = 0; warp < 4; ++warp) {
            const auto& value = timing.kv[warp][tile];
            cute::print(
                "MXFP8_TIME kv warp=%d tile=%d start=%llu sv_free_wait=%llu "
                "indices_ready=%llu indices_sync_wait=%llu "
                "transaction_ready=%llu transaction_sync_wait=%llu "
                "tma_issued=%llu\n",
                warp + 4,
                tile,
                static_cast<unsigned long long>(value.tile_start_ns),
                static_cast<unsigned long long>(value.sv_free_wait_ns),
                static_cast<unsigned long long>(value.indices_ready_ns),
                static_cast<unsigned long long>(value.indices_sync_wait_ns),
                static_cast<unsigned long long>(value.transaction_ready_ns),
                static_cast<unsigned long long>(value.transaction_sync_wait_ns),
                static_cast<unsigned long long>(value.tma_issued_ns)
            );
        }
        const auto& mask = timing.mask[tile];
        cute::print(
            "MXFP8_TIME mask warp=9 tile=%d start=%llu free_wait=%llu "
            "arrived=%llu\n",
            tile,
            static_cast<unsigned long long>(mask.tile_start_ns),
            static_cast<unsigned long long>(mask.buffer_free_wait_ns),
            static_cast<unsigned long long>(mask.arrived_ns)
        );
        CUTE_UNROLL
        for (int warp = 0; warp < 2; ++warp) {
            const auto& scale = timing.scale[warp][tile];
            cute::print(
                "MXFP8_TIME scale warp=%d tile=%d start=%llu free_wait=%llu "
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
            "MXFP8_TIME mma warp=8 iter=%d start=%llu p_free_wait=%llu "
            "kv_scale_wait=%llu k_scale_ready=%llu kv_wait=%llu "
            "kv_ready=%llu qk_committed=%llu s_ready_wait=%llu "
            "s_ready=%llu s_scale_ready=%llu sv_committed=%llu\n",
            iter,
            static_cast<unsigned long long>(value.iter_start_ns),
            static_cast<unsigned long long>(value.p_free_wait_ns),
            static_cast<unsigned long long>(value.kv_scale_wait_ns),
            static_cast<unsigned long long>(value.k_scale_ready_ns),
            static_cast<unsigned long long>(value.kv_wait_ns),
            static_cast<unsigned long long>(value.kv_ready_ns),
            static_cast<unsigned long long>(value.qk_committed_ns),
            static_cast<unsigned long long>(value.s_ready_wait_ns),
            static_cast<unsigned long long>(value.s_ready_ns),
            static_cast<unsigned long long>(value.s_scale_ready_ns),
            static_cast<unsigned long long>(value.sv_committed_ns)
        );
    }
}
#else
#define MXFP8_TIMED_WAIT(sample, destination, wait_expression)                  \
    do {                                                                        \
        wait_expression;                                                        \
    } while (0)
#define MXFP8_TIMEPOINT(sample, destination) do { } while (0)
#endif

CUTE_DEVICE
float ue8m0_ratio_to_float(uint8_t numerator_bits, uint8_t denominator_bits) {
    TRAP_ONLY_DEVICE_ASSERT(numerator_bits != 0xff && denominator_bits != 0xff);
    int exponent = static_cast<int>(numerator_bits) - static_cast<int>(denominator_bits);
    TRAP_ONLY_DEVICE_ASSERT(exponent >= -149 && exponent <= 127);
    if (exponent >= -126 && exponent <= 127) {
        return __uint_as_float(static_cast<uint32_t>(exponent + 127) << 23);
    }
    if (exponent >= -149 && exponent <= -127) {
        return __uint_as_float(1u << (exponent + 149));
    }
    return exponent < -149 ? 0.0f : CUDART_INF_F;
}

CUTE_DEVICE
void transpose_bf16_8x8(uint32_t values[4], int lane_idx) {
    // Swap the lane and packed-element index bits within each 8-lane group.
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) {
        const uint32_t peer = __shfl_xor_sync(
            0xffffffffu, values[i], 1, 8
        );
        values[i] = (lane_idx & 1) == 0
            ? (values[i] & 0x0000ffffu) | (peer << 16)
            : (values[i] & 0xffff0000u) | (peer >> 16);
    }

    const uint32_t peer0 = __shfl_xor_sync(
        0xffffffffu, values[0], 2, 8
    );
    const uint32_t peer1 = __shfl_xor_sync(
        0xffffffffu, values[1], 2, 8
    );
    const uint32_t peer2 = __shfl_xor_sync(
        0xffffffffu, values[2], 2, 8
    );
    const uint32_t peer3 = __shfl_xor_sync(
        0xffffffffu, values[3], 2, 8
    );
    if ((lane_idx & 2) == 0) {
        values[1] = peer0;
        values[3] = peer2;
    } else {
        values[0] = peer1;
        values[2] = peer3;
    }

    const uint32_t peer02 = __shfl_xor_sync(
        0xffffffffu, values[0], 4, 8
    );
    const uint32_t peer12 = __shfl_xor_sync(
        0xffffffffu, values[1], 4, 8
    );
    const uint32_t peer22 = __shfl_xor_sync(
        0xffffffffu, values[2], 4, 8
    );
    const uint32_t peer32 = __shfl_xor_sync(
        0xffffffffu, values[3], 4, 8
    );
    if ((lane_idx & 4) == 0) {
        values[2] = peer02;
        values[3] = peer12;
    } else {
        values[0] = peer22;
        values[1] = peer32;
    }
}

template<int B_H, int B_H_TMEM, int TMEM_COL_START>
CUTE_DEVICE
void rescale_O_block_t(float scale[B_H], int dv_block) {
    float o[B_H_TMEM];
    ku::tmem_ld_32dp32bNx<B_H_TMEM>(
        TMEM_COL_START + dv_block * B_H_TMEM, o
    );
    cutlass::arch::fence_view_async_tmem_load();
    CUTE_UNROLL
    for (int i = 0; i < B_H; ++i) {
        o[i] *= scale[i];
    }
    ku::tmem_st_32dp32bNx<B_H_TMEM>(
        TMEM_COL_START + dv_block * B_H_TMEM, o
    );
    cutlass::arch::fence_view_async_tmem_store();
}

using FwdMode = SparseAttnFwdMode;

template<typename TmaParams>
__global__ void __launch_bounds__(NUM_THREADS, 1, 1)
sparse_attn_fwd_kernel(__grid_constant__ const MxFp8SparseAttnFwdParams params, __grid_constant__ const TmaParams tma_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))

    const int s_q_idx = blockIdx.x;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = threadIdx.x % 32;
    const int warpgroup_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
    const int idx_in_warpgroup = threadIdx.x % 128;
    const int requested_topk_length = params.topk_length != nullptr ? __ldg(params.topk_length + s_q_idx) : params.topk;
    const int topk_length = max(0, min(requested_topk_length, params.topk));
    const int num_k_blocks = max(cute::ceil_div(topk_length, (int)B_TOPK), 1);  // num_k_blocks always >= 1

    // Define shared tensors
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

    Tensor tP0 = partition_fragment_C(tiled_mma_P, Shape<Int<QK_M>, Int<QK_N>>{});
    Tensor tP1 = partition_fragment_C(tiled_mma_P, Shape<Int<QK_M>, Int<QK_N>>{});
    Tensor tQ_scale = make_tensor<typename TiledMMA_P::FrgTypeSFA>(shape(SmemLayoutPScaleAAtom{}));
    Tensor tK_scale = make_tensor<typename TiledMMA_P::FrgTypeSFB>(shape(SmemLayoutPScaleBAtom{}));
    Tensor tV_scale = make_tensor<typename TiledMMA_O::FrgTypeSFA>(shape(SmemLayoutOScaleAAtom{}));
    Tensor tS_scale = make_tensor<typename TiledMMA_O::FrgTypeSFB>(shape(SmemLayoutOScaleBAtom{}));

    tP0.data().get() = tmem_cols::P0;
    tP1.data().get() = tmem_cols::P1;
    tQ_scale.data().get() = tmem_cols::Q_Scale;
    tK_scale.data().get() = tmem_cols::K_Scale;
    tV_scale.data().get() = tmem_cols::V_Scale;
    tS_scale.data().get() = tmem_cols::S_Scale;

    if (warp_idx == 0) {
        if(elect_one_sync()) {
            plan.bar_prologue_q.init(1);
            plan.bar_prologue_q_scale.init(1);
            fence_barrier_init();

            // Q is stored as e4m3 data followed by e8m0 block scales.
            plan.bar_prologue_q.arrive_and_expect_tx(QK_M*D_Q*sizeof(e4m3));
            Tensor gQ = flat_divide(
                tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx),
                Tile<Int<B_H>, Int<Q_TMA_K>>{}
            );
            Tensor sQ = flat_divide(
                make_tensor(make_smem_ptr(plan.q.data()), SmemLayoutQDuplicated{}),
                Tile<Int<B_H>, Int<Q_TMA_K>>{}
            );
            // The canonical 128-row UMMA layout interleaves the two Q views
            // at K=128 granularity, so populate its exact 64x128 subtiles.
            CUTE_UNROLL
            for (int d_block = 0; d_block < D_Q / Q_TMA_K; ++d_block) {
                CUTE_UNROLL
                for (int q_view = 0; q_view < QK_M / B_H; ++q_view) {
                    ku::launch_tma_copy(
                        tma_params.tma_Q,
                        gQ(_, _, _0{}, d_block),
                        sQ(_, _, q_view, d_block),
                        plan.bar_prologue_q,
                        TMA::CacheHintSm90::EVICT_FIRST
                    );
                }
            }
        CUTE_UNROLL
        for (int i = 0; i < NUM_P_BUFS; ++i) {
            plan.bar_qk_done[i].init(1);
            plan.bar_p_free[i].init(128);
        }
        CUTE_UNROLL
        for (int i = 0; i < NUM_BUFS; ++i) {
            CUTE_UNROLL
            for (int dv_block = 0; dv_block < NUM_SV_TMEM_BLOCKS - 1; ++dv_block) {
                plan.bar_sv_block_done[i][dv_block].init(1);
            }
            plan.bar_sv_done[i].init(1);
            plan.bar_kv_ready[i].init(1);
            plan.bar_kv_scale_ready[i].init(2);
            plan.bar_k_valid_ready[i].init(B_TOPK/8);
            plan.bar_k_valid_free[i].init(128);
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
#if defined(MXFP8_FWD_BARRIER_TIMING)
    if (s_q_idx == 0) {
        static_assert(sizeof(MxFp8BarrierTiming) % sizeof(uint64_t) == 0);
        uint64_t* timing_words = reinterpret_cast<uint64_t*>(
            &plan.barrier_timing
        );
        CUTE_UNROLL
        for (
            int i = threadIdx.x;
            i < sizeof(MxFp8BarrierTiming) / sizeof(uint64_t);
            i += NUM_THREADS
        ) {
            timing_words[i] = 0;
        }
    }
    __syncthreads();
    if (s_q_idx == 0 && threadIdx.x == 0) {
        plan.barrier_timing.origin_ns = mxfp8_timing_now_ns();
    }
    __syncthreads();
#endif
    MXFP8_MARK_WARP(
        "MXFP8_MARK 00 post_init warp=%d wg=%d",
        warp_idx,
        warpgroup_idx
    );

    if (warpgroup_idx == 0) {
        // The scale-factor fragment is replicated across all four physical
        // TMEM warp rows. WG0 fills those replicas directly from GMEM.
        constexpr int Q_SCALE_WORDS = Q_SCALE_BYTES / sizeof(uint32_t);
        const uint8_t* q_scale_bytes = reinterpret_cast<const uint8_t*>(params.q)
            + static_cast<int64_t>(s_q_idx) * params.stride_q_s_q
            + D_Q;
        uint32_t q_scale_lo[Q_SCALE_WORDS];
        uint32_t q_scale_hi[Q_SCALE_WORDS];
        CUTE_UNROLL
        for (int i = 0; i < Q_SCALE_WORDS; ++i) {
            q_scale_lo[i] = __ldg(reinterpret_cast<const uint32_t*>(
                q_scale_bytes + lane_idx * params.stride_q_h_q
            ) + i);
            q_scale_hi[i] = __ldg(reinterpret_cast<const uint32_t*>(
                q_scale_bytes + (lane_idx + 32) * params.stride_q_h_q
            ) + i);
        }
        uint32_t q_scale_words_lo[8] = {
            q_scale_lo[0], q_scale_hi[0], q_scale_lo[0], q_scale_hi[0],
            q_scale_lo[1], q_scale_hi[1], q_scale_lo[1], q_scale_hi[1]
        };
        uint32_t q_scale_words_hi[8] = {
            q_scale_lo[2], q_scale_hi[2], q_scale_lo[2], q_scale_hi[2],
            q_scale_lo[3], q_scale_hi[3], q_scale_lo[3], q_scale_hi[3]
        };
        ku::tmem_st_32dp32bNx<8>(
            tmem_cols::Q_Scale, q_scale_words_lo
        );
        ku::tmem_st_32dp32bNx<8>(
            tmem_cols::Q_Scale + 8, q_scale_words_hi
        );
        cutlass::arch::fence_view_async_tmem_store();
        MXFP8_TIMEPOINT(
            s_q_idx == 0 && idx_in_warpgroup == 0,
            plan.barrier_timing.q_scale_tmem_committed_ns
        );

        NamedBarrier::arrive_and_wait(128, NamedBarriers::q_scale_sync);
        ku::tcgen05_after_thread_sync();
        if (warp_idx == 0 && elect_one_sync()) {
            plan.bar_prologue_q_scale.arrive();
        }

        // U(t) is folded into S. Each 64-D group shares W(g), while the
        // four bytes in a TMEM word replicate that scale over token groups.
        const uint64_t w_scale_bits = __ldg(
            reinterpret_cast<const uint64_t*>(params.kv_scale_w)
        );
        uint32_t v_scale_words[V_SCALE_TMEM_COLS];
        CUTE_UNROLL
        for (int dv_block = 0; dv_block < NUM_SV_TMEM_BLOCKS; ++dv_block) {
            const uint32_t w_lo = static_cast<uint8_t>(
                w_scale_bits >> (2 * dv_block * 8)
            ) * 0x01010101u;
            const uint32_t w_hi = static_cast<uint8_t>(
                w_scale_bits >> ((2 * dv_block + 1) * 8)
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
        MXFP8_TIMEPOINT(
            s_q_idx == 0 && idx_in_warpgroup == 0,
            plan.barrier_timing.v_scale_tmem_committed_ns
        );
        // math instructions 
        if (idx_in_warpgroup < B_H) {
            plan.head_mi[idx_in_warpgroup] = MAX_INIT_VAL;
            plan.head_li[idx_in_warpgroup] = 0.0f;
            plan.head_real_mi[idx_in_warpgroup] = -CUDART_INF_F;
        }
        NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);

        static constexpr int NUM_ELEMS_PER_THREAD = B_TOPK / 2;

        CUTE_NO_UNROLL
        for (int k = 0; k < num_k_blocks; ++k) {
#if defined(MXFP8_FWD_BARRIER_TIMING)
            const bool trace_wg0_tile = s_q_idx == 0
                && lane_idx == 0 && k < MXFP8_TIMING_MAX_TILES;
#endif
            MXFP8_TIMEPOINT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].tile_start_ns
            );
            MXFP8_TIMED_WAIT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].tile_sync_wait_ns,
                (NamedBarrier::arrive_and_wait(
                    128, NamedBarriers::wg0_sync
                ))
            );
            int cur_buf = k % NUM_BUFS;
            int p_stage = k % NUM_P_BUFS;
            MXFP8_TIMED_WAIT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].qk_wait_ns,
                plan.bar_qk_done[p_stage].wait((k / NUM_P_BUFS) & 1)
            );
            MXFP8_TIMED_WAIT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].valid_wait_ns,
                plan.bar_k_valid_ready[cur_buf].wait((k / NUM_BUFS) & 1)
            );
            MXFP8_TIMEPOINT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].waits_done_ns
            );
            MXFP8_MARK_WARP(
                "MXFP8_MARK 10 wg0_deps_ready warp=%d tile=%d",
                warp_idx,
                k
            );
            ku::tcgen05_after_thread_sync();

            float p[NUM_ELEMS_PER_THREAD];
            int h = idx_in_warpgroup % B_H;
            int token_group = idx_in_warpgroup / B_H;
            int token_base = token_group * NUM_ELEMS_PER_THREAD;
            int p_tmem_col = (p_stage == 0 ? tmem_cols::P0 : tmem_cols::P1)
                + token_base;
            ku::tmem_ld_32dp32bNx<NUM_ELEMS_PER_THREAD>(p_tmem_col, p);
            cutlass::arch::fence_view_async_tmem_load();
            ku::tcgen05_before_thread_sync();
            plan.bar_p_free[p_stage].arrive();
            MXFP8_TIMEPOINT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].p_released_ns
            );

            Tensor sS_out = make_tensor(make_smem_ptr(plan.s_q_scale.s), SmemLayoutS{});
            Tensor sS_scale = make_tensor(make_smem_ptr(plan.s_scale.data()), SmemLayoutOScaleBAtom{});
            float local_pi_max = -CUDART_INF_F;
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD; ++i) {
                int kk = token_base + i;
                bool is_valid = ((plan.is_k_valid[cur_buf][kk / 8] >> (kk & 7)) & 1) != 0;
                float p_val = is_valid ? p[i] * params.sm_scale_div_log2 : -CUDART_INF_F;
                p[i] = p_val;
                local_pi_max = max(local_pi_max, p_val);
            }

            plan.rowwise_max_buf[idx_in_warpgroup] = local_pi_max;
            fence_view_async_shared();
            MXFP8_TIMEPOINT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].p_prepared_ns
            );
            MXFP8_TIMED_WAIT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].rowmax_wait_ns,
                (NamedBarrier::arrive_and_wait(
                    128, NamedBarriers::wg0_sync
                ))
            );

            float cur_pi_max = max(
                local_pi_max,
                plan.rowwise_max_buf[idx_in_warpgroup ^ B_H]
            );
            float old_mi = plan.head_mi[h];
            bool should_scale_o = cur_pi_max - old_mi > 6.0f;
            float new_max = should_scale_o ? max(cur_pi_max, old_mi) : old_mi;
            float scale_for_old = should_scale_o ? exp2f(old_mi - new_max) : 1.0f;
            MXFP8_TIMEPOINT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].rowmax_ready_ns
            );

            float local_sum = 0.0f;
            float scaled_s_absmax = 0.0f;
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD; ++i) {
                int kk = token_base + i;
                float s_val = exp2f(p[i] - new_max);
                local_sum += s_val;

                // Absorb U(t) before quantization; li uses the unscaled S.
                float scaled_s = s_val * plan.kv_u_scale[cur_buf][kk];
                p[i] = scaled_s;
                scaled_s_absmax = max(scaled_s_absmax, scaled_s);
            }

            float scale_f = scaled_s_absmax > 0.0f
                ? scaled_s_absmax / FP8_MAX
                : 1.0f;
            e8m0 scale_g = e8m0(scale_f);
            sS_scale(
                h,
                _0{},
                make_coord(token_group, _0{})
            ) = scale_g;
            CUTE_UNROLL
            for (int i = 0; i < NUM_ELEMS_PER_THREAD; ++i) {
                sS_out(h, token_base + i) = e4m3(p[i] / float(scale_g));
            }
            MXFP8_TIMEPOINT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].s_quantized_ns
            );

            plan.rowwise_li_buf[idx_in_warpgroup] = local_sum;
            fence_view_async_shared();
            MXFP8_TIMED_WAIT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].li_wait_ns,
                (NamedBarrier::arrive_and_wait(
                    128, NamedBarriers::wg0_sync
                ))
            );
            MXFP8_TIMEPOINT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].softmax_ready_ns
            );

            if (token_group == 0) {
                float cur_sum = local_sum + plan.rowwise_li_buf[idx_in_warpgroup + B_H];
                plan.head_scale[h] = scale_for_old;
                plan.head_mi[h] = new_max;
                plan.head_real_mi[h] = max(plan.head_real_mi[h], cur_pi_max);
                plan.head_li[h] = fma(plan.head_li[h], scale_for_old, cur_sum);
            }
            MXFP8_TIMEPOINT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].head_updated_ns
            );

            plan.bar_k_valid_free[cur_buf].arrive();
            fence_view_async_shared();
            MXFP8_TIMED_WAIT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].pre_rescale_wait_ns,
                (NamedBarrier::arrive_and_wait(
                    128, NamedBarriers::wg0_sync
                ))
            );
            MXFP8_TIMEPOINT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].pre_rescale_ready_ns
            );

            if (k > 0) {
                const int prev_buf = (k - 1) % NUM_BUFS;
                const int prev_phase = ((k - 1) / NUM_BUFS) & 1;

                // Commit each SV stripe independently.  WG0 can rescale a
                // completed 128-DV stripe while warp 8 is still issuing the
                // remaining SV MMAs for this tile.
                CUTE_UNROLL
                for (int dv_block = 0; dv_block < NUM_SV_TMEM_BLOCKS; ++dv_block) {
                    if (dv_block == 0) {
                        MXFP8_TIMED_WAIT(
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

                    ku::tcgen05_after_thread_sync();
                    rescale_O_block_t<B_H, B_H_TMEM, tmem_cols::O>(
                        plan.head_scale, dv_block
                    );
                    ku::tcgen05_before_thread_sync();

                    if (dv_block + 1 < NUM_SV_TMEM_BLOCKS) {
                        NamedBarrier::arrive_and_wait(
                            128, NamedBarriers::wg0_sync
                        );
                    } else {
                        MXFP8_TIMEPOINT(
                            trace_wg0_tile,
                            plan.barrier_timing.wg0[warp_idx][k].rescale_done_ns
                        );
                        MXFP8_TIMED_WAIT(
                            trace_wg0_tile,
                            plan.barrier_timing.wg0[warp_idx][k].rescale_wait_ns,
                            (NamedBarrier::arrive_and_wait(
                                128, NamedBarriers::wg0_sync
                            ))
                        );
                    }
                }
            } else {
                MXFP8_TIMEPOINT(
                    trace_wg0_tile,
                    plan.barrier_timing.wg0[warp_idx][k].rescale_done_ns
                );
            }

            if (idx_in_warpgroup == 0) {
                plan.bar_so_ready.arrive();
            }
            MXFP8_TIMEPOINT(
                trace_wg0_tile,
                plan.barrier_timing.wg0[warp_idx][k].s_arrived_ns
            );
            MXFP8_MARK_WARP(
                "MXFP8_MARK 11 wg0_s_ready warp=%d tile=%d",
                warp_idx,
                k
            );
        }
#if defined(MXFP8_FWD_BARRIER_TIMING)
        const bool trace_epilogue = s_q_idx == 0 && lane_idx == 0;
#endif
        NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
        if (idx_in_warpgroup < B_H) {
            float mi = plan.head_mi[idx_in_warpgroup];
            float li = plan.head_li[idx_in_warpgroup];
            float real_mi = plan.head_real_mi[idx_in_warpgroup];
            if (real_mi == -CUDART_INF_F) {
                li = 0.0f;
                mi = -CUDART_INF_F;
                plan.head_li[idx_in_warpgroup] = li;
                plan.head_mi[idx_in_warpgroup] = mi;
            }
            int global_index = s_q_idx*params.h_q + idx_in_warpgroup;
            float cur_lse = fmaf(mi, CUDART_LN2_F, logf(li));
            cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;
            params.max_logits[global_index] = real_mi*CUDART_LN2_F;
            params.lse[global_index] = cur_lse;
        }
        MXFP8_TIMEPOINT(
            trace_epilogue,
            plan.barrier_timing.epilogue[warp_idx].stats_stored_ns
        );

        // The softmax statistics are final before the last SV MMA finishes,
        // so prepare the output scale now and drain each completed SV stripe
        // directly into SMEM instead of waiting for all four stripes.
        if (idx_in_warpgroup < B_H) {
            int h = idx_in_warpgroup;
            float attn_sink = params.attn_sink == nullptr ? -CUDART_INF_F : __ldg(params.attn_sink + h)*CUDART_L2E_F;
            float output_scale = plan.head_li[h] == 0.0f
                ? 0.0f
                : __fdividef(1.0f, plan.head_li[h] + exp2f(attn_sink - plan.head_mi[h]));
            plan.head_scale[h] = output_scale;
        }
        MXFP8_TIMED_WAIT(
            trace_epilogue,
            plan.barrier_timing.epilogue[warp_idx].scale_sync_wait_ns,
            (NamedBarrier::arrive_and_wait(
                128, NamedBarriers::wg0_sync
            ))
        );
        MXFP8_TIMEPOINT(
            trace_epilogue,
            plan.barrier_timing.epilogue[warp_idx].scale_ready_ns
        );

        constexpr int B_EPI = 64;
        const int final_buf = (num_k_blocks - 1) % NUM_BUFS;
        const int final_phase = ((num_k_blocks - 1) / NUM_BUFS) & 1;
        const int scratch_lo_buf = (final_buf + 1) % NUM_BUFS;
        const int scratch_hi_buf = (final_buf + 2) % NUM_BUFS;
        Tensor tma_gO = flat_divide(
            tma_params.tma_O.get_tma_tensor(tma_params.shape_O)(_, _, s_q_idx),
            Shape<Int<B_H>, Int<B_EPI>>{}
        )(_, _, _0{}, _);
        // kvo.o aliases the final V tile.  Use the two inactive KV buffers
        // instead, with one 64-DV slot per SV stripe, so TMA can drain a
        // stripe while later SV MMAs still read the final V buffer.
        Tensor sO_lo = make_tensor(
            make_smem_ptr(reinterpret_cast<bf16*>(
                plan.kvo.kv.kv[scratch_lo_buf].data()
            )),
            SmemLayoutOTiles<NUM_SV_TMEM_BLOCKS>{}
        );
        Tensor sO_hi = make_tensor(
            make_smem_ptr(reinterpret_cast<bf16*>(
                plan.kvo.kv.kv[scratch_hi_buf].data()
            )),
            SmemLayoutOTiles<NUM_SV_TMEM_BLOCKS>{}
        );
        Tensor sO_lo_divided = flat_divide(
            sO_lo,
            Shape<Int<B_H>, Int<B_EPI>>{}
        )(_, _, _0{}, _);
        Tensor sO_hi_divided = flat_divide(
            sO_hi,
            Shape<Int<B_H>, Int<B_EPI>>{}
        )(_, _, _0{}, _);
        auto thr_tma = tma_params.tma_O.get_slice(_0{});

        float o_head[B_H_TMEM];
        CUTE_UNROLL
        for (int dv_block = 0; dv_block < NUM_SV_TMEM_BLOCKS; ++dv_block) {
            if (dv_block == 0) {
                MXFP8_TIMED_WAIT(
                    trace_epilogue,
                    plan.barrier_timing.epilogue[warp_idx].final_sv_wait_ns,
                    plan.bar_sv_block_done[final_buf][dv_block].wait(
                        final_phase
                    )
                );
            } else if (dv_block + 1 < NUM_SV_TMEM_BLOCKS) {
                plan.bar_sv_block_done[final_buf][dv_block].wait(final_phase);
            } else {
                plan.bar_sv_done[final_buf].wait(final_phase);
            }
            if (dv_block == 0) {
                MXFP8_TIMEPOINT(
                    trace_epilogue,
                    plan.barrier_timing.epilogue[warp_idx].final_sv_ready_ns
                );
            }
            ku::tcgen05_after_thread_sync();

            ku::tmem_ld_32dp32bNx<B_H_TMEM>(
                tmem_cols::O + dv_block * B_H_TMEM, o_head
            );
            cutlass::arch::fence_view_async_tmem_load();
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
                const int h = h_base + (lane_idx & 7);
                const int d = dv_block * B_EPI + d_vec;
                if (warp_idx < 2) {
                    *reinterpret_cast<uint4*>(&sO_lo(h, d)) = vector;
                } else {
                    *reinterpret_cast<uint4*>(&sO_hi(h, d)) = vector;
                }
            }

            fence_view_async_shared();
            if (dv_block + 1 < NUM_SV_TMEM_BLOCKS) {
                NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
            } else {
                MXFP8_TIMEPOINT(
                    trace_epilogue,
                    plan.barrier_timing.epilogue[warp_idx].o_staged_ns
                );
                MXFP8_TIMED_WAIT(
                    trace_epilogue,
                    plan.barrier_timing.epilogue[warp_idx].o_sync_wait_ns,
                    (NamedBarrier::arrive_and_wait(
                        128, NamedBarriers::wg0_sync
                    ))
                );
            }

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
        MXFP8_TIMEPOINT(
            trace_epilogue,
            plan.barrier_timing.epilogue[warp_idx].store_issued_ns
        );

        if (warp_idx == 0) {
            cute::TMEM::Allocator1Sm().free(0, 512);
        }
        MXFP8_MARK_WARP(
            "MXFP8_MARK 12 wg0_epilogue_done warp=%d",
            warp_idx
        );
    } else if (warpgroup_idx == 1) {
        // Producer warp for KV
        int producer_warp_idx = cutlass::canonical_warp_idx_sync() - 4;
        constexpr int NUM_LOCAL_ROWS_PER_WARP = (B_TOPK/4)/NUM_KV_PRODUCER_WARPS;
        CUTE_NO_UNROLL
        for (int k = 0; k < num_k_blocks; ++k) {
#if defined(MXFP8_FWD_BARRIER_TIMING)
            const bool trace_kv_tile = s_q_idx == 0
                && lane_idx == 0 && k < MXFP8_TIMING_MAX_TILES;
#endif
            MXFP8_TIMEPOINT(
                trace_kv_tile,
                plan.barrier_timing.kv[producer_warp_idx][k].tile_start_ns
            );
            int4 indices[NUM_LOCAL_ROWS_PER_WARP];
            if (elect_one_sync()) {
                // Copy NoPE data with gather4.
                int cur_buf = k%NUM_BUFS;
                MXFP8_TIMED_WAIT(
                    trace_kv_tile,
                    plan.barrier_timing.kv[producer_warp_idx][k].sv_free_wait_ns,
                    plan.bar_sv_done[cur_buf].wait((k / NUM_BUFS) & 1 ^ 1)
                );

                bool has_valid_index = false;
                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                    int vector_idx = local_row*NUM_KV_PRODUCER_WARPS + producer_warp_idx;
                    indices[local_row] = __ldg(reinterpret_cast<int4*>(gIndices + k*B_TOPK) + vector_idx);
                    CUTE_UNROLL
                    for (int i = 0; i < 4; ++i) {
                        int row = vector_idx*4 + i;
                        int& index = reinterpret_cast<int*>(&indices[local_row])[i];
                        if (index < 0 || index >= params.s_kv || k*B_TOPK + row >= topk_length) {
                            index = -1;
                        } else {
                            has_valid_index = true;
                        }
                    }
                }
                plan.kv_warp_has_valid[cur_buf][producer_warp_idx] = has_valid_index;
                MXFP8_TIMEPOINT(
                    trace_kv_tile,
                    plan.barrier_timing.kv[producer_warp_idx][k].indices_ready_ns
                );
            }
            fence_view_async_shared();
            MXFP8_TIMED_WAIT(
                trace_kv_tile,
                plan.barrier_timing.kv[producer_warp_idx][k].indices_sync_wait_ns,
                (NamedBarrier::arrive_and_wait(
                    128, NamedBarriers::wg1_tma_sync
                ))
            );

            if (elect_one_sync()) {
                int cur_buf = k%NUM_BUFS;
                if (producer_warp_idx == 0) {
                    bool all_invalid = true;
                    CUTE_UNROLL
                    for (int producer = 0; producer < NUM_KV_PRODUCER_WARPS; ++producer) {
                        all_invalid &= !plan.kv_warp_has_valid[cur_buf][producer];
                    }
                    plan.kv_skip_tma[cur_buf] = all_invalid;
                    plan.bar_kv_ready[cur_buf].arrive_and_expect_tx(B_TOPK*D_K*sizeof(e4m3));
                }
            }
            fence_view_async_shared();
            MXFP8_TIMEPOINT(
                trace_kv_tile,
                plan.barrier_timing.kv[producer_warp_idx][k].transaction_ready_ns
            );
            MXFP8_TIMED_WAIT(
                trace_kv_tile,
                plan.barrier_timing.kv[producer_warp_idx][k].transaction_sync_wait_ns,
                (NamedBarrier::arrive_and_wait(
                    128, NamedBarriers::wg1_tma_sync
                ))
            );

            int cur_buf = k%NUM_BUFS;
            uint8_t* sK_base = reinterpret_cast<uint8_t*>(plan.kvo.kv.kv[cur_buf].data());
            if (plan.kv_skip_tma[cur_buf]) {
                // A zero S fragment can still propagate NaN from uninitialized
                // E4M3 V data. Materialize a zero tile before manually
                // completing the skipped TMA transaction.
                uint4* sK_vec = reinterpret_cast<uint4*>(sK_base);
                constexpr int NUM_KV_VECS = B_TOPK*D_K/sizeof(uint4);
                for (int vec = idx_in_warpgroup; vec < NUM_KV_VECS; vec += 128) {
                    sK_vec[vec] = make_uint4(0, 0, 0, 0);
                }
                fence_view_async_shared();
                NamedBarrier::arrive_and_wait(128, NamedBarriers::wg1_tma_sync);
                if (producer_warp_idx == 0 && elect_one_sync()) {
                    plan.bar_kv_ready[cur_buf].complete_transaction(
                        B_TOPK*D_K*sizeof(e4m3)
                    );
                }
            }

            if (elect_one_sync()) {
                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_LOCAL_ROWS_PER_WARP; ++local_row) {
                    if (!plan.kv_skip_tma[cur_buf]) {
                        CUTE_UNROLL
                        for (int local_col = 0; local_col < D_K/TMA_K_CHUNK_BYTES; ++local_col) {
                            ku::tma_gather4(
                                &(tma_params.tensor_map_kv),
                                plan.bar_kv_ready[cur_buf],
                                sK_base + local_col * B_TOPK * TMA_K_CHUNK_BYTES
                                        + (
                                            local_row * NUM_KV_PRODUCER_WARPS
                                            + producer_warp_idx
                                        ) * 4 * TMA_K_CHUNK_BYTES,
                                local_col*TMA_K_CHUNK_ELEMS,
                                indices[local_row],
                                (int64_t)TMA::CacheHintSm90::EVICT_LAST
                            );
                        }
                    }

                }
            }
            MXFP8_TIMEPOINT(
                trace_kv_tile,
                plan.barrier_timing.kv[producer_warp_idx][k].tma_issued_ns
            );
            MXFP8_MARK_WARP(
                "MXFP8_MARK 20 kv_tma_issued warp=%d tile=%d",
                warp_idx,
                k
            );
        }
    } else {
        if (warp_idx == 8 && elect_one_sync()) {

            Tensor sQ_dup = make_tensor(make_smem_ptr(plan.q.data()), SmemLayoutQDuplicated{});
            MXFP8_MARK_ONE("MXFP8_MARK 30 mma_start");
            MXFP8_TIMED_WAIT(
                s_q_idx == 0,
                plan.barrier_timing.q_tma_wait_ns,
                plan.bar_prologue_q.wait(0)
            );

            MXFP8_TIMED_WAIT(
                s_q_idx == 0,
                plan.barrier_timing.q_scale_sync_wait_ns,
                plan.bar_prologue_q_scale.wait(0)
            );

            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks+1; ++k) {
#if defined(MXFP8_FWD_BARRIER_TIMING)
                const bool trace_mma_iter = s_q_idx == 0
                    && k <= MXFP8_TIMING_MAX_TILES;
#endif
                MXFP8_TIMEPOINT(
                    trace_mma_iter,
                    plan.barrier_timing.mma[k].iter_start_ns
                );
                if (k < num_k_blocks) {
                    // Pi = QKi^T
                    int cur_buf = k % NUM_BUFS;
                    int p_stage = k % NUM_P_BUFS;
                    Tensor sK = make_tensor(
                        make_smem_ptr(plan.kvo.kv.kv[cur_buf].data()),
                        SmemLayoutK_TiledMMA{}
                    );

                    MXFP8_TIMED_WAIT(
                        trace_mma_iter,
                        plan.barrier_timing.mma[k].p_free_wait_ns,
                        plan.bar_p_free[p_stage].wait(
                            ((k / NUM_P_BUFS) & 1) ^ 1
                        )
                    );
                    ku::tcgen05_after_thread_sync();

                    MXFP8_TIMED_WAIT(
                        trace_mma_iter,
                        plan.barrier_timing.mma[k].kv_scale_wait_ns,
                        plan.bar_kv_scale_ready[cur_buf].wait(
                            (k / NUM_BUFS) & 1
                        )
                    );
                    CUTE_UNROLL
                    for (int sf_block = 0; sf_block < QK_K / 128; ++sf_block) {
                        Tensor sK_block = make_tensor(
                            make_smem_ptr(
                                plan.kvo.kv.kv_scale[cur_buf].data()
                                + sf_block * cosize_v<SmemLayoutPScaleBBlockAtom>
                            ),
                            SmemLayoutPScaleBBlockAtom{}
                        );
                        Tensor tK_block = make_tensor<typename TiledMMA_P::FrgTypeSFB>(
                            shape(SmemLayoutPScaleBBlockAtom{})
                        );
                        tK_block.data().get() = tmem_cols::K_Scale
                            + sf_block * TMEM_SCALE_K128_STRIDE;
                        auto sK_compact = make_tensor(
                            sK_block.data(), filter_zeros(sK_block.layout())
                        );
                        auto tK_compact = make_tensor(
                            tK_block.data(), filter_zeros(tK_block.layout())
                        );
                        auto copy_K_scale = make_utccp_copy(
                            SM100_UTCCP_4x32dp128bit_1cta{}, tK_compact
                        );
                        auto thr_K = copy_K_scale.get_slice(0);
                        auto src_K = get_utccp_smem_desc_tensor<SM100_UTCCP_4x32dp128bit_1cta>(
                            thr_K.partition_S(sK_compact)
                        );
                        auto dst_K = thr_K.partition_D(tK_compact);
                        cute::copy(copy_K_scale, src_K, dst_K);
                    }
                    MXFP8_TIMEPOINT(
                        trace_mma_iter,
                        plan.barrier_timing.mma[k].k_scale_ready_ns
                    );

                    MXFP8_TIMED_WAIT(
                        trace_mma_iter,
                        plan.barrier_timing.mma[k].kv_wait_ns,
                        plan.bar_kv_ready[cur_buf].wait((k / NUM_BUFS) & 1)
                    );
                    MXFP8_TIMEPOINT(
                        trace_mma_iter,
                        plan.barrier_timing.mma[k].kv_ready_ns
                    );
                    ku::tcgen05_after_thread_sync();

                    ku::utcmma_blockscaled_ss(
                        tiled_mma_P,
                        sQ_dup,
                        sK,
                        tQ_scale,
                        tK_scale,
                        p_stage == 0 ? tP0 : tP1,
                        true
                    );

                    ku::umma_arrive_noelect(plan.bar_qk_done[p_stage]);
                    MXFP8_TIMEPOINT(
                        trace_mma_iter,
                        plan.barrier_timing.mma[k].qk_committed_ns
                    );
                    MXFP8_MARK_ONE(
                        "MXFP8_MARK 31 qk_committed tile=%d",
                        k
                    );
                }

                if (k > 0) {
                    // O += S(i-1)V(i-1)
                    int cur_buf = (k-1)%NUM_BUFS;

                    Tensor sS = make_tensor(make_smem_ptr(plan.s_q_scale.s), SmemLayoutS{});
                    Tensor sS_scale = make_tensor(make_smem_ptr(plan.s_scale.data()), SmemLayoutOScaleBAtom{});
                    Tensor sV = make_tensor(make_smem_ptr(plan.kvo.kv.kv[cur_buf].data()), SmemLayoutV{});

                    // Wait for S(i-1) and O to be scaled
                    MXFP8_TIMED_WAIT(
                        trace_mma_iter,
                        plan.barrier_timing.mma[k].s_ready_wait_ns,
                        plan.bar_so_ready.wait((k - 1) & 1)
                    );
                    MXFP8_TIMEPOINT(
                        trace_mma_iter,
                        plan.barrier_timing.mma[k].s_ready_ns
                    );
                    ku::tcgen05_after_thread_sync();

                    auto sS_compact = make_tensor(sS_scale.data(), filter_zeros(sS_scale.layout()));
                    auto tS_compact = make_tensor(tS_scale.data(), filter_zeros(tS_scale.layout()));
                    auto copy_S_scale = make_utccp_copy(SM100_UTCCP_4x32dp128bit_1cta{}, tS_compact);
                    auto thr_S = copy_S_scale.get_slice(0);
                    auto src_S = get_utccp_smem_desc_tensor<SM100_UTCCP_4x32dp128bit_1cta>(
                        thr_S.partition_S(sS_compact) // TODO: change here to tcgen05.ld instead of tcgen05.cp
                    );
                    auto dst_S = thr_S.partition_D(tS_compact);
                    cute::copy(copy_S_scale, src_S, dst_S);
                    ku::tcgen05_after_thread_sync();
                    MXFP8_TIMEPOINT(
                        trace_mma_iter,
                        plan.barrier_timing.mma[k].s_scale_ready_ns
                    );

                    // O += sS @ sV
                    CUTE_UNROLL
                    for (int dv_block = 0; dv_block < NUM_SV_TMEM_BLOCKS; ++dv_block) {
                        Tensor tO_block = partition_fragment_C(tiled_mma_O, Shape<Int<SV_M>, Int<B_H>>{});
                        tO_block.data().get() = tmem_cols::O + dv_block*B_H_TMEM;
                        tV_scale.data().get() = tmem_cols::V_Scale
                            + dv_block * SV_SCALE_TMEM_COLS;
                        ku::utcmma_blockscaled_ss(
                            tiled_mma_O, sV(make_coord(_, dv_block), _), sS,
                            tV_scale, tS_scale, tO_block, k == 1
                        );
                        if (dv_block + 1 < NUM_SV_TMEM_BLOCKS) {
                            ku::umma_arrive_noelect(
                                plan.bar_sv_block_done[cur_buf][dv_block]
                            );
                        }
                    }
                    // Keep the final completion barrier as the existing
                    // whole-SV handoff used by KV buffer reuse.
                    ku::umma_arrive_noelect(plan.bar_sv_done[cur_buf]);
                    MXFP8_TIMEPOINT(
                        trace_mma_iter,
                        plan.barrier_timing.mma[k].sv_committed_ns
                    );
                    MXFP8_MARK_ONE(
                        "MXFP8_MARK 32 sv_committed tile=%d",
                        k - 1
                    );
                }

            }
            MXFP8_MARK_ONE("MXFP8_MARK 33 mma_done");

        } else if (warp_idx == 9) {
            // KV valid loading warp
            if (lane_idx < B_TOPK/8) {
                CUTE_NO_UNROLL
                for (int k = 0; k < num_k_blocks; ++k) {
#if defined(MXFP8_FWD_BARRIER_TIMING)
                    const bool trace_mask_tile = s_q_idx == 0
                        && lane_idx == 0 && k < MXFP8_TIMING_MAX_TILES;
#endif
                    MXFP8_TIMEPOINT(
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
                    MXFP8_TIMED_WAIT(
                        trace_mask_tile,
                        plan.barrier_timing.mask[k].buffer_free_wait_ns,
                        plan.bar_k_valid_free[cur_buf].wait(
                            (k / NUM_BUFS) & 1 ^ 1
                        )
                    );
                    plan.is_k_valid[cur_buf][lane_idx] = k_validness_mask;
                    plan.bar_k_valid_ready[cur_buf].arrive();
                    MXFP8_TIMEPOINT(
                        trace_mask_tile,
                        plan.barrier_timing.mask[k].arrived_ns
                    );
                    MXFP8_MARK_WARP(
                        "MXFP8_MARK 40 mask_arrived tile=%d",
                        k
                    );
                }
            }
        } else if (warp_idx == 10 || warp_idx == 11) {
            // Two warps cooperatively gather the page-tail KV scales and
            // scatter them into the block-scaled MMA layout.
            constexpr int NUM_SCALE_PRODUCER_WARPS = 2;
            constexpr int NUM_ROWS_PER_SCALE_WARP = B_TOPK / NUM_SCALE_PRODUCER_WARPS;
            constexpr int NUM_ROWS_PER_SCALE_LANE = NUM_ROWS_PER_SCALE_WARP / 32;
            const int scale_warp_idx = warp_idx - 10;
            const uint8_t* kv_scale_base = reinterpret_cast<const uint8_t*>(params.kv)
                + static_cast<int64_t>(params.s_kv) * params.h_kv * D_K;

            int w_anchor_bits = 0;
            if (lane_idx == 0) {
                uint64_t w_scale_bits = __ldg(
                    reinterpret_cast<const uint64_t*>(params.kv_scale_w)
                );
                w_anchor_bits = static_cast<uint8_t>(
                    w_scale_bits >> (KV_SCALE_ANCHOR * 8)
                );
            }
            w_anchor_bits = __shfl_sync(0xffffffff, w_anchor_bits, 0);

            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks; ++k) {
#if defined(MXFP8_FWD_BARRIER_TIMING)
                const bool trace_scale_tile = s_q_idx == 0
                    && lane_idx == 0 && k < MXFP8_TIMING_MAX_TILES;
#endif
                MXFP8_TIMEPOINT(
                    trace_scale_tile,
                    plan.barrier_timing.scale[scale_warp_idx][k].tile_start_ns
                );
                int cur_buf = k % NUM_BUFS;
                MXFP8_TIMED_WAIT(
                    trace_scale_tile,
                    plan.barrier_timing.scale[scale_warp_idx][k].buffer_free_wait_ns,
                    plan.bar_sv_done[cur_buf].wait(
                        (k / NUM_BUFS) & 1 ^ 1
                    )
                );
                Tensor sK_scale = make_tensor(
                    make_smem_ptr(plan.kvo.kv.kv_scale[cur_buf].data()),
                    SmemLayoutPScaleBAtom{}
                );

                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_ROWS_PER_SCALE_LANE; ++local_row) {
                    int row = scale_warp_idx * NUM_ROWS_PER_SCALE_WARP
                        + local_row * 32 + lane_idx;
                    int src_idx = __ldg(gIndices + k * B_TOPK + row);
                    if (src_idx < 0 || src_idx >= params.s_kv || k * B_TOPK + row >= topk_length) {
                        src_idx = -1;
                    }

                    alignas(8) e8m0 scale[K_SCALE_BYTES];
                    if (src_idx >= 0) {
                        const e8m0* src_scale = reinterpret_cast<const e8m0*>(kv_scale_base)
                            + static_cast<int64_t>(src_idx) * params.h_kv * K_SCALE_BYTES;
                        *reinterpret_cast<uint64_t*>(scale) = __ldg(
                            reinterpret_cast<const uint64_t*>(src_scale)
                        );
                        uint8_t anchor_bits = reinterpret_cast<uint8_t*>(scale)[KV_SCALE_ANCHOR];
                        plan.kv_u_scale[cur_buf][row] = ue8m0_ratio_to_float(
                            anchor_bits, static_cast<uint8_t>(w_anchor_bits)
                        );
                    } else {
                        *reinterpret_cast<uint64_t*>(scale) = 0;
                        plan.kv_u_scale[cur_buf][row] = 1.0f;
                    }

                    constexpr int K_SCALE_GROUPS = QK_K / MXFP8_SCALE_VEC_SIZE;
                    CUTE_UNROLL
                    for (int dst_sf_idx = 0; dst_sf_idx < K_SCALE_GROUPS; ++dst_sf_idx) {
                        int src_sf_idx = dst_sf_idx / K_SCALE_DUP;
                        sK_scale(
                            row,
                            _0{},
                            make_coord(
                                dst_sf_idx % SCALE_GROUPS_PER_TMEM_BLOCK,
                                dst_sf_idx / SCALE_GROUPS_PER_TMEM_BLOCK
                            )
                        ) = scale[src_sf_idx];
                    }
                }
                fence_view_async_shared();
                if (elect_one_sync()) {
                    plan.bar_kv_scale_ready[cur_buf].arrive();
                }
                MXFP8_TIMEPOINT(
                    trace_scale_tile,
                    plan.barrier_timing.scale[scale_warp_idx][k].arrived_ns
                );
                MXFP8_MARK_WARP(
                    "MXFP8_MARK 50 scale_arrived warp=%d tile=%d",
                    warp_idx,
                    k
                );
            }
        }
    }

#if defined(MXFP8_FWD_BARRIER_TIMING)
    if (s_q_idx == 0 && lane_idx == 0) {
        plan.barrier_timing.branch_end_ns[warp_idx] = mxfp8_timing_now_ns()
            - plan.barrier_timing.origin_ns;
    }
    __syncthreads();
    if (s_q_idx == 0 && threadIdx.x == 0) {
        print_mxfp8_barrier_timing(plan, num_k_blocks);
    }
#endif

#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100");
    }
#endif
}

#undef MXFP8_MARK_WARP
#undef MXFP8_MARK_ONE
#undef MXFP8_TIMED_WAIT
#undef MXFP8_TIMEPOINT

template<int D_QK>
void run_mxfp8_fwd_phase1_kernel(const MxFp8SparseAttnFwdParams& params) {
    KU_ASSERT(params.s_q > 0);
    KU_ASSERT(params.s_kv > 0);
    KU_ASSERT(params.topk > 0);
    KU_ASSERT(params.topk % B_TOPK == 0, "topk (%d) mod B_TOPK (%d) must be 0", params.topk, B_TOPK);
    KU_ASSERT(params.h_q == B_H);
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.d_qk == D_QK);
    KU_ASSERT(params.d_v == D_V);
    KU_ASSERT(params.stride_kv_h_kv == KV_BYTES_PER_TOKEN,
        "packed KV storage must use a %d-byte logical token envelope", KV_BYTES_PER_TOKEN);
    KU_ASSERT(params.stride_kv_s_kv == params.h_kv * KV_BYTES_PER_TOKEN,
        "packed KV storage must be contiguous across tokens");
    KU_ASSERT(reinterpret_cast<int64_t>(params.q) % 16 == 0, "q must be 16-byte aligned");
    KU_ASSERT(params.stride_q_h_q % alignof(uint32_t) == 0,
        "stride_q_h_q must be 4-byte aligned for direct Q-scale loads");
    KU_ASSERT(params.stride_q_s_q % alignof(uint32_t) == 0,
        "stride_q_s_q must be 4-byte aligned for direct Q-scale loads");
    KU_ASSERT(reinterpret_cast<int64_t>(params.kv) % 16 == 0, "kv must be 16-byte aligned");
    KU_ASSERT(params.kv_scale_w != nullptr);
    KU_ASSERT(reinterpret_cast<int64_t>(params.kv_scale_w) % 8 == 0, "kv_scale_w must be 8-byte aligned");
    static_assert(D_QK == D_Q);

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
        SmemLayoutQBlock{}
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

    auto kernel = &sparse_attn_fwd_kernel<decltype(tma_params)>;

    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    kernel<<<params.s_q, NUM_THREADS, smem_size, params.stream>>>(params, tma_params);
    KU_CHECK_KERNEL_LAUNCH();
}

}
