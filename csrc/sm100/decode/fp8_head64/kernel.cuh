#include "kernel.h"

#include <cstdint>

#include <math_constants.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/barrier.h>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/tensor.hpp>

#include "kerutils/kerutils.cuh"

#include "config.h"
#include "sm100/helpers.h"
#include "utils.h"

namespace sm100::decode::fp8_head64 {

// Instruction-family ablations mirror sparse prefill.  Each replacement keeps
// a live data dependency into the existing memory/synchronization skeleton so
// disabling one family does not let ptxas remove another family as dead code.
CUTE_DEVICE
float fp8_decode_exp2(float value) {
#if defined(FP8_FWD_DISABLE_SFU)
    uint32_t bits = __float_as_uint(value);
    asm volatile("mov.b32 %0, %0;" : "+r"(bits));
    return __uint_as_float(bits);
#else
    float result;
    asm volatile(
        "ex2.approx.ftz.f32 %0, %1;"
        : "=f"(result)
        : "f"(value)
    );
    return result;
#endif
}

CUTE_DEVICE
float fp8_decode_log(float value) {
    float result;
#if defined(FP8_FWD_DISABLE_SFU)
    uint32_t bits = __float_as_uint(value);
    asm volatile("mov.b32 %0, %0;" : "+r"(bits));
    result = __uint_as_float(bits);
#else
    asm volatile(
        "lg2.approx.ftz.f32 %0, %1;"
        : "=f"(result)
        : "f"(value)
    );
#endif
    return result * CUDART_LN2_F;
}

CUTE_DEVICE
float fp8_decode_log2(float value) {
#if defined(FP8_FWD_DISABLE_SFU)
    uint32_t bits = __float_as_uint(value);
    asm volatile("mov.b32 %0, %0;" : "+r"(bits));
    return __uint_as_float(bits);
#else
    float result;
    asm volatile(
        "lg2.approx.ftz.f32 %0, %1;"
        : "=f"(result)
        : "f"(value)
    );
    return result;
#endif
}

CUTE_DEVICE
float fp8_decode_rcp(float value) {
#if defined(FP8_FWD_DISABLE_SFU)
    uint32_t bits = __float_as_uint(value);
    asm volatile("mov.b32 %0, %0;" : "+r"(bits));
    return __uint_as_float(bits);
#else
    float result;
    asm volatile(
        "rcp.approx.ftz.f32 %0, %1;"
        : "=f"(result)
        : "f"(value)
    );
    return result;
#endif
}

CUTE_DEVICE
uint32_t fp8_decode_exp2_quad_packed(float a, float b, float c, float d) {
#if defined(FP8_FWD_DISABLE_SFU)
    uint32_t packed = __float_as_uint(d);
    asm volatile("mov.b32 %0, %0;" : "+r"(packed));
    return packed;
#else
    uint32_t packed;
    asm volatile(
        "{\n"
        "  .reg .f32 s0, s1, s2, s3;\n"
        "  .reg .b32 b0, b1, b2, b3, ab, cd;\n"
        "  ex2.approx.ftz.f32 s0, %1;\n"
        "  ex2.approx.ftz.f32 s1, %2;\n"
        "  ex2.approx.ftz.f32 s2, %3;\n"
        "  ex2.approx.ftz.f32 s3, %4;\n"
        "  mov.b32 b0, s0;\n"
        "  mov.b32 b1, s1;\n"
        "  mov.b32 b2, s2;\n"
        "  mov.b32 b3, s3;\n"
        "  prmt.b32 ab, b0, b1, 0x4040;\n"
        "  prmt.b32 cd, b2, b3, 0x4040;\n"
        "  prmt.b32 %0, ab, cd, 0x5410;\n"
        "}"
        : "=r"(packed)
        : "f"(a), "f"(b), "f"(c), "f"(d)
    );
    return packed;
#endif
}

CUTE_DEVICE
float fp8_decode_bit_passthrough(float value) {
    uint32_t result;
    asm volatile(
        "prmt.b32 %0, %1, %1, 0x3210;"
        : "=r"(result)
        : "r"(__float_as_uint(value))
    );
    return __uint_as_float(result);
}

CUTE_DEVICE
float ue8m0_bits_to_float(uint8_t bits) {
    if (bits == 0) {
        return __uint_as_float(0x00400000u);
    }
    return __uint_as_float(static_cast<uint32_t>(bits) << 23);
}

CUTE_DEVICE
void rescale_o_tmem_stripe(
    float scale,
    uint32_t tmem_col
) {
    float2 o[SV_M / 4];
    const float2 scale2 = make_float2(scale, scale);

    ku::tmem_ld_32dp32bNx<SV_M / 2>(tmem_col, o);
    cutlass::arch::fence_view_async_tmem_load();
    CUTE_UNROLL
    for (int i = 0; i < SV_M / 4; ++i) {
#if !defined(FP8_FWD_DISABLE_NON_SFU_NON_GEMM)
        o[i] = ku::float2_mul(o[i], scale2);
#endif
    }
#if defined(FP8_FWD_DISABLE_NON_SFU_NON_GEMM)
    o[0].x = scale;
#endif
    ku::tmem_st_32dp32bNx<SV_M / 2>(tmem_col, o);
    cutlass::arch::fence_view_async_tmem_store();
}

template<typename TmaParam>
__global__ void __launch_bounds__(NUM_THREADS, 1, 1)
flash_fwd_splitkv_mla_fp8_sparse_kernel(
    __grid_constant__ const SparseAttnFp8DecodeParams params,
    __grid_constant__ const TmaParam tma_params
) {
#if defined(KERUTILS_ENABLE_SM100A)
    const int s_q_idx = blockIdx.x;
    const int partition_idx = blockIdx.y;
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int idx_in_warpgroup = threadIdx.x % 128;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = threadIdx.x % 32;

    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan *>(wksp_buf);

    if (warp_idx == 0 && elect_one_sync()) {
        cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_O.get_tma_descriptor());
        cute::prefetch_tma_descriptor(&tma_params.tensor_map_kv);
        if (params.extra_topk > 0) {
            cute::prefetch_tma_descriptor(&tma_params.tensor_map_extra_kv);
        }
    }

    if (warp_idx == 0) {
        if (elect_one_sync()) {
            plan.bar_last_store_done.init(128);
            plan.bar_q_tma.init(1);
            plan.bar_q_utccp.init(1);
            CUTE_UNROLL
            for (int i = 0; i < NUM_BUFS; ++i) {
                // WG2 contributes one arrival per thread; all 128 threads
                // issue one gather4 transaction for this KV stage.
                plan.bar_kv_ready[i].init(128);
                // Warp 5 and warp 6 each publish one half of the token-scale
                // tile for the corresponding KV stage.
                plan.bar_kv_scale_ready[i].init(2);
                plan.bar_qk_done[i].init(1);
                plan.bar_so_ready[i].init(128);
                plan.bar_sv_done[i].init(1);
                plan.bar_o_ready[i].init(128);
            }
            CUTE_UNROLL
            for (int i = 0; i < NUM_INDEX_BUFS; ++i) {
                // Warp 7 produces one four-index packet per lane.  WG0 and
                // WG2 each retain the packet until their 128 arrivals land.
                plan.bar_valid_coord_scale_ready[i].init(32);
                plan.bar_valid_coord_scale_free[i].init(256);
            }
            cutlass::arch::fence_barrier_init();
        }
        cute::TMEM::Allocator1Sm().allocate(512, plan.tmem_start_addr.data());
        KU_TRAP_ONLY_DEVICE_ASSERT(plan.tmem_start_addr.data()[0] == 0);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }
    __syncthreads();

    struct MainLoopArgs {
        int batch_idx;
        int start_block_idx;
        int end_block_idx;
        bool is_no_split;
        int n_split_idx;
        bool bar_phase_batch_rel;
        int topk_length;
        int extra_topk_length;
        int num_orig_kv_blocks;
        bool is_last_batch;
    };

    auto run_main_loop = [&](auto f) {
        // Keeping scheduler state inside each specialization avoids register
        // spilling into the WG0 softmax and WG2 TMA paths.
        DecodingSchedMeta sched_meta;
        KU_LDG_256(
            params.tile_scheduler_metadata_ptr + partition_idx,
            &sched_meta,
            ".nc",
            "no_allocate",
            "evict_normal",
            "256B"
        );

        if (sched_meta.begin_req_idx >= params.b) {
            return;
        }

        bool bar_phase_batch_rel = false;
        #pragma unroll 1
        for (int batch_idx = sched_meta.begin_req_idx;
             batch_idx <= sched_meta.end_req_idx;
             ++batch_idx, bar_phase_batch_rel = !bar_phase_batch_rel) {
            const int topk_length = params.topk_length
                ? __ldg(params.topk_length + batch_idx)
                : params.topk;
            const int orig_topk_padded = max(
                ku::ceil(topk_length, (int)B_TOPK), (int)B_TOPK
            );
            const int extra_topk_length = params.extra_topk_length
                ? __ldg(params.extra_topk_length + batch_idx)
                : params.extra_topk;
            const int total_topk_padded = orig_topk_padded
                + ku::ceil(extra_topk_length, (int)B_TOPK);
            const int start_block_idx = batch_idx == sched_meta.begin_req_idx
                ? sched_meta.begin_block_idx
                : 0;
            const int end_block_idx = batch_idx == sched_meta.end_req_idx
                ? sched_meta.end_block_idx
                : total_topk_padded / B_TOPK;
            const bool is_split = batch_idx == sched_meta.begin_req_idx
                ? sched_meta.is_first_req_splitted
                : (batch_idx == sched_meta.end_req_idx
                    ? sched_meta.is_last_req_splitted
                    : false);
            const int n_split_idx = batch_idx == sched_meta.begin_req_idx
                ? __ldg(params.num_splits_ptr + batch_idx)
                    + sched_meta.begin_split_idx
                : __ldg(params.num_splits_ptr + batch_idx);

            f(MainLoopArgs{
                batch_idx,
                start_block_idx,
                end_block_idx,
                !is_split,
                n_split_idx,
                bar_phase_batch_rel,
                topk_length,
                extra_topk_length,
                orig_topk_padded / B_TOPK,
                batch_idx == sched_meta.end_req_idx,
            });
            NamedBarrier(NUM_THREADS, NamedBarriers::everyone_sync)
                .arrive_and_wait_unaligned();
        }
    };

    struct RingState {
        int buf_idx = 0;
        bool bar_phase = false;
        int index_buf_idx = 0;
        bool index_bar_phase = false;

        CUTE_DEVICE void update() {
            bar_phase ^= buf_idx == NUM_BUFS - 1;
            buf_idx = (buf_idx + 1) % NUM_BUFS;
            index_bar_phase ^= index_buf_idx == NUM_INDEX_BUFS - 1;
            index_buf_idx = (index_buf_idx + 1) % NUM_INDEX_BUFS;
        }
    };
    RingState rs;

    if (warpgroup_idx == 0) {
        // WG0 owns online softmax, S quantization, O rescaling, and the
        // output epilogue.
        cutlass::arch::warpgroup_reg_alloc<256>();

        constexpr int B_EPI = 64;
        Tensor sO = make_tensor(
            make_smem_ptr(plan.qkvo.o.o_buf.data()), SmemLayoutOBuf{}
        );
        Tensor sS = make_tensor(make_smem_ptr(plan.s.data()), SmemLayoutS{});
        bf16 *sO_bases[B_EPI / 8];
        CUTE_UNROLL
        for (int i = 0; i < B_EPI / 8; ++i) {
            sO_bases[i] = &sO(idx_in_warpgroup % B_H, i * 8);
        }

        const float attn_sink = params.attn_sink == nullptr
            ? -CUDART_INF_F
            : __ldg(params.attn_sink + idx_in_warpgroup % B_H) * CUDART_L2E_F;

        run_main_loop([&](const MainLoopArgs &args) {
            cute::tma_store_wait<0>();
            plan.bar_last_store_done.arrive();

            float mi = MAX_INIT_VAL;
            float li = 0.0f;
            float real_mi = -CUDART_INF_F;
            const int h = idx_in_warpgroup % B_H;
            const int token_base = (idx_in_warpgroup / B_H) * (B_TOPK / 2);
            const uint8_t *q_scale_base =
                reinterpret_cast<const uint8_t *>(params.q)
                + static_cast<int64_t>(args.batch_idx)
                    * params.stride_q_b
                + static_cast<int64_t>(s_q_idx) * params.stride_q_s_q
                + B_H * D_Q;
            const float q_head_scale = ue8m0_bits_to_float(
                __ldg(q_scale_base + h)
            );
#if defined(FP8_FWD_DISABLE_NON_SFU_NON_GEMM)
            const float qk_base_scale = q_head_scale;
#else
            const float qk_base_scale =
                q_head_scale * params.sm_scale_div_log2;
#endif
            const float2 qk_base_scale2 = make_float2(
                qk_base_scale,
                qk_base_scale
            );

            CUTE_NO_UNROLL
            for (int block_idx = args.start_block_idx;
                 block_idx < args.end_block_idx;
                 ++block_idx) {
                NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
                plan.bar_valid_coord_scale_ready[rs.index_buf_idx].wait(
                    rs.index_bar_phase
                );
                plan.bar_qk_done[rs.buf_idx].wait(rs.bar_phase);
                ku::tcgen05_after_thread_sync();

                // The FP8 QK MMA produces the complete 64 x 128 P tile.  In
                // contrast to the BF16 dual-GEMM decode path, no peer-P merge
                // is necessary before masking and softmax.
                float p[B_TOPK / 2];
                ku::tmem_ld_32dp32bNx<B_TOPK / 2>(tmem_cols::P, p);
                cutlass::arch::fence_view_async_tmem_load();
                ku::tcgen05_before_thread_sync();
                plan.bar_kv_scale_ready[rs.buf_idx].wait(rs.bar_phase);

                // Each WG0 lane initially owns four token scales.  Shuffles
                // expose the complete half-tile to the thread that owns the
                // corresponding P values, matching the prefill pipeline.
                float lane_kv_scale[B_TOPK / 64];
                CUTE_UNROLL
                for (int i = 0; i < B_TOPK / 64; ++i) {
                    lane_kv_scale[i] = plan.kv_token_scale[rs.buf_idx][
                        token_base + i * 32 + lane_idx
                    ];
                }

                const uint64_t valid_mask = *(
                    reinterpret_cast<const uint64_t *>(
                        plan.is_token_valid[rs.index_buf_idx]
                    ) + idx_in_warpgroup / B_H
                );
                CUTE_UNROLL
                for (int i = 0; i < B_TOPK / 2; ++i) {
                    if (((valid_mask >> i) & 1ull) == 0) {
                        p[i] = -CUDART_INF_F;
                    }
                }

                // Keep both the token scale and complete QK scale in
                // registers for the whole softmax + quant path.  This is
                // the prefill arrangement: the S loop consumes these values
                // directly and does not perform another shuffle.
                float kv_scale[B_TOPK / 2];
                float cur_pi_max = -CUDART_INF_F;
#if !defined(FP8_FWD_DISABLE_NON_SFU_NON_GEMM)
                float2 qk_scale[B_TOPK / 4];
                CUTE_UNROLL
                for (int i = 0; i < B_TOPK / 4; ++i) {
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

                CUTE_UNROLL
                for (int i = 0; i < B_TOPK / 2; i += 2) {
                    const float2 scaled_p = ku::float2_mul(
                        make_float2(p[i + 0], p[i + 1]),
                        qk_scale[i / 2]
                    );
                    p[i + 0] = scaled_p.x;
                    p[i + 1] = scaled_p.y;
                    cur_pi_max = max(cur_pi_max, scaled_p.x);
                    cur_pi_max = max(cur_pi_max, scaled_p.y);
                }
#else
                // Retain the TMEM load, validity path, shared-memory
                // handoffs, and SFU work while bypassing CUDA-core math.
                cur_pi_max = p[0];
#endif

                plan.rowwise_max_buf[idx_in_warpgroup] = cur_pi_max;
                NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
                plan.bar_valid_coord_scale_free[rs.index_buf_idx].arrive();
#if defined(FP8_FWD_DISABLE_NON_SFU_NON_GEMM)
                cur_pi_max = plan.rowwise_max_buf[idx_in_warpgroup ^ B_H];
                real_mi = cur_pi_max;
                const bool should_scale_o = true;
                const float scale_for_old = fp8_decode_exp2(mi);
                const float new_max = scale_for_old;
#else
                cur_pi_max = max(
                    cur_pi_max, plan.rowwise_max_buf[idx_in_warpgroup ^ B_H]
                );
                real_mi = max(real_mi, cur_pi_max);

                const bool should_scale_o = __any_sync(
                    0xffffffff, cur_pi_max - mi > 6.0f
                );
                const float new_max = should_scale_o
                    ? max(cur_pi_max, mi)
                    : mi;
                const float scale_for_old = should_scale_o
                    ? fp8_decode_exp2(mi - new_max)
                    : 1.0f;
#endif
                mi = new_max;

                float cur_sum = 0.0f;
#if defined(FP8_FWD_DISABLE_NON_SFU_NON_GEMM)
                uint32_t s[B_TOPK / 8];
                CUTE_UNROLL
                for (int i = 0; i < B_TOPK / 2; i += 4) {
                    const uint32_t packed_s = fp8_decode_exp2_quad_packed(
                        p[i + 0], p[i + 1], p[i + 2], p[i + 3]
                    );
                    s[i / 4] = packed_s;
                    cur_sum = __uint_as_float(packed_s);
                }
#else
                CUTE_UNROLL
                for (int i = 0; i < B_TOPK / 2; i += 2) {
                    const float2 softmax_s = make_float2(
                        fp8_decode_exp2(p[i + 0] - new_max),
                        fp8_decode_exp2(p[i + 1] - new_max)
                    );
                    cur_sum += softmax_s.x + softmax_s.y;
                    const float2 s_pair = ku::float2_mul(
                        softmax_s,
                        make_float2(kv_scale[i + 0], kv_scale[i + 1])
                    );
                    p[i + 0] = s_pair.x;
                    p[i + 1] = s_pair.y;
                }

                // Match prefill's FP32x2 quantization path: convert the
                // scaled softmax values to packed E4M3 bits before writing S.
                constexpr float inv_current_s_scale = FP8_MAX;
                const float2 inv_current_s_scale2 = make_float2(
                    inv_current_s_scale,
                    inv_current_s_scale
                );
                uint32_t s[B_TOPK / 8];
                CUTE_UNROLL
                for (int i = 0; i < B_TOPK / 2; i += 4) {
                    const uint16_t s01 = ku::float2_to_e4m3x2_bits(
                        ku::float2_mul(
                            make_float2(p[i + 0], p[i + 1]),
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
#endif
                CUTE_UNROLL
                for (int i = 0; i < B_TOPK / 2; i += 16) {
                    *reinterpret_cast<uint4 *>(&sS(h, token_base + i)) =
                        make_uint4(
                            s[i / 4],
                            s[i / 4 + 1],
                            s[i / 4 + 2],
                            s[i / 4 + 3]
                        );
                }
#if defined(FP8_FWD_DISABLE_NON_SFU_NON_GEMM)
                li = cur_sum;
#else
                li = fma(li, scale_for_old, cur_sum);
#endif

                fence_view_async_shared();
                plan.bar_so_ready[rs.buf_idx].arrive();

                // Rescale the accumulated O tile before WG1 issues this
                // stage's SV MMA.  The branch is warp-uniform, matching the
                // former WG3 path, while scale_for_old remains rowwise.
                if (block_idx != args.start_block_idx && should_scale_o) {
                    const int prev_buf =
                        (rs.buf_idx + NUM_BUFS - 1) % NUM_BUFS;
                    const bool prev_phase = rs.buf_idx == 0
                        ? rs.bar_phase ^ true
                        : rs.bar_phase;
                    plan.bar_sv_done[prev_buf].wait(prev_phase);

                    ku::tcgen05_after_thread_sync();
                    CUTE_UNROLL
                    for (int dv_block = 0;
                         dv_block < D_V / SV_M;
                         ++dv_block) {
                        rescale_o_tmem_stripe(
                            scale_for_old,
                            tmem_cols::O + dv_block * (SV_M / 2)
                        );
                    }
                    ku::tcgen05_before_thread_sync();
                }
                plan.bar_o_ready[rs.buf_idx].arrive();

                if (block_idx != args.end_block_idx - 1) {
                    rs.update();
                }
            }

            if (real_mi == -CUDART_INF_F) {
                li = 0.0f;
                mi = -CUDART_INF_F;
            }

            plan.rowwise_max_buf[idx_in_warpgroup] = li;
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
#if defined(FP8_FWD_DISABLE_NON_SFU_NON_GEMM)
            li = plan.rowwise_max_buf[idx_in_warpgroup ^ B_H];
#else
            li += plan.rowwise_max_buf[idx_in_warpgroup ^ B_H];
#endif

            if (idx_in_warpgroup < B_H) {
                if (args.is_no_split) {
#if defined(FP8_FWD_DISABLE_NON_SFU_NON_GEMM)
                    float lse = fp8_decode_log(li);
#else
                    float lse = fmaf(mi, CUDART_LN2_F, fp8_decode_log(li));
#endif
                    lse = lse == -CUDART_INF_F ? CUDART_INF_F : lse;
                    params.lse[
                        args.batch_idx * params.stride_lse_b
                        + s_q_idx * params.stride_lse_s_q
                        + idx_in_warpgroup
                    ] = lse;
                } else {
                    params.lse_accum[
                        args.n_split_idx * params.stride_lse_accum_split
                        + s_q_idx * params.stride_lse_accum_s_q
                        + idx_in_warpgroup
#if defined(FP8_FWD_DISABLE_NON_SFU_NON_GEMM)
                    ] = fp8_decode_log2(li);
#else
                    ] = fp8_decode_log2(li) + mi;
#endif
                }
            }

            plan.bar_sv_done[rs.buf_idx].wait(rs.bar_phase);
            rs.update();
            ku::tcgen05_after_thread_sync();

            if (args.is_last_batch) {
                cudaTriggerProgrammaticLaunchCompletion();
            }

#if defined(FP8_FWD_DISABLE_NON_SFU_NON_GEMM)
            const float o_scale = fp8_decode_rcp(
                args.is_no_split
                    ? fp8_decode_bit_passthrough(fp8_decode_exp2(attn_sink))
                    : fp8_decode_bit_passthrough(li)
            );
#else
            const float o_scale = li == 0.0f
                ? 0.0f
                : S_FP8_SCALE * fp8_decode_rcp(
                    args.is_no_split
                        ? li + fp8_decode_exp2(attn_sink - mi)
                        : li
                );
#endif
            if (args.is_no_split) {
                Tensor tma_gO = flat_divide(
                    tma_params.tma_O.get_tma_tensor(tma_params.shape_O)(
                        _, _, s_q_idx, args.batch_idx
                    ),
                    Shape<Int<B_H>, Int<B_EPI>>{}
                )(_, _, _0{}, _);
                Tensor tma_sO = flat_divide(
                    sO, Shape<Int<B_H>, Int<B_EPI>>{}
                )(_, _, _0{}, _);
                auto thr_tma = tma_params.tma_O.get_slice(_0{});

                float2 o[B_EPI / 2];
                __nv_bfloat162 o_bf16[B_EPI / 2];
                CUTE_UNROLL
                for (int c = 0; c < 2; ++c) {
                    CUTE_UNROLL
                    for (int k = 0; k < (D_V / 4) / B_EPI; ++k) {
                        ku::tmem_ld_32dp32bNx<B_EPI>(
                            tmem_cols::O + c * 128 + k * B_EPI, o
                        );
                        cutlass::arch::fence_view_async_tmem_load();
                        const int d_group = c * 4 + k * 2
                            + idx_in_warpgroup / B_H;
#if defined(FP8_FWD_DISABLE_NON_SFU_NON_GEMM)
                        const float output_dequant_scale = o_scale;
#else
                        const float output_dequant_scale = o_scale
                            * ue8m0_bits_to_float(
                                __ldg(params.kv_scale_w + d_group)
                            );
#endif
                        const float2 output_dequant_scale2 = make_float2(
                            output_dequant_scale, output_dequant_scale
                        );
                        CUTE_UNROLL
                        for (int j = 0; j < B_EPI / 2; ++j) {
#if !defined(FP8_FWD_DISABLE_NON_SFU_NON_GEMM)
                            o[j] = ku::float2_mul(
                                o[j], output_dequant_scale2
                            );
                            o_bf16[j] = __float22bfloat162_rn(o[j]);
#endif
                        }
                        CUTE_UNROLL
                        for (int j = 0; j < B_EPI / 8; ++j) {
#if defined(FP8_FWD_DISABLE_NON_SFU_NON_GEMM)
                            const float4 o_bits = make_float4(
                                output_dequant_scale,
                                o[j * 4 + 0].y,
                                o[j * 4 + 1].x,
                                o[j * 4 + 1].y
                            );
                            *reinterpret_cast<__int128_t *>(
                                sO_bases[j] + d_group * B_EPI * B_H
                            ) = reinterpret_cast<const __int128_t &>(o_bits);
#else
                            *reinterpret_cast<__int128_t *>(
                                sO_bases[j] + d_group * B_EPI * B_H
                            ) = *reinterpret_cast<__int128_t *>(&o_bf16[j * 4]);
#endif
                        }

                        fence_view_async_shared();
                        NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
                        if (warp_idx == 0 && elect_one_sync()) {
                            const int epi_chunk_idx = c * 4 + k * 2;
                            cute::copy(
                                tma_params.tma_O,
                                thr_tma.partition_S(tma_sO(_, _, epi_chunk_idx)),
                                thr_tma.partition_D(tma_gO(_, _, epi_chunk_idx))
                            );
                        }
                        if (warp_idx == 1 && elect_one_sync()) {
                            const int epi_chunk_idx = c * 4 + k * 2 + 1;
                            cute::copy(
                                tma_params.tma_O,
                                thr_tma.partition_S(tma_sO(_, _, epi_chunk_idx)),
                                thr_tma.partition_D(tma_gO(_, _, epi_chunk_idx))
                            );
                        }
                    }
                }
                cute::tma_store_arrive();
            } else {
                Tensor sO_accum = make_tensor(
                    make_smem_ptr(plan.qkvo.o.o_accum_buf.data()),
                    SmemLayoutOAccumBuf{}
                );
                float2 o[B_EPI / 2];
                CUTE_UNROLL
                for (int c = 0; c < 2; ++c) {
                    CUTE_UNROLL
                    for (int k = 0; k < (D_V / 4) / B_EPI; ++k) {
                        ku::tmem_ld_32dp32bNx<B_EPI>(
                            tmem_cols::O + c * 128 + k * B_EPI, o
                        );
                        cutlass::arch::fence_view_async_tmem_load();
                        const int d_group = c * 4 + k * 2
                            + idx_in_warpgroup / B_H;
#if defined(FP8_FWD_DISABLE_NON_SFU_NON_GEMM)
                        const float output_dequant_scale = o_scale;
#else
                        const float output_dequant_scale = o_scale
                            * ue8m0_bits_to_float(
                                __ldg(params.kv_scale_w + d_group)
                            );
#endif
                        const float2 output_dequant_scale2 = make_float2(
                            output_dequant_scale, output_dequant_scale
                        );
                        CUTE_UNROLL
                        for (int j = 0; j < B_EPI / 2; ++j) {
#if !defined(FP8_FWD_DISABLE_NON_SFU_NON_GEMM)
                            o[j] = ku::float2_mul(
                                o[j], output_dequant_scale2
                            );
#endif
                        }
                        CUTE_UNROLL
                        for (int j = 0; j < B_EPI / 4; ++j) {
#if defined(FP8_FWD_DISABLE_NON_SFU_NON_GEMM)
                            const float4 o_bits = make_float4(
                                output_dequant_scale,
                                o[j * 2 + 0].y,
                                o[j * 2 + 1].x,
                                o[j * 2 + 1].y
                            );
                            *reinterpret_cast<__int128_t *>(
                                &sO_accum(h, d_group * B_EPI + j * 4)
                            ) = reinterpret_cast<const __int128_t &>(o_bits);
#else
                            *reinterpret_cast<__int128_t *>(
                                &sO_accum(h, d_group * B_EPI + j * 4)
                            ) = *reinterpret_cast<__int128_t *>(&o[j * 2]);
#endif
                        }
                    }
                }
                fence_view_async_shared();
                NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
                if (elect_one_sync()) {
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < B_H / 4; ++local_row) {
                        const int smem_row = local_row * 4 + warp_idx;
                        SM90_BULK_COPY_S2G::copy(
                            &sO_accum(smem_row, _0{}),
                            params.o_accum
                                + args.n_split_idx * params.stride_o_accum_split
                                + s_q_idx * params.stride_o_accum_s_q
                                + smem_row * params.stride_o_accum_h_q,
                            D_V * sizeof(float)
                        );
                    }
                    cute::tma_store_arrive();
                }
            }
        });

        if (warp_idx == 0) {
            cute::TMEM::Allocator1Sm().free(0, 512);
        }
    } else if (warpgroup_idx == 1) {
        // Warp 4 remains the Q producer/MMA issuer and warp 7 remains the
        // index-coordinate producer.  Warps 5 and 6 load the per-token KV
        // scales in parallel with the gather path.
        cutlass::arch::warpgroup_reg_dealloc<80>();
        const int local_warp_idx = cutlass::canonical_warp_idx_sync();

        if (local_warp_idx == 4 && elect_one_sync()) {
            run_main_loop([&](const MainLoopArgs &args) {
                KU_TRAP_ONLY_DEVICE_ASSERT(
                    args.start_block_idx < args.end_block_idx
                );

                // Q shares qkvo storage with the prior batch's O buffer.
                // WG0 publishes this only after its asynchronous stores finish.
                plan.bar_last_store_done.wait(args.bar_phase_batch_rel);
                plan.bar_q_tma.arrive_and_expect_tx(B_H * D_Q * sizeof(e4m3));
                Tensor gQ = tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(
                    _, _, s_q_idx, args.batch_idx
                );
                Tensor sQ = make_tensor(
                    make_smem_ptr(plan.qkvo.q.data()), SmemLayoutQ_SW128{}
                );
                ku::launch_tma_copy(
                    tma_params.tma_Q,
                    gQ,
                    sQ,
                    plan.bar_q_tma,
                    TMA::CacheHintSm90::EVICT_FIRST
                );
                plan.bar_q_tma.wait(args.bar_phase_batch_rel);
                ku::tcgen05_after_thread_sync();

                // Copy the 64 x 512 FP8 Q tile into its 128 TMEM columns.
                UMMA::SmemDescriptor sQ_desc = UMMA::make_umma_desc<
                    UMMA::Major::K
                >(
                    make_tensor(
                        make_smem_ptr(plan.qkvo.q.data()),
                        tile_to_shape(
                            UMMA::Layout_K_SW128_Atom<e4m3>{},
                            Shape<Int<B_H>, Int<128>>{}
                        )
                    )
                );
                CUTE_UNROLL
                for (int tile_idx = 0; tile_idx < D_Q / 128; ++tile_idx) {
                    CUTE_UNROLL
                    for (int subtile_idx = 0; subtile_idx < 8; ++subtile_idx) {
                        SM100_UTCCP_2x64dp128bitlw0213_1cta::copy(
                            sQ_desc + (
                                tile_idx * B_H * 128 + subtile_idx * 16
                            ) / 16,
                            tmem_cols::Q + tile_idx * 32 + subtile_idx * 4
                        );
                    }
                }
                ku::umma_arrive_noelect(plan.bar_q_utccp);

                TiledMMA tiled_mma_P = TiledMMA_P{};
                TiledMMA tiled_mma_O = TiledMMA_O{};
                Tensor tQ = tiled_mma_P.get_slice(_0{}).make_fragment_A(
                    partition_shape_A(
                        tiled_mma_P, Shape<Int<B_H>, Int<D_K>>{}
                    )
                );
                Tensor tP = partition_fragment_C(
                    tiled_mma_P, Shape<Int<B_H>, Int<B_TOPK>>{}
                );
                Tensor tO = partition_fragment_C(
                    tiled_mma_O, Shape<Int<B_H>, Int<SV_M>>{}
                );
                tQ.data().get() = tmem_cols::Q;
                tP.data().get() = tmem_cols::P;
                tO.data().get() = tmem_cols::O;

                plan.bar_q_utccp.wait(args.bar_phase_batch_rel);
                ku::tcgen05_after_thread_sync();

                CUTE_NO_UNROLL
                for (int block_idx = args.start_block_idx;
                     block_idx < args.end_block_idx;
                     ++block_idx) {
                    plan.bar_kv_ready[rs.buf_idx].wait(rs.bar_phase);
                    ku::tcgen05_after_thread_sync();

                    Tensor sK = make_tensor(
                        make_smem_ptr(plan.qkvo.kv[rs.buf_idx].data()),
                        SmemLayoutKTiles_SW128<D_K / 64>{}
                    );
#if !defined(FP8_FWD_DISABLE_GEMM)
                    ku::utcmma_ts(tiled_mma_P, tQ, sK, tP, true);
#endif
                    ku::umma_arrive_noelect(plan.bar_qk_done[rs.buf_idx]);

                    plan.bar_so_ready[rs.buf_idx].wait(rs.bar_phase);
                    plan.bar_o_ready[rs.buf_idx].wait(rs.bar_phase);
                    ku::tcgen05_after_thread_sync();

                    Tensor sS = make_tensor(
                        make_smem_ptr(plan.s.data()), SmemLayoutS{}
                    );
                    Tensor sV = make_tensor(
                        make_smem_ptr(plan.qkvo.kv[rs.buf_idx].data()),
                        SmemLayoutKTilesTransposed_SW128<D_V / 64>{}
                    );
                    Tensor sV_divided = flat_divide(
                        sV, Tile<Int<SV_M>, Int<B_TOPK>>{}
                    )(_, _, _, _0{});
                    CUTE_UNROLL
                    for (int dv_block = 0; dv_block < D_V / SV_M; ++dv_block) {
                        tO.data().get() = tmem_cols::O
                            + dv_block * (SV_M / 2);
#if !defined(FP8_FWD_DISABLE_GEMM)
                        ku::utcmma_ss(
                            tiled_mma_O,
                            sS,
                            sV_divided(_, _, dv_block),
                            tO,
                            block_idx == args.start_block_idx
                        );
#endif
                    }
                    ku::umma_arrive_noelect(plan.bar_sv_done[rs.buf_idx]);
                    rs.update();
                }
            });
        } else if (local_warp_idx == 7) {
            // Each lane publishes four sparse indices, their TMA coordinates,
            // and four validity bits.  No cache-scale path is needed for raw
            // E4M3 operands.
            constexpr int TOKENS_PER_LANE = B_TOPK / 32;
            static_assert(TOKENS_PER_LANE == 4);
            constexpr int TMA_COORDS_PER_TOKEN = 1;
            static_assert(TMA_COORDS_PER_TOKEN == 1);
            const int tma_coords_per_block = params.stride_kv_block / TMA_K_STRIDE;
            const int extra_tma_coords_per_block =
                params.stride_extra_kv_block / TMA_K_STRIDE;

            run_main_loop([&](const MainLoopArgs &args) {
                int *indices = params.indices
                    + args.batch_idx * params.stride_indices_b
                    + s_q_idx * params.stride_indices_s_q;
                int *extra_indices = params.extra_topk > 0
                    ? params.extra_indices
                        + args.batch_idx * params.stride_extra_indices_b
                        + s_q_idx * params.stride_extra_indices_s_q
                    : nullptr;

                auto process_one_block = [&](int block_idx, bool is_extra) {
                    const int local_block_idx = is_extra
                        ? block_idx - args.num_orig_kv_blocks
                        : block_idx;
                    const int current_block_size = is_extra
                        ? params.extra_page_block_size
                        : params.page_block_size;
                    const int current_num_blocks = is_extra
                        ? params.extra_num_blocks
                        : params.num_blocks;
                    const int current_tma_coords_per_block = is_extra
                        ? extra_tma_coords_per_block
                        : tma_coords_per_block;
                    const int current_topk_length = is_extra
                        ? args.extra_topk_length
                        : args.topk_length;
                    const int abs_pos = local_block_idx * B_TOPK
                        + lane_idx * TOKENS_PER_LANE;
                    const int4 indices_vec = __ldg(
                        reinterpret_cast<const int4 *>(
                            (is_extra ? extra_indices : indices) + abs_pos
                        )
                    );

                    plan.bar_valid_coord_scale_free[rs.index_buf_idx].wait(
                        rs.index_bar_phase ^ true
                    );

                    const int *my_indices = reinterpret_cast<const int *>(&indices_vec);
                    int4 coords_vec;
                    int *coords = reinterpret_cast<int *>(&coords_vec);
                    uint32_t valid_mask = 0;
                    CUTE_UNROLL
                    for (int i = 0; i < TOKENS_PER_LANE; ++i) {
                        const int token_idx = my_indices[i];
                        const bool is_valid = token_idx >= 0
                            && static_cast<int64_t>(token_idx)
                                < static_cast<int64_t>(current_num_blocks)
                                    * current_block_size
                            && abs_pos + i < current_topk_length;
                        valid_mask |= static_cast<uint32_t>(is_valid) << i;
                        if (is_valid) {
                            const int page_idx = token_idx / current_block_size;
                            const int idx_in_page = token_idx % current_block_size;
                            coords[i] = page_idx * current_tma_coords_per_block
                                + idx_in_page * TMA_COORDS_PER_TOKEN;
                        } else {
                            // TMA gather4 treats -1 as an invalid row and
                            // therefore avoids reading a padded top-k entry.
                            coords[i] = -1;
                        }
                    }

                    valid_mask <<= (lane_idx & 1) * TOKENS_PER_LANE;
                    valid_mask |= __shfl_xor_sync(
                        0xffffffff, valid_mask, 1
                    );
                    *reinterpret_cast<int4 *>(
                        plan.tma_coord[rs.index_buf_idx]
                            + lane_idx * TOKENS_PER_LANE
                    ) = coords_vec;
                    if ((lane_idx & 1) == 0) {
                        plan.is_token_valid[rs.index_buf_idx][lane_idx / 2]
                            = static_cast<char>(valid_mask);
                    }

                    plan.bar_valid_coord_scale_ready[rs.index_buf_idx].arrive();
                    rs.update();
                };

                CUTE_NO_UNROLL
                for (int block_idx = args.start_block_idx;
                     block_idx < min(args.num_orig_kv_blocks, args.end_block_idx);
                     ++block_idx) {
                    process_one_block(block_idx, false);
                }
                CUTE_NO_UNROLL
                for (int block_idx = max(
                        args.start_block_idx, args.num_orig_kv_blocks
                    );
                     block_idx < args.end_block_idx;
                     ++block_idx) {
                    process_one_block(block_idx, true);
                }
            });
        } else if (local_warp_idx == 5 || local_warp_idx == 6) {
            const int scale_warp_idx = local_warp_idx - 5;
            constexpr int TOKENS_PER_WARP = B_TOPK / 2;
            constexpr int TOKENS_PER_LANE = TOKENS_PER_WARP / 32;
            static_assert(TOKENS_PER_LANE == 2);

            run_main_loop([&](const MainLoopArgs &args) {
                const int token_base = scale_warp_idx * TOKENS_PER_WARP;

                auto process_one_block = [&](int, bool is_extra) {
                    const uint8_t *current_kv = reinterpret_cast<const uint8_t *>(
                        is_extra ? params.extra_kv : params.kv
                    );

                    // Wait until warp 7 has published this index packet and
                    // until the previous consumer has released this stage.
                    plan.bar_valid_coord_scale_ready[rs.index_buf_idx].wait(
                        rs.index_bar_phase
                    );
                    plan.bar_sv_done[rs.buf_idx].wait(rs.bar_phase ^ true);

                    // Warp 7 writes -1 for invalid tokens and otherwise
                    // publishes the flattened TMA row coordinate.  Reuse it
                    // directly: coord * TMA_K_STRIDE points at the physical
                    // 528-byte KV row containing the scale.
                    CUTE_UNROLL
                    for (int i = 0; i < TOKENS_PER_LANE; ++i) {
                        const int row = token_base + i * 32 + lane_idx;
                        const int coord =
                            plan.tma_coord[rs.index_buf_idx][row];
                        float scale = 1.0f;
                        if (coord >= 0) {
                            const uint8_t *scale_ptr = current_kv
                                + static_cast<int64_t>(coord) * TMA_K_STRIDE
                                + D_K;
                            scale = ue8m0_bits_to_float(__ldg(scale_ptr));
                        }
                        plan.kv_token_scale[rs.buf_idx][row] = scale;
                    }
                    fence_view_async_shared();
                    if (elect_one_sync()) {
                        plan.bar_kv_scale_ready[rs.buf_idx].arrive();
                    }
                    rs.update();
                };

                CUTE_NO_UNROLL
                for (int block_idx = args.start_block_idx;
                     block_idx < min(args.num_orig_kv_blocks, args.end_block_idx);
                     ++block_idx) {
                    process_one_block(block_idx, false);
                }
                CUTE_NO_UNROLL
                for (int block_idx = max(
                        args.start_block_idx, args.num_orig_kv_blocks
                    );
                     block_idx < args.end_block_idx;
                     ++block_idx) {
                    process_one_block(block_idx, true);
                }
            });
        } else {
            run_main_loop([&](const MainLoopArgs &) {});
        }
    } else if (warpgroup_idx == 2) {
        // WG2 loads KV with all 128 threads.  Four neighboring threads own
        // one group of four token rows, and each thread issues one 128-byte
        // K-dimension gather4 chunk for that group.  This preserves the
        // 128B-swizzled destination layout while replacing the old 32-thread
        // loader with a full-warpgroup producer.
        cutlass::arch::warpgroup_reg_dealloc<80>();
        constexpr int ROWS_PER_LOADER = 4;
        constexpr int NUM_ROW_GROUPS = B_TOPK / ROWS_PER_LOADER;
        constexpr int BYTES_PER_GATHER =
            ROWS_PER_LOADER * TMA_K_CHUNK_BYTES;
        static_assert(NUM_ROW_GROUPS == 32);

        run_main_loop([&](const MainLoopArgs &args) {
            plan.bar_q_utccp.wait(args.bar_phase_batch_rel);
            plan.bar_last_store_done.wait(args.bar_phase_batch_rel);

            CUTE_NO_UNROLL
            for (int block_idx = args.start_block_idx;
                 block_idx < args.end_block_idx;
                 ++block_idx) {
                plan.bar_valid_coord_scale_ready[rs.index_buf_idx].wait(
                    rs.index_bar_phase
                );
                // The KV stage aliases Q/O storage, so it cannot be reused
                // until the prior SV MMA for this stage has committed.
                plan.bar_sv_done[rs.buf_idx].wait(rs.bar_phase ^ true);

                // Thread t maps to row group t/4 and K chunk t%4.  Every
                // thread contributes one transaction-barrier arrival and one
                // gather4 operation, so the total expected bytes remain
                // B_TOPK * D_K while all 128 WG2 threads are useful.
                const int row_group = idx_in_warpgroup / 4;
                const int row = row_group * ROWS_PER_LOADER;
                const int col = idx_in_warpgroup % 4;
                const int4 row_coords = *reinterpret_cast<const int4 *>(
                    plan.tma_coord[rs.index_buf_idx] + row
                );
                plan.bar_kv_ready[rs.buf_idx].arrive_and_expect_tx(
                    BYTES_PER_GATHER * sizeof(e4m3)
                );
                ku::tma_gather4(
                    block_idx >= args.num_orig_kv_blocks
                        ? &tma_params.tensor_map_extra_kv
                        : &tma_params.tensor_map_kv,
                    plan.bar_kv_ready[rs.buf_idx],
                    plan.qkvo.kv[rs.buf_idx].data()
                        + col * B_TOPK * TMA_K_CHUNK_BYTES
                        + row * TMA_K_CHUNK_BYTES,
                    col * TMA_K_CHUNK_ELEMS,
                    row_coords,
                    (int64_t)TMA::CacheHintSm90::EVICT_LAST
                );

                plan.bar_valid_coord_scale_free[rs.index_buf_idx].arrive();
                rs.update();
            }
        });
    }
#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100 through sm119");
    }
#endif
}

void run_flash_splitkv_mla_fp8_sparse_kernel(
    const SparseAttnFp8DecodeParams &params
) {
    KU_ASSERT(params.model_type == ModelType::MODEL1);
    KU_ASSERT(params.topk % B_TOPK == 0,
        "topk (%d) mod B_TOPK (%d) must be 0", params.topk, B_TOPK);
    KU_ASSERT(params.extra_topk % B_TOPK == 0,
        "extra_topk (%d) mod B_TOPK (%d) must be 0",
        params.extra_topk, B_TOPK);
    KU_ASSERT(params.h_q == B_H);
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.d_qk == D_Q);
    KU_ASSERT(params.d_v == D_V);
    KU_ASSERT(params.kv_scale_w != nullptr);
    KU_ASSERT(reinterpret_cast<uintptr_t>(params.kv_scale_w) % 8 == 0,
        "kv_scale_w must be 8-byte aligned");
    KU_ASSERT(params.stride_q_h_q == D_Q,
        "FP8 Q head stride must equal the FP8 data width");
    KU_ASSERT(params.stride_kv_row == KV_BYTES_PER_TOKEN,
        "FP8 KV rows must be contiguous (%d bytes), got %d",
        KV_BYTES_PER_TOKEN, params.stride_kv_row);
    KU_ASSERT(params.stride_kv_block % TMA_K_STRIDE == 0,
        "stride_kv_block (%d) must be a multiple of %d",
        params.stride_kv_block, TMA_K_STRIDE);
    KU_ASSERT(reinterpret_cast<uintptr_t>(params.q) % 16 == 0,
        "q must be 16-byte aligned for SM100 TMA");
    KU_ASSERT(reinterpret_cast<uintptr_t>(params.kv) % 16 == 0,
        "kv must be 16-byte aligned for SM100 TMA");
    if (params.extra_topk > 0) {
        KU_ASSERT(params.extra_kv != nullptr);
        KU_ASSERT(params.extra_indices != nullptr);
        KU_ASSERT(params.stride_extra_kv_row == KV_BYTES_PER_TOKEN,
            "extra KV rows must be contiguous (%d bytes), got %d",
            KV_BYTES_PER_TOKEN, params.stride_extra_kv_row);
        KU_ASSERT(params.stride_extra_kv_block % TMA_K_STRIDE == 0,
            "stride_extra_kv_block (%d) must be a multiple of %d",
            params.stride_extra_kv_block, TMA_K_STRIDE);
        KU_ASSERT(reinterpret_cast<uintptr_t>(params.extra_kv) % 16 == 0,
            "extra_kv must be 16-byte aligned for SM100 TMA");
    }

    auto shape_Q = make_shape(B_H, D_Q, params.s_q, params.b);
    auto tma_Q = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr(reinterpret_cast<e4m3 *>(params.q)),
            make_layout(
                shape_Q,
                make_stride(
                    params.stride_q_h_q,
                    _1{},
                    params.stride_q_s_q,
                    params.stride_q_b
                )
            )
        ),
        SmemLayoutQ_SW128{}
    );

    auto shape_O = make_shape(B_H, D_V, params.s_q, params.b);
    auto tma_O = cute::make_tma_copy(
        SM90_TMA_STORE{},
        make_tensor(
            make_gmem_ptr(params.out),
            make_layout(
                shape_O,
                make_stride(
                    params.stride_o_h_q,
                    _1{},
                    params.stride_o_s_q,
                    params.stride_o_b
                )
            )
        ),
        SmemLayoutOBuf_TMA{}
    );

    auto make_kv_tensormap = [&](void *kv, int num_blocks, int block_stride) {
        return ku::make_tensor_map(
            {D_K / 8,
             static_cast<uint64_t>(num_blocks)
                 * (block_stride / TMA_K_STRIDE)},
            {TMA_K_STRIDE},
            {TMA_K_CHUNK_ELEMS, 1},
            kv,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT64,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B
        );
    };

    CUtensorMap tensor_map_kv = make_kv_tensormap(
        params.kv, params.num_blocks, params.stride_kv_block
    );
    CUtensorMap tensor_map_extra_kv{};
    if (params.extra_topk > 0) {
        tensor_map_extra_kv = make_kv_tensormap(
            params.extra_kv,
            params.extra_num_blocks,
            params.stride_extra_kv_block
        );
    }

    TmaParams<
        decltype(shape_Q), decltype(tma_Q),
        decltype(shape_O), decltype(tma_O)
    > tma_params{
        shape_Q,
        tma_Q,
        shape_O,
        tma_O,
        tensor_map_kv,
        tensor_map_extra_kv,
    };

    auto kernel = &flash_fwd_splitkv_mla_fp8_sparse_kernel<decltype(tma_params)>;
    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size
    ));
    kernel<<<
        dim3(params.s_q, params.num_sm_parts, 1),
        dim3(NUM_THREADS, 1, 1),
        smem_size,
        params.stream
    >>>(params, tma_params);
    KU_CHECK_KERNEL_LAUNCH();
}

} // namespace sm100::decode::fp8_head64
