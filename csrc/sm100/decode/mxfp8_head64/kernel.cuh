#include "kernel.h"

#include <math_constants.h>
#include <cutlass/barrier.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

#include "kerutils/kerutils.cuh"

#include "utils.h"
#include "sm100/helpers.h"

#include "config.h"

namespace sm100::decode::mxfp8_head64 {

template<ModelType MODEL_TYPE>
template<typename TmaParam>
__device__ void
KernelTemplate<MODEL_TYPE>
::flash_fwd_splitkv_mla_mxfp8_sparse_kernel_devfunc(const SparseAttnMxfp8DecodeParams &params, const TmaParam &tma_params) {
#if defined(KERUTILS_ENABLE_SM100A)
    const int s_q_idx = blockIdx.x;
    const int partition_idx = blockIdx.y;
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int idx_in_warpgroup = threadIdx.x % 128;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = threadIdx.x % 32;

    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);

    if (warp_idx == 0 && elect_one_sync()) {
        cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_O.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_Q_scale.get_tma_descriptor());
        cute::prefetch_tma_descriptor(&tma_params.tensor_map_kv);
        cute::prefetch_tma_descriptor(&tma_params.tensor_map_extra_kv);
    }

    if (warp_idx == 0) {
        if (elect_one_sync()) {
            plan.bar_last_store_done.init(128);
            plan.bar_q_tma.init(1);
            plan.bar_q_utccp.init(1);
            plan.bar_q_scale_tma.init(1);
            plan.bar_q_scale_utccp.init(1);
            for (int i = 0; i < NUM_BUFS; ++i) {
                plan.bar_kv_ready[i].init(1);
                plan.bar_kv_scale_ready[i].init(1);
                plan.bar_qk_done[i].init(1);
                plan.bar_so_ready[i].init(128);
                plan.bar_sv_done[i].init(1);
            }
            for (int i = 0; i < NUM_INDEX_BUFS; ++i) {
                plan.bar_valid_coord_scale_ready[i].init(32);
                plan.bar_valid_coord_scale_free[i].init(128+1+1);
            }
            cutlass::arch::fence_barrier_init();
        }
        cute::TMEM::Allocator1Sm().allocate(512, plan.tmem_start_addr.data());
        KU_TRAP_ONLY_DEVICE_ASSERT(plan.tmem_start_addr.data()[0] == 0);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }
    __syncthreads();

    struct MainLoopArgs {
        int batch_idx, start_block_idx, end_block_idx;
        bool is_no_split; int n_split_idx;
        bool bar_phase_batch_rel;    // Bar phase of barriers that are used once per batch
        int topk_length, extra_topk_length, num_orig_kv_blocks;
        bool is_last_batch;
    };

    auto run_main_loop = [&](auto f) {
        // NOTE Putting the following code outside the warpgroup specialization switch results in register spilling.
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

        bool bar_phase_batch_rel = 0;
        #pragma unroll 1
        for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx, bar_phase_batch_rel ^= 1) {
            int topk_length = params.topk_length ? __ldg(params.topk_length + batch_idx) : params.topk;
            int orig_topk_padded = max(ku::ceil(topk_length, (int)B_TOPK), (int)B_TOPK);
            int extra_topk_length = params.extra_topk_length ? __ldg(params.extra_topk_length + batch_idx) : params.extra_topk;
            int total_topk_padded = orig_topk_padded + ku::ceil(extra_topk_length, (int)B_TOPK);    // % B_TOPK == 0
            int start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
            int end_block_idx = batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx : total_topk_padded / B_TOPK;
            bool is_split = batch_idx == sched_meta.begin_req_idx ? sched_meta.is_first_req_splitted : (batch_idx == sched_meta.end_req_idx ? sched_meta.is_last_req_splitted : false);
            int n_split_idx = batch_idx == sched_meta.begin_req_idx ? (__ldg(params.num_splits_ptr+batch_idx) + sched_meta.begin_split_idx) : __ldg(params.num_splits_ptr+batch_idx);

            MainLoopArgs args = {
                batch_idx, start_block_idx, end_block_idx,
                !is_split, n_split_idx,
                bar_phase_batch_rel,
                topk_length, extra_topk_length,
                orig_topk_padded / B_TOPK,
                batch_idx == sched_meta.end_req_idx
            };

            f(args);
            NamedBarrier(NUM_THREADS, NamedBarriers::everyone_sync).arrive_and_wait_unaligned();
        }
    };

    struct RingState {
        int buf_idx = 0;
        bool bar_phase = 0;
        int index_buf_idx = 0;
        bool index_bar_phase = 0;
        CUTE_DEVICE void update() {
            bar_phase ^= (buf_idx == NUM_BUFS-1);
            buf_idx = (buf_idx+1) % NUM_BUFS;
            index_bar_phase ^= (index_buf_idx == NUM_INDEX_BUFS-1);
            index_buf_idx = (index_buf_idx+1) % NUM_INDEX_BUFS;
        }
    };
    RingState rs;

    if (warpgroup_idx == 0) {
        // ============================================================
        // WG0: Scale & Exp + mxfp8 quantize of S
        // ============================================================
        cutlass::arch::warpgroup_reg_alloc<224>();

        constexpr int B_EPI = 64;   // Must be equal to the size of the swizzle atom
        Tensor sO = make_tensor(make_smem_ptr(plan.u.qo.o.o_buf.data()), SmemLayoutOBuf{});
        bf16* sO_bases[B_EPI/8];   // 64 is the size of the swizzle atom (in number of elements) while 8 is the width of each write
        CUTE_UNROLL
        for (int i = 0; i < B_EPI/8; ++i)
            sO_bases[i] = &sO(idx_in_warpgroup%64, (idx_in_warpgroup/64)*128 + i*8);

        Tensor sS = make_tensor(make_smem_ptr(plan.s_p.s.data()), SmemLayoutS{});
        Tensor sS_scale = make_tensor(make_smem_ptr(plan.s_scale.data()), SmemLayoutSscale{});

        float attn_sink = params.attn_sink == nullptr ? -CUDART_INF_F : __ldg((float*)params.attn_sink + (idx_in_warpgroup%64)) * CUDART_L2E_F;

        run_main_loop([&](const MainLoopArgs &args) {
            cute::tma_store_wait<0>();
            plan.bar_last_store_done.arrive();

            float mi = MAX_INIT_VAL;
            float li = 0.0f;
            float real_mi = -CUDART_INF_F;

            CUTE_NO_UNROLL
            for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
                NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);  // Make sure all intermediate buffers are free
                plan.bar_valid_coord_scale_ready[rs.index_buf_idx].wait(rs.index_bar_phase);    // Put the barrier wait here for more code reordering space
                plan.bar_qk_done[rs.buf_idx].wait(rs.bar_phase);
                ku::tcgen05_after_thread_sync();

                // Load P (KQ^T output) from tmem, shape (B_TOPK, B_H) in tmem.
                // We load the right half and the left half into p and p_peer.
                float p[B_TOPK/2], p_peer[B_TOPK/2];
                if (warp_idx < 2) {
                    ku::tmem_ld_32dp32bNx<B_TOPK/2>(tmem_cols::P, p);
                    ku::tmem_ld_32dp32bNx<B_TOPK/2>(tmem_cols::P+32, p_peer);
                } else {
                    ku::tmem_ld_32dp32bNx<B_TOPK/2>(tmem_cols::P, p_peer);
                    ku::tmem_ld_32dp32bNx<B_TOPK/2>(tmem_cols::P+32, p);
                }
                cutlass::arch::fence_view_async_tmem_load();
                ku::tcgen05_before_thread_sync();

                // Reduce within shared mem (transpose of P)
                {
                    // Store
                    CUTE_UNROLL
                    for (int i = 0; i < (B_TOPK/2)/4; ++i)
                        plan.s_p.p_exchange_buf[warp_idx^2][i*32 + lane_idx] = *(float4*)(p_peer + i*4);
                    NamedBarrier::arrive_and_wait(64, NamedBarriers::wg0_warp02_sync+(warp_idx&1));
                    // Load
                    CUTE_UNROLL
                    for (int i = 0; i < (B_TOPK/2)/4; ++i) {
                        float2 t[2];
                        *(float4*)t = plan.s_p.p_exchange_buf[warp_idx][i*32 + lane_idx];
                        float2* cur_p = (float2*)(p + i*4);
                        cur_p[0] = ku::float2_add(cur_p[0], t[0]);
                        cur_p[1] = ku::float2_add(cur_p[1], t[1]);
                    }
                }

                // After exchange, p[i] holds the i-th column of the (B_TOPK, B_H) P matrix.
                // In register layout, this corresponds to S[h, t] = P[t, h], which is what
                // the VS MMA will consume (S in K-major (B_H, B_TOPK) layout).

                // Mask
                uint32_t valid_mask = *((uint32_t*)plan.is_token_valid[rs.index_buf_idx] + (idx_in_warpgroup>=64?1:0));
                CUTE_UNROLL
                for (int i = 0; i < B_TOPK/2; i += 1) {
                    if (!(valid_mask>>i&1))
                        p[i] = -CUDART_INF_F;
                }

                // Get rowwise max of Pi
                float cur_pi_max = -CUDART_INF_F;
                CUTE_UNROLL
                for (int i = 0; i < (B_TOPK/2); i += 1) {
                    cur_pi_max = max(cur_pi_max, p[i]);
                }
                cur_pi_max *= params.sm_scale_div_log2;

                plan.rowwise_max_buf[idx_in_warpgroup] = cur_pi_max;
                NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
                plan.bar_valid_coord_scale_free[rs.index_buf_idx].arrive();
                cur_pi_max = max(cur_pi_max, plan.rowwise_max_buf[idx_in_warpgroup^64]);
                real_mi = max(real_mi, cur_pi_max);
                bool should_scale_o = __any_sync(0xffffffff, cur_pi_max - mi > 6.0f);

                // Calc scale factor, and scale li
                float new_max, scale_for_old;
                if (!should_scale_o) {
                    scale_for_old = 1.0f;
                    new_max = mi;
                } else {
                    new_max = max(cur_pi_max, mi);
                    scale_for_old = exp2f(mi - new_max);
                }
                mi = new_max;

                // Threads h and h+64 own the two 64-token halves of one head.
                constexpr int NUM_LOCAL_GROUPS = (B_TOPK / 2) / MXFP8_SCALE_VEC_SIZE;
                int h = idx_in_warpgroup % B_H;
                int token_base = (idx_in_warpgroup / B_H) * (B_TOPK / 2);
                float cur_sum = 0.0f;
                CUTE_UNROLL
                for (int g = 0; g < NUM_LOCAL_GROUPS; ++g) {
                    float group_max_logit = -CUDART_INF_F;
                    CUTE_UNROLL
                    for (int i = 0; i < MXFP8_SCALE_VEC_SIZE; ++i) {
                        int local_token = g * MXFP8_SCALE_VEC_SIZE + i;
                        group_max_logit = max(group_max_logit, p[local_token] * params.sm_scale_div_log2);
                    }

                    float group_max_s = exp2f(group_max_logit - new_max);
                    e8m0 scale_g = e8m0(group_max_s > 0.0f ? group_max_s / FP8_MAX : 1.0f);
                    float inv_scale = 1.0f / float(scale_g);
                    int group_token = token_base + g * MXFP8_SCALE_VEC_SIZE;
                    sS_scale(h, group_token, _0{}) = scale_g;

                    CUTE_UNROLL
                    for (int i = 0; i < MXFP8_SCALE_VEC_SIZE; ++i) {
                        int local_token = g * MXFP8_SCALE_VEC_SIZE + i;
                        float s_val = exp2f(p[local_token] * params.sm_scale_div_log2 - new_max);
                        cur_sum += s_val;
                        sS(h, group_token + i) = e4m3(s_val * inv_scale);
                    }
                }
                li = fma(li, scale_for_old, cur_sum);

                // Scale O in tmem (only when should_scale_o and not first block)
                if (block_idx != args.start_block_idx && should_scale_o) {
                    float2 scale_for_old_float2 = float2 {scale_for_old, scale_for_old};
                    ku::tcgen05_after_thread_sync();

                    static constexpr int CHUNK_SIZE = 64;
                    float2 o[CHUNK_SIZE/2];
                    CUTE_UNROLL
                    for (int chunk_idx = 0; chunk_idx < (D_V/2)/CHUNK_SIZE; ++chunk_idx) {
                        ku::tmem_ld_32dp32bNx<CHUNK_SIZE>(tmem_cols::O + chunk_idx*CHUNK_SIZE, o);
                        cutlass::arch::fence_view_async_tmem_load();

                        CUTE_UNROLL
                        for (int i = 0; i < CHUNK_SIZE/2; ++i) {
                            o[i] = ku::float2_mul(o[i], scale_for_old_float2);
                        }

                        ku::tmem_st_32dp32bNx<CHUNK_SIZE>(tmem_cols::O + chunk_idx*CHUNK_SIZE, o);
                        cutlass::arch::fence_view_async_tmem_store();
                    }
                    ku::tcgen05_before_thread_sync();
                }

                fence_view_async_shared();
                plan.bar_so_ready[rs.buf_idx].arrive();

                if (block_idx != args.end_block_idx-1) {
                    rs.update();
                }
            }

            if (real_mi == -CUDART_INF_F) {
                li = 0.0f;
                mi = -CUDART_INF_F;
            }

            // Exchange li
            plan.rowwise_max_buf[idx_in_warpgroup] = li;
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
            li += plan.rowwise_max_buf[idx_in_warpgroup^64];

            // Store li
            if (idx_in_warpgroup < B_H) {
                if (args.is_no_split) {
                    float cur_lse = fmaf(mi, CUDART_LN2_F, logf(li));
                    cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;
                    float* gSoftmaxLse = (float*)params.lse + args.batch_idx*params.stride_lse_b + s_q_idx*params.stride_lse_s_q + idx_in_warpgroup;
                    *gSoftmaxLse = cur_lse;
                } else {
                    float cur_lse = log2f(li) + mi;
                    float* gSoftmaxLseAccum = (float*)params.lse_accum + args.n_split_idx*params.stride_lse_accum_split + s_q_idx*params.stride_lse_accum_s_q + idx_in_warpgroup;
                    *gSoftmaxLseAccum = cur_lse;
                }
            }

            plan.bar_sv_done[rs.buf_idx].wait(rs.bar_phase);
            rs.update();
            ku::tcgen05_after_thread_sync();

            if (args.is_last_batch) {
                cudaTriggerProgrammaticLaunchCompletion();
            }

            if (args.is_no_split) {
                Tensor tma_gO = flat_divide(
                    tma_params.tma_O.get_tma_tensor(tma_params.shape_O)(_, _, s_q_idx, args.batch_idx),
                    Shape<Int<B_H>, Int<64>>{}
                )(_, _, _0{}, _);
                auto thr_tma = tma_params.tma_O.get_slice(_0{});
                Tensor tma_sO = flat_divide(
                    sO,
                    Shape<Int<B_H>, Int<64>>{}
                )(_, _, _0{}, _);

                float o_scale = li == 0.0f ? 0.0f : __fdividef(1.0f, li + exp2f(attn_sink - mi));
                float2 o_scale_float2 = {o_scale, o_scale};
                float2 o[B_EPI/2];
                __nv_bfloat162 o_bf16[B_EPI/2];
                CUTE_UNROLL
                for (int i = 0; i < (D_V/2) / B_EPI; ++i) {
                    ku::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::O + i*B_EPI, o);
                    cutlass::arch::fence_view_async_tmem_load();
                    CUTE_UNROLL
                    for (int j = 0; j < B_EPI/2; ++j) {
                        o[j] = ku::float2_mul(o[j], o_scale_float2);
                        o_bf16[j] = __float22bfloat162_rn(o[j]);
                    }
                    int col_base = (i*B_EPI>=D_V/4 ? D_V/2 : 0) + (i*B_EPI%(D_V/4));
                    CUTE_UNROLL
                    for (int j = 0; j < B_EPI / 8; ++j)
                        *(__int128_t*)(sO_bases[j] + col_base*B_H) = *(__int128_t*)(&o_bf16[j*4]);
                    fence_view_async_shared();
                    NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
                    if (warp_idx == 0 && elect_one_sync()) {
                        cute::copy(
                            tma_params.tma_O,
                            thr_tma.partition_S(tma_sO(_, _, col_base/64)),
                            thr_tma.partition_D(tma_gO(_, _, col_base/64))
                        );
                    }
                    if (warp_idx == 1 && elect_one_sync()) {
                        cute::copy(
                            tma_params.tma_O,
                            thr_tma.partition_S(tma_sO(_, _, col_base/64 + (D_V/4)/64)),
                            thr_tma.partition_D(tma_gO(_, _, col_base/64 + (D_V/4)/64))
                        );
                    }
                }
                cute::tma_store_arrive();
            } else {
                float o_scale = li == 0.0f ? 0.0f : __fdividef(1.0f, li);   // leave attn_sink to the combine kernel
                float2 o_scale_float2 = {o_scale, o_scale};
                constexpr int B_EPI = 64;
                float2 o[B_EPI/2];
                Tensor sO = make_tensor(make_smem_ptr(plan.u.qo.o.o_accum_buf.data()), SmemLayoutOAccumBuf{});
                CUTE_UNROLL
                for (int i = 0; i < (D_V/2) / B_EPI; ++i) {
                    ku::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::O + i*B_EPI, o);
                    cutlass::arch::fence_view_async_tmem_load();
                    CUTE_UNROLL
                    for (int j = 0; j < B_EPI/2; ++j)
                        o[j] = ku::float2_mul(o[j], o_scale_float2);
                    int col_base = (idx_in_warpgroup/64)*128 + (i*B_EPI >= D_V/4 ? D_V/2 : 0) + (i*B_EPI%(D_V/4));
                    CUTE_UNROLL
                    for (int j = 0; j < B_EPI / 4; ++j)
                        *(__int128_t*)&sO(idx_in_warpgroup%64, col_base + j*4) = *(__int128_t*)(&o[j*2]);
                }
                fence_view_async_shared();
                NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
                if (elect_one_sync()) {
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < B_H/4; ++local_row) {
                        int smem_row = local_row*4 + warp_idx;
                        SM90_BULK_COPY_S2G::copy(
                            &sO(smem_row, _0{}),
                            (float*)params.o_accum + args.n_split_idx*params.stride_o_accum_split + s_q_idx*params.stride_o_accum_s_q + smem_row*params.stride_o_accum_h_q,
                            D_V*sizeof(float)
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
        // ============================================================
        // WG1: MMA, KV load, index/scale producer
        // ============================================================
        cutlass::arch::warpgroup_reg_dealloc<72>();
        const int warp_idx = cutlass::canonical_warp_idx_sync();

        if (warp_idx == 4 && elect_one_sync()) {
            // ===== MMA + Q-scale UTCCP warp =====
            run_main_loop([&](const MainLoopArgs &args) {
                if (args.start_block_idx >= args.end_block_idx) {
                    ku::trap();
                }

                // ===== Prologue: Q TMA (e4m3) =====
                plan.bar_q_tma.arrive_and_expect_tx(B_H*D_Q*sizeof(e4m3));
                plan.bar_q_scale_tma.arrive_and_expect_tx(B_H*Q_SCALE_BYTES*sizeof(e8m0));
                {
                    Tensor gQ = tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx, args.batch_idx);
                    Tensor sQ = make_tensor(make_smem_ptr(plan.u.qo.q.data()), SmemLayoutQ_SW128{});
                    ku::launch_tma_copy(
                        tma_params.tma_Q,
                        gQ,
                        sQ,
                        plan.bar_q_tma,
                        TMA::CacheHintSm90::EVICT_FIRST
                    );
                }
                // ===== Prologue: Q-scale TMA (e8m0) =====
                {
                    Tensor gQ_scale = tma_params.tma_Q_scale.get_tma_tensor(tma_params.shape_Q_scale)(_, _, s_q_idx, args.batch_idx);
                    Tensor sQ_scale = make_tensor(make_smem_ptr(plan.u.qo.q_scale.data()), SmemLayoutQScaleTMA{});
                    ku::launch_tma_copy(
                        tma_params.tma_Q_scale,
                        gQ_scale,
                        sQ_scale,
                        plan.bar_q_scale_tma,
                        TMA::CacheHintSm90::EVICT_FIRST
                    );
                }
                plan.bar_q_tma.wait(args.bar_phase_batch_rel);
                plan.bar_q_scale_tma.wait(args.bar_phase_batch_rel);
                ku::tcgen05_after_thread_sync();

                // ===== Prologue: Q-scale UTCCP smem -> tmem =====
                {
                    Tensor sQ_scale = make_tensor(
                        make_smem_ptr(plan.u.qo.q_scale.data()),
                        SmemLayoutQScale{}
                    );
                    auto sQ_compact = make_tensor(sQ_scale.data(), filter_zeros(sQ_scale.layout()));
                    Tensor tQ_scale = make_tensor<typename TiledMMA_P::FrgTypeSFB>(shape(SmemLayoutPScaleBAtom{}));
                    tQ_scale.data().get() = tmem_cols::QScale;

                    auto tQ_compact = make_tensor(tQ_scale.data(), filter_zeros(tQ_scale.layout()));
                    auto copy_Q_scale = make_utccp_copy(SM100_UTCCP_4x32dp128bit_1cta{}, tQ_compact);
                    auto thr_Q = copy_Q_scale.get_slice(0);
                    auto src_Q = get_utccp_smem_desc_tensor<SM100_UTCCP_4x32dp128bit_1cta>(
                        thr_Q.partition_S(sQ_compact)
                    );
                    auto dst_Q = thr_Q.partition_D(tQ_compact);
                    cute::copy(copy_Q_scale, src_Q, dst_Q);
                }
                ku::umma_arrive_noelect(plan.bar_q_utccp);
                plan.bar_q_scale_utccp.arrive();

                // ===== Allocate tmem tensors =====
                TiledMMA tiled_mma_P = TiledMMA_P{};
                TiledMMA tiled_mma_O = TiledMMA_O{};
                // tP is the KQ output of shape (B_TOPK, B_H) — the MMA's M=B_TOPK, N=B_H.
                Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<B_TOPK>, Int<B_H>>{});
                tP.data().get() = tmem_cols::P;

                plan.bar_q_utccp.wait(args.bar_phase_batch_rel);
                ku::tcgen05_after_thread_sync();

                // ===== Main loop =====
                CUTE_NO_UNROLL
                for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
                    // Wait for K data + K scales to land
                    plan.bar_kv_ready[rs.buf_idx].wait(rs.bar_phase);
                    plan.bar_kv_scale_ready[rs.buf_idx].wait(rs.bar_phase);
                    ku::tcgen05_after_thread_sync();

                    // UTCCP K scale smem -> tmem
                    {
                        Tensor sK_scale = make_tensor(
                            make_smem_ptr(plan.u.kv.kv_scale[rs.buf_idx].data()),
                            SmemLayoutKScale{}
                        );
                        auto sK_compact = make_tensor(sK_scale.data(), filter_zeros(sK_scale.layout()));
                        Tensor tK_scale = make_tensor<typename TiledMMA_P::FrgTypeSFA>(shape(SmemLayoutPScaleAAtom{}));
                        tK_scale.data().get() = tmem_cols::KScale;

                        auto tK_compact = make_tensor(tK_scale.data(), filter_zeros(tK_scale.layout()));
                        auto copy_K_scale = make_utccp_copy(SM100_UTCCP_4x32dp128bit_1cta{}, tK_compact);
                        auto thr_K = copy_K_scale.get_slice(0);
                        auto src_K = get_utccp_smem_desc_tensor<SM100_UTCCP_4x32dp128bit_1cta>(
                            thr_K.partition_S(sK_compact)
                        );
                        auto dst_K = thr_K.partition_D(tK_compact);
                        cute::copy(copy_K_scale, src_K, dst_K);
                    }
                    ku::tcgen05_after_thread_sync();

                    // KQ^T MMA: P = K @ Q^T (both operands in smem, tmem accum P)
                    Tensor tQ_scale = make_tensor<typename TiledMMA_P::FrgTypeSFB>(shape(SmemLayoutPScaleBAtom{}));
                    tQ_scale.data().get() = tmem_cols::QScale;
                    Tensor tK_scale = make_tensor<typename TiledMMA_P::FrgTypeSFA>(shape(SmemLayoutPScaleAAtom{}));
                    tK_scale.data().get() = tmem_cols::KScale;

                    Tensor sK = make_tensor(make_smem_ptr(plan.u.kv.kv[rs.buf_idx].data()), SmemLayoutKTiles_SW128<D_K/64>{});
                    Tensor sQ = make_tensor(make_smem_ptr(plan.u.qo.q.data()), SmemLayoutQ_SW128{});
                    ku::utcmma_blockscaled_ss(
                        tiled_mma_P, sK, sQ, tK_scale, tQ_scale,
                        tP, true
                    );
                    ku::umma_arrive_noelect(plan.bar_qk_done[rs.buf_idx]);

                    // Wait for WG0 to produce S + S scales
                    plan.bar_so_ready[rs.buf_idx].wait(rs.bar_phase);
                    ku::tcgen05_after_thread_sync();

                    // UTCCP S scale smem -> tmem
                    {
                        Tensor sS_scale = make_tensor(
                            make_smem_ptr(plan.s_scale.data()),
                            SmemLayoutSscale{}
                        );
                        auto sS_compact = make_tensor(sS_scale.data(), filter_zeros(sS_scale.layout()));
                        Tensor tS_scale = make_tensor<typename TiledMMA_O::FrgTypeSFB>(shape(SmemLayoutSscale{}));
                        tS_scale.data().get() = tmem_cols::SScale;

                        auto tS_compact = make_tensor(tS_scale.data(), filter_zeros(tS_scale.layout()));
                        auto copy_S_scale = make_utccp_copy(SM100_UTCCP_4x32dp128bit_1cta{}, tS_compact);
                        auto thr_S = copy_S_scale.get_slice(0);
                        auto src_S = get_utccp_smem_desc_tensor<SM100_UTCCP_4x32dp128bit_1cta>(
                            thr_S.partition_S(sS_compact)
                        );
                        auto dst_S = thr_S.partition_D(tS_compact);
                        cute::copy(copy_S_scale, src_S, dst_S);
                    }
                    ku::tcgen05_after_thread_sync();

                    // VS MMA: O_T = V @ S^T (V in MN-major (D_V, B_TOPK) via composition, S in K-major (B_TOPK, B_H))
                    // V scale reuses tmem_cols::KScale (same data, since K and V share buffer in MLA).
                    // tV_scale uses the O atom's SFA layout (different from K's SFA layout, since the
                    // K-dim of the O MMA is B_TOPK=128, not D_K=512), so we must construct tV_scale
                    // with `shape(SmemLayoutOScaleAAtom{})` for correct address calculation.
                    Tensor tV_scale = make_tensor<typename TiledMMA_O::FrgTypeSFA>(shape(SmemLayoutOScaleAAtom{}));
                    tV_scale.data().get() = tmem_cols::KScale;
                    Tensor tS_scale = make_tensor<typename TiledMMA_O::FrgTypeSFB>(shape(SmemLayoutSscale{}));
                    tS_scale.data().get() = tmem_cols::SScale;

                    Tensor sS = make_tensor(make_smem_ptr(plan.s_p.s.data()), SmemLayoutS{});
                    Tensor sV = make_tensor(make_smem_ptr(plan.u.kv.kv[rs.buf_idx].data()), SmemLayoutKTilesTransposed_SW128<D_V/64>{});

                    CUTE_UNROLL
                    for (int dv_block = 0; dv_block < D_V/B_TOPK; ++dv_block) {
                        Tensor tO_block = partition_fragment_C(tiled_mma_O, Shape<Int<B_TOPK>, Int<B_H>>{});
                        tO_block.data().get() = tmem_cols::O + dv_block*B_H;
                        ku::utcmma_blockscaled_ss(
                            tiled_mma_O, sV(make_coord(_, dv_block), _), sS,
                            tV_scale, tS_scale, tO_block, block_idx == args.start_block_idx
                        );
                    }
                    ku::umma_arrive_noelect(plan.bar_sv_done[rs.buf_idx]);

                    rs.update();
                }
            });
        } else if (warp_idx == 5 && elect_one_sync()) {
            // ===== Raw KV e4m3 retrieval warp (TMA-gather4) =====
            run_main_loop([&](const MainLoopArgs &args) {
                plan.bar_q_utccp.wait(args.bar_phase_batch_rel);
                plan.bar_last_store_done.wait(args.bar_phase_batch_rel);
                CUTE_NO_UNROLL
                for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
                    plan.bar_valid_coord_scale_ready[rs.index_buf_idx].wait(rs.index_bar_phase);
                    plan.bar_kv_ready[rs.buf_idx].arrive_and_expect_tx(B_TOPK*D_K*sizeof(e4m3));
                    int4 cur_indices = *(int4*)(plan.tma_coord[rs.index_buf_idx] + 0);
                    int4 nxt_cur_indices;
                    CUTE_UNROLL
                    for (int row = 0; row < B_TOPK; row += 4) {
                        if (row+4 < B_TOPK)
                            nxt_cur_indices = *(int4*)(plan.tma_coord[rs.index_buf_idx] + row + 4);
                        ku::tma_gather4(
                            block_idx >= args.num_orig_kv_blocks ? &tma_params.tensor_map_extra_kv : &tma_params.tensor_map_kv,
                            plan.bar_kv_ready[rs.buf_idx],
                            plan.u.kv.kv[rs.buf_idx].data() + D_K*row,
                            0,
                            cur_indices,
                            (int64_t)TMA::CacheHintSm90::EVICT_LAST
                        );
                        cur_indices = nxt_cur_indices;
                    }
                    plan.bar_valid_coord_scale_free[rs.index_buf_idx].arrive();
                    rs.update();
                }
            });
        } else if (warp_idx == 6) {
            // ===== K scale layout producer warp =====
            // Reads 8 raw e8m0 scales per token and duplicates each one into
            // `plan.u.kv.kv_scale[buf]` for tcgen05's 32-element scale vectors.
            // The smem layout is 3D (B_TOPK, D_K, 1)
            // and the scales occupy positions row, g*32 in the (row, K) plane.
            run_main_loop([&](const MainLoopArgs &args) {
                plan.bar_q_utccp.wait(args.bar_phase_batch_rel);
                plan.bar_last_store_done.wait(args.bar_phase_batch_rel);

                CUTE_NO_UNROLL
                for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
                    int cur_buf = rs.buf_idx;
                    int ib = rs.index_buf_idx;
                    plan.bar_valid_coord_scale_ready[ib].wait(rs.index_bar_phase);
                    // The K data TMA (warp 5) and the K scale layout producer (this warp) are
                    // independent — we just need the K scales from `plan.scales[ib]` to land
                    // (which warp 7's `bar_valid_coord_scale_ready` ensures).
                    Tensor sK_scale = make_tensor(
                        make_smem_ptr(plan.u.kv.kv_scale[cur_buf].data()),
                        SmemLayoutKScale{}
                    );
                    e8m0 (*scales_base)[K_SCALE_BYTES] = plan.scales[ib];

                    // 32 threads * 4 rows = 128 = B_TOPK
                    int rows_per_thread = B_TOPK / 32;  // 4
                    int row_base = lane_idx * rows_per_thread;
                    CUTE_UNROLL
                    for (int lr = 0; lr < rows_per_thread; ++lr) {
                        int row = row_base + lr;
                        e8m0* src = scales_base[row];
                        CUTE_UNROLL
                        for (int g = 0; g < K_SCALE_BYTES; ++g) {
                            CUTE_UNROLL
                            for (int dup = 0; dup < K_SCALE_DUP; ++dup) {
                                int dst_group = g * K_SCALE_DUP + dup;
                                sK_scale(row, dst_group * MXFP8_SCALE_VEC_SIZE, _0{}) = src[g];
                            }
                        }
                    }
                    fence_view_async_shared();
                    __syncwarp();
                    if (elect_one_sync()) {
                        plan.bar_kv_scale_ready[cur_buf].arrive();
                        plan.bar_valid_coord_scale_free[ib].arrive();
                    }
                    rs.update();
                }
            });
        } else if (warp_idx == 7) {
            // ===== Indices / K scale / valid-mask producer =====
            // The e8m0 K scales live at the END of each KV page (offset
            // `page_block_size * D_K` from the page start), 8 bytes per token.
            // Each lane handles four rows, covering all 128 rows in one warp.
            // The raw scales are packed into `plan.scales[ib][row]` alongside
            // the TMA coordinates and validity mask.
            static_assert(B_TOPK == 128);
            constexpr int TOKENS_PER_LANE = B_TOPK / 32;
            static_assert(TOKENS_PER_LANE == 4);
            int tma_coords_step_per_token = D_K / TMA_K_STRIDE;  // = 1
            int tma_coords_step_per_block = params.stride_kv_block / TMA_K_STRIDE;
            int tma_coords_step_per_extra_block = params.stride_extra_kv_block / TMA_K_STRIDE;
            uint8_t* k_scales_ptr = (uint8_t*)params.kv;  // base of all KV data
            uint8_t* extra_k_scales_ptr = (uint8_t*)params.extra_kv;

            run_main_loop([&](const MainLoopArgs &args) {
                int* indices = (int*)params.indices + params.stride_indices_b*args.batch_idx + params.stride_indices_s_q*s_q_idx;
                int* extra_indices = (int*)params.extra_indices + params.stride_extra_indices_b*args.batch_idx + params.stride_extra_indices_s_q*s_q_idx;

                struct IsOrigBlock {};
                struct IsExtraBlock {};
                auto process_one_block = [&](int block_idx, auto is_extra_block_t) {
                    static constexpr bool IS_EXTRA_BLOCK = std::is_same_v<decltype(is_extra_block_t), IsExtraBlock>;
                    int cur_block_size = IS_EXTRA_BLOCK ? params.extra_page_block_size : params.page_block_size;
                    int cur_num_blocks = IS_EXTRA_BLOCK ? params.extra_num_blocks : params.num_blocks;
                    int64_t cur_k_block_stride = IS_EXTRA_BLOCK ? params.stride_extra_kv_block : params.stride_kv_block;
                    uint8_t* cur_k_scales_ptr = IS_EXTRA_BLOCK ? extra_k_scales_ptr : k_scales_ptr;
                    int cur_tma_coords_step_per_block = IS_EXTRA_BLOCK ? tma_coords_step_per_extra_block : tma_coords_step_per_block;
                    int64_t page_scale_offset = (int64_t)cur_block_size * D_K;  // offset to scale region from page start

                    int abs_pos;
                    int4 my_indices_vec;
                    if (!IS_EXTRA_BLOCK) {
                        abs_pos = block_idx*B_TOPK + lane_idx*TOKENS_PER_LANE;
                        my_indices_vec = __ldg(reinterpret_cast<int4*>(indices + abs_pos));
                    } else {
                        abs_pos = (block_idx-args.num_orig_kv_blocks)*B_TOPK + lane_idx*TOKENS_PER_LANE;
                        my_indices_vec = __ldg(reinterpret_cast<int4*>(extra_indices + abs_pos));
                    }
                    plan.bar_valid_coord_scale_free[rs.index_buf_idx].wait(rs.index_bar_phase^1);

                    int* my_indices = reinterpret_cast<int*>(&my_indices_vec);
                    int4 tma_coords_vec;
                    int* tma_coords = reinterpret_cast<int*>(&tma_coords_vec);
                    uint32_t valid_mask = 0;
                    CUTE_UNROLL
                    for (int i = 0; i < TOKENS_PER_LANE; ++i) {
                        int page_idx, idx_in_block;
                        page_idx = (unsigned int)my_indices[i] / cur_block_size;
                        idx_in_block = (unsigned int)my_indices[i] % cur_block_size;
                        bool is_token_valid = my_indices[i] >= 0
                            && my_indices[i] < cur_num_blocks * cur_block_size
                            && (abs_pos+i < (IS_EXTRA_BLOCK?args.extra_topk_length:args.topk_length));
                        valid_mask |= is_token_valid << i;
                        tma_coords[i] = is_token_valid ? page_idx*cur_tma_coords_step_per_block + idx_in_block*tma_coords_step_per_token : -1;
                        // K scales at the end of the page
                        int64_t scale_offset = page_idx*cur_k_block_stride + page_scale_offset + idx_in_block*K_SCALE_BYTES;
                        *reinterpret_cast<uint64_t*>(plan.scales[rs.index_buf_idx][lane_idx*TOKENS_PER_LANE + i]) = is_token_valid
                            ? __ldg(reinterpret_cast<const uint64_t*>(cur_k_scales_ptr + scale_offset))
                            : uint64_t(0);
                    }
                    valid_mask <<= (lane_idx%2)*TOKENS_PER_LANE;
                    valid_mask |= __shfl_xor_sync(0xFFFFFFFF, valid_mask, 0x1);
                    *reinterpret_cast<int4*>(plan.tma_coord[rs.index_buf_idx] + lane_idx*TOKENS_PER_LANE) = tma_coords_vec;
                    if (lane_idx%2 == 0)
                        plan.is_token_valid[rs.index_buf_idx][lane_idx/2] = static_cast<char>(valid_mask);

                    plan.bar_valid_coord_scale_ready[rs.index_buf_idx].arrive();
                    rs.update();
                };

                CUTE_NO_UNROLL
                for (int block_idx = args.start_block_idx; block_idx < min(args.num_orig_kv_blocks, args.end_block_idx); ++block_idx) {
                    process_one_block(block_idx, IsOrigBlock{});
                }

                CUTE_NO_UNROLL
                for (int block_idx = max(args.start_block_idx, args.num_orig_kv_blocks); block_idx < args.end_block_idx; ++block_idx) {
                    process_one_block(block_idx, IsExtraBlock{});
                }
            });
        } else {
            run_main_loop([&](const MainLoopArgs &args) {});
        }
    } else {
        // ============================================================
        // WG2: Q TMA producer + Q scale TMA producer
        // ============================================================
        // The MMA warp (warp 4 of WG1) issues the Q TMA copies inside the
        // run_main_loop of warp 4. The Q data and Q scale are loaded by warp 4
        // itself in the prologue. This WG2 is therefore unused; warps 8–11
        // simply sit in their barrier init / __syncthreads() and never enter
        // the main loop.
        run_main_loop([&](const MainLoopArgs &args) {});
    }
#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100");
    }
#endif
}

template<typename Kernel, typename TmaParams>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, 1)
flash_fwd_splitkv_mla_mxfp8_sparse_kernel(__grid_constant__ const SparseAttnMxfp8DecodeParams params, __grid_constant__ const TmaParams tma_params) {
    Kernel::flash_fwd_splitkv_mla_mxfp8_sparse_kernel_devfunc(params, tma_params);
}

template<ModelType MODEL_TYPE>
void KernelTemplate<MODEL_TYPE>::run(const SparseAttnMxfp8DecodeParams &params) {
    KU_ASSERT(params.topk % B_TOPK == 0, "topk (%d) mod B_TOPK (%d) must be 0", params.topk, B_TOPK);
    KU_ASSERT(params.extra_topk % B_TOPK == 0, "extra_topk (%d) mod B_TOPK (%d) must be 0", params.extra_topk, B_TOPK);
    KU_ASSERT(params.h_q == B_H);
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.d_qk == D_Q);
    KU_ASSERT(params.d_v == D_V);
    // Each Q token is D_Q e4m3 + D_Q/32 e8m0 = 512 + 16 = 528 bytes
    constexpr int Q_BYTES_PER_TOKEN = D_Q + Q_SCALE_BYTES;
    KU_ASSERT(params.stride_q_h_q == Q_BYTES_PER_TOKEN,
        "stride_q_h_q must equal Q_BYTES_PER_TOKEN (D_Q + Q_SCALE_BYTES = %d), got %d",
        Q_BYTES_PER_TOKEN, params.stride_q_h_q);
    // Each KV page stores all e4m3 rows followed by all per-row UE8M0 scales.
    // The K data must be 16-byte aligned and stride_kv_block must be a multiple of TMA_K_STRIDE
    KU_ASSERT(params.stride_kv_block % TMA_K_STRIDE == 0,
        "stride_kv_block (%d) must be a multiple of TMA_K_STRIDE (%d); padding may be necessary",
        params.stride_kv_block, TMA_K_STRIDE);
    KU_ASSERT((int64_t)params.kv % 16 == 0, "kv base address must be 16B aligned for sm100 mxfp8");

    auto shape_Q = make_shape(B_H, D_Q, params.s_q, params.b);
    auto tma_Q = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((uint8_t*)params.q),
            make_layout(
                shape_Q,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q, params.stride_q_b)
            )
        ),
        SmemLayoutQ_SW128{}
    );

    auto shape_Q_scale = make_shape(B_H, Q_SCALE_BYTES, params.s_q, params.b);
    auto tma_Q_scale = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((uint8_t*)params.q + D_Q),
            make_layout(
                shape_Q_scale,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q, params.stride_q_b)
            )
        ),
        SmemLayoutQScaleTMA{}
    );

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

    auto get_kv_tensormap = [&](void* k_ptr, int num_blocks, int64_t k_block_stride) {
        return ku::make_tensor_map(
            {D_K/8, (uint64_t)num_blocks * (k_block_stride/TMA_K_STRIDE)},
            {TMA_K_STRIDE},
            {D_K/8, 1},
            k_ptr,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT64,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B
        );  // pack 8 e4m3 into 1 INT64 (TMA box dim < 256)
    };

    CUtensorMap tensor_map_kv = get_kv_tensormap(params.kv, params.num_blocks, params.stride_kv_block);
    CUtensorMap tensor_map_extra_kv{};
    if (params.extra_topk > 0) {
        tensor_map_extra_kv = get_kv_tensormap(params.extra_kv, params.extra_num_blocks, params.stride_extra_kv_block);
    }

    TmaParams<
        decltype(shape_Q), decltype(tma_Q),
        decltype(shape_Q_scale), decltype(tma_Q_scale),
        decltype(shape_O), decltype(tma_O)
    > tma_params = {
        shape_Q, tma_Q,
        shape_Q_scale, tma_Q_scale,
        shape_O, tma_O,
        tensor_map_kv,
        tensor_map_extra_kv
    };
    auto mla_kernel = &flash_fwd_splitkv_mla_mxfp8_sparse_kernel<KernelTemplate<MODEL_TYPE>, decltype(tma_params)>;

    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    static_assert(smem_size < 227*1024);
    KU_CUDA_CHECK(cudaFuncSetAttribute(mla_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    // NOTE Don't use PDL because of potential compiler bugs!
    mla_kernel<<<dim3(params.s_q, params.num_sm_parts, 1), dim3(NUM_THREADS, 1, 1), smem_size, params.stream>>>(params, tma_params);
    KU_CHECK_KERNEL_LAUNCH();
}

template<ModelType MODEL_TYPE>
void run_flash_splitkv_mla_mxfp8_sparse_kernel(const SparseAttnMxfp8DecodeParams &params) {
    KernelTemplate<MODEL_TYPE>::run(params);
}

}
