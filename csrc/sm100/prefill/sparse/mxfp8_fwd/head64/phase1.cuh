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

CUTE_DEVICE
float ue8m0_bits_to_float(uint8_t bits) {
    TRAP_ONLY_DEVICE_ASSERT(bits != 0xff);
    if (bits == 0) {
        return __uint_as_float(0x00400000u); // UE8M0 0x00 is 2^-127.
    }
    return __uint_as_float(static_cast<uint32_t>(bits) << 23);
}

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

template<int B_H, int B_H_TMEM, int TMEM_COL_START, int D_V>
CUTE_DEVICE
void rescale_O_t(float scale[B_H]) {
    float o[B_H_TMEM];
    CUTE_UNROLL
    for (int tile = 0; tile < D_V/SV_M; ++tile) {
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
        cute::prefetch_tma_descriptor(tma_params.tma_Q_scale.get_tma_descriptor());
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
            Tensor gQ_scale = tma_params.tma_Q_scale.get_tma_tensor(tma_params.shape_Q_scale)(_, _, s_q_idx);
            Tensor sQ_scale = make_tensor(make_smem_ptr(plan.s_q_scale.q_scale.compact.data()), SmemLayoutQScaleTMA{});
            plan.bar_prologue_q_scale.arrive_and_expect_tx(B_H*Q_SCALE_BYTES*sizeof(e8m0));
            ku::launch_tma_copy(tma_params.tma_Q_scale, gQ_scale, sQ_scale, plan.bar_prologue_q_scale, TMA::CacheHintSm90::EVICT_FIRST);

        CUTE_UNROLL
        for (int i = 0; i < NUM_P_BUFS; ++i) {
            plan.bar_qk_done[i].init(1);
            plan.bar_p_free[i].init(128);
        }
        CUTE_UNROLL
        for (int i = 0; i < NUM_BUFS; ++i) {
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

    if (warpgroup_idx == 0) {
        const uint64_t w_scale_bits = __ldg(
            reinterpret_cast<const uint64_t*>(params.kv_scale_w)
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
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
            int cur_buf = k % NUM_BUFS;
            int p_stage = k % NUM_P_BUFS;
            plan.bar_qk_done[p_stage].wait((k / NUM_P_BUFS) & 1);
            plan.bar_k_valid_ready[cur_buf].wait((k / NUM_BUFS) & 1);
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
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);

            float cur_pi_max = max(
                local_pi_max,
                plan.rowwise_max_buf[idx_in_warpgroup ^ B_H]
            );
            float old_mi = plan.head_mi[h];
            bool should_scale_o = cur_pi_max - old_mi > 6.0f;
            float new_max = should_scale_o ? max(cur_pi_max, old_mi) : old_mi;
            float scale_for_old = should_scale_o ? exp2f(old_mi - new_max) : 1.0f;

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

            plan.rowwise_li_buf[idx_in_warpgroup] = local_sum;
            fence_view_async_shared();
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);

            if (token_group == 0) {
                float cur_sum = local_sum + plan.rowwise_li_buf[idx_in_warpgroup + B_H];
                plan.head_scale[h] = scale_for_old;
                plan.head_mi[h] = new_max;
                plan.head_real_mi[h] = max(plan.head_real_mi[h], cur_pi_max);
                plan.head_li[h] = fma(plan.head_li[h], scale_for_old, cur_sum);
                if (k > 0) {
                    plan.bar_sv_done[(k - 1) % NUM_BUFS].wait(((k - 1) / NUM_BUFS) & 1);
                }
            }

            plan.bar_k_valid_free[cur_buf].arrive();
            fence_view_async_shared();
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);

            if (k > 0) {
                ku::tcgen05_after_thread_sync();
                rescale_O_t<B_H, B_H_TMEM, tmem_cols::O, D_V>(plan.head_scale);
                ku::tcgen05_before_thread_sync();
                NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);
            }

            if (idx_in_warpgroup == 0) {
                plan.bar_so_ready.arrive();
            }
        }
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

        plan.bar_sv_done[(num_k_blocks-1)%NUM_BUFS].wait(((num_k_blocks-1)/NUM_BUFS)&1);
        ku::tcgen05_after_thread_sync();

        if (idx_in_warpgroup < B_H) {
            int h = idx_in_warpgroup;
            float attn_sink = params.attn_sink == nullptr ? -CUDART_INF_F : __ldg(params.attn_sink + h)*CUDART_L2E_F;
            float output_scale = plan.head_li[h] == 0.0f
                ? 0.0f
                : __fdividef(1.0f, plan.head_li[h] + exp2f(attn_sink - plan.head_mi[h]));
            plan.head_scale[h] = output_scale;
        }
        NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);

        {
            Tensor sO = make_tensor(make_smem_ptr(plan.kvo.o.data()), SmemLayoutO{});
            float o_head[B_H_TMEM];
            CUTE_UNROLL
            for (int tile = 0; tile < D_V/SV_M; ++tile) {
                int dv = tile*SV_M + idx_in_warpgroup;
                ku::tmem_ld_32dp32bNx<B_H_TMEM>(tmem_cols::O + tile*B_H_TMEM, o_head);
                cutlass::arch::fence_view_async_tmem_load();
                int w_group = dv / K_QUANT_GROUP_SIZE;
                uint8_t w_bits = static_cast<uint8_t>(w_scale_bits >> (w_group * 8));
                float w_scale = ue8m0_bits_to_float(w_bits);
                CUTE_UNROLL
                for (int h = 0; h < B_H; ++h) {
                    sO(h, dv) = bf16(o_head[h] * plan.head_scale[h] * w_scale);
                }
            }
        }
        NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync);

        // Store O using TMA
        constexpr int B_EPI = 64;
        Tensor sO = make_tensor(make_smem_ptr(plan.kvo.o.data()), SmemLayoutO{});
        Tensor tma_gO = flat_divide(
            tma_params.tma_O.get_tma_tensor(tma_params.shape_O)(_, _, s_q_idx),
            Shape<Int<B_H>, Int<B_EPI>>{}
        )(_, _, _0{}, _);
        Tensor sO_divided = flat_divide(
            sO,
            Shape<Int<B_H>, Int<B_EPI>>{}
        )(_, _, _0{}, _);
        auto thr_tma = tma_params.tma_O.get_slice(_0{});

        CUTE_UNROLL
        for (int k = 0; k < D_V/B_EPI; ++k) {
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
        int producer_warp_idx = cutlass::canonical_warp_idx_sync() - 4;
        constexpr int NUM_LOCAL_ROWS_PER_WARP = (B_TOPK/4)/NUM_KV_PRODUCER_WARPS;
        CUTE_NO_UNROLL
        for (int k = 0; k < num_k_blocks; ++k) {
            int4 indices[NUM_LOCAL_ROWS_PER_WARP];
            if (elect_one_sync()) {
                // Copy NoPE data with gather4.
                int cur_buf = k%NUM_BUFS;
                plan.bar_sv_done[cur_buf].wait((k/NUM_BUFS)&1^1);

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
            }
            fence_view_async_shared();
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg1_tma_sync);

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
            NamedBarrier::arrive_and_wait(128, NamedBarriers::wg1_tma_sync);

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
        }
    } else {
        if (warp_idx == 8 && elect_one_sync()) {

            Tensor sQ_dup = make_tensor(make_smem_ptr(plan.q.data()), SmemLayoutQDuplicated{});
            plan.bar_prologue_q.wait(0);

            plan.bar_prologue_q_scale.wait(0);
            Tensor sQ_scale_tma = make_tensor(
                make_smem_ptr(plan.s_q_scale.q_scale.compact.data()),
                SmemLayoutQScaleTMA{}
            );
            Tensor sQ_scale_mma = make_tensor(
                make_smem_ptr(plan.s_q_scale.q_scale.mma.data()),
                SmemLayoutPScaleAAtom{}
            );
            constexpr int QK_SCALE_GROUPS = QK_K / MXFP8_SCALE_VEC_SIZE;
            CUTE_UNROLL
            for (int row = 0; row < QK_M; ++row) {
                CUTE_UNROLL
                for (int g = 0; g < QK_SCALE_GROUPS; ++g) {
                    auto dst_coord = make_coord(
                        g % SCALE_GROUPS_PER_TMEM_BLOCK,
                        g / SCALE_GROUPS_PER_TMEM_BLOCK
                    );
                    sQ_scale_mma(row, _0{}, dst_coord) =
                        sQ_scale_tma(row % B_H, g);
                }
            }
            fence_view_async_shared();
            auto copy_q_scale_to_tmem = [&](e8m0* scale_storage) {
                CUTE_UNROLL
                for (int sf_block = 0; sf_block < QK_K / 128; ++sf_block) {
                    Tensor sQ_block = make_tensor(
                        make_smem_ptr(
                            scale_storage
                            + sf_block * cosize_v<SmemLayoutPScaleABlockAtom>
                        ),
                        SmemLayoutPScaleABlockAtom{}
                    );
                    Tensor tQ_block = make_tensor<typename TiledMMA_P::FrgTypeSFA>(
                        shape(SmemLayoutPScaleABlockAtom{})
                    );
                    tQ_block.data().get() = tmem_cols::Q_Scale
                        + sf_block * TMEM_SCALE_K128_STRIDE;
                    auto sQ_compact = make_tensor(
                        sQ_block.data(), filter_zeros(sQ_block.layout())
                    );
                    auto tQ_compact = make_tensor(
                        tQ_block.data(), filter_zeros(tQ_block.layout())
                    );
                    auto copy_Q_scale = make_utccp_copy(
                        SM100_UTCCP_4x32dp128bit_1cta{}, tQ_compact
                    );
                    auto thr_Q = copy_Q_scale.get_slice(0);
                    auto src_Q = get_utccp_smem_desc_tensor<SM100_UTCCP_4x32dp128bit_1cta>(
                        thr_Q.partition_S(sQ_compact)
                    );
                    auto dst_Q = thr_Q.partition_D(tQ_compact);
                    cute::copy(copy_Q_scale, src_Q, dst_Q);
                }
            };
            copy_q_scale_to_tmem(plan.s_q_scale.q_scale.mma.data());

            // V's token-dependent U(t) is folded into S before S is
            // quantized. The remaining W(g) is applied in the epilogue, so
            // every logical V scale consumed by the SV MMA is one.
            Tensor sV_scale_one = make_tensor(
                make_smem_ptr(plan.s_scale.data()),
                SmemLayoutOScaleAAtom{}
            );
            uint8_t* sV_scale_storage = reinterpret_cast<uint8_t*>(plan.s_scale.data());
            CUTE_NO_UNROLL
            for (int i = 0; i < cosize_v<SmemLayoutOScaleAAtom>; ++i) {
                sV_scale_storage[i] = UE8M0_ONE_BITS;
            }
            fence_view_async_shared();
            auto sV_compact = make_tensor(sV_scale_one.data(), filter_zeros(sV_scale_one.layout()));
            auto tV_compact = make_tensor(tV_scale.data(), filter_zeros(tV_scale.layout()));
            auto copy_V_scale = make_utccp_copy(SM100_UTCCP_4x32dp128bit_1cta{}, tV_compact);
            auto thr_V = copy_V_scale.get_slice(0);
            auto src_V = get_utccp_smem_desc_tensor<SM100_UTCCP_4x32dp128bit_1cta>(
                thr_V.partition_S(sV_compact)
            );
            auto dst_V = thr_V.partition_D(tV_compact);
            cute::copy(copy_V_scale, src_V, dst_V);

            CUTE_NO_UNROLL
            for (int k = 0; k < num_k_blocks+1; ++k) {
                if (k < num_k_blocks) {
                    // Pi = QKi^T
                    int cur_buf = k % NUM_BUFS;
                    int p_stage = k % NUM_P_BUFS;
                    Tensor sK = make_tensor(
                        make_smem_ptr(plan.kvo.kv.kv[cur_buf].data()),
                        SmemLayoutK_TiledMMA{}
                    );

                    plan.bar_p_free[p_stage].wait(((k / NUM_P_BUFS) & 1) ^ 1);
                    ku::tcgen05_after_thread_sync();

                    plan.bar_kv_scale_ready[cur_buf].wait((k/NUM_BUFS)&1);
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

                    plan.bar_kv_ready[cur_buf].wait((k/NUM_BUFS)&1);
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
                }

                if (k > 0) {
                    // O += S(i-1)V(i-1)
                    int cur_buf = (k-1)%NUM_BUFS;

                    Tensor sS = make_tensor(make_smem_ptr(plan.s_q_scale.s), SmemLayoutS{});
                    Tensor sS_scale = make_tensor(make_smem_ptr(plan.s_scale.data()), SmemLayoutOScaleBAtom{});
                    Tensor sV = make_tensor(make_smem_ptr(plan.kvo.kv.kv[cur_buf].data()), SmemLayoutV{});

                    // Wait for S(i-1) and O to be scaled
                    plan.bar_so_ready.wait((k-1)&1);
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

                    // O += sS @ sV
                    CUTE_UNROLL
                    for (int dv_block = 0; dv_block < D_V/SV_M; ++dv_block) {
                        Tensor tO_block = partition_fragment_C(tiled_mma_O, Shape<Int<SV_M>, Int<B_H>>{});
                        tO_block.data().get() = tmem_cols::O + dv_block*B_H_TMEM;
                        ku::utcmma_blockscaled_ss(
                            tiled_mma_O, sV(make_coord(_, dv_block), _), sS,
                            tV_scale, tS_scale, tO_block, k == 1
                        );
                    }
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
                int cur_buf = k % NUM_BUFS;
                plan.bar_sv_done[cur_buf].wait((k / NUM_BUFS) & 1 ^ 1);
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

    auto shape_Q_scale = make_shape(B_H, Q_SCALE_BYTES, params.s_q);
    auto tma_Q_scale = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((e8m0*)((uint8_t*)params.q + D_Q)),
            make_layout(
                shape_Q_scale,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q)
            )
        ),
        SmemLayoutQScaleTMA{}
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
        decltype(shape_Q), decltype(tma_Q),
        decltype(shape_Q_scale), decltype(tma_Q_scale)
    > tma_params = {
        shape_O, tma_O,
        shape_Q, tma_Q,
        shape_Q_scale, tma_Q_scale,
        tensor_map_kv
    };

    auto kernel = &sparse_attn_fwd_kernel<decltype(tma_params)>;

    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    kernel<<<params.s_q, NUM_THREADS, smem_size, params.stream>>>(params, tma_params);
    KU_CHECK_KERNEL_LAUNCH();
}

}
