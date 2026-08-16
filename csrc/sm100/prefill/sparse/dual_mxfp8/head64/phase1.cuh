#pragma once
#include "phase1.h"

#include <math_constants.h>
#include <cute/arch/copy_sm80.hpp>
#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/arch.h>

#include "params.h"
#include "utils.h"
#include "sm100/prefill/sparse/common_subroutine.h"
#include "sm100/helpers.h"

#include "config.h"

namespace sm100::dual_mxfp8::head64 {

using namespace cute;

using FwdMode = SparseAttnFwdMode;

template<class TiledMMA, class SmemScaleLayout>
CUTE_DEVICE void copy_sfa_smem_to_tmem(
        typename TiledMMA::ValTypeSFA* smem_ptr, uint32_t tmem_col) {
    Tensor sScale = make_tensor(make_smem_ptr(smem_ptr), SmemScaleLayout{});
    Tensor tScale = make_tensor<typename TiledMMA::FrgTypeSFA>(shape(SmemScaleLayout{}));
    tScale.data().get() = tmem_col;

    auto sScaleCompact = make_tensor(sScale.data(), filter_zeros(sScale.layout()));
    auto tScaleCompact = make_tensor(tScale.data(), filter_zeros(tScale.layout()));
    auto tiled_copy = make_utccp_copy(
        SM100_UTCCP_4x32dp128bit_2cta{}, tScaleCompact
    );
    auto thr_copy = tiled_copy.get_slice(0);
    auto src = get_utccp_smem_desc_tensor<SM100_UTCCP_4x32dp128bit_2cta>(
        thr_copy.partition_S(sScaleCompact)
    );
    auto dst = thr_copy.partition_D(tScaleCompact);
    cute::copy(tiled_copy, src, dst);
}

CUTE_DEVICE UMMA::SmemDescriptor make_utccp_scale_desc(void* smem_ptr) {
    // Same descriptor used by DeepGEMM for a post-transpose scale tile:
    // eight 16-byte atoms along MN, one atom along K.
    UMMA::SmemDescriptor desc = {};
    desc.version_ = 1;
    desc.lbo_mode_ = 0;
    desc.layout_type_ = static_cast<uint8_t>(UMMA::LayoutType::SWIZZLE_NONE);
    desc.start_address_ = static_cast<uint16_t>(cast_smem_ptr_to_uint(smem_ptr) >> 4);
    desc.base_offset_ = 0;
    desc.stride_byte_offset_ = (8 * 16) >> 4;
    desc.leading_byte_offset_ = 0;
    return desc;
}

CUTE_DEVICE void copy_q_scale_smem_to_tmem(
        void* smem_ptr, uint32_t tmem_col) {
    UMMA::SmemDescriptor desc = make_utccp_scale_desc(nullptr);
    CUTE_UNROLL
    for (int sf_block = 0; sf_block < 2; ++sf_block) {
        // Global -> SMEM already writes DeepGEMM's post-transpose physical
        // order. Each K=128 scale tile is 128 uint32_t words (512 bytes).
        void* block_ptr = reinterpret_cast<uint8_t*>(smem_ptr) + sf_block * 512;
        desc.start_address_ = static_cast<uint16_t>(
            cast_smem_ptr_to_uint(block_ptr) >> 4
        );
        SM100_UTCCP_4x32dp128bit_2cta::copy(
            desc, tmem_col + sf_block * 4
        );
    }
}

CUTE_DEVICE void copy_k_scale_smem_to_tmem(
        void* smem_ptr, uint32_t tmem_col) {
    UMMA::SmemDescriptor desc = make_utccp_scale_desc(nullptr);
    desc.start_address_ = static_cast<uint16_t>(
        cast_smem_ptr_to_uint(smem_ptr) >> 4
    );
    // The warp10/warp11 repack produces the post-transpose 64x16B source
    // expected by the 2x2 SFB layout. This copy atom maps the paired rows into
    // the two 32-bit words selected by the duplicated2by2 TMEM layout.
    SM100_UTCCP_2x64dp128bitlw0123_2cta::copy(desc, tmem_col);
}

template<class TiledMMA, class ScaleLayout>
CUTE_DEVICE void fill_sfa_from_registers(uint32_t tmem_col, int warp_in_warpgroup) {
    Tensor tScale = make_tensor<typename TiledMMA::FrgTypeSFA>(shape(ScaleLayout{}));
    tScale.data().get() = tmem_col;

    // UE8M0 is byte-packed in TMEM.  For the all-one bring-up scale, each
    // participating lane supplies four 0x7f bytes from a register.  The SFA
    // fragment has four 32-DP replications in each 128-DP set.  Consecutive
    // four scale factors packed in the register fill the K=0,32,64,96 UE8M0
    // subwords.  The four 32-row groups occupy consecutive TMEM columns.
    uint32_t one4 = 0x7f7f7f7f;
    uint32_t warp_dp_addr =
        tmem_col + warp_in_warpgroup * 32 * cute::TMEM::DP<uint32_t>::value;
    CUTE_UNROLL
    for (int row_group = 0; row_group < 4; ++row_group) {
        SM100_TMEM_STORE_32dp32b1x::copy(one4, warp_dp_addr + row_group);
    }
}

template<FwdMode FWD_MODE, int D_QK>
__device__ void
KernelTemplate<FWD_MODE, D_QK>::sparse_attn_fwd_kernel_devfunc(const ArgT &params, const TmaParams &tma_params) {
#ifdef KERUTILS_ENABLE_SM100A
    // Each 2-CTA cluster handles one adjacent query-token pair, and each CTA
    // owns one native [64, D_Q] tile.  For an odd decode s_q, CTA 1 in the
    // final cluster uses the last real Q as a dummy operand and skips stores.
    // Cluster shape: [2, 1, 1]
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = threadIdx.x % 32;
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int idx_in_warpgroup = threadIdx.x % 128;
    const int cta_idx = block_id_in_cluster().x;

    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &smem = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);

    if (warp_idx == 0 && elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tma_params.tensor_map_q);
        cute::prefetch_tma_descriptor(&tma_params.tensor_map_o);
        cute::prefetch_tma_descriptor(&tma_params.tensor_map_kv);
        if constexpr (IS_PREFILL && !DUAL_MXFP8_K_SCALE_CP_ASYNC) {
            cute::prefetch_tma_descriptor(&tma_params.tensor_map_k_scale);
        }
    } else if (warp_idx == 1 && elect_one_sync()) {
        smem.bar_sQ_full.init(1);
        // CTA0's barrier gathers the two warpgroup-0 Q-scale producers before
        // CTA0 issues the single cta_group::2 UTCCP operation.
        smem.bar_Q_scale_ready.init(2);
        smem.bar_tQ_empty.init(1);
        smem.bar_tQ_full.init(1);
        smem.bar_tOut_full.init(1);
        smem.bar_tOut_empty.init(256);
        smem.bar_P_empty.init(256);
        smem.bar_SV_done.init(1);
        smem.bar_S_O_full.init(256);
        smem.bar_li_full.init(H_Q/2);
        smem.bar_li_empty.init(128);
        if constexpr (FWD_MODE != FwdMode::DecodeWithSplitKV) {
            smem.bar_clc_full.init(1);
            smem.bar_clc_empty.init(NUM_WORKER_THREADS);
        }
        fence_barrier_init();
    } else if (warp_idx == 2) {
        cute::TMEM::Allocator2Sm().allocate(512, smem.tmem_start_addr.data());
        KU_TRAP_ONLY_DEVICE_ASSERT(smem.tmem_start_addr.data()[0] == 0);
        cute::TMEM::Allocator2Sm().release_allocation_lock();
    } else if (warp_idx == 3 && elect_one_sync()) {
        CUTE_UNROLL
        for (int i = 0; i < NUM_K_BUFS; ++i) {
            smem.bar_KV_full[i].init(IS_PREFILL ? 1 : (128/32)*2+1);
            smem.bar_KV_empty[i].init(1);
#if !DUAL_MXFP8_K_SCALE_CP_ASYNC
            smem.bar_K_scale_raw_full[i].init(4);
#endif
            // CTA0's barrier gathers completion reports from warp10/warp11 in
            // both CTAs after the final K-scale source is visible.
            smem.bar_K_scale_copy_ready[i].init(4);
            // The MMA completion releases this stage's scale source and is
            // also consumed by the softmax workers as QK-ready.
            smem.bar_QK_done[i].init(1);
        }
        CUTE_UNROLL
        for (int i = 0; i < NUM_INDEX_BUFS; ++i) {
            smem.bar_valid_coord_scales_full[i].init(IS_PREFILL ? B_TOPK/8 : 32);
            smem.bar_valid_coord_scales_empty[i].init(IS_PREFILL ? 128 : (128 + (cta_idx==1) + 2 + 128));
        }
        fence_barrier_init();
    }

    ku::barrier_cluster_arrive_relaxed();
    ku::barrier_cluster_wait_acquire();

    // w1/w2 each pack four UE8M0 values. One 32-bit TMEM store writes all
    // four scale-factor IDs. Four warps cover the four 32-DP subpartitions.
    if (warpgroup_idx == 0) {
        const uint32_t packed_v_scales = __float_as_uint(
            cta_idx == 0 ? params.w1 : params.w2
        );
        const uint32_t warp_dp_addr = tmem_cols::V_scale
            + warp_idx * 32 * cute::TMEM::DP<uint32_t>::value;
        CUTE_UNROLL
        for (int col = 0; col < 4; ++col) {
            SM100_TMEM_STORE_32dp32b1x::copy(
                packed_v_scales, warp_dp_addr + col
            );
        }
        cutlass::arch::fence_view_async_tmem_store();
    }
    ku::barrier_cluster_arrive_relaxed();
    ku::barrier_cluster_wait_acquire();

    struct OuterloopArgs {
        bool outer_loop_phase;
        int batch_idx, s_q_idx;
        int start_block_idx, end_block_idx;
        int topk_length;

        int extra_topk_length, num_orig_kv_blocks;  // extra-KV related
        bool is_no_split; int n_split_idx;  // splitkv related
    };

    auto run_outer_loop = [&](auto loop_body) -> bool {
        int outer_loop_phase = false;
        if constexpr (FWD_MODE == FwdMode::DecodeWithSplitKV) {
            int s_q_idx = blockIdx.x / 2; // adjacent-token pair index
            DecodingSchedMeta sched_meta;
            KU_LDG_256(
                params.tile_scheduler_metadata_ptr + blockIdx.y,
                &sched_meta,
                ".nc",
                "no_allocate",
                "evict_normal",
                "256B"
            );
            if (sched_meta.begin_req_idx >= params.b) {
                return 0;
            }

            #pragma unroll 1
            for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
                int topk_length = params.topk_length ? __ldg(params.topk_length + batch_idx) : params.topk;
                int orig_topk_padded = max(ku::ceil(topk_length, (int)B_TOPK), (int)B_TOPK);
                int extra_topk_length = params.extra_topk_length ? __ldg(params.extra_topk_length + batch_idx) : params.extra_topk;
                int total_topk_padded = orig_topk_padded + ku::ceil(extra_topk_length, (int)B_TOPK);    // % B_TOPK == 0
                int start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
                int end_block_idx = batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx : total_topk_padded / B_TOPK;
                bool is_split = batch_idx == sched_meta.begin_req_idx ? sched_meta.is_first_req_splitted : (batch_idx == sched_meta.end_req_idx ? sched_meta.is_last_req_splitted : false);
                int n_split_idx = batch_idx == sched_meta.begin_req_idx ? (__ldg(params.num_splits_ptr+batch_idx) + sched_meta.begin_split_idx) : __ldg(params.num_splits_ptr+batch_idx);

                    // start_block_idx = 0;
                    // end_block_idx = total_topk_padded / B_TOPK;
                    // is_split = false;
                    // n_split_idx = 0;

                OuterloopArgs args = {
                    (bool)outer_loop_phase,
                    batch_idx, s_q_idx,
                    start_block_idx, end_block_idx,
                    topk_length,

                    extra_topk_length, orig_topk_padded / B_TOPK,
                    !is_split, n_split_idx
                };

                loop_body(args);
                outer_loop_phase ^= 1;
            }
        } else {
            // Prefill mode. Use CLC to allocate different s_q (for decoding, different batches + s_q) to different workers
            ku::CLCResult next_job = {true, (int)blockIdx.x, IS_PREFILL ? 0 : (int)blockIdx.y, 0};
            CUTE_NO_UNROLL
            while (next_job.is_valid) {
                int s_q_idx = next_job.x / 2; // pair index for this head64 prefill kernel
                int batch_idx = IS_PREFILL ? 0 : next_job.y;
                int topk_length = params.topk_length != nullptr
                    ? __ldg(params.topk_length + (IS_PREFILL ? s_q_idx : batch_idx))
                    : params.topk;

                if constexpr (IS_PREFILL) {
                    int num_k_blocks = max(cute::ceil_div(topk_length, (int)B_TOPK), 1);  // num_k_blocks always >= 1
                    OuterloopArgs args = {
                        (bool)outer_loop_phase,
                        0, s_q_idx,
                        0, num_k_blocks,
                        topk_length
                    };
                    loop_body(args);
                } else {
                    int orig_topk_padded = max(ku::ceil(topk_length, (int)B_TOPK), (int)B_TOPK);
                    int extra_topk_length = params.extra_topk_length ? __ldg(params.extra_topk_length + batch_idx) : params.extra_topk;
                    int total_topk_padded = orig_topk_padded + ku::ceil(extra_topk_length, (int)B_TOPK);    // % B_TOPK == 0

                    OuterloopArgs args = {
                        (bool)outer_loop_phase,
                        batch_idx, s_q_idx,
                        0, total_topk_padded / B_TOPK,
                        topk_length,

                        extra_topk_length, orig_topk_padded / B_TOPK,
                        false, 0
                    };
                    loop_body(args);
                }

                smem.bar_clc_full.wait(outer_loop_phase);
                next_job = ku::get_clc_query_response<true>(smem.clc_response_obj);
                smem.bar_clc_empty.arrive(0u);

                outer_loop_phase ^= 1;
            }
        }
        return outer_loop_phase;
    };

    if (warpgroup_idx == 0) {
        // Q fetching and O writing back warpgroup
        cutlass::arch::warpgroup_reg_alloc<176>();

        bf16* sO_addrs[B_EPI/8];
        CUTE_UNROLL
        for (int i = 0; i < B_EPI/8; ++i) {
            Tensor sO = make_tensor(make_smem_ptr(reinterpret_cast<bf16*>(smem.Q.data())), ku::make_umma_canonical_k_major_layout<H_Q/2, D_V, 128>());
            sO_addrs[i] = &sO(idx_in_warpgroup%64, (idx_in_warpgroup/64)*(D_V/2) + i*8);
        }

        float* sO_accum_addrs[B_EPI_SPLITKV/4];
        if constexpr (FWD_MODE == FwdMode::DecodeWithSplitKV) {
            // If split-KV is enabled, we need to store back O in float32
            // We view Q buffer (with shape 64 x 512, bf16) as 4 buffers with shape (H_Q/2) x (B_EPI_SPLITKV*2), float32
            Tensor sO_accum = make_tensor(make_smem_ptr((float*)smem.Q.data()), ku::make_umma_canonical_k_major_layout<H_Q/2, D_V, 128, float>());
            CUTE_UNROLL
            for (int i = 0; i < B_EPI_SPLITKV/4; ++i) {
                sO_accum_addrs[i] = &sO_accum(idx_in_warpgroup%64, i*4) + (idx_in_warpgroup >= 64 ? (H_Q/2)*B_EPI_SPLITKV : 0);
            }
        }

        auto perform_o_copy_out = [&](const OuterloopArgs &args, bool is_last_o) {
            // outer_loop_phase is the loop phase corresponding to s_q_idx

            // Get li (output_scale actually)
            smem.bar_li_full.wait(args.outer_loop_phase);
            float output_scale = smem.rowwise_li_buf[idx_in_warpgroup%64];
            float2 output_scale_float2 = float2 {output_scale, output_scale};
            smem.bar_li_empty.arrive();

            // Retrieve and store O, and calculate delta := sum(O*dO, dim=-1) if FWD_MODE is Recompute
            smem.bar_tOut_full.wait(args.outer_loop_phase);
            if (is_last_o && elect_one_sync()) {
                cudaTriggerProgrammaticLaunchCompletion();
            }

            if (FWD_MODE != FwdMode::DecodeWithSplitKV || args.is_no_split) {
                CUTE_UNROLL
                for (int k = 0; k < (D_V/2)/B_EPI; ++k) {
                    float2 o[B_EPI/2];
                    ku::tmem_ld_32dp32bNx<B_EPI>(tmem_cols::O + k*B_EPI, o);
                    cutlass::arch::fence_view_async_tmem_load();
                    if (k == (D_V/2)/B_EPI-1) {
                        smem.bar_tOut_empty.arrive(0u);
                    }
                    CUTE_UNROLL
                    for (int i = 0; i < B_EPI/8; ++i) {
                        nv_bfloat162 o_bf16[4];
                        CUTE_UNROLL
                        for (int j = 0; j < 4; ++j) {
                            o[i*4+j] = ku::float2_mul(o[i*4+j], output_scale_float2);
                            o_bf16[j] = __float22bfloat162_rn(o[i*4+j]);
                        }
                        bf16* o_do_addr = sO_addrs[i] + k*B_EPI*(H_Q/2);
                            if (k == 0 && i == 0) {
                                smem.bar_tQ_full.wait(args.outer_loop_phase^1^is_last_o);    // Wait for sQ's availability
                            }
                        ku::st_shared(o_do_addr, *(__int128_t*)o_bf16);
                    }
                }

                fence_view_async_shared();
                NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);
                int q_token_idx = 2*args.s_q_idx + cta_idx;
                bool is_real_q_token = q_token_idx < params.s_q;
                if (warp_idx == 0 && elect_one_sync() && is_real_q_token) {
                    SM90_TMA_STORE_5D::copy(
                        &tma_params.tensor_map_o,
                        smem.Q.data(),
                        0,
                        0,
                        0,
                        q_token_idx,
                        IS_DECODE ? args.batch_idx : 0
                    );
                    cute::tma_store_arrive();
                }
            } else {
                CUTE_UNROLL
                for (int k = 0; k < (D_V/2)/B_EPI_SPLITKV; ++k) {
                    int cur_buf_idx = k % NUM_EPI_SPLITKV_BUFS;
                    if (k == 0) {
                        cute::tma_store_wait<0>();
                    } else {
                        cute::tma_store_wait<NUM_EPI_SPLITKV_BUFS-1>();
                    }
                    NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);

                    float o[B_EPI_SPLITKV];
                    ku::tmem_ld_32dp32bNx<B_EPI_SPLITKV>(tmem_cols::O + k*B_EPI_SPLITKV, o);
                    cutlass::arch::fence_view_async_tmem_load();
                    if (k == (D_V/2)/B_EPI_SPLITKV-1) {
                        smem.bar_tOut_empty.arrive(0u);
                    }
                    CUTE_UNROLL
                    for (int i = 0; i < B_EPI_SPLITKV/4; ++i) {
                        CUTE_UNROLL
                        for (int j = 0; j < 4; j += 2) {
                            *(float2*)(o + i*4 + j) = ku::float2_mul(float2 {o[i*4+j], o[i*4+j+1]}, output_scale_float2);
                        }
                        if (k == 0 && i == 0) {
                            smem.bar_tQ_full.wait(args.outer_loop_phase^1^is_last_o);    // Wait for sQ's availability
                        }
                        ku::st_shared(
                            sO_accum_addrs[i] + cur_buf_idx*((H_Q/2)*B_EPI_SPLITKV*2),
                            *(__int128_t*)(o + i*4)
                        );
                    }

                    fence_view_async_shared();
                    NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);
                    if constexpr (IS_DECODE) {  // Otherwise nvcc complains about `tma_params` doesn't have `tensor_map_o_accum`
                        float* cur_buf_base = (float*)smem.Q.data() + cur_buf_idx*((H_Q/2)*B_EPI_SPLITKV*2);
                        int q_token_idx = 2*args.s_q_idx + cta_idx;
                        bool is_real_q_token = q_token_idx < params.s_q;
                        if (warp_idx == 0 && elect_one_sync() && is_real_q_token) {
                            SM90_TMA_STORE_5D::copy(
                                &tma_params.tensor_map_o_accum,
                                cur_buf_base,
                                0, 0, k*(B_EPI_SPLITKV/32), q_token_idx, args.n_split_idx
                            );
                            cute::tma_store_arrive();
                        } else if (warp_idx == 1 && elect_one_sync() && is_real_q_token) {
                            SM90_TMA_STORE_5D::copy(
                                &tma_params.tensor_map_o_accum,
                                cur_buf_base + (H_Q/2)*B_EPI_SPLITKV,
                                0, 0, k*(B_EPI_SPLITKV/32) + (D_V/2)/32, q_token_idx, args.n_split_idx
                            );
                            cute::tma_store_arrive();
                        }
                    }
                }
            }
        };

        OuterloopArgs last_args;
        last_args.batch_idx = -1;

        bool final_outer_loop_phase = \
        run_outer_loop([&](const OuterloopArgs &args) {
            if constexpr (!IS_DECODE) {
                // One WG0 thread loads one complete replicated 16B scale slot
                // for each Q head. Keep the physical post-transpose SMEM
                // layout expected by the existing Q-scale UTCCP path.
                const int head_idx = idx_in_warpgroup;
                if (head_idx < TOKEN_H_Q) {
                    const int q_token_idx = 2 * args.s_q_idx + cta_idx;
                    const int q_load_token_idx = min(q_token_idx, params.s_q - 1);
                    const uint8_t* q_scale_base =
                        reinterpret_cast<const uint8_t*>(params.q)
                        + q_load_token_idx * params.stride_q_s_q
                        + head_idx * params.stride_q_h_q
                        + D_Q;
                    uint4 packed_scale;
                    KU_LDG_128(
                        q_scale_base,
                        &packed_scale,
                        ".nc",
                        "evict_first",
                        "128B"
                    );

                    const int row_base =
                        (head_idx % 32) * 16 + (head_idx / 32) * 4;
                    fp8_e8m0* dst = smem.q_scale_mma.data() + row_base;
                    *reinterpret_cast<uint32_t*>(dst + 0 * 512) = packed_scale.x;
                    *reinterpret_cast<uint32_t*>(dst + 1 * 512) = packed_scale.y;
                    *reinterpret_cast<uint32_t*>(dst + 2 * 512) = packed_scale.z;
                    *reinterpret_cast<uint32_t*>(dst + 3 * 512) = packed_scale.w;
                    *reinterpret_cast<uint32_t*>(dst + 0 * 512 + 8) = packed_scale.x;
                    *reinterpret_cast<uint32_t*>(dst + 1 * 512 + 8) = packed_scale.y;
                    *reinterpret_cast<uint32_t*>(dst + 2 * 512 + 8) = packed_scale.z;
                    *reinterpret_cast<uint32_t*>(dst + 3 * 512 + 8) = packed_scale.w;
                }
                fence_view_async_shared();
                NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);
                if (warp_idx == 0 && elect_one_sync()) {
                    smem.bar_Q_scale_ready.arrive(uint32_t{0}, 1u);
                }
            }

            // Copy Q for this round
            if constexpr (FWD_MODE == FwdMode::DecodeWithSplitKV) {
                cute::tma_store_wait<0>();
                NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);  // Since we use two warps to issue TMA during FwdMode::DecodeWithSplitKV
            }
            if (warp_idx == 0 && elect_one_sync()) {
                // Wait for sQ to become empty, and issue G -> S copy for Q
                if constexpr (FWD_MODE != FwdMode::DecodeWithSplitKV) {
                    cute::tma_store_wait<0>();  // This thread must be the same one as o copy out thread (since `elect_one_sync()` always returns the same thread for the same `mask`, according to PTX document)
                }
                int stride_q_b_div_stride_q_s_q = 0;
                if constexpr (IS_DECODE) {
                    stride_q_b_div_stride_q_s_q = params.stride_q_b / params.stride_q_s_q;
                }
                int q_token_idx = 2*args.s_q_idx + cta_idx;
                // A 2-SM operation cannot drop the second CTA.  Clamp the odd
                // tail CTA to the final real token so it receives a valid Q
                // tile and can participate in every transaction and barrier.
                int q_load_token_idx = min(q_token_idx, params.s_q - 1);
                SM100_TMA_2SM_LOAD_5D_NOSPLIT::copy(
                    &tma_params.tensor_map_q,
                    (uint64_t*)&smem.bar_sQ_full,
                    (uint64_t)TMA::CacheHintSm90::EVICT_FIRST,
                    smem.Q.data(),
                    0,
                    0,
                    0,
                    0,
                    (IS_DECODE ? args.batch_idx*stride_q_b_div_stride_q_s_q : 0) + q_load_token_idx
                );

                // Q's 5D TMA layout groups the two 256-value halves into a
                // 128-row dual-map source. E4M3 needs eight 16-byte copies per
                // 128-value tile to populate the full duplicated TS fragment.
                if (cta_idx == 0) {
                    smem.bar_sQ_full.arrive_and_expect_tx(H_Q*D_Q*sizeof(fp8_e4m3));
                    smem.bar_sQ_full.wait(args.outer_loop_phase);
                    smem.bar_tQ_empty.wait(args.outer_loop_phase^1);

                    ku::tcgen05_after_thread_sync();
                    UMMA::SmemDescriptor sQ_desc = UMMA::make_umma_desc<UMMA::Major::K>(
                        make_tensor(
                            make_smem_ptr(smem.Q.data()),
                            ku::make_umma_canonical_k_major_layout<(H_Q/2)*2, 128, 128, fp8_e4m3>()
                        )
                    );
                    CUTE_UNROLL
                    for (int tile_idx = 0; tile_idx < D_Q / 128; ++tile_idx) {
                        CUTE_UNROLL
                        for (int subtile_idx = 0; subtile_idx < 8; ++subtile_idx) {
                            UMMA::SmemDescriptor cur_sQ_desc = sQ_desc;
                            cur_sQ_desc.start_address_ +=
                                tile_idx * ((H_Q/2) * 128 * 2) / 16
                                + subtile_idx;
                            SM100_UTCCP_128dp128bit_2cta::copy(
                                cur_sQ_desc,
                                tmem_cols::Q + tile_idx * 32 + subtile_idx * 4
                            );
                        }
                    }
                    ku::umma_arrive_multicast_2x1SM_noelect(smem.bar_tQ_full, 1|2);
                }
            }

            if (last_args.batch_idx != -1) {
                perform_o_copy_out(last_args, false);
            } else {
                smem.bar_tQ_full.wait(args.outer_loop_phase);   // To prevent double arrive
            }
            last_args = args;
        });
        if (last_args.batch_idx != -1) {
            cute::tma_store_wait<0>();
            NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);
            perform_o_copy_out(last_args, true);
        }

        if (warp_idx == 0) {
            cute::TMEM::Allocator2Sm().free(0, 512);
        }
    } else if (warpgroup_idx == 1) {
        // KV fetching threads for prefill, dequant threads for decoding
        cutlass::arch::warpgroup_reg_dealloc<80>();
        RingBufferState rs;

        if constexpr (!IS_DECODE) {
            const int warp_idx = cutlass::canonical_warp_idx();    // Using `warp_idx` without `__shfl_sync` is faster
            if (elect_one_sync()) {
                // KV fetching threads
                run_outer_loop([&](const OuterloopArgs &args) {
                    int* gIndices = params.indices + args.s_q_idx*params.stride_indices_s_q;
                    int64_t cache_hint = ku::create_simple_cache_policy<ku::CacheHint::EVICT_LAST>();

                    static constexpr int NUM_ROWS_PER_THREAD = B_TOPK / 4;

                    CUTE_NO_UNROLL
                    for (int k = args.start_block_idx; k < args.end_block_idx; ++k) {
                        auto [k_buf_idx, k_bar_phase] = rs.get<NUM_K_BUFS>();

                        int cur_indices[NUM_ROWS_PER_THREAD];
                        CUTE_UNROLL
                        for (int local_row = 0; local_row < NUM_ROWS_PER_THREAD/8; local_row += 1) {
                            int row = local_row*(4*8) + (warp_idx-4)*8;
                            KU_LDG_256(
                                gIndices + k*B_TOPK + row,
                                cur_indices + local_row*8,
                                ".nc",
                                "no_allocate",
                                "evict_first",
                                "256B"
                            );
                        }
                        smem.bar_KV_empty[k_buf_idx].wait(k_bar_phase^1);

#if !DUAL_MXFP8_K_SCALE_CP_ASYNC
                        // Gather complete replicated 16B token slots into a
                        // padded raw buffer. Warp10/warp11 later transpose
                        // token r and r+32 into the CUTLASS SFB source rows.
                        smem.bar_K_scale_raw_full[k_buf_idx].arrive_and_expect_tx(
                            (NUM_ROWS_PER_THREAD / K_SCALE_GATHER_ROWS)
                                * K_SCALE_GATHER_BYTES
                        );
                        CUTE_UNROLL
                        for (int local_group = 0;
                             local_group < NUM_ROWS_PER_THREAD / K_SCALE_GATHER_ROWS;
                             ++local_group) {
                            // cur_indices positions 0..7 map to physical
                            // rows [warp*8, warp*8+7], and positions 8..15
                            // map to [32+warp*8, 32+warp*8+7]. Pair rows are
                            // laid out as two 4-token groups per half.
                            const int pair_row = (warp_idx - 4) * 2
                                + (local_group / 2) * (B_TOPK / 8)
                                + (local_group % 2);
                            int4 token_indices = *reinterpret_cast<int4*>(
                                cur_indices + local_group * K_SCALE_GATHER_ROWS
                            );
                            ku::tma_gather4(
                                &tma_params.tensor_map_k_scale,
                                smem.bar_K_scale_raw_full[k_buf_idx],
                                &smem.k_scale_pair_raw[k_buf_idx][pair_row][0],
                                0,
                                token_indices,
                                cache_hint
                            );
                        }
#endif

                        CUTE_UNROLL
                        for (int local_row = 0; local_row < NUM_ROWS_PER_THREAD/4; local_row += 1) {
                            int row = (warp_idx-4)*8 + (local_row/2)*(4*8) + (local_row%2)*4;
                            int4 indices = *(int4*)(cur_indices+local_row*4);
                            static_assert(D_K == 512);
                            CUTE_UNROLL
                            for (int local_col = 0; local_col < (D_K/128)/2; ++local_col) {
                                ku::tma_gather4_cta_group_2<true>(
                                    &tma_params.tensor_map_kv,
                                    smem.bar_KV_full[k_buf_idx],
                                    smem.K[k_buf_idx].data() + row*128 + local_col*128*B_TOPK,
                                    local_col*128 + cta_idx*(D_K/2),
                                    indices,
                                    cache_hint
                                );
                            }
                        }
                        rs.update();
                    }
                });
            }

        } else {
#if 0  // Decode-only legacy path; dual_mxfp8 currently instantiates prefill only.
            // 8 threads per token
            struct IsCTA0 {};
            struct IsCTA1 {};

            auto launch_dequant_wg = [&](auto cta_id_t) {
                static constexpr bool IS_CTA1 = std::is_same<decltype(cta_id_t), IsCTA1>::value;
                constexpr int GROUP_SIZE = 8, NUM_GROUPS = 128/8, ROWS_PER_GROUP = B_TOPK / NUM_GROUPS, COLS_PER_GROUP = (IS_CTA1 ? 256-64 : 256) / (GROUP_SIZE*8);
                int group_idx = idx_in_warpgroup/GROUP_SIZE, idx_in_group = idx_in_warpgroup%GROUP_SIZE;
                Tensor nope0 = make_tensor(make_smem_ptr(smem.K[0].data()), ku::make_umma_canonical_k_major_layout<B_TOPK, D_K/2, 128>());
                bf16* nope0_base = &nope0(group_idx, idx_in_group*8);
                fp8_e4m3* raw_nope0_base = smem.K_raw[0].data() + group_idx*(D_K/2) + idx_in_group*8;
                run_outer_loop([&](const OuterloopArgs &args) {
                    CUTE_NO_UNROLL
                    for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
                        auto [k_buf_idx, k_bar_phase] = rs.get<NUM_K_BUFS>();
                        auto [raw_k_buf_idx, raw_k_bar_phase] = rs.get<NUM_RAW_K_BUFS>();
                        auto [index_buf_idx, index_bar_phase] = rs.get<NUM_INDEX_BUFS>();
                        fp8_e4m3* raw_nope_base = raw_nope0_base + raw_k_buf_idx * (B_TOPK*(D_K/2));
                        auto get_raw_fp8 = [&](int local_row_idx, int local_col_idx) -> uint64_t {
                            return *(uint64_t*)(raw_nope_base + local_row_idx*NUM_GROUPS*(D_K/2) + local_col_idx*(GROUP_SIZE*8));
                        };
                        bf16* nope_base = nope0_base + k_buf_idx * (B_TOPK*(D_K/2));
                        uint32_t cur_nope_base_uint_addr = cute::cast_smem_ptr_to_uint(nope_base);
                        auto st_128b = [&](int local_row_idx, int local_col_idx, __int128_t &data) {
                            asm volatile ("st.weak.shared::cta.b128 [%0], %1;\n"
                                :
                                : "r"(cur_nope_base_uint_addr + 2*(local_row_idx*NUM_GROUPS*64 + local_col_idx*B_TOPK*64)), "q"(data)   // 2 for sizeof(bf16)
                            );  // We have this `asm volatile` here, otherwise the compiler generates ST.E instead of STS
                        };

                        smem.bar_valid_coord_scales_full[index_buf_idx].wait(index_bar_phase);
                        smem.bar_raw_KV_full[raw_k_buf_idx].wait(raw_k_bar_phase);

                        CUTE_UNROLL
                        for (int local_row_idx = 0; local_row_idx < ROWS_PER_GROUP; ++local_row_idx) {
                            int row_idx = local_row_idx*NUM_GROUPS + group_idx;
                            bf16 scales[4];
                            fp8_e8m0 scales_e8m0[4];
                            *(uint32_t*)scales_e8m0 = *(uint32_t*)(smem.scales[index_buf_idx][row_idx]);
                            *(__nv_bfloat162_raw*)(scales+0) = __nv_cvt_e8m0x2_to_bf162raw(*(unsigned short*)(scales_e8m0+0));
                            *(__nv_bfloat162_raw*)(scales+2) = __nv_cvt_e8m0x2_to_bf162raw(*(unsigned short*)(scales_e8m0+2));

                            uint64_t cur_data_fp8x8 = get_raw_fp8(local_row_idx, 0);
                            CUTE_UNROLL
                            for (int local_col_idx = 0; local_col_idx < COLS_PER_GROUP; ++local_col_idx) {
                                ku::nve4m3x2 data_fp8[4];
                                ku::nvbf16x2 data_bf16[4];
                                *(uint64_t*)data_fp8 = cur_data_fp8x8;
                                if (local_col_idx+1 < COLS_PER_GROUP)
                                    cur_data_fp8x8 = get_raw_fp8(local_row_idx, local_col_idx+1);
                                bf16 scale = scales[local_col_idx];
                                CUTE_UNROLL
                                for (int i = 0; i < 4; ++i) {
                                    data_bf16[i] = fp8x2_to_bf16x2_with_scale(data_fp8[i], *(ku::nvbf16*)(&scale));
                                }
                                if (local_row_idx == 0 && local_col_idx == 0) {
                                    smem.bar_KV_empty[k_buf_idx].wait(k_bar_phase^1);
                                }
                                st_128b(local_row_idx, local_col_idx, *(__int128_t*)data_bf16);
                            }
                        }

                        fence_view_async_shared();  // NOTE Should we use shared::cluster here?
                        __syncwarp();
                        smem.bar_valid_coord_scales_empty[index_buf_idx].arrive();
                        smem.bar_raw_KV_empty[raw_k_buf_idx].arrive();
                        if (elect_one_sync()) {
                            smem.bar_KV_full[k_buf_idx].arrive(0u);
                        }
                        rs.update();
                    }
                });
            };
            if (cta_idx == 0) {
                launch_dequant_wg(IsCTA0{});
            } else {
                launch_dequant_wg(IsCTA1{});
            }
#endif
        }
    } else if (warpgroup_idx == 2) {
        cutlass::arch::warpgroup_reg_dealloc<80>();

        RingBufferState rs;
        if (warp_idx == 8 && cta_idx == 0 && elect_one_sync()) {
            // UMMA thread
            TiledMMA tiled_mma_P = TiledMMA_P{};
            TiledMMA tiled_mma_O = TiledMMA_O{};
            Tensor tP = partition_fragment_C(tiled_mma_P, Shape<Int<H_Q/2>, Int<B_TOPK*2>>{});
            Tensor tO = partition_fragment_C(tiled_mma_O, Shape<Int<H_Q/2>, Int<D_V>>{});
            Tensor tQ = tiled_mma_P.get_slice(_0{}).make_fragment_A(
                partition_shape_A(tiled_mma_P, Shape<Int<H_Q/2>, Int<D_Q/2>>{})
            );
            tP.data().get() = tmem_cols::P;
            tO.data().get() = tmem_cols::O;
            tQ.data().get() = tmem_cols::Q;

            run_outer_loop([&](const OuterloopArgs &args) {
                smem.bar_tQ_full.wait(args.outer_loop_phase);
                ku::tcgen05_after_thread_sync();
                smem.bar_Q_scale_ready.wait(args.outer_loop_phase);
                copy_q_scale_smem_to_tmem(
                    smem.q_scale_mma.data(), tmem_cols::Q_scale
                );
                ku::tcgen05_before_thread_sync();

                // Issue P = Q K^T
                auto issue_P = [&](int k, int rs_offset) {
                    auto [k_buf_idx, k_bar_phase] = rs.offset_by(rs_offset).get<NUM_K_BUFS>();
                    auto [_, bar_phase] = rs.offset_by(rs_offset).get<1>();
                    smem.bar_P_empty.wait(bar_phase^1);
                    if constexpr (IS_PREFILL) {
                        smem.bar_KV_full[k_buf_idx].arrive_and_expect_tx(B_TOPK*D_K*sizeof(fp8_e4m3));
                    } else {
                        // RoPE only
                        smem.bar_KV_full[k_buf_idx].arrive_and_expect_tx(B_TOPK*D_ROPE*sizeof(bf16));
                    }
                    smem.bar_KV_full[k_buf_idx].wait(k_bar_phase);
                    ku::tcgen05_after_thread_sync();

                    smem.bar_K_scale_copy_ready[k_buf_idx].wait(k_bar_phase);
                    copy_k_scale_smem_to_tmem(
                        smem.k_scale_mma[k_buf_idx].data(), tmem_cols::K_scale
                    );
                    ku::tcgen05_before_thread_sync();
                    Tensor sK = make_tensor(
                        make_smem_ptr(smem.K[k_buf_idx].data()),
                        ku::make_umma_canonical_k_major_layout<B_TOPK, D_K/2, 128, fp8_e4m3>()
                    );
                    Tensor tQ_scale = make_tensor<typename TiledMMA_P::FrgTypeSFA>(shape(SmemLayoutPScaleA{}));
                    Tensor tK_scale = make_tensor<typename TiledMMA_P::FrgTypeSFB>(shape(SmemLayoutPScaleB{}));
                    tQ_scale.data().get() = tmem_cols::Q_scale;
                    tK_scale.data().get() = tmem_cols::K_scale;
                    ku::utcmma_blockscaled_ts_explicit_sf_ids<K_QUANT_GROUP_SIZE>(
                        tiled_mma_P, tQ, sK, tQ_scale, tK_scale, tP, true
                    );
                    ku::umma_arrive_multicast_2x1SM_noelect(
                        smem.bar_QK_done[k_buf_idx], 1|2
                    );
                };

                // Issue O += S V
                auto issue_O = [&](int k, int rs_offset) {
                    auto [k_buf_idx, k_bar_phase] = rs.offset_by(rs_offset).get<NUM_K_BUFS>();
                    auto [_, bar_phase] = rs.offset_by(rs_offset).get<1>();
                    smem.bar_S_O_full.wait(bar_phase);
                    if (k == args.start_block_idx) {
                        smem.bar_tOut_empty.wait(args.outer_loop_phase^1);
                    }
                    ku::tcgen05_after_thread_sync();
                    Tensor sS = make_tensor(
                        make_smem_ptr(smem.S.data()),
                        ku::make_umma_canonical_k_major_layout<H_Q/2, B_TOPK, 0, fp8_e4m3>()
                    );
                    Tensor sV = make_tensor(
                        make_smem_ptr(smem.K[k_buf_idx].data()),
                        ku::make_umma_canonical_mn_major_layout<D_V/2, B_TOPK, 128, fp8_e4m3>()
                    );
                    Tensor tS_scale = make_tensor<typename TiledMMA_O::FrgTypeSFA>(shape(SmemLayoutOScaleA{}));
                    Tensor tV_scale = make_tensor<typename TiledMMA_O::FrgTypeSFB>(shape(SmemLayoutOScaleB{}));
                    tS_scale.data().get() = tmem_cols::S_scale;
                    tV_scale.data().get() = tmem_cols::V_scale;
                    ku::utcmma_blockscaled_ss(tiled_mma_O, sS, sV, tS_scale, tV_scale, tO, k == args.start_block_idx);
                    ku::umma_arrive_multicast_2x1SM_noelect(smem.bar_SV_done, 1|2);
                    ku::umma_arrive_multicast_2x1SM_noelect(smem.bar_KV_empty[k_buf_idx], 1|2);
                };

                CUTE_NO_UNROLL
                for (int k = args.start_block_idx; k < args.end_block_idx+1; ++k) {
                    if (k < args.end_block_idx) {
                        issue_P(k, 0);
                    }
                    if (k == args.end_block_idx-1) {
                        ku::umma_arrive_2x1SM_noelect(smem.bar_tQ_empty);
                    }

                    if (k > args.start_block_idx) {
                        issue_O(k-1, -1);
                    }

                    if (k != args.end_block_idx) {
                        rs.update();
                    }
                }
                ku::tcgen05_before_thread_sync();
                ku::umma_arrive_multicast_2x1SM_noelect(smem.bar_tOut_full, 1|2);
            });
        } else if (warp_idx == 9) {
            // KV validness loading warp (for prefill), Indices transformation warp (for decode, Responsible for generating: TMA coordinates, scale factors, and valid masks)
            if constexpr (IS_PREFILL) {
                if (lane_idx < B_TOPK/8) {
                    run_outer_loop([&](const OuterloopArgs &args) {
                        int* gIndices = params.indices + args.s_q_idx*params.stride_indices_s_q;
                        CUTE_NO_UNROLL
                        for (int k = args.start_block_idx; k < args.end_block_idx; ++k) {
                            char k_validness_mask = load_indices_and_generate_mask(
                                lane_idx,
                                gIndices + k*B_TOPK,
                                params.s_kv,
                                k*B_TOPK,
                                args.topk_length
                            );

                            auto [indices_buf_idx, indices_bar_phase] = rs.get<NUM_INDEX_BUFS>();
                            smem.bar_valid_coord_scales_empty[indices_buf_idx].wait(indices_bar_phase^1);
                            smem.is_k_valid[indices_buf_idx][lane_idx] = k_validness_mask;
                            smem.bar_valid_coord_scales_full[indices_buf_idx].arrive();

                            rs.update();
                        }
                    });
                }
            } else {
                static_assert(B_TOPK == 64);
                // Each thread is responsible for 2 tokens
                static constexpr int tma_coords_step_per_token = 576/TMA_K_STRIDE_FOR_DECODING;
                int tma_coords_step_per_block = params.stride_kv_block / TMA_K_STRIDE_FOR_DECODING; // must < 2G since k_batch_stride < 1T and TMA_K_STRIDE_FOR_DECODING > 512
                int tma_coords_step_per_extra_block = params.stride_extra_kv_block / TMA_K_STRIDE_FOR_DECODING;
                uint8_t* k_scales_ptr = (uint8_t*)params.kv + params.page_block_size*(D_NOPE+2*D_ROPE);
                uint8_t* extra_k_scales_ptr = (uint8_t*)params.extra_kv + params.extra_page_block_size*(D_NOPE+2*D_ROPE);

                run_outer_loop([&](const OuterloopArgs &args) {
                    int* indices = (int*)params.indices + params.stride_indices_b*args.batch_idx + params.stride_indices_s_q*args.s_q_idx;
                    int* extra_indices = (int*)params.extra_indices + params.stride_extra_indices_b*args.batch_idx + params.stride_extra_indices_s_q*args.s_q_idx;

                    struct IsOrigBlock {};
                    struct IsExtraBlock {};
                    auto process_one_block = [&](int block_idx, auto is_extra_block_t) {
                        auto [index_buf_idx, index_bar_phase] = rs.get<NUM_INDEX_BUFS>();
                        static constexpr bool IS_EXTRA_BLOCK = std::is_same_v<decltype(is_extra_block_t), IsExtraBlock>;
                        int cur_block_size = IS_EXTRA_BLOCK ? params.extra_page_block_size : params.page_block_size;
                        int64_t cur_k_block_stride = IS_EXTRA_BLOCK ? params.stride_extra_kv_block : params.stride_kv_block;
                        [[maybe_unused]] int cur_k_row_stride = IS_EXTRA_BLOCK ? params.stride_extra_kv_row : params.stride_kv_row;
                        uint8_t* cur_k_scales_ptr = IS_EXTRA_BLOCK ? extra_k_scales_ptr : k_scales_ptr;
                        int cur_tma_coords_step_per_block = IS_EXTRA_BLOCK ? tma_coords_step_per_extra_block : tma_coords_step_per_block;

                        int abs_pos, my_indices[2];
                        if (!IS_EXTRA_BLOCK) {
                            abs_pos = block_idx*B_TOPK + lane_idx*2;
                            *(int2*)my_indices = __ldg((int2*)(indices + abs_pos));
                        } else {
                            abs_pos = (block_idx-args.num_orig_kv_blocks)*B_TOPK + lane_idx*2;
                            *(int2*)my_indices = __ldg((int2*)(extra_indices + abs_pos));
                        }
                        smem.bar_valid_coord_scales_empty[index_buf_idx].wait(index_bar_phase^1);

                        int tma_coords[2];
                        fp8_e8m0 scales[2*(NUM_SCALES_EACH_TOKEN/2)];
                        char valid_mask = 0;
                        CUTE_UNROLL
                        for (int i = 0; i < 2; ++i) {
                            int block_idx, idx_in_block;
                            block_idx = (unsigned int)my_indices[i] / cur_block_size;
                            idx_in_block = (unsigned int)my_indices[i] % cur_block_size;
                            bool is_token_valid = my_indices[i] != -1 && (abs_pos+i < (IS_EXTRA_BLOCK?args.extra_topk_length:args.topk_length));
                            valid_mask |= is_token_valid << i;
                            tma_coords[i] = is_token_valid ? block_idx*cur_tma_coords_step_per_block + idx_in_block*tma_coords_step_per_token : -1; // If the token is invalid because it topk position exceeds topk_length, we must manually fill tma_coords with -1 to avoid copying-in NaN.

                            int64_t offset = block_idx*cur_k_block_stride + (idx_in_block*8 + (cta_idx == 1 ? 4 : 0)); // Each token has 7 scale factors with an extra 1B padding
                            uint32_t scalesx4 = is_token_valid ? __ldg((uint32_t*)(cur_k_scales_ptr + offset)) : 0;
                            *(uint32_t*)(scales+i*(NUM_SCALES_EACH_TOKEN/2)) = scalesx4;
                        }
                        valid_mask <<= lane_idx%4*2;
                        valid_mask |= __shfl_xor_sync(0xFFFFFFFF, valid_mask, 0x1);
                        valid_mask |= __shfl_xor_sync(0xFFFFFFFF, valid_mask, 0x2);
                        *(uint64_t*)(smem.scales[index_buf_idx] + lane_idx*2) = *(uint64_t*)scales;
                        *(int2*)(smem.tma_coord[index_buf_idx] + lane_idx*2) = *(int2*)tma_coords;
                        if (lane_idx%4 == 0)
                            smem.is_k_valid[index_buf_idx][lane_idx/4] = valid_mask;

                        smem.bar_valid_coord_scales_full[index_buf_idx].arrive();
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
            }
        } else if (warp_idx == 10 || warp_idx == 11) {
            if constexpr (IS_PREFILL) {
                // Warp10 handles rows 0..31 and warp11 handles rows 32..63.
                // Together they form the two K=256 source halves and the
                // r/r+32 pair rows required by duplicated2by2. Each scale
                // byte is kept once; explicit b_sf_id reuses it for two
                // consecutive K=32 MMAs.
                const int scale_warp = warp_idx - 10;
                RingBufferState scale_rs;
                run_outer_loop([&](const OuterloopArgs &args) {
                    CUTE_NO_UNROLL
                    for (int k = args.start_block_idx; k < args.end_block_idx; ++k) {
                        auto [k_buf_idx, k_bar_phase] = scale_rs.get<NUM_K_BUFS>();
#if DUAL_MXFP8_K_SCALE_CP_ASYNC
                        // The final UTCCP source is ring-buffered. Wait until
                        // the previous QK MMA has released this stage before
                        // cp.async starts writing it again.
                        smem.bar_QK_done[k_buf_idx].wait(k_bar_phase ^ 1);

                        const int row = scale_warp * (B_TOPK / 2) + lane_idx;
                        const int* g_indices = params.indices
                            + args.s_q_idx * params.stride_indices_s_q
                            + k * B_TOPK;
                        const int token_idx = __ldg(g_indices + row);
                        const bool token_valid =
                            token_idx >= 0 && token_idx < params.s_kv;
                        const int safe_token_idx = token_valid ? token_idx : 0;
                        const uint8_t* k_scale_base =
                            reinterpret_cast<const uint8_t*>(params.kv)
                            + static_cast<int64_t>(params.s_kv)
                                * params.h_kv * D_K;
                        const uint32_t* src = reinterpret_cast<const uint32_t*>(
                            k_scale_base
                            + static_cast<int64_t>(safe_token_idx)
                                * K_SCALE_SLOT_BYTES
                        );

                        // Each lane owns one token. The replicated GMEM slot
                        // is [s0..s3][s4..s7][s0..s3][s4..s7]. Write those
                        // four words directly into the r/r+32 interleaving
                        // required by the duplicated2by2 UTCCP source.
                        const int token_lane = row % 32;
                        const int token_half = row / 32;
                        fp8_e8m0* dst_lo = smem.k_scale_mma[k_buf_idx].data()
                            + token_lane * 16 + token_half * 4;
                        fp8_e8m0* dst_hi = smem.k_scale_mma[k_buf_idx].data()
                            + (32 + token_lane) * 16 + token_half * 4;
                        uint32_t* dst_lo_words =
                            reinterpret_cast<uint32_t*>(dst_lo);
                        uint32_t* dst_hi_words =
                            reinterpret_cast<uint32_t*>(dst_hi);
                        using CpAsync4 =
                            cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint32_t>;
                        CpAsync4::copy(src[0], dst_lo_words[0], token_valid);
                        CpAsync4::copy(src[2], dst_lo_words[2], token_valid);
                        CpAsync4::copy(src[1], dst_hi_words[0], token_valid);
                        CpAsync4::copy(src[3], dst_hi_words[2], token_valid);
                        cute::cp_async_fence();
                        cute::cp_async_wait<0>();
                        __syncwarp();

                        // The first copied byte is also the token multiplier
                        // applied to S before its E4M3 conversion.
                        smem.v_token_scale[k_buf_idx][row] = dst_lo[0];
                        fence_view_async_shared();
                        __syncwarp();
#else
                        smem.bar_K_scale_raw_full[k_buf_idx].wait(k_bar_phase);

                        // TMA gather4 wrote complete 16B slots in token order.
                        // Read the first 8B of the selected slot; the second
                        // 8B is the producer-side replication of the same
                        // eight scale factors. Keep one byte per K=64 group:
                        // explicit b_sf_id reuses each byte across two K=32
                        // MMAs, so we must not duplicate bytes here.
                        CUTE_UNROLL
                        for (int local_row = 0;
                             local_row < B_TOPK / (2 * 32); ++local_row) {
                            const int row = scale_warp * (B_TOPK / 2)
                                + local_row * 32 + lane_idx;
                            const int raw_row = row / K_SCALE_GATHER_ROWS;
                            const int raw_slot = row % K_SCALE_GATHER_ROWS;
                            uint64_t packed = *reinterpret_cast<const uint64_t*>(
                                &smem.k_scale_pair_raw[k_buf_idx]
                                    [raw_row][raw_slot * K_SCALE_SLOT_BYTES]
                            );
                            fp8_e8m0* src = reinterpret_cast<fp8_e8m0*>(&packed);
                            fp8_e8m0* dst = smem.k_scale_token[row];
                            // K scale[0] is the per-token scale used by V.
                            // Publish it in the same ring stage consumed by
                            // the softmax/SV workers.
                            smem.v_token_scale[k_buf_idx][row] = src[0];
                            CUTE_UNROLL
                            for (int group = 0; group < K_SCALE_BYTES; ++group)
                                dst[group] = src[group];
                        }
                        fence_view_async_shared();
                        __syncwarp();

                        // Do not overwrite a ring-stage source until its prior
                        // asynchronous QK MMA has completed.
                        smem.bar_QK_done[k_buf_idx].wait(k_bar_phase ^ 1);
                        CUTE_UNROLL
                        for (int local_row = 0;
                             local_row < B_TOPK / (2 * 32); ++local_row) {
                            const int row = scale_warp * (B_TOPK / 2)
                                + local_row * 32 + lane_idx;
                            // Source rows 0..31 carry the first K=256 half;
                            // rows 32..63 carry the second half. Within each
                            // source row, words 0/2 are token r and words
                            // 1/3 are token r+32.
                            const int token_lane = row % 32;
                            const int token_half = row / 32;
                            fp8_e8m0* dst_lo = smem.k_scale_mma[k_buf_idx].data()
                                + token_lane * 16 + token_half * 4;
                            fp8_e8m0* dst_hi = smem.k_scale_mma[k_buf_idx].data()
                                + (32 + token_lane) * 16 + token_half * 4;
                            CUTE_UNROLL
                            for (int group_in_tile = 0;
                                 group_in_tile < SCALE_GROUPS_PER_TMEM_BLOCK;
                                 ++group_in_tile) {
                                dst_lo[0 + group_in_tile] = smem.k_scale_token[row][
                                    0 + group_in_tile
                                ];
                                dst_lo[8 + group_in_tile] = smem.k_scale_token[row][
                                    group_in_tile
                                ];
                                dst_hi[0 + group_in_tile] = smem.k_scale_token[row][
                                    4 + group_in_tile
                                ];
                                dst_hi[8 + group_in_tile] = smem.k_scale_token[row][
                                    4 + group_in_tile
                                ];
                            }
                        }
                        fence_view_async_shared();
                        __syncwarp();
#endif
                        if (elect_one_sync()) {
                            smem.bar_K_scale_copy_ready[k_buf_idx].arrive(uint32_t{0}, 1u);
                        }
                        scale_rs.update();
                    }
                    // The CLC producer must run inside this same outer-loop
                    // body. A second run_outer_loop after scale completion
                    // would wait for the very query it is supposed to issue.
                    if (warp_idx == 10 && elect_one_sync()) {
                        if (cta_idx == 0) {
                            smem.bar_clc_empty.wait(args.outer_loop_phase^1);
                            ku::issue_clc_query_multicast_cluster_all(smem.bar_clc_full, smem.clc_response_obj);
                        }
                        smem.bar_clc_full.arrive_and_expect_tx(sizeof(smem.clc_response_obj));
                    }
                });
            } else {
#if 0  // Decode-only legacy path; dual_mxfp8 currently instantiates prefill only.
                // Raw KV NoPE Producer thread
                run_outer_loop([&](const OuterloopArgs &args) {
                    CUTE_NO_UNROLL
                    for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
                        auto [raw_k_buf_idx, raw_k_bar_phase] = rs.get<NUM_RAW_K_BUFS>();
                        auto [index_buf_idx, index_bar_phase] = rs.get<NUM_INDEX_BUFS>();
                        smem.bar_valid_coord_scales_full[index_buf_idx].wait(index_bar_phase);
                        smem.bar_raw_KV_empty[raw_k_buf_idx].wait(raw_k_bar_phase^1);

                        int4 nxt_indices = *(int4*)(smem.tma_coord[index_buf_idx] + (warp_idx == 10 ? 0 : 4));
                        CUTE_UNROLL
                        for (int row = (warp_idx == 10 ? 0 : 4); row < B_TOPK; row += 8) {
                            int4 cur_indices = nxt_indices;
                            if (row+8 < B_TOPK)
                                nxt_indices = *(int4*)(smem.tma_coord[index_buf_idx] + row + 8);
                            ku::tma_gather4(
                                block_idx >= args.num_orig_kv_blocks ? &tma_params.tensor_map_extra_kv_nope : &tma_params.tensor_map_kv_nope,
                                smem.bar_raw_KV_full[raw_k_buf_idx],
                                smem.K_raw[raw_k_buf_idx].data() + row*(D_K/2),
                                cta_idx*(D_K/2),
                                cur_indices,
                                (int64_t)TMA::CacheHintSm90::EVICT_LAST
                            );
                        }
                        if (warp_idx == 10) {
                            smem.bar_raw_KV_full[raw_k_buf_idx].arrive_and_expect_tx(B_TOPK*(D_K/2)*sizeof(fp8_e4m3));
                        }
                        smem.bar_valid_coord_scales_empty[index_buf_idx].arrive();
                        rs.update();
                    }
                });
#endif
            }
        }
    } else {
        // Scale & Exp threads
        cutlass::arch::warpgroup_reg_alloc<176>();

        int local_warp_idx = warp_idx - 12;
        fill_sfa_from_registers<TiledMMA_O, SmemLayoutOScaleA>(
            tmem_cols::S_scale, local_warp_idx
        );
        cutlass::arch::fence_view_async_tmem_store();

        Tensor sS_store = make_tensor(
            make_smem_ptr(smem.S.data()),
            ku::make_umma_canonical_k_major_layout<
                H_Q/2, B_TOPK, 0, fp8_e4m3
            >()
        );
        int s_row = idx_in_warpgroup % (H_Q/2);
        int s_col_base = (local_warp_idx >= 2 ? B_TOPK/2 : 0);

        RingBufferState rs;
        run_outer_loop([&](const OuterloopArgs &args) {
            // For definition and consistency about `mi`, `li`, and `real_mi`, plz refer to head64 prefill
            float mi = MAX_INIT_VAL;
            float li = 0.0f;
            float real_mi = -CUDART_INF_F;
            static constexpr int NUM_ELEMS_PER_THREAD = B_TOPK / 2;

            CUTE_NO_UNROLL
            for (int k = args.start_block_idx; k < args.end_block_idx; ++k) {
                auto [k_buf_idx, k_bar_phase] = rs.get<NUM_K_BUFS>();
                auto [indices_buf_idx, indices_bar_phase] = rs.get<NUM_INDEX_BUFS>();
                auto [_, bar_phase] = rs.get<1>();
                // NOTE We don't need to sync for Prefill mode, since we have two synchronizations inside the loop body (one for p_exchange_buf sync, another one for rowwise_max_buf sync). The latter one guarantees the emptyness of p_exchange_buf and the former one guarantees the emptyness of rowwise_max_buf
                smem.bar_valid_coord_scales_full[indices_buf_idx].wait(indices_bar_phase);

                // Get P from TMEM
                float p[NUM_ELEMS_PER_THREAD];
                smem.bar_QK_done[k_buf_idx].wait(k_bar_phase);
                ku::tcgen05_after_thread_sync();
                retrieve_mask_and_reduce_p<
                    NUM_ELEMS_PER_THREAD,
                    tmem_cols::P,
                    barrier_ids::WG2_WARP02_SYNC,
                    barrier_ids::WG2_WARP13_SYNC,
                    false
                >(
                    smem.is_k_valid[indices_buf_idx],
                    local_warp_idx,
                    lane_idx,
                    [&]() {smem.bar_P_empty.arrive(0u);},
                    smem.P_exchange,
                    p
                );

                // Get rowwise max of P
                float cur_pi_max = get_max<NUM_ELEMS_PER_THREAD>(p);
                cur_pi_max *= params.sm_scale_div_log2;

                smem.rowwise_max_buf[idx_in_warpgroup] = cur_pi_max;
                NamedBarrier::arrive_and_wait(64, barrier_ids::WG2_WARP02_SYNC + (local_warp_idx&1));
                cur_pi_max = max(cur_pi_max, smem.rowwise_max_buf[idx_in_warpgroup^64]);
                real_mi = max(real_mi, cur_pi_max);
                bool should_scale_o = __any_sync(0xffffffff, cur_pi_max - mi > 6.0f);


                // Calc scale factor, and scale li
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

                // Calculate S
                fp8_e4m3 s[NUM_ELEMS_PER_THREAD];
                float cur_sum = 0.0f;
                CUTE_UNROLL
                for (int i = 0; i < NUM_ELEMS_PER_THREAD; ++i) {
                    float s_value = exp2f(fmaf(p[i], params.sm_scale_div_log2, -new_max));
                    cur_sum += s_value;
                    // Apply the selected K token's scale to all H elements in
                    // this S column before the direct E4M3 conversion. S's
                    // TMEM scale remains one, so this multiplication carries
                    // the token scale into the S operand itself.
                    const int token = s_col_base + i;
                    s[i] = fp8_e4m3(
                        s_value * float(smem.v_token_scale[k_buf_idx][token])
                    );
                }
                li = fmaf(li, scale_for_old, cur_sum);

                // Store S
                smem.bar_SV_done.wait(bar_phase^1);
                CUTE_UNROLL
                for (int i = 0; i < NUM_ELEMS_PER_THREAD/8; ++i) {
                    fp8_e4m3* sS_base = &sS_store(
                        s_row, s_col_base + i * 8
                    );
                    *reinterpret_cast<uint64_t*>(sS_base) =
                        *reinterpret_cast<uint64_t*>(s + i*8);
                }

                // Rescale O
                if (k > 0 && should_scale_o) {
                    ku::tcgen05_after_thread_sync();
                    rescale_O<D_V, 32, tmem_cols::O>(scale_for_old);
                    ku::tcgen05_before_thread_sync();
                }

                fence_view_async_shared();
                smem.bar_S_O_full.arrive(0u);
                smem.bar_valid_coord_scales_empty[indices_buf_idx].arrive();

                rs.update();
            }

            if (real_mi == -CUDART_INF_F) {
                // real_mi == -CUDART_INF_F <=> No valid TopK indices
                // We set li to 0 to fit the definition that li := exp(x[i] - mi)
                li = 0.0f;
                mi = -CUDART_INF_F;
            }

            // Reduce li
            smem.bar_li_empty.wait(args.outer_loop_phase^1);
            smem.rowwise_li_buf[idx_in_warpgroup^64] = li;
            NamedBarrier::arrive_and_wait(128, barrier_ids::WG2_SYNC);
            li += smem.rowwise_li_buf[idx_in_warpgroup];

            if (idx_in_warpgroup < H_Q/2) {
                // Calculate output_scale and save
                int head_idx = idx_in_warpgroup;
                float attn_sink = params.attn_sink == nullptr ? -CUDART_INF_F : __ldg(params.attn_sink + head_idx);
                float output_scale;
                if (FWD_MODE != FwdMode::DecodeWithSplitKV || args.is_no_split) {
                    output_scale = __fdividef(1.0f, li + exp2f(fmaf(attn_sink, CUDART_L2E_F, -mi)));
                } else {
                    output_scale = __fdividef(1.0f, li);
                }
                smem.rowwise_li_buf[idx_in_warpgroup] = li == 0.0f ? 0.0f : output_scale;
                smem.bar_li_full.arrive();

                float cur_lse = fmaf(mi, CUDART_LN2_F, logf(li));
                cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;
                if constexpr (IS_PREFILL) {
                    int q_token_idx = 2*args.s_q_idx + cta_idx;
                    int global_index = q_token_idx*params.h_q + head_idx;
                    params.max_logits[global_index] = real_mi*CUDART_LN2_F;
                    params.lse[global_index] = cur_lse;
                } else {
                    int q_token_idx = 2*args.s_q_idx + cta_idx;
                    if (q_token_idx < params.s_q) {
                        if (FWD_MODE != FwdMode::DecodeWithSplitKV || args.is_no_split) {
                            params.lse[args.batch_idx*params.stride_lse_b + q_token_idx*params.stride_lse_s_q + head_idx] = cur_lse;
                        } else {
                            float cur_lse_2base = log2f(li) + mi;
                            params.lse_accum[args.n_split_idx*params.stride_lse_accum_split + q_token_idx*params.stride_lse_accum_s_q + head_idx] = cur_lse_2base;
                        }
                    }
                }

            }
        });
    }

    ku::barrier_cluster_arrive_relaxed();
    ku::barrier_cluster_wait_acquire();

#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100");
    }
#endif
}

// We have two launchers with different kernel names to distinguish prefill and decode

template<typename Kernel>
static __global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, 2)
dual_mxfp8_sparse_attn_fwd_kernel(__grid_constant__ const typename Kernel::ArgT params, __grid_constant__ const typename Kernel::TmaParams tma_params) {
    Kernel::sparse_attn_fwd_kernel_devfunc(params, tma_params);
}

template<typename Kernel>
static __global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, 2)
flash_fwd_splitkv_mla_fp8_sparse_kernel(__grid_constant__ const typename Kernel::ArgT params, __grid_constant__ const typename Kernel::TmaParams tma_params) {
    Kernel::sparse_attn_fwd_kernel_devfunc(params, tma_params);
}

template<FwdMode FWD_MODE, int D_QK>
void KernelTemplate<FWD_MODE, D_QK>::run(const ArgT& params) {
    static_assert(D_QK == 576 || D_QK == 512);

    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk % B_TOPK == 0);   // To save some boundry checkings
    KU_ASSERT(params.h_q == TOKEN_H_Q);
    KU_ASSERT(params.s_q > 0);
    static_assert(IS_PREFILL, "dual_mxfp8/head64 currently implements the prefill path only");
    if constexpr (IS_PREFILL) {
        KU_ASSERT(params.s_q % 2 == 0);
    }
    KU_ASSERT(params.d_qk == D_QK);

    static_assert(D_Q == 512);
    if constexpr (IS_PREFILL) {
        KU_ASSERT(params.stride_q_h_q == Q_BYTES_PER_HEAD,
            "Q head stride must be 512 e4m3 bytes + an aligned 16-byte scale slot");
        KU_ASSERT(params.stride_kv_h_kv == KV_BYTES_PER_TOKEN,
            "KV token envelope must be 512 e4m3 bytes + a replicated 16-byte scale slot");
        KU_ASSERT(params.stride_kv_s_kv == params.h_kv * KV_BYTES_PER_TOKEN,
            "packed KV storage must be contiguous across logical token envelopes");
        KU_ASSERT(reinterpret_cast<int64_t>(params.q) % 16 == 0);
        KU_ASSERT(reinterpret_cast<int64_t>(params.kv) % 16 == 0);
    }

    CUtensorMap tensor_map_q;
    if constexpr (IS_DECODE) {
        KU_ASSERT(params.stride_q_b % params.stride_q_s_q == 0, "In decode mode for MODEL1 sparse fp8 decoding on sm100f, q.stride(0) (on the batch dimension) must be divisible by q.stride(1) (on the sequence dimension).");
        tensor_map_q = ku::make_tensor_map(
            {128ul, TOKEN_H_Q, 2ul, (D_Q/128ul)/2ul, (unsigned long)params.b * (params.stride_q_b / params.stride_q_s_q)},
            ku::make_stride_helper<int>({params.stride_q_h_q, D_Q/2, 128, params.stride_q_s_q}, sizeof(fp8_e4m3)),
            {128, TOKEN_H_Q, 2, (D_Q/128)/2, 1},
            params.q,
            CU_TENSOR_MAP_DATA_TYPE_UINT8,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        );
    } else {
        tensor_map_q = ku::make_tensor_map(
            {128ul, TOKEN_H_Q, 2ul, (D_Q/128ul)/2ul, (unsigned long)params.s_q},
            ku::make_stride_helper<int>({params.stride_q_h_q, D_Q/2, 128, params.stride_q_s_q}, sizeof(fp8_e4m3)),
            {128, TOKEN_H_Q, 2, (D_Q/128)/2, 1},
            params.q,
            CU_TENSOR_MAP_DATA_TYPE_UINT8,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        );  // Interleave the two 256-value halves for the dual-map Q copy.

    }

    CUtensorMap tensor_map_kv;
    [[maybe_unused]] CUtensorMap tensor_map_k_scale = {};
    CUtensorMap tensor_map_extra_kv = {};
    if constexpr (IS_DECODE) {
        auto get_kv_tensormap = [&](bool is_extra, void* k_ptr, int num_blocks, int64_t stride_kv_block, int64_t stride_kv_row) -> CUtensorMap {
            KU_ASSERT((int64_t)k_ptr % 16 == 0, "The base address of %sk_ptr (%p) must be 16B aligned for sparse fp8 attention on sm100f", is_extra?"extra_":"", k_ptr);
            KU_ASSERT(stride_kv_block % TMA_K_STRIDE_FOR_DECODING == 0, "%sk_cache.stride(0) (%ld) must be a multiple of %d. Padding might be necessary", is_extra?"extra_":"", stride_kv_block, TMA_K_STRIDE_FOR_DECODING);
            CUtensorMap tensor_map_kv = ku::make_tensor_map(
                {D_NOPE + D_ROPE*2, (uint64_t)num_blocks * (stride_kv_block/TMA_K_STRIDE_FOR_DECODING)},
                {TMA_K_STRIDE_FOR_DECODING},
                {D_K/2, 1},
                k_ptr,
                CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
                CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B
            ); 
            return tensor_map_kv;
        };
        tensor_map_kv = get_kv_tensormap(false, params.kv, params.num_blocks, params.stride_kv_block, params.stride_kv_row);
        if (params.extra_topk > 0)
            tensor_map_extra_kv = get_kv_tensormap(true, params.extra_kv, params.extra_num_blocks, params.stride_extra_kv_block, params.stride_extra_kv_row);
    } else {
        // The storage tensor is exposed as 528-byte logical envelopes, but
        // inside the packed page all 512-byte E4M3 rows are contiguous and
        // the replicated 16-byte scale slots form a separate tail plane.
        tensor_map_kv = ku::make_tensor_map(
            {D_QK, (unsigned long)params.s_kv},
            {(unsigned long)D_K},
            {128, 1},
            params.kv,
            CU_TENSOR_MAP_DATA_TYPE_UINT8,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        );
        // The scale tail contains one replicated 16B UE8M0 slot per K token.
#if !DUAL_MXFP8_K_SCALE_CP_ASYNC
        uint8_t* k_scale_base = reinterpret_cast<uint8_t*>(params.kv)
            + static_cast<int64_t>(params.s_kv) * params.h_kv * D_K;
        tensor_map_k_scale = ku::make_tensor_map(
            {static_cast<uint64_t>(K_SCALE_SLOT_BYTES),
             static_cast<uint64_t>(params.s_kv)},
            {static_cast<uint64_t>(K_SCALE_SLOT_BYTES)},
            {K_SCALE_SLOT_BYTES, 1},
            k_scale_base,
            CU_TENSOR_MAP_DATA_TYPE_UINT8,
            // Warp10/warp11 consume the gathered slots with ordinary SMEM
            // loads before building the post-transpose UTCCP source, so the
            // raw gather map must preserve linear 16B slot order.
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_L2_128B
        );
#endif
    }

    CUtensorMap tensor_map_o;
    if constexpr (IS_DECODE) {
        tensor_map_o = ku::make_tensor_map(
            {64, TOKEN_H_Q, D_V/64, (unsigned long)params.s_q, (unsigned long)params.b},
            ku::make_stride_helper<int>({params.stride_o_h_q, 64, params.stride_o_s_q, params.stride_o_b}, sizeof(bf16)),
            {64, TOKEN_H_Q, D_V/64, 1, 1},
            params.out,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        );
    } else {
        tensor_map_o = ku::make_tensor_map(
            {64, TOKEN_H_Q, D_V/64, (unsigned long)params.s_q, 1ul},
            ku::make_stride_helper<int>({D_V, 64, TOKEN_H_Q*D_V, TOKEN_H_Q*D_V}, sizeof(bf16)),
            {64, TOKEN_H_Q, D_V/64, 1, 1},
            params.out,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        );
    }


    CUtensorMap tensor_map_o_accum = {};
    if constexpr (FWD_MODE == FwdMode::DecodeWithSplitKV) {
        tensor_map_o_accum = ku::make_tensor_map(
            {32, TOKEN_H_Q, D_V/32, (unsigned long)params.s_q, (unsigned long)params.num_sm_parts + params.b},
            ku::make_stride_helper<int>({params.stride_o_accum_h_q, 32, params.stride_o_accum_s_q, params.stride_o_accum_split}, sizeof(float)),
            {32, TOKEN_H_Q, B_EPI_SPLITKV/32, 1, 1},
            params.o_accum,
            CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        );
    }

    TmaParams tma_params;
    if constexpr (IS_DECODE) {
        tma_params = {
            tensor_map_q,
            tensor_map_o,
            tensor_map_o_accum,
            tensor_map_kv,
            tensor_map_extra_kv
        };
    } else {
        tma_params = {
            tensor_map_q,
            tensor_map_kv,
            tensor_map_k_scale,
            tensor_map_o
        };
    }

    auto kernel = IS_PREFILL ? &dual_mxfp8_sparse_attn_fwd_kernel<KernelTemplate<FWD_MODE, D_QK>> : &flash_fwd_splitkv_mla_fp8_sparse_kernel<KernelTemplate<FWD_MODE, D_QK>>;
    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 grid_shape;
    if constexpr (IS_DECODE) {
        grid_shape = dim3(2*cute::ceil_div(params.s_q, 2), params.num_sm_parts, 1);
    } else {
        // grid_shape counts CTAs, while each logical query pair is one
        // two-CTA cluster.
        grid_shape = dim3(params.s_q, 1, 1);
    }

    cutlass::ClusterLaunchParams launch_params = {
        grid_shape,
        dim3(NUM_THREADS, 1, 1),
        dim3(2, 1, 1),
        smem_size,
        params.stream
    };
    KU_CUTLASS_CHECK(cutlass::launch_kernel_on_cluster(
        launch_params, (void*)kernel, params, tma_params
    ));
}

template<FwdMode FWD_MODE, int D_QK>
void run_dual_mxfp8_phase1_kernel(
        const std::conditional_t<
            is_decode_v<FWD_MODE>,
            SparseAttnDecodeParams,
            MxFp8SparseAttnFwdParams
        >& params) {
    using Kernel = KernelTemplate<FWD_MODE, D_QK>;
    Kernel::run(params);
}

}
