#pragma once

#include <cute/tensor.hpp>

#include "kerutils/device/common.h"

namespace kerutils {

// Perform SS UTCMMA
// sA and sB should be shared memory tensors (i.e. make_tensor(make_shared_ptr(XXX), XXX)) while tC_frag should be tmem fragment
template<
    typename TiledMMA,
    typename TensorA,
    typename TensorB,
    typename TensorFragC
>
CUTE_DEVICE
void utcmma_ss(
    TiledMMA &tiled_mma,
    TensorA sA,
    TensorB sB,
    TensorFragC tC_frag,
    bool clear_accum
) {
    using namespace cute;
    tiled_mma.accumulate_ = clear_accum ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
    ThrMMA thr_mma = tiled_mma.get_slice(_0{}); // Since A/B/C are already CTA-local tiles, this number does not matter
    auto sA_frag = thr_mma.partition_fragment_A(sA);
    auto sB_frag = thr_mma.partition_fragment_B(sB);
    static_assert(size<2>(sA_frag) == size<2>(sB_frag));
    static_assert(size<1>(sA_frag) == size<1>(tC_frag));
    static_assert(size<1>(sB_frag) == size<2>(tC_frag));
    CUTE_UNROLL
    for (int k = 0; k < size<2>(sA_frag); ++k) {
        cute::gemm(
            tiled_mma,
            sA_frag(_, _, k),
            sB_frag(_, _, k),
            tC_frag
        );
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }
}

template<
    typename TiledMMA,
    typename TensorA,
    typename TensorB,
    typename TensorSFA,
    typename TensorSFB,
    typename TensorFragC
>
CUTE_DEVICE
void utcmma_blockscaled_ss(
    TiledMMA &tiled_mma,
    TensorA sA,
    TensorB sB,
    TensorSFA tSFA_frag,
    TensorSFB tSFB_frag,
    TensorFragC tC_frag,
    bool clear_accum
) {
    using namespace cute;
    tiled_mma.accumulate_ = clear_accum ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
    ThrMMA thr_mma = tiled_mma.get_slice(_0{}); // Since A/B/C are already CTA-local tiles, this number does not matter
    auto sA_frag = thr_mma.partition_fragment_A(sA);
    auto sB_frag = thr_mma.partition_fragment_B(sB);
    static_assert(size<2>(sA_frag) == size<2>(sB_frag));
    static_assert(size<1>(sA_frag) == size<1>(tC_frag));
    static_assert(size<1>(sB_frag) == size<2>(tC_frag));
    CUTE_UNROLL
    for (int k = 0; k < size<2>(sA_frag); ++k) {
        auto tiled_mma_with_scale = tiled_mma.with(
            tiled_mma.accumulate_,
            tSFA_frag(_, _, k),
            tSFB_frag(_, _, k)
        );
        cute::gemm(
            tiled_mma_with_scale,
            sA_frag(_, _, k),
            sB_frag(_, _, k),
            tC_frag
        );
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }
}

// Perform TS UTCMMA
// sB should be shared memory tensors (i.e. make_tensor(make_shared_ptr(XXX), XXX)) while tA_frag and tC_frag should be tmem fragment
template<
    typename TiledMMA,
    typename TensorA,
    typename TensorB,
    typename TensorFragC
>
CUTE_DEVICE
void utcmma_ts(
    TiledMMA &tiled_mma,
    TensorA tA_frag,
    TensorB sB,
    TensorFragC tC_frag,
    bool clear_accum
) {
    using namespace cute;
    tiled_mma.accumulate_ = clear_accum ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
    ThrMMA thr_mma = tiled_mma.get_slice(_0{}); // Since A/B/C are already CTA-local tiles, this number does not matter
    auto sB_frag = thr_mma.partition_fragment_B(sB);
    static_assert(size<2>(tA_frag) == size<2>(sB_frag));
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tA_frag); ++k) {
        cute::gemm(
            tiled_mma,
            tA_frag(_, _, k),
            sB_frag(_, _, k),
            tC_frag
        );
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }
}

template<
    typename TiledMMA,
    typename TensorA,
    typename TensorB,
    typename TensorSFA,
    typename TensorSFB,
    typename TensorFragC
>
CUTE_DEVICE
void utcmma_blockscaled_ts(
    TiledMMA &tiled_mma,
    TensorA tA_frag,
    TensorB sB,
    TensorSFA tSFA_frag,
    TensorSFB tSFB_frag,
    TensorFragC tC_frag,
    bool clear_accum
) {
    using namespace cute;
    tiled_mma.accumulate_ = clear_accum ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
    ThrMMA thr_mma = tiled_mma.get_slice(_0{}); // Since A/B/C are already CTA-local tiles, this number does not matter
    auto sB_frag = thr_mma.partition_fragment_B(sB);
    static_assert(size<2>(tA_frag) == size<2>(sB_frag));
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tA_frag); ++k) {
        auto tiled_mma_with_scale = tiled_mma.with(
            tiled_mma.accumulate_,
            tSFA_frag(_, _, k),
            tSFB_frag(_, _, k)
        );
        cute::gemm(
            tiled_mma_with_scale,
            tA_frag(_, _, k),
            sB_frag(_, _, k),
            tC_frag
        );
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }
}

// Perform TS UTCMMA with explicit UE8M0 byte selectors.  This mirrors
// DeepGEMM's SM100 block-scaled loop: each K=32 instruction selects one of the
// four scale bytes packed in the current TMEM word, and the fragment advances
// to the next word after four instructions.
template<
    typename TiledMMA,
    typename TensorA,
    typename TensorB,
    typename TensorSFA,
    typename TensorSFB,
    typename TensorFragC
>
CUTE_DEVICE
void utcmma_blockscaled_ts_explicit_sf_ids(
    TiledMMA &tiled_mma,
    TensorA tA_frag,
    TensorB sB,
    TensorSFA tSFA_frag,
    TensorSFB tSFB_frag,
    TensorFragC tC_frag,
    bool clear_accum
) {
    using namespace cute;
    tiled_mma.accumulate_ = clear_accum ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
    ThrMMA thr_mma = tiled_mma.get_slice(_0{});
    auto sB_frag = thr_mma.partition_fragment_B(sB);
    static_assert(size<2>(tA_frag) == size<2>(sB_frag));
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tA_frag); ++k) {
        uint32_t sf_id = uint32_t(k) & 3u;
        // Keep the address fixed for four K=32 instructions and rotate only
        // the scale-factor ID. UTCCP destinations are four-column aligned, so
        // each K=128 scale tile occupies the next four-column slot for both
        // operands even when the compact SFB fragment aliases fewer columns.
        // Keep the M128/2x2 fragment's physical TMEM column stride.  In this
        // mode the SFB fragment advances by two TMEM columns per K=128 tile;
        // flattening it to a four-column stride aliases the second datapath.
        auto tSFA = tSFA_frag(_, _, k);
        auto tSFB = tSFB_frag(_, _, k);
        tSFA.data().get() = raw_pointer_cast(tSFA.data());
        tSFB.data().get() = raw_pointer_cast(tSFB.data());
        auto tiled_mma_with_scale = tiled_mma.with(
            tiled_mma.accumulate_,
            tSFA,
            tSFB,
            sf_id,
            sf_id
        );
        cute::gemm(
            tiled_mma_with_scale,
            tA_frag(_, _, k),
            sB_frag(_, _, k),
            tC_frag
        );
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }
}

template<int MN, int K, int SWIZZLE, typename T = bf16>
static constexpr auto make_umma_canonical_k_major_layout() {
    using namespace cute;
    using base_atom_type = \
        std::conditional_t<SWIZZLE == 0 || SWIZZLE == 16, 
            UMMA::Layout_K_INTER_Atom<T>,
            std::conditional_t<SWIZZLE == 32,
                UMMA::Layout_K_SW32_Atom<T>,
                std::conditional_t<SWIZZLE == 64,
                    UMMA::Layout_K_SW64_Atom<T>,
                    std::conditional_t<SWIZZLE == 128,
                        UMMA::Layout_K_SW128_Atom<T>,
                        void
                    >
                >
            >
        >;
    static_assert(!std::is_same_v<base_atom_type, void>, "Invalid SWIZZLE value");
    return coalesce(tile_to_shape(
        base_atom_type{},
        Shape<Int<MN>, Int<K>>{},
        Step<_1, _2>{}
    ), Shape<_1, _1>{});
}

template<int MN, int K, int SWIZZLE, typename T = bf16>
static constexpr auto make_umma_canonical_mn_major_layout() {
    using namespace cute;
    using base_atom_type = \
        std::conditional_t<SWIZZLE == 0 || SWIZZLE == 16, 
            UMMA::Layout_MN_INTER_Atom<T>,
            std::conditional_t<SWIZZLE == 32,
                UMMA::Layout_MN_SW32_Atom<T>,
                std::conditional_t<SWIZZLE == 64,
                    UMMA::Layout_MN_SW64_Atom<T>,
                    std::conditional_t<SWIZZLE == 128,
                        UMMA::Layout_MN_SW128_Atom<T>,
                        void
                    >
                >
            >
        >;
    static_assert(!std::is_same_v<base_atom_type, void>, "Invalid SWIZZLE value");
    return coalesce(tile_to_shape(
        base_atom_type{},
        Shape<Int<MN>, Int<K>>{},
        Step<_2, _1>{}
    ), Shape<_1, _1>{});
}

template<cute::UMMA::Major MAJOR, int MN, int K, int SWIZZLE, typename T = bf16>
auto make_umma_canonical_layout() {
    if constexpr (MAJOR == cute::UMMA::Major::K) {
        return make_umma_canonical_k_major_layout<MN, K, SWIZZLE, T>();
    } else {
        return make_umma_canonical_mn_major_layout<MN, K, SWIZZLE, T>();
    }
}

}
