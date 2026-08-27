#include <cstdio>
#include <cute/tensor.hpp>
#include <kerutils/device/sm100/helpers.cuh>
#include <kerutils/device/sm100/gemm.cuh>

int main() {
  using namespace cute;
  using TiledMMA = decltype(make_tiled_mma(
      SM100_MMA_MXF8F6F4_2x1SM_TS_NOELECT<
          cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
          cutlass::float_ue8m0_t, 128, 256,
          UMMA::Major::K, UMMA::Major::K>{}));
  using TiledMMASS = decltype(make_tiled_mma(
      SM100_MMA_MXF8F6F4_2x1SM_SS_NOELECT<
          cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
          cutlass::float_ue8m0_t, 128, 128,
          UMMA::Major::K, UMMA::Major::K>{}));
  TiledMMA tiled_mma;
  auto part_shape_64x256 = partition_shape_A(tiled_mma, Shape<_64, _256>{});
  auto part_shape_128x256 = partition_shape_A(tiled_mma, Shape<_128, _256>{});
  auto part_shape_256x256 = partition_shape_A(tiled_mma, Shape<_256, _256>{});
  auto tQ_64x256 = tiled_mma.get_slice(_0{}).make_fragment_A(part_shape_64x256);
  auto tQ_128x256 = tiled_mma.get_slice(_0{}).make_fragment_A(part_shape_128x256);
  auto tQ_256x256 = tiled_mma.get_slice(_0{}).make_fragment_A(part_shape_256x256);
  auto tP_64x128 = partition_fragment_C(tiled_mma, Shape<_64, _128>{});
  auto tP_128x128 = partition_fragment_C(tiled_mma, Shape<_128, _128>{});
  auto tP_256x128 = partition_fragment_C(tiled_mma, Shape<_256, _128>{});
  print("part_shape_A(64x256):  "); print(part_shape_64x256);
  print("\ntQ(64x256):             "); print(tQ_64x256.layout());
  print("\npart_shape_A(128x256): "); print(part_shape_128x256);
  print("\ntQ(128x256):            "); print(tQ_128x256.layout()); print("\n");
  print("part_shape_A(256x256): "); print(part_shape_256x256);
  print("\ntQ(256x256):            "); print(tQ_256x256.layout());
  print("\ntP(64x128):             "); print(tP_64x128.layout());
  print("\ntP(128x128):            "); print(tP_128x128.layout());
  print("\ntP(256x128):            "); print(tP_256x128.layout()); print("\n");
  auto physical = kerutils::make_umma_canonical_k_major_layout<64, 512, 128, cutlass::float_e4m3_t>();
  auto logical = kerutils::make_umma_canonical_k_major_layout<128, 256, 128, cutlass::float_e4m3_t>();
  auto alternate = coalesce(tile_to_shape(
      UMMA::Layout_K_SW128_Atom<cutlass::float_e4m3_t>{},
      Shape<_128, _256>{}, Step<_2, _1>{}), Shape<_1, _1>{});
  auto reshape = composition(
      physical, Layout<Shape<_128, _256>, Stride<_256, _1>>{});
  auto q_global_to_physical = Layout<
      Shape<Shape<_64, _2, _2>, _256>,
      Stride<Stride<_1, _0, Int<64 * 256>>, _64>>{};
  auto ss_forged = composition(physical, q_global_to_physical);
  TiledMMASS tiled_mma_ss;
  auto sQ_forged = make_tensor(
      make_smem_ptr(static_cast<cutlass::float_e4m3_t*>(nullptr)), ss_forged);
  auto sQ_forged_frag = tiled_mma_ss.get_slice(_0{}).partition_fragment_A(sQ_forged);
  print("physical:  "); print(physical); print("\nlogical:   "); print(logical);
  print("\nalternate: "); print(alternate); print("\nreshape:   "); print(reshape); print("\n");
  print("ss_forged: "); print(ss_forged);
  print("\nss_forged_frag: "); print(sQ_forged_frag.layout()); print("\n");
  for (int half = 0; half < 2; ++half) {
    for (int head : {0, 1, 8, 63}) {
      for (int k : {0, 16, 32, 128, 255}) {
        int p = physical(make_coord(head, k + half * 256));
        int l = logical(make_coord(head + half * 64, k));
        int a = alternate(make_coord(head + half * 64, k));
        int r = reshape(make_coord(head * 2 + half, k));
        std::printf("half=%d head=%d k=%d physical=%d logical=%d alternate=%d reshape=%d\n", half, head, k, p, l, a, r);
      }
    }
  }
}
