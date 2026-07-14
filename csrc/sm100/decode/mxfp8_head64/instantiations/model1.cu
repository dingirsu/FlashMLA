#include "../kernel.cuh"

namespace sm100::decode::mxfp8_head64 {

template
void run_flash_splitkv_mla_mxfp8_sparse_kernel<ModelType::MODEL1>(const SparseAttnMxfp8DecodeParams &params);

}
