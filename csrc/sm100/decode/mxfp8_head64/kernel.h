#pragma once

#include "params.h"

namespace sm100::decode::mxfp8_head64 {

// D=512 MXFP8 sparse decode: e4m3 data with one UE8M0 scale per 32 values.
template<ModelType MODEL_TYPE>
void run_flash_splitkv_mla_mxfp8_sparse_kernel(const SparseAttnMxfp8DecodeParams &params);

}
