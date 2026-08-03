#pragma once

#include "params.h"

namespace sm100::decode::fp8_head64 {

void run_flash_splitkv_mla_fp8_sparse_kernel(
    const SparseAttnFp8DecodeParams &params
);

} // namespace sm100::decode::fp8_head64
