#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
PYTHON=${PYTHON:-"$ROOT/.venv/bin/python"}
TORCH_ROOT="$($PYTHON -c 'import torch; print(torch.__path__[0])')"
PYTHON_INCLUDE="$($PYTHON -c 'import sysconfig; print(sysconfig.get_path("include"))')"
TORCH_LIB="$TORCH_ROOT/lib"
NVCC="$CUDA_HOME/bin/nvcc"
CXX=${CXX:-c++}

INCLUDES=(
    -I"$ROOT/csrc" -I"$ROOT/csrc/kerutils/include" -I"$ROOT/csrc/sm100"
    -I"$ROOT/csrc/cutlass/include" -I"$ROOT/csrc/cutlass/tools/util/include"
    -I"$TORCH_ROOT/include" -I"$TORCH_ROOT/include/torch/csrc/api/include"
    -I"$PYTHON_INCLUDE" -I"$CUDA_HOME/include"
)
COMMON=("${INCLUDES[@]}" -O3 -std=c++20 -DNDEBUG -D_USE_MATH_DEFINES
    -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-deprecated-declarations
    -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__
    -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__
    --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math
    --compiler-options=-fPIC)
NVCC_FLAGS=("${COMMON[@]}" -c --ptxas-options=-v,--register-usage-level=10,--warn-on-spills
    -lineinfo -gencode arch=compute_100f,code=sm_100f --threads "${NVCC_THREADS:-16}")
if [[ "${DECODE_HEAD64_BARRIER_TIMING:-0}" == "1" ]]; then
    NVCC_FLAGS+=( -DDECODE_HEAD64_BARRIER_TIMING=1 )
fi
if [[ "${DECODE_HEAD64_TIMING_PRINT:-0}" == "1" ]]; then
    NVCC_FLAGS+=( -DDECODE_HEAD64_TIMING_PRINT=1 )
fi
if [[ "${DECODE_HEAD64_TIMING_FINAL_SYNC:-0}" == "1" ]]; then
    NVCC_FLAGS+=( -DDECODE_HEAD64_TIMING_FINAL_SYNC=1 )
fi

"$CXX" "${INCLUDES[@]}" -O3 -std=c++20 -DNDEBUG -D_GLIBCXX_USE_CXX11_ABI=1 \
    -DTORCH_EXTENSION_NAME=head64_decode_test_ext -fPIC -c \
    "$ROOT/tests/head64_decode_test_ext.cpp" -o /tmp/head64_decode_api.o
"$NVCC" "${NVCC_FLAGS[@]}" "$ROOT/csrc/sm100/decode/head64/instantiations/model1.cu" -o /tmp/head64_decode_kernel.o

"$CXX" -shared /tmp/head64_decode_api.o /tmp/head64_decode_kernel.o \
    -L"$TORCH_LIB" -Wl,-rpath,"$TORCH_LIB" -ltorch_python -ltorch_cuda -ltorch_cpu -ltorch \
    -lc10_cuda -lc10 -L"$CUDA_HOME/lib64" -Wl,-rpath,"$CUDA_HOME/lib64" -lcudart \
    -o /tmp/head64_decode_test_ext.so
echo "built /tmp/head64_decode_test_ext.so"
