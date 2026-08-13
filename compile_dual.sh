#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
PYTHON=${PYTHON:-"$ROOT/.venv/bin/python"}
TORCH_ROOT=${TORCH_ROOT:-$("$PYTHON" -c 'import torch; print(torch.__path__[0])')}
PYTHON_INCLUDE=${PYTHON_INCLUDE:-$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_path("include"))')}
TORCH_LIB="$TORCH_ROOT/lib"
NVCC="$CUDA_HOME/bin/nvcc"
CXX=${CXX:-c++}
BUILD_DIR=${DUAL_BUILD_DIR:-"$ROOT/build/dual_head64"}
OUTPUT=${DUAL_EXTENSION_PATH:-"$ROOT/build/dual_head64_test_ext.so"}

mkdir -p "$BUILD_DIR"

INCLUDES=(
    -I"$ROOT/csrc"
    -I"$ROOT/csrc/kerutils/include"
    -I"$ROOT/csrc/sm100"
    -I"$ROOT/csrc/cutlass/include"
    -I"$ROOT/csrc/cutlass/tools/util/include"
    -I"$TORCH_ROOT/include"
    -I"$TORCH_ROOT/include/torch/csrc/api/include"
    -I"$PYTHON_INCLUDE"
    -I"$CUDA_HOME/include"
)

NVCC_FLAGS=(
    "${INCLUDES[@]}"
    -c -O3 -std=c++20 -DNDEBUG -D_USE_MATH_DEFINES
    -D_GLIBCXX_USE_CXX11_ABI=1
    -Wno-deprecated-declarations
    -U__CUDA_NO_HALF_OPERATORS__
    -U__CUDA_NO_HALF_CONVERSIONS__
    -U__CUDA_NO_HALF2_OPERATORS__
    -U__CUDA_NO_BFLOAT16_CONVERSIONS__
    --expt-relaxed-constexpr
    --expt-extended-lambda
    --use_fast_math
    --ptxas-options=-v,--register-usage-level=10,--warn-on-spills,--warn-on-local-memory-usage
    -lineinfo
    -gencode arch=compute_100f,code=sm_100f
    --threads "${NVCC_THREADS:-16}"
    --compiler-options=-fPIC
)

"$CXX" \
    "${INCLUDES[@]}" \
    -O3 -std=c++20 -DNDEBUG -Wno-deprecated-declarations \
    -D_GLIBCXX_USE_CXX11_ABI=1 \
    -DTORCH_EXTENSION_NAME=dual_head64_test_ext \
    -fPIC -c \
    "$ROOT/csrc/sm100/prefill/sparse/dual_fwd/head64/binding.cpp" \
    -o "$BUILD_DIR/binding.o"

for dim in 512 576; do
    "$NVCC" "${NVCC_FLAGS[@]}" \
        "$ROOT/csrc/sm100/prefill/sparse/dual_fwd/head64/instantiations/phase1_k${dim}.cu" \
        -o "$BUILD_DIR/phase1_k${dim}.o"
done

"$CXX" -shared \
    "$BUILD_DIR/binding.o" \
    "$BUILD_DIR/phase1_k512.o" \
    "$BUILD_DIR/phase1_k576.o" \
    -L"$TORCH_LIB" -Wl,-rpath,"$TORCH_LIB" \
    -ltorch_python -ltorch_cuda -ltorch_cpu -ltorch -lc10_cuda -lc10 \
    -L"$CUDA_HOME/lib64" -Wl,-rpath,"$CUDA_HOME/lib64" -lcudart \
    -o "$OUTPUT"

echo "Built $OUTPUT"
