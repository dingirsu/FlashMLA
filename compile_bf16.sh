#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CUDA_HOME=/usr/local/cuda
PYTHON="$ROOT/.venv/bin/python"
TORCH_ROOT=${TORCH_ROOT:-$("$PYTHON" -c 'import torch; print(torch.__path__[0])')}
PYTHON_INCLUDE=${PYTHON_INCLUDE:-$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_path("include"))')}
TORCH_LIB="$TORCH_ROOT/lib"
NVCC="$CUDA_HOME/bin/nvcc"
CXX=${CXX:-c++}

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
    -c
    -O3
    -std=c++20
    -DNDEBUG
    -D_USE_MATH_DEFINES
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
    -gencode
    arch=compute_100f,code=sm_100f
    --threads
    16
    --compiler-options=-fPIC
)

if [[ "${BF16_BARRIER_TIMING:-0}" == "1" ]]; then
    NVCC_FLAGS+=( -DBF16_FWD_BARRIER_TIMING=1 )
fi

"$CXX" \
    "${INCLUDES[@]}" \
    -O3 -std=c++20 -DNDEBUG -Wno-deprecated-declarations \
    -D_GLIBCXX_USE_CXX11_ABI=1 \
    -DTORCH_EXTENSION_NAME=bf16_test_ext \
    -fPIC -c \
    "$ROOT/tests/bf16_test_ext.cpp" \
    -o /tmp/bf16_api_pic.o

"$NVCC" "${NVCC_FLAGS[@]}" \
    "$ROOT/csrc/sm100/prefill/sparse/fwd/head64/instantiations/phase1_k512.cu" \
    -o /tmp/bf16_prefill_pic.o

"$CXX" -shared \
    /tmp/bf16_api_pic.o \
    /tmp/bf16_prefill_pic.o \
    -L"$TORCH_LIB" \
    -Wl,-rpath,"$TORCH_LIB" \
    -ltorch_python \
    -ltorch_cuda \
    -ltorch_cpu \
    -ltorch \
    -lc10_cuda \
    -lc10 \
    -L"$CUDA_HOME/lib64" \
    -Wl,-rpath,"$CUDA_HOME/lib64" \
    -lcudart \
    -o /tmp/bf16_test_ext.so
