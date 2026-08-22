#!/usr/bin/env bash
set -euo pipefail

# Build the standalone SM100 test extensions.
# Usage: ./compile_sm100.sh [all|bf16|dual|dual_mxfp8|decode_head64]

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
PYTHON=${PYTHON:-"$ROOT/.venv/bin/python"}
TORCH_ROOT=${TORCH_ROOT:-$("$PYTHON" -c 'import torch; print(torch.__path__[0])')}
PYTHON_INCLUDE=${PYTHON_INCLUDE:-$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_path("include"))')}
TORCH_LIB="$TORCH_ROOT/lib"
NVCC="$CUDA_HOME/bin/nvcc"
CXX=${CXX:-c++}
BUILD_ROOT=${SM100_BUILD_DIR:-"$ROOT/build"}

mkdir -p "$BUILD_ROOT"

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

COMMON=(
    "${INCLUDES[@]}" -O3 -std=c++20 -DNDEBUG -D_USE_MATH_DEFINES
    -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-deprecated-declarations
    -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__
    -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__
    --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math
    --compiler-options=-fPIC
)

link_extension() {
    local output="$1"
    shift
    "$CXX" -shared "$@" \
        -L"$TORCH_LIB" -Wl,-rpath,"$TORCH_LIB" \
        -ltorch_python -ltorch_cuda -ltorch_cpu -ltorch \
        -lc10_cuda -lc10 -L"$CUDA_HOME/lib64" \
        -Wl,-rpath,"$CUDA_HOME/lib64" -lcudart -o "$output"
}

build_bf16() {
    local dir="$BUILD_ROOT/bf16"
    local output="${BF16_EXTENSION_OUTPUT:-$BUILD_ROOT/bf16_test_ext.so}"
    mkdir -p "$dir"
    "$CXX" "${INCLUDES[@]}" -O3 -std=c++20 -DNDEBUG \
        -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-deprecated-declarations \
        -DTORCH_EXTENSION_NAME=bf16_test_ext -fPIC -c \
        "$ROOT/tests/bf16_test_ext.cpp" -o "$dir/api.o"
    local nvcc_flags=("${COMMON[@]}" -c --ptxas-options=-v,--register-usage-level=10,--warn-on-spills,--warn-on-local-memory-usage -lineinfo -gencode arch=compute_100f,code=sm_100f --threads "${NVCC_THREADS:-16}")
    if [[ "${BF16_BARRIER_TIMING:-0}" == "1" ]]; then nvcc_flags+=( -DBF16_FWD_BARRIER_TIMING=1 ); fi
    "$NVCC" "${nvcc_flags[@]}" "$ROOT/csrc/sm100/prefill/sparse/fwd/head64/instantiations/phase1_k512.cu" -o "$dir/prefill.o"
    link_extension "$output" "$dir/api.o" "$dir/prefill.o"
    echo "built $output"
}

build_dual() {
    local dir="$BUILD_ROOT/dual_head64"
    local output="${DUAL_EXTENSION_PATH:-$BUILD_ROOT/dual_head64_test_ext.so}"
    mkdir -p "$dir"
    "$CXX" "${INCLUDES[@]}" -O3 -std=c++20 -DNDEBUG -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-deprecated-declarations -DTORCH_EXTENSION_NAME=dual_head64_test_ext -fPIC -c "$ROOT/csrc/sm100/prefill/sparse/dual_fwd/head64/binding.cpp" -o "$dir/binding.o"
    local nvcc_flags=("${COMMON[@]}" -c --ptxas-options=-v,--register-usage-level=10,--warn-on-spills,--warn-on-local-memory-usage -lineinfo -gencode arch=compute_100f,code=sm_100f --threads "${NVCC_THREADS:-16}")
    for dim in 512 576; do
        "$NVCC" "${nvcc_flags[@]}" "$ROOT/csrc/sm100/prefill/sparse/dual_fwd/head64/instantiations/phase1_k${dim}.cu" -o "$dir/phase1_k${dim}.o"
    done
    link_extension "$output" "$dir/binding.o" "$dir/phase1_k512.o" "$dir/phase1_k576.o"
    echo "built $output"
}

build_dual_mxfp8() {
    local dir="$BUILD_ROOT/dual_mxfp8"
    local output="${DUAL_MXFP8_EXTENSION_PATH:-$BUILD_ROOT/dual_mxfp8_test_ext.so}"
    mkdir -p "$dir"
    "$CXX" "${INCLUDES[@]}" -O3 -std=c++20 -DNDEBUG -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-deprecated-declarations -DTORCH_EXTENSION_NAME=dual_mxfp8_test_ext -fPIC -c "$ROOT/csrc/sm100/prefill/sparse/dual_mxfp8/head64/binding.cpp" -o "$dir/binding.o"
    local nvcc_flags=("${COMMON[@]}" -c --ptxas-options=-v,--register-usage-level=10,--warn-on-spills,--warn-on-local-memory-usage -lineinfo -gencode arch=compute_100f,code=sm_100f --threads "${NVCC_THREADS:-16}")
    if [[ "${DUAL_MXFP8_K_SCALE_CP_ASYNC:-0}" == "1" ]]; then
        nvcc_flags+=( -DDUAL_MXFP8_K_SCALE_CP_ASYNC=1 )
    fi
    "$NVCC" "${nvcc_flags[@]}" "$ROOT/csrc/sm100/prefill/sparse/dual_mxfp8/head64/instantiations/phase1_k512.cu" -o "$dir/phase1_k512.o"
    "$NVCC" "${nvcc_flags[@]}" "$ROOT/csrc/sm100/prefill/sparse/dual_mxfp8/head64/instantiations/phase1_decode_k512.cu" -o "$dir/phase1_decode_k512.o"
    "$NVCC" "${nvcc_flags[@]}" "$ROOT/csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu" -o "$dir/get_decoding_sched_meta.o"
    "$NVCC" "${nvcc_flags[@]}" "$ROOT/csrc/smxx/decode/combine/combine.cu" -o "$dir/combine.o"
    link_extension "$output" "$dir/binding.o" "$dir/phase1_k512.o" "$dir/phase1_decode_k512.o" "$dir/get_decoding_sched_meta.o" "$dir/combine.o"
    echo "built $output"
}

build_decode_head64() {
    local dir="$BUILD_ROOT/head64_decode"
    local output="${HEAD64_DECODE_EXT:-$BUILD_ROOT/head64_decode_test_ext.so}"
    mkdir -p "$dir"
    "$CXX" "${INCLUDES[@]}" -O3 -std=c++20 -DNDEBUG -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-deprecated-declarations -DTORCH_EXTENSION_NAME=head64_decode_test_ext -fPIC -c "$ROOT/tests/head64_decode_test_ext.cpp" -o "$dir/api.o"
    local nvcc_flags=("${COMMON[@]}" -c --ptxas-options=-v,--register-usage-level=10,--warn-on-spills -lineinfo -gencode arch=compute_100f,code=sm_100f --threads "${NVCC_THREADS:-16}")
    if [[ "${DECODE_HEAD64_BARRIER_TIMING:-0}" == "1" ]]; then nvcc_flags+=( -DDECODE_HEAD64_BARRIER_TIMING=1 ); fi
    if [[ "${DECODE_HEAD64_TIMING_PRINT:-0}" == "1" ]]; then nvcc_flags+=( -DDECODE_HEAD64_TIMING_PRINT=1 ); fi
    if [[ "${DECODE_HEAD64_TIMING_FINAL_SYNC:-0}" == "1" ]]; then nvcc_flags+=( -DDECODE_HEAD64_TIMING_FINAL_SYNC=1 ); fi
    "$NVCC" "${nvcc_flags[@]}" "$ROOT/csrc/sm100/decode/head64/instantiations/model1.cu" -o "$dir/kernel.o"
    link_extension "$output" "$dir/api.o" "$dir/kernel.o"
    echo "built $output"
}

case "${1:-all}" in
    all) build_bf16; build_dual; build_dual_mxfp8; build_decode_head64 ;;
    bf16) build_bf16 ;;
    dual) build_dual ;;
    dual_mxfp8) build_dual_mxfp8 ;;
    decode_head64) build_decode_head64 ;;
    *) echo "usage: $0 [all|bf16|dual|dual_mxfp8|decode_head64]" >&2; exit 2 ;;
esac
