#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/home/jintao/weijia/FlashMLA
CUDA_HOME=/share/home/jintao/nvidia
DSV4=/share/home/jintao/miniconda3/envs/dsv4
TORCH_ROOT="$DSV4/lib/python3.13/site-packages/torch"
TORCH_LIB="$TORCH_ROOT/lib"
NVCC="$CUDA_HOME/bin/nvcc"
CXX=c++

INCLUDES=(
    -I"$ROOT/csrc"
    -I"$ROOT/csrc/kerutils/include"
    -I"$ROOT/csrc/sm100"
    -I"$ROOT/csrc/cutlass/include"
    -I"$ROOT/csrc/cutlass/tools/util/include"
    -I"$TORCH_ROOT/include"
    -I"$TORCH_ROOT/include/torch/csrc/api/include"
    -I"$DSV4/include/python3.13"
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

# "$CXX" \
#     "${INCLUDES[@]}" \
#     -O3 -std=c++20 -DNDEBUG -Wno-deprecated-declarations \
#     -D_GLIBCXX_USE_CXX11_ABI=1 \
#     -DTORCH_EXTENSION_NAME=mxfp8_test_ext \
#     -fPIC -c \
#     "$ROOT/tests/mxfp8_test_ext.cpp" \
#     -o /tmp/mxfp8_api_pic.o

"$NVCC" "${NVCC_FLAGS[@]}" \
    "$ROOT/csrc/sm100/decode/mxfp8_head64/instantiations/model1.cu" \
    -o /tmp/mxfp8_decode_pic.o

# "$NVCC" "${NVCC_FLAGS[@]}" \
#     -DMXFP8_PREFILL_LOAD_KV=1 \
#     "$ROOT/csrc/sm100/prefill/sparse/mxfp8_fwd/head64/instantiations/phase1_k512.cu" \
#     -o /tmp/mxfp8_prefill_pic.o

# "$NVCC" "${NVCC_FLAGS[@]}" \
#     "$ROOT/csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu" \
#     -o /tmp/mxfp8_sched_pic.o

# "$NVCC" "${NVCC_FLAGS[@]}" \
#     "$ROOT/csrc/smxx/decode/combine/combine.cu" \
#     -o /tmp/mxfp8_combine_pic.o

"$CXX" -shared \
    /tmp/mxfp8_api_pic.o \
    /tmp/mxfp8_decode_pic.o \
    /tmp/mxfp8_prefill_pic.o \
    /tmp/mxfp8_sched_pic.o \
    /tmp/mxfp8_combine_pic.o \
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
    -o /tmp/mxfp8_test_ext.so
