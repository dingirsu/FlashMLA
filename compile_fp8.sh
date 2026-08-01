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

if [[ "${FP8_DEBUG_MARKERS:-0}" == "1" ]]; then
    NVCC_FLAGS+=( -DFP8_FWD_DEBUG_MARKERS=1 )
fi

if [[ "${FP8_BARRIER_TIMING:-0}" == "1" ]]; then
    NVCC_FLAGS+=( -DFP8_FWD_BARRIER_TIMING=1 )
fi

# Diagnostic-only switches for separating KV global-memory costs from the
# rest of the producer pipeline.  They default to the numerically-correct path.
if [[ "${FP8_BENCH_KV_TOKEN_SCALE_CONST:-0}" == "1" ]]; then
    NVCC_FLAGS+=( -DFP8_FWD_BENCH_KV_TOKEN_SCALE_CONST=1 )
fi

if [[ "${FP8_BENCH_KV_DATA_CONST:-0}" == "1" ]]; then
    NVCC_FLAGS+=( -DFP8_FWD_BENCH_KV_DATA_CONST=1 )
fi

# Diagnostic-only O-rescale ablations. They intentionally do not preserve
# numerical correctness and are used only to isolate pipeline costs.
if [[ "${FP8_BENCH_O_RESCALE_SKIP_MUL:-0}" == "1" ]]; then
    NVCC_FLAGS+=( -DFP8_BENCH_O_RESCALE_SKIP_MUL=1 )
fi

if [[ "${FP8_BENCH_O_RESCALE_SKIP_TMEM:-0}" == "1" ]]; then
    NVCC_FLAGS+=( -DFP8_BENCH_O_RESCALE_SKIP_TMEM=1 )
fi

# Batch all O TMEM rescale stripes after the final SV commit. Keep this opt-in
# until functional and timing validation establish it as the release default.
if [[ "${FP8_FWD_WHOLE_O_RESCALE:-0}" == "1" ]]; then
    NVCC_FLAGS+=( -DFP8_FWD_WHOLE_O_RESCALE=1 )
fi

# Load the aligned 16-byte token-scale slot instead of a scalar byte. This is
# opt-in while NCU confirms the wider random load reduces sector pressure.
if [[ "${FP8_FWD_VECTOR_KV_SCALE_LOAD:-0}" == "1" ]]; then
    NVCC_FLAGS+=( -DFP8_FWD_VECTOR_KV_SCALE_LOAD=1 )
fi

"$CXX" \
    "${INCLUDES[@]}" \
    -O3 -std=c++20 -DNDEBUG -Wno-deprecated-declarations \
    -D_GLIBCXX_USE_CXX11_ABI=1 \
    -DTORCH_EXTENSION_NAME=fp8_test_ext \
    -fPIC -c \
    "$ROOT/tests/fp8_test_ext.cpp" \
    -o /tmp/fp8_api_pic.o

"$NVCC" "${NVCC_FLAGS[@]}" \
    "$ROOT/csrc/sm100/prefill/sparse/fp8_fwd/head64/instantiations/phase1_k512.cu" \
    -o /tmp/fp8_prefill_k512_pic.o

"$NVCC" "${NVCC_FLAGS[@]}" \
    "$ROOT/csrc/sm100/prefill/sparse/fp8_fwd/head64/instantiations/phase1_k576.cu" \
    -o /tmp/fp8_prefill_k576_pic.o

"$CXX" -shared \
    /tmp/fp8_api_pic.o \
    /tmp/fp8_prefill_k512_pic.o \
    /tmp/fp8_prefill_k576_pic.o \
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
    -o /tmp/fp8_test_ext.so
