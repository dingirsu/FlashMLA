#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CUDA_HOME=/usr/local/cuda
PYTHON="$ROOT/.venv/bin/python"
TORCH_ROOT="$("$PYTHON" -c 'import torch; print(torch.__path__[0])')"
PYTHON_INCLUDE="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
TORCH_LIB="$TORCH_ROOT/lib"
NVCC="$CUDA_HOME/bin/nvcc"
CXX=c++
BUILD_DIR=${FP8_BUILD_DIR:-"$ROOT/build"}

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

if [[ "${FP8_TIMING_PRINT:-0}" == "1" ]]; then
    NVCC_FLAGS+=( -DFP8_FWD_BARRIER_TIMING=1 -DFP8_FWD_TIMING_PRINT=1 )
fi

if [[ "${FP8_TIMING_FINAL_SYNC:-0}" == "1" ]]; then
    NVCC_FLAGS+=( -DFP8_FWD_BARRIER_TIMING=1 -DFP8_FWD_TIMING_FINAL_SYNC=1 )
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

# Orthogonal instruction-family ablations.  These are benchmark-only and do
# not preserve numerical results.
if [[ "${FP8_DISABLE_GEMM:-0}" == "1" ]]; then
    NVCC_FLAGS+=( -DFP8_FWD_DISABLE_GEMM=1 )
fi

if [[ "${FP8_DISABLE_SFU:-0}" == "1" ]]; then
    NVCC_FLAGS+=( -DFP8_FWD_DISABLE_SFU=1 )
fi

if [[ "${FP8_DISABLE_NON_SFU_NON_GEMM:-0}" == "1" ]]; then
    NVCC_FLAGS+=( -DFP8_FWD_DISABLE_NON_SFU_NON_GEMM=1 )
fi

# Independent TMA ablations.  These are benchmark-only and do not preserve
# numerical results.
if [[ "${FP8_DISABLE_Q_TMA_LOAD:-0}" == "1" ]]; then
    NVCC_FLAGS+=( -DFP8_FWD_DISABLE_Q_TMA_LOAD=1 )
fi

if [[ "${FP8_DISABLE_KV_TMA_GATHER:-0}" == "1" ]]; then
    NVCC_FLAGS+=( -DFP8_FWD_DISABLE_KV_TMA_GATHER=1 )
fi

if [[ "${FP8_DISABLE_O_TMA_STORE:-0}" == "1" ]]; then
    NVCC_FLAGS+=( -DFP8_FWD_DISABLE_O_TMA_STORE=1 )
fi

if [[ "${FP8_COMPILE_PREFILL_ONLY:-0}" == "1" ]]; then
    FP8_PREFILL_OUT_DIR="${FP8_PREFILL_OUT_DIR:-$BUILD_DIR}"
    FP8_PREFILL_D_QK="${FP8_PREFILL_D_QK:-all}"
    mkdir -p "$FP8_PREFILL_OUT_DIR"
    if [[ "$FP8_PREFILL_D_QK" == "all" || "$FP8_PREFILL_D_QK" == "512" ]]; then
        "$NVCC" "${NVCC_FLAGS[@]}" \
            "$ROOT/csrc/sm100/prefill/sparse/fp8_fwd/head64/instantiations/phase1_k512.cu" \
            -o "$FP8_PREFILL_OUT_DIR/fp8_prefill_k512_pic.o"
    fi
    if [[ "$FP8_PREFILL_D_QK" == "all" || "$FP8_PREFILL_D_QK" == "576" ]]; then
        "$NVCC" "${NVCC_FLAGS[@]}" \
            "$ROOT/csrc/sm100/prefill/sparse/fp8_fwd/head64/instantiations/phase1_k576.cu" \
            -o "$FP8_PREFILL_OUT_DIR/fp8_prefill_k576_pic.o"
    fi
    exit 0
fi

# "$CXX" \
#     "${INCLUDES[@]}" \
#     -O3 -std=c++20 -DNDEBUG -Wno-deprecated-declarations \
#     -D_GLIBCXX_USE_CXX11_ABI=1 \
#     -DTORCH_EXTENSION_NAME=fp8_test_ext \
#     -fPIC -c \
#     "$ROOT/tests/fp8_test_ext.cpp" \
#     -o "$BUILD_DIR/fp8_api_pic.o"

# "$NVCC" "${NVCC_FLAGS[@]}" \
#     "$ROOT/csrc/sm100/prefill/sparse/fp8_fwd/head64/instantiations/phase1_k512.cu" \
#     -o "$BUILD_DIR/fp8_prefill_k512_pic.o"

# "$NVCC" "${NVCC_FLAGS[@]}" \
#     "$ROOT/csrc/sm100/prefill/sparse/fp8_fwd/head64/instantiations/phase1_k576.cu" \
#     -o "$BUILD_DIR/fp8_prefill_k576_pic.o"

FP8_DECODE_OBJECT="${FP8_DECODE_OBJECT:-$BUILD_DIR/fp8_decode_pic.o}"
"$NVCC" "${NVCC_FLAGS[@]}" \
    "$ROOT/csrc/sm100/decode/fp8_head64/instantiations/model1.cu" \
    -o "$FP8_DECODE_OBJECT"

if [[ "${FP8_COMPILE_DECODE_ONLY:-0}" == "1" ]]; then
    exit 0
fi

# "$NVCC" "${NVCC_FLAGS[@]}" \
#     "$ROOT/csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu" \
#     -o "$BUILD_DIR/fp8_sched_pic.o"

# "$NVCC" "${NVCC_FLAGS[@]}" \
#     "$ROOT/csrc/smxx/decode/combine/combine.cu" \
#     -o "$BUILD_DIR/fp8_combine_pic.o"

FP8_EXTENSION_OUTPUT="${FP8_EXTENSION_OUTPUT:-$BUILD_DIR/fp8_test_ext.so}"
"$CXX" -shared \
    "$BUILD_DIR/fp8_api_pic.o" \
    "$BUILD_DIR/fp8_prefill_k512_pic.o" \
    "$BUILD_DIR/fp8_prefill_k576_pic.o" \
    "$FP8_DECODE_OBJECT" \
    "$BUILD_DIR/fp8_sched_pic.o" \
    "$BUILD_DIR/fp8_combine_pic.o" \
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
    -o "$FP8_EXTENSION_OUTPUT"
