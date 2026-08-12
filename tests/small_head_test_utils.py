import os
from functools import lru_cache
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def require_sm100_family() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("small-head CUDA tests require an SM100-family GPU")


@lru_cache(maxsize=1)
def get_extension():
    cuda_13 = Path("/share/apps/cuda/13.3.0")
    requested_cuda = os.getenv("FLASH_MLA_TEST_CUDA_HOME")
    if requested_cuda:
        os.environ["CUDA_HOME"] = requested_cuda
    elif torch.version.cuda and torch.version.cuda.startswith("13.") and cuda_13.exists():
        os.environ["CUDA_HOME"] = str(cuda_13)

    import torch.utils.cpp_extension as cpp_extension

    # cpp_extension may have been imported before this helper selected CUDA.
    cpp_extension.CUDA_HOME = os.environ.get("CUDA_HOME")

    sources = [
        ROOT / "tests/small_head_test_ext.cpp",
        ROOT / "csrc/sm100/prefill/sparse/fwd/head_small/instantiations/phase1_k128.cu",
        ROOT / "csrc/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_prefill_k512.cu",
        ROOT / "csrc/sm100/prefill/sparse/fwd_for_small_topk/head64/instantiations/phase1_prefill_head64_k512.cu",
        ROOT / "csrc/sm100/prefill/sparse/bwd/head_small/phase1.cu",
        ROOT / "csrc/sm100/prefill/sparse/bwd/head_small/convert.cu",
    ]
    include_paths = [
        ROOT / "csrc",
        ROOT / "csrc/kerutils/include",
        ROOT / "csrc/sm90",
        ROOT / "csrc/sm100",
        ROOT / "csrc/cutlass/include",
        ROOT / "csrc/cutlass/tools/util/include",
    ]
    nvcc_flags = [
        "-O3",
        "-std=c++20",
        "-DNDEBUG",
        "-D_USE_MATH_DEFINES",
        "-Wno-deprecated-declarations",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-gencode=arch=compute_100f,code=sm_100f",
        "--threads=8",
    ]
    return cpp_extension.load(
        name="flash_mla_small_head_test_ext",
        sources=[str(path) for path in sources],
        extra_include_paths=[str(path) for path in include_paths],
        extra_cflags=["-O3", "-std=c++20", "-DNDEBUG"],
        extra_cuda_cflags=nvcc_flags,
        with_cuda=True,
        verbose=os.getenv("FLASH_MLA_TEST_VERBOSE", "0") == "1",
    )
