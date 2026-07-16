#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA="${CONDA:-/share/home/jintao/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-dsv4}"

export FLASH_MLA_TEST_CUDA_HOME="${FLASH_MLA_TEST_CUDA_HOME:-/share/apps/cuda/13.3.0}"
export FLASH_MLA_TEST_VERBOSE="${FLASH_MLA_TEST_VERBOSE:-1}"
export MAX_JOBS="${MAX_JOBS:-4}"

cd "$ROOT"

case "${1:-build}" in
    build)
        "$CONDA" run -n "$CONDA_ENV" python -c \
            'import sys; sys.path.insert(0, "tests"); from small_head_test_utils import get_extension; print(get_extension().__file__)'
        ;;
    fwd)
        "$CONDA" run -n "$CONDA_ENV" python -m pytest -q \
            tests/test_sm100_sparse_small_topk_fwd.py -s
        ;;
    bwd)
        "$CONDA" run -n "$CONDA_ENV" python -m pytest -q \
            tests/test_sm100_sparse_head_small_bwd.py -s
        ;;
    test)
        "$CONDA" run -n "$CONDA_ENV" python -m pytest -q \
            tests/test_sm100_sparse_small_topk_fwd.py \
            tests/test_sm100_sparse_head_small_bwd.py -s
        ;;
    *)
        echo "usage: $0 [build|fwd|bwd|test]" >&2
        exit 2
        ;;
esac
