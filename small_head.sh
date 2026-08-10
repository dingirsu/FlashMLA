#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

export FLASH_MLA_TEST_CUDA_HOME="${FLASH_MLA_TEST_CUDA_HOME:-/usr/local/cuda}"
export FLASH_MLA_TEST_VERBOSE="${FLASH_MLA_TEST_VERBOSE:-1}"
export MAX_JOBS="${MAX_JOBS:-4}"

cd "$ROOT"

case "${1:-build}" in
    build)
        "$PYTHON" -c \
            'import sys; sys.path.insert(0, "tests"); from small_head_test_utils import get_extension; print(get_extension().__file__)'
        ;;
    fwd)
        "$PYTHON" -m pytest -q \
            tests/test_sm100_sparse_small_topk_fwd.py -s
        ;;
    bwd)
        "$PYTHON" -m pytest -q \
            tests/test_sm100_sparse_head_small_bwd.py -s
        ;;
    test)
        "$PYTHON" -m pytest -q \
            tests/test_sm100_sparse_small_topk_fwd.py \
            tests/test_sm100_sparse_head_small_bwd.py -s
        ;;
    *)
        echo "usage: $0 [build|fwd|bwd|test]" >&2
        exit 2
        ;;
esac
