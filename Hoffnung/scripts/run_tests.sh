#!/usr/bin/env bash
# Run all GBDT tests (C++ and Python).
# Usage:
#   ./scripts/run_tests.sh          # run all tests
#   ./scripts/run_tests.sh cpp      # C++ only
#   ./scripts/run_tests.sh py       # Python only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build"

# ── Build ────────────────────────────────────────────────
echo "=== Building ==="
mkdir -p "$BUILD_DIR"
cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$BUILD_DIR" -j "$(nproc)" 2>&1

# ── C++ tests ────────────────────────────────────────────
if [[ "$#" -eq 0 || "$1" == "cpp" ]]; then
    echo ""
    echo "=== C++ tests ==="
    "$BUILD_DIR/test_tree"
fi

# ── Python tests ─────────────────────────────────────────
if [[ "$#" -eq 0 || "$1" == "py" ]]; then
    echo ""
    echo "=== Python tests ==="
    PYTHONPATH="${BUILD_DIR}:${PROJECT_DIR}/python:${PYTHONPATH:-}" \
        python -m pytest "${PROJECT_DIR}/tests/test_basic.py" -v "$@"
fi
