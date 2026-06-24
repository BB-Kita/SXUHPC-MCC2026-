#!/usr/bin/env bash
set -eo pipefail
# CRITICAL: DTK 25.04 + gfx906:xnack- for Z200SM_71 DCU compatibility
# The cluster DCUs were upgraded from Z100L to Z200SM_71 (gfx906:sramecc+:xnack-)
# Must use DTK >= 24.04 with xnack- flag; older DTK versions produce incompatible code objects
source /public/software/compiler/rocm/dtk-25.04/env.sh 2>/dev/null || true
module load mathlib/netcdf/4.4.1/gcc-7.3.1 2>/dev/null || true
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SRC_DIR}/build"
OUT_NAME="mcc_baseline"
OUT="${BUILD_DIR}/${OUT_NAME}"
mkdir -p "${BUILD_DIR}"
CXX="hipcc"
echo "[compiler] $(hipcc --version 2>&1 | head -1)"
echo "[target] gfx906:xnack- (Z200SM_71 DCU)"
# CRITICAL: --offload-arch=gfx906:xnack- to match Z200SM_71 hardware
# Default gfx906 compiles for xnack+ which is incompatible
${CXX} -std=c++14 -O3 -march=native -fopenmp \
    --offload-arch=gfx906:xnack- \
    "${SRC_DIR}/main.cpp" "${SRC_DIR}/io_handler.cpp" "${SRC_DIR}/compute_dcu.cpp" \
    -I"${SRC_DIR}" \
    -I/public/home/fujiake/miniconda3/envs/lsd/include \
    -L/public/home/fujiake/miniconda3/envs/lsd/lib -lnetcdf -lhdf5 \
    -Wl,-rpath,/public/home/fujiake/miniconda3/envs/lsd/lib \
    -o "${OUT}"
echo "Build success: ${OUT}"
ls -lh "${OUT}"
