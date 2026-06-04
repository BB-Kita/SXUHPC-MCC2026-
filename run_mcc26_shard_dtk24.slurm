#!/usr/bin/env bash
#SBATCH -J mcc-shard
#SBATCH -p kshdmcc2026
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --exclusive
#SBATCH --gres=dcu:4
#SBATCH --time=00:10:00
#SBATCH -o /public/home/fujiake/fjk/MCC26_SXU_two_node_exp_20260603/logs/mcc_shard_%j.out
#SBATCH -e /public/home/fujiake/fjk/MCC26_SXU_two_node_exp_20260603/logs/mcc_shard_%j.err

set -euo pipefail

WORK_DIR="${MCC_WORK_DIR:-/public/home/fujiake/fjk/MCC26_SXU_two_node_exp_20260603}"
BUILD_DIR="${WORK_DIR}/build"
SHARD_NAME="${MCC_SHARD_NAME:-single}"
BIN="${BUILD_DIR}/mcc_${SHARD_NAME}_${SLURM_JOB_ID}"

cd "${WORK_DIR}"

source /public/home/fujiake/miniconda3/bin/activate
conda activate lsd
set +u
source /public/software/compiler/rocm/dtk-24.04.3/env.sh
set -u

mkdir -p "${WORK_DIR}/logs" "${BUILD_DIR}" "${MCC_OUTPUT_DIR}"

echo "============================================"
echo "  MCC26 sharded run"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Node:   $(hostname)"
echo "  Shard:  ${SHARD_NAME}"
echo "  Range:  ${MCC_TARGET_DOY_BEGIN:-152}..${MCC_TARGET_DOY_END:-243}"
echo "  Output: ${MCC_OUTPUT_DIR}"
echo "============================================"

hipcc -O2 -std=c++17 -fopenmp --offload-arch=gfx906 main.cpp io_handler.cpp compute_dcu.cpp \
  -I"${WORK_DIR}" \
  -I"${CONDA_PREFIX}/include" \
  -L"${CONDA_PREFIX}/lib" \
  -lnetcdf \
  -lhdf5 \
  -Wl,-rpath,"${CONDA_PREFIX}/lib" \
  -o "${BIN}"

export MCC_PRELOAD_SEASON="${MCC_PRELOAD_SEASON:-1}"
export MCC_PRELOAD_DIRECT_H2D="${MCC_PRELOAD_DIRECT_H2D:-1}"
export MCC_COMPACT_PER_SLOT="${MCC_COMPACT_PER_SLOT:-1}"
export MCC_PRELOAD_COPY_THREADS="${MCC_PRELOAD_COPY_THREADS:-8}"
export MCC_ASYNC_DCU_INIT="${MCC_ASYNC_DCU_INIT:-1}"
export MCC_IO_THREADS="${MCC_IO_THREADS:-32}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_MAX_ACTIVE_LEVELS=1

time "${BIN}"
