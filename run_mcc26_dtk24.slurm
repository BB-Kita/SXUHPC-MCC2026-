#!/usr/bin/env bash
#SBATCH -J mcc-dtk24
#SBATCH -p kshdmcc2026
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --exclusive
#SBATCH --gres=dcu:4
#SBATCH --time=02:00:00
#SBATCH -o /public/home/fujiake/fjk/MCC26_SXU_gpuopt_exp_20260601/logs/mcc_dtk24_%j.out
#SBATCH -e /public/home/fujiake/fjk/MCC26_SXU_gpuopt_exp_20260601/logs/mcc_dtk24_%j.err

set -euo pipefail

WORK_DIR="/public/home/fujiake/fjk/MCC26_SXU_gpuopt_exp_20260601"
BUILD_DIR="${WORK_DIR}/build"
BIN="${BUILD_DIR}/mcc_baseline_inc"

cd "${WORK_DIR}"

source /public/home/fujiake/miniconda3/bin/activate
conda activate lsd
set +u
source /public/software/compiler/rocm/dtk-24.04.3/env.sh
set -u

mkdir -p "${WORK_DIR}/logs" "${BUILD_DIR}"

echo "============================================"
echo "  MCC26 DTK24 incremental run"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Node:   $(hostname)"
echo "  CWD:    $(pwd)"
echo "  Binary: ${BIN}"
echo "  Output: ${MCC_OUTPUT_DIR:-${WORK_DIR}/output_full_rawpread_t32}"
echo "  I/O threads: ${MCC_IO_THREADS:-32}"
echo "============================================"

hipcc -O2 -std=c++17 -fopenmp --offload-arch=gfx906 main.cpp io_handler.cpp compute_dcu.cpp \
  -I"${WORK_DIR}" \
  -I"${CONDA_PREFIX}/include" \
  -L"${CONDA_PREFIX}/lib" \
  -lnetcdf \
  -lhdf5 \
  -Wl,-rpath,"${CONDA_PREFIX}/lib" \
  -o "${BIN}"

export MCC_OUTPUT_DIR="${MCC_OUTPUT_DIR:-${WORK_DIR}/output_full_rawpread_t32}"
mkdir -p "${MCC_OUTPUT_DIR}"

export MCC_IO_THREADS="${MCC_IO_THREADS:-32}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_MAX_ACTIVE_LEVELS=1

time "${BIN}"
