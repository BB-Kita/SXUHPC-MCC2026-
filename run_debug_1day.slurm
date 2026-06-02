#!/bin/bash
#SBATCH -J mcc26_debug1
#SBATCH -p kshdmcc2026
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --exclusive
#SBATCH --gres=dcu:4
#SBATCH --time=00:20:00
#SBATCH -o /public/home/fujiake/fjk/MCC26_SXU_gpuopt_exp_20260601/logs/debug1_%j.out
#SBATCH -e /public/home/fujiake/fjk/MCC26_SXU_gpuopt_exp_20260601/logs/debug1_%j.err

set -euo pipefail

WORK_DIR=/public/home/fujiake/fjk/MCC26_SXU_gpuopt_exp_20260601
BUILD_DIR=${WORK_DIR}/build
BIN=${BUILD_DIR}/mcc_baseline_inc
OUT_DIR=${WORK_DIR}/output_debug1

mkdir -p "${BUILD_DIR}" "${WORK_DIR}/logs" "${OUT_DIR}"
cd "${WORK_DIR}"

source /public/home/fujiake/miniconda3/bin/activate
conda activate lsd
set +u
source /public/software/compiler/rocm/dtk-24.04.3/env.sh
set -u

echo "============================================"
echo "  MCC26 debug correctness probe"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Node:   ${SLURMD_NODENAME}"
echo "  CWD:    ${PWD}"
echo "  Binary: ${BIN}"
echo "  Output: ${OUT_DIR}"
echo "============================================"

hipcc -O2 -std=c++17 -fopenmp --offload-arch=gfx906 main.cpp io_handler.cpp compute_dcu.cpp \
  -I"${WORK_DIR}" \
  -I"${CONDA_PREFIX}/include" \
  -L"${CONDA_PREFIX}/lib" \
  -lnetcdf \
  -Wl,-rpath,"${CONDA_PREFIX}/lib" \
  -o "${BIN}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export MCC_TARGET_DOY_END=152
export MCC_DEBUG_VALIDATE=1
export MCC_OUTPUT_DIR="${OUT_DIR}"

"${BIN}"
