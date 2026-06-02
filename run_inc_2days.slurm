#!/usr/bin/env bash
#SBATCH -J mcc-inc2
#SBATCH -p kshdmcc2026
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --exclusive
#SBATCH --gres=dcu:4
#SBATCH --time=00:20:00
#SBATCH -o /public/home/fujiake/fjk/MCC26_SXU_gpuopt_exp_20260601/logs/inc2_%j.out
#SBATCH -e /public/home/fujiake/fjk/MCC26_SXU_gpuopt_exp_20260601/logs/inc2_%j.err

set -euo pipefail

WORK_DIR="/public/home/fujiake/fjk/MCC26_SXU_gpuopt_exp_20260601"
cd "${WORK_DIR}"

source /public/home/fujiake/miniconda3/bin/activate
conda activate lsd
set +u
source /public/software/compiler/rocm/dtk-24.04.3/env.sh
set -u

mkdir -p "${WORK_DIR}/logs" "${WORK_DIR}/output_inc2_dtk24" "${WORK_DIR}/build"

hipcc -O2 -std=c++17 -fopenmp --offload-arch=gfx906 main.cpp io_handler.cpp compute_dcu.cpp \
  -I"${WORK_DIR}" \
  -I"${CONDA_PREFIX}/include" \
  -L"${CONDA_PREFIX}/lib" \
  -lnetcdf \
  -Wl,-rpath,"${CONDA_PREFIX}/lib" \
  -o "${WORK_DIR}/build/mcc_baseline_inc"

export MCC_TARGET_DOY_END=153
export MCC_OUTPUT_DIR="${WORK_DIR}/output_inc2_dtk24"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "============================================"
echo "  MCC26 incremental DCU smoke test"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Node:   $(hostname)"
echo "  CWD:    $(pwd)"
echo "  Binary: ${WORK_DIR}/build/mcc_baseline_inc"
echo "  Output: ${MCC_OUTPUT_DIR}"
echo "============================================"

time "${WORK_DIR}/build/mcc_baseline_inc"
