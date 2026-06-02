#!/usr/bin/env bash
#SBATCH -J mcc-iosweep
#SBATCH -p kshdmcc2026
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --exclusive
#SBATCH --gres=dcu:4
#SBATCH --time=00:40:00
#SBATCH -o /public/home/fujiake/fjk/MCC26_SXU_gpuopt_exp_20260601/logs/iosweep_%j.out
#SBATCH -e /public/home/fujiake/fjk/MCC26_SXU_gpuopt_exp_20260601/logs/iosweep_%j.err

set -euo pipefail

WORK_DIR="/public/home/fujiake/fjk/MCC26_SXU_gpuopt_exp_20260601"
BUILD_DIR="${WORK_DIR}/build"
BIN="${BUILD_DIR}/mcc_iosweep"

cd "${WORK_DIR}"

source /public/home/fujiake/miniconda3/bin/activate
conda activate lsd
set +u
source /public/software/compiler/rocm/dtk-24.04.3/env.sh
set -u

mkdir -p "${WORK_DIR}/logs" "${BUILD_DIR}"

hipcc -O2 -std=c++17 -fopenmp --offload-arch=gfx906 main.cpp io_handler.cpp compute_dcu.cpp \
  -I"${WORK_DIR}" \
  -I"${CONDA_PREFIX}/include" \
  -L"${CONDA_PREFIX}/lib" \
  -lnetcdf \
  -lhdf5 \
  -Wl,-rpath,"${CONDA_PREFIX}/lib" \
  -o "${BIN}"

export MCC_TARGET_DOY_END=153
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_MAX_ACTIVE_LEVELS=1

echo "============================================"
echo "  MCC26 raw pread I/O thread sweep"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Node:   $(hostname)"
echo "  OMP:    ${OMP_NUM_THREADS}"
echo "============================================"

for threads in 1 2 4 8 16 32; do
  export MCC_IO_THREADS="${threads}"
  export MCC_OUTPUT_DIR="${WORK_DIR}/output_iosweep_t${threads}"
  mkdir -p "${MCC_OUTPUT_DIR}"

  echo
  echo "===== MCC_IO_THREADS=${threads} ====="
  time "${BIN}"
done
