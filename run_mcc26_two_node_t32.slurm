#!/usr/bin/env bash
#SBATCH -J mcc-2node
#SBATCH -p kshdmcc2026
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --gres=dcu:4
#SBATCH --time=02:00:00
#SBATCH -o /public/home/fujiake/fjk/MCC26_SXU_gpuopt_exp_20260601/logs/mcc_2node_%j.out
#SBATCH -e /public/home/fujiake/fjk/MCC26_SXU_gpuopt_exp_20260601/logs/mcc_2node_%j.err

set -euo pipefail

WORK_DIR="/public/home/fujiake/fjk/MCC26_SXU_gpuopt_exp_20260601"
BUILD_DIR="${WORK_DIR}/build"
BIN="${BUILD_DIR}/mcc_2node_t32"
OUT_DIR="${MCC_OUTPUT_DIR:-${WORK_DIR}/output_full_2node_t32}"

cd "${WORK_DIR}"

source /public/home/fujiake/miniconda3/bin/activate
conda activate lsd
set +u
source /public/software/compiler/rocm/dtk-24.04.3/env.sh
set -u

mkdir -p "${WORK_DIR}/logs" "${BUILD_DIR}" "${OUT_DIR}"

hipcc -O2 -std=c++17 -fopenmp --offload-arch=gfx906 main.cpp io_handler.cpp compute_dcu.cpp \
  -I"${WORK_DIR}" \
  -I"${CONDA_PREFIX}/include" \
  -L"${CONDA_PREFIX}/lib" \
  -lnetcdf \
  -lhdf5 \
  -Wl,-rpath,"${CONDA_PREFIX}/lib" \
  -o "${BIN}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_MAX_ACTIVE_LEVELS=1
export MCC_IO_THREADS="${MCC_IO_THREADS:-32}"

echo "============================================"
echo "  MCC26 two-node raw pread + IO t32 run"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Nodes:  ${SLURM_JOB_NODELIST}"
echo "  Output: ${OUT_DIR}"
echo "============================================"

time (
  srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-32}" --gres=dcu:4 --exclusive \
    env MCC_OUTPUT_DIR="${OUT_DIR}" MCC_IO_THREADS="${MCC_IO_THREADS}" MCC_TARGET_DOY_BEGIN=152 MCC_TARGET_DOY_END=197 "${BIN}" \
    > "${WORK_DIR}/logs/mcc_2node_${SLURM_JOB_ID}_152_197.out" \
    2> "${WORK_DIR}/logs/mcc_2node_${SLURM_JOB_ID}_152_197.err" &
  pid1=$!

  srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-32}" --gres=dcu:4 --exclusive \
    env MCC_OUTPUT_DIR="${OUT_DIR}" MCC_IO_THREADS="${MCC_IO_THREADS}" MCC_TARGET_DOY_BEGIN=198 MCC_TARGET_DOY_END=243 "${BIN}" \
    > "${WORK_DIR}/logs/mcc_2node_${SLURM_JOB_ID}_198_243.out" \
    2> "${WORK_DIR}/logs/mcc_2node_${SLURM_JOB_ID}_198_243.err" &
  pid2=$!

  wait "${pid1}"
  wait "${pid2}"
)

echo "two-node output count:"
find "${OUT_DIR}" -maxdepth 1 -name '*.nc' | wc -l
