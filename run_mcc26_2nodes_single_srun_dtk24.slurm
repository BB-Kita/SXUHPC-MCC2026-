#!/usr/bin/env bash
#SBATCH -J mcc-2n-1srun
#SBATCH -p kshdmcc2026
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --exclusive
#SBATCH --gres=dcu:4
#SBATCH --exclude=f16r4n03
#SBATCH --time=00:10:00
#SBATCH -o /public/home/fujiake/fjk/MCC26_SXU_two_node_exp_20260603/logs/mcc_2nodes_single_%j.out
#SBATCH -e /public/home/fujiake/fjk/MCC26_SXU_two_node_exp_20260603/logs/mcc_2nodes_single_%j.err

set -euo pipefail

WORK_DIR="${MCC_WORK_DIR:-/public/home/fujiake/fjk/MCC26_SXU_two_node_exp_20260603}"
OUTPUT_DIR="${MCC_OUTPUT_DIR:-${WORK_DIR}/output_2nodes_single_srun}"
BUILD_DIR="${WORK_DIR}/build"
BIN="${BUILD_DIR}/mcc_2nodes_single_${SLURM_JOB_ID}"

cd "${WORK_DIR}"

source /public/home/fujiake/miniconda3/bin/activate
conda activate lsd
set +u
source /public/software/compiler/rocm/dtk-24.04.3/env.sh
set -u

mkdir -p "${WORK_DIR}/logs" "${BUILD_DIR}" "${OUTPUT_DIR}"

mapfile -t ALLOC_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")

echo "============================================"
echo "  MCC26 two-node single-srun sharded run"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Nodes:  ${ALLOC_NODES[*]}"
echo "  Output: ${OUTPUT_DIR}"
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
export MCC_PARALLEL_DCU_INIT="${MCC_PARALLEL_DCU_INIT:-0}"
export MCC_SKIP_FINAL_CLEANUP="${MCC_SKIP_FINAL_CLEANUP:-1}"
export MCC_OUTPUT_FLOAT="${MCC_OUTPUT_FLOAT:-1}"
export MCC_OUTPUT_CLASSIC="${MCC_OUTPUT_CLASSIC:-0}"
export MCC_MINIMAL_OUTPUT="${MCC_MINIMAL_OUTPUT:-0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_MAX_ACTIVE_LEVELS=1

time srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=32 --exclusive \
  bash -c '
    set -euo pipefail
    if [[ "${SLURM_PROCID}" == "0" ]]; then
      SHARD_NAME=A
      BEGIN_DOY=152
      END_DOY=197
    else
      SHARD_NAME=B
      BEGIN_DOY=198
      END_DOY=243
    fi

    OUT_LOG="'"${WORK_DIR}"'/logs/mcc_2nodes_single_'"${SLURM_JOB_ID}"'_${SHARD_NAME}.out"
    ERR_LOG="'"${WORK_DIR}"'/logs/mcc_2nodes_single_'"${SLURM_JOB_ID}"'_${SHARD_NAME}.err"

    {
      echo "============================================"
      echo "  MCC26 single-srun shard"
      echo "  Job ID: ${SLURM_JOB_ID}"
      echo "  Proc:   ${SLURM_PROCID}"
      echo "  Node:   $(hostname)"
      echo "  Shard:  ${SHARD_NAME}"
      echo "  Range:  ${BEGIN_DOY}..${END_DOY}"
      echo "  Output: '"${OUTPUT_DIR}"'"
      echo "============================================"
      cd "'"${WORK_DIR}"'"
      export MCC_OUTPUT_DIR="'"${OUTPUT_DIR}"'"
      export MCC_TARGET_DOY_BEGIN="${BEGIN_DOY}"
      export MCC_TARGET_DOY_END="${END_DOY}"
      time "'"${BIN}"'"
    } >"${OUT_LOG}" 2>"${ERR_LOG}"
  '
