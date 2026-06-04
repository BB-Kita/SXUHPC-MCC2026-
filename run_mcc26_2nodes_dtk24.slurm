#!/usr/bin/env bash
#SBATCH -J mcc-2nodes
#SBATCH -p kshdmcc2026
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --exclusive
#SBATCH --gres=dcu:4
#SBATCH --exclude=f16r4n03
#SBATCH --time=00:10:00
#SBATCH -o /public/home/fujiake/fjk/MCC26_SXU_two_node_exp_20260603/logs/mcc_2nodes_%j.out
#SBATCH -e /public/home/fujiake/fjk/MCC26_SXU_two_node_exp_20260603/logs/mcc_2nodes_%j.err

set -euo pipefail

WORK_DIR="${MCC_WORK_DIR:-/public/home/fujiake/fjk/MCC26_SXU_two_node_exp_20260603}"
OUTPUT_DIR="${MCC_OUTPUT_DIR:-${WORK_DIR}/output_2nodes_injob_preload_direct}"
BUILD_DIR="${WORK_DIR}/build"
BIN="${BUILD_DIR}/mcc_2nodes_${SLURM_JOB_ID}"

cd "${WORK_DIR}"

source /public/home/fujiake/miniconda3/bin/activate
conda activate lsd
set +u
source /public/software/compiler/rocm/dtk-24.04.3/env.sh
set -u

mkdir -p "${WORK_DIR}/logs" "${BUILD_DIR}" "${OUTPUT_DIR}"

mapfile -t ALLOC_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if (( ${#ALLOC_NODES[@]} < 2 )); then
  echo "[error] expected 2 allocated nodes, got ${#ALLOC_NODES[@]}: ${SLURM_JOB_NODELIST}" >&2
  exit 1
fi

echo "============================================"
echo "  MCC26 two-node sharded run"
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
export MCC_SKIP_HOST_WINDOW_ZERO="${MCC_SKIP_HOST_WINDOW_ZERO:-0}"
export MCC_PRELOAD_SOURCE_WINDOW="${MCC_PRELOAD_SOURCE_WINDOW:-0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_MAX_ACTIVE_LEVELS=1

run_shard() {
  local shard_name="$1"
  local node_name="$2"
  local begin_doy="$3"
  local end_doy="$4"
  local out_log="${WORK_DIR}/logs/mcc_2nodes_${SLURM_JOB_ID}_${shard_name}.out"
  local err_log="${WORK_DIR}/logs/mcc_2nodes_${SLURM_JOB_ID}_${shard_name}.err"

  echo "[launch] shard=${shard_name} node=${node_name} range=${begin_doy}..${end_doy}"
  srun --nodes=1 --ntasks=1 --cpus-per-task=32 --nodelist="${node_name}" --exclusive \
    bash -c "
      set -euo pipefail
      cd '${WORK_DIR}'
      export MCC_OUTPUT_DIR='${OUTPUT_DIR}'
      export MCC_TARGET_DOY_BEGIN='${begin_doy}'
      export MCC_TARGET_DOY_END='${end_doy}'
      export MCC_PRELOAD_SEASON='${MCC_PRELOAD_SEASON}'
      export MCC_PRELOAD_DIRECT_H2D='${MCC_PRELOAD_DIRECT_H2D}'
      export MCC_COMPACT_PER_SLOT='${MCC_COMPACT_PER_SLOT}'
      export MCC_PRELOAD_COPY_THREADS='${MCC_PRELOAD_COPY_THREADS}'
      export MCC_ASYNC_DCU_INIT='${MCC_ASYNC_DCU_INIT}'
      export MCC_IO_THREADS='${MCC_IO_THREADS}'
      export MCC_PARALLEL_DCU_INIT='${MCC_PARALLEL_DCU_INIT}'
      export MCC_SKIP_FINAL_CLEANUP='${MCC_SKIP_FINAL_CLEANUP}'
      export MCC_OUTPUT_FLOAT='${MCC_OUTPUT_FLOAT}'
      export MCC_OUTPUT_CLASSIC='${MCC_OUTPUT_CLASSIC}'
      export MCC_MINIMAL_OUTPUT='${MCC_MINIMAL_OUTPUT}'
      export MCC_SKIP_HOST_WINDOW_ZERO='${MCC_SKIP_HOST_WINDOW_ZERO}'
      export MCC_PRELOAD_SOURCE_WINDOW='${MCC_PRELOAD_SOURCE_WINDOW}'
      export OMP_NUM_THREADS='${OMP_NUM_THREADS}'
      export OMP_PROC_BIND='${OMP_PROC_BIND}'
      export OMP_PLACES='${OMP_PLACES}'
      export OMP_MAX_ACTIVE_LEVELS='${OMP_MAX_ACTIVE_LEVELS}'
      time '${BIN}'
    " >"${out_log}" 2>"${err_log}" &
}

run_parallel() {
  run_shard A "${ALLOC_NODES[0]}" 152 197
  local pid_a=$!
  run_shard B "${ALLOC_NODES[1]}" 198 243
  local pid_b=$!

  wait "${pid_a}"
  local status_a=$?
  wait "${pid_b}"
  local status_b=$?
  echo "[profile] shard_exit A=${status_a} B=${status_b}"
  (( status_a == 0 && status_b == 0 ))
}

set +e
time run_parallel
run_status=$?
set -e

exit "${run_status}"
