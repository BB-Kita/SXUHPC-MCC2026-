#!/usr/bin/env bash
#SBATCH -J mcc
#SBATCH -p kshdmcc2026
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --exclusive
#SBATCH --gres=dcu:4
#SBATCH --time=02:00:00
#SBATCH -o /public/home/fujiake/fjk/MCC26_SXU/logs/mcc26_%j.out
#SBATCH -e /public/home/fujiake/fjk/MCC26_SXU/logs/mcc26_%j.err

set -euo pipefail

WORK_DIR="/public/home/fujiake/fjk/MCC26_SXU"
cd "${WORK_DIR}"

echo "============================================"
echo "  MCC26_SXU Slurm Job"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Run at: $(date)"
echo "  Node:   $(hostname)"
echo "  CWD:    $(pwd)"
echo "============================================"

# --- 1. 激活 conda 环境 ---
source /public/home/fujiake/miniconda3/bin/activate
conda activate lsd

echo "[ENV] CONDA_PREFIX=${CONDA_PREFIX}"
echo "[ENV] which hipcc => $(which hipcc 2>/dev/null || echo N/A)"

# --- 2. 编译 ---
mkdir -p "${WORK_DIR}/logs" "${WORK_DIR}/build" "${WORK_DIR}/output"
BUILD_DIR="${WORK_DIR}/build"
OUT="${BUILD_DIR}/mcc_baseline"

# Force rebuild if source is newer than binary
if [ ! -f "${OUT}" ] || [ "${WORK_DIR}/io_handler.cpp" -nt "${OUT}" ] || [ "${WORK_DIR}/compute_dcu.cpp" -nt "${OUT}" ] || [ "${WORK_DIR}/main.cpp" -nt "${OUT}" ]; then
    CXX="hipcc"
    echo "[BUILD] Compiler: $(${CXX} --version 2>&1 | head -1)"
    ${CXX} -std=c++14 -O3 -fopenmp \
        "${WORK_DIR}/main.cpp" \
        "${WORK_DIR}/io_handler.cpp" \
        "${WORK_DIR}/compute_dcu.cpp" \
        -I"${WORK_DIR}" \
        -I"${CONDA_PREFIX}/include" \
        -L"${CONDA_PREFIX}/lib" \
        -lnetcdf \
        -Wl,-rpath,"${CONDA_PREFIX}/lib" \
        -o "${OUT}"
    echo "[BUILD] Output: ${OUT}"
    ls -lh "${OUT}"
else
    echo "[BUILD] Binary exists, skipping: ${OUT}"
fi

# --- 3. 运行 ---
echo ""
echo "[RUN] Starting DCU computation..."
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

time "${OUT}"

echo ""
echo "============================================"
echo "  Job finished at: $(date)"
echo "============================================"
