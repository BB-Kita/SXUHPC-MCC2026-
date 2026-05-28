#!/usr/bin/env bash
#SBATCH -J mcc
#SBATCH -p kshdmcccc2026
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH -o logs/mcc26_%j.out
#SBATCH -e logs/mcc26_%j.err

# ============================================================
# MCC26_SXU 海洋热浪阈值计算 - Slurm 提交脚本
# 使用 conda lsd 环境编译 + 运行（不提交，仅保存）
# ============================================================

set -euo pipefail

echo "============================================"
echo "  MCC26_SXU Slurm Job"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Run at: $(date)"
echo "  Node:   $(hostname)"
echo "============================================"

# --- 1. 激活 conda lsd 环境 ---
source /public/home/fujiake/miniconda3/bin/activate
conda activate lsd

echo "[ENV] CONDA_PREFIX=${CONDA_PREFIX}"
echo "[ENV] which g++  => $(which g++ 2>/dev/null || echo N/A)"
echo "[ENV] which hipcc => $(which hipcc 2>/dev/null || echo N/A)"

# --- 2. 编译（已手动编译，跳过）---
# 使用 hipcc (海光 DCU 编译器) + conda lsd 的 NetCDF
# SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
# BUILD_DIR="${SRC_DIR}/build"
# OUT="${BUILD_DIR}/mcc_baseline"

mkdir -p logs

# CXX="hipcc"

# echo ""
# echo "[BUILD] Compiler: $(${CXX} --version 2>&1 | head -1)"
# echo "[BUILD] NetCDF:   $(nc-config --version 2>/dev/null || echo unknown)"

# ${CXX} -std=c++14 -O3 -fopenmp \
#     "${SRC_DIR}/main.cpp" \
#     "${SRC_DIR}/io_handler.cpp" \
#     "${SRC_DIR}/compute_dcu.cpp" \
#     -I"${SRC_DIR}" \
#     -I"${CONDA_PREFIX}/include" \
#     -L"${CONDA_PREFIX}/lib" \
#     -lnetcdf \
#     -Wl,-rpath,"${CONDA_PREFIX}/lib" \
#     -o "${OUT}"

# echo "[BUILD] Output: ${OUT}"
# ls -lh "${OUT}"

# --- 3. 运行 ---
echo ""
echo "[RUN] Starting DCU computation..."
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

time "./build/mcc_baseline"

echo ""
echo "============================================"
echo "  Job finished at: $(date)"
echo "============================================"
