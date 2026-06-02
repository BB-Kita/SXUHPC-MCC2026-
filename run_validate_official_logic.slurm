#!/usr/bin/env bash
#SBATCH -J mcc-verify
#SBATCH -p kshdmcc2026
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --exclusive
#SBATCH --gres=dcu:1
#SBATCH --time=02:00:00
#SBATCH -o /public/home/fujiake/fjk/MCC26_SXU_gpuopt_exp_20260601/logs/verify_%j.out
#SBATCH -e /public/home/fujiake/fjk/MCC26_SXU_gpuopt_exp_20260601/logs/verify_%j.err

set -euo pipefail

WORK_DIR="/public/home/fujiake/fjk/MCC26_SXU_gpuopt_exp_20260601"
cd "${WORK_DIR}"

export PATH=/public/home/achwjznh4b/install//bin:$PATH

export MCC_REF_CLIM_PATH="${MCC_REF_CLIM_PATH:-/public/home/achwjznh4b/ERA5/Climatology}"
export MCC_CONTESTANT_CLIM_PATH="${MCC_CONTESTANT_CLIM_PATH:-${WORK_DIR}/output}"
export MCC_VERIFY_SAVE_PATH="${MCC_VERIFY_SAVE_PATH:-${WORK_DIR}/verification}"

mkdir -p "${MCC_VERIFY_SAVE_PATH}" "${WORK_DIR}/logs"

echo "============================================"
echo "  MCC official-logic verification"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Node:   $(hostname)"
echo "  CWD:    $(pwd)"
echo "  Ref:    ${MCC_REF_CLIM_PATH}"
echo "  Test:   ${MCC_CONTESTANT_CLIM_PATH}"
echo "  Save:   ${MCC_VERIFY_SAVE_PATH}"
echo "  Files:  ${MCC_VALIDATE_FILES:-all 92}"
echo "============================================"

time matlab -nodisplay -nosplash -nodesktop < validate_official_logic.m
