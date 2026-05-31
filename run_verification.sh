#!/bin/bash
# ============================================================
# RMSE 验证提交脚本 (CPU, 无 DCU)
# ============================================================
#SBATCH -p kshdmcc2026
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --time=00:20:00
#SBATCH --job-name=clim_verify
#SBATCH --output=verify_%j.out
#SBATCH --error=verify_%j.err

set -eo pipefail

source /public/home/fujiake/miniconda3/etc/profile.d/conda.sh
conda activate nb

WORK_DIR=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}
cd "$WORK_DIR"
export PYTHONUNBUFFERED=1

echo "[$(date)] Starting verification..."
time python clim_verification.py \
    --clim-path /public/home/fujiake/data/ \
    --contestant-path /public/home/fujiake/feng/fast/output/ \
    --save-path /public/home/fujiake/feng/fast/verification/ \
    --workers 8
echo "[$(date)] Done."
