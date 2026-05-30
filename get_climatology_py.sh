#!/bin/bash
#SBATCH -p kshdmcc2026
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --gres=dcu:4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -o /public/home/fujiake/nan/python/climatology_%j.out
#SBATCH -e /public/home/fujiake/nan/python/climatology_%j.err

# ============================================================
# SST 气候态计算 SLURM 提交脚本（夏季 6-8 月，高性能版）
# 用法: sbatch get_climatology_py.sh
# ============================================================

source /public/home/fujiake/nan/env.sh

cd /public/home/fujiake/nan/python

echo "=========================================="
echo "开始 SST 气候态计算（夏季 DOY 152-243）"
echo "开始时间: $(date)"
echo "节点: $(hostname)"
echo "Python: $(which python)"
echo "=========================================="

/public/home/fujiake/nan/python/.venv/bin/python get_climatology.py \
    --start_doy 152 \
    --end_doy 243 \
    --doy_workers 4 \
    --compute_workers 1 \
    --nc_path /public/home/achwjznh4b/Newdata/ \
    --save_path /public/home/fujiake/nan/output

echo "=========================================="
echo "结束时间: $(date)"
echo "计算完成！"
echo "=========================================="
