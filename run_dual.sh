#!/bin/bash
# ============================================================
# 双节点提交: 把 92 天 (doy 152-243) 平均切成两段, 各占一台机。
# 两段写入同一个 output 目录, 文件名按 MMDD 互不冲突。
# 用法: bash run_dual.sh
# ============================================================
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

# doy 152..243 共 92 天, 切成 152-197 / 198-243
sbatch --export=ALL,DOY_START=152,DOY_END=197 --job-name=clim_fast_a run_climatology.sh
sbatch --export=ALL,DOY_START=198,DOY_END=243 --job-name=clim_fast_b run_climatology.sh

echo "已提交两个作业 (doy 152-197 / 198-243)。两者完成后再跑 run_verification.sh。"
