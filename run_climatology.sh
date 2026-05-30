#!/bin/bash
# ============================================================
# SST 气候态计算 —— CPU 优化版提交脚本 (单节点, 无 DCU)
# 集群: kshdmcc2026
# ============================================================
#SBATCH -p kshdmcc2026
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --job-name=clim_fast
#SBATCH --output=clim_fast_%j.out
#SBATCH --error=clim_fast_%j.err

set -eo pipefail

# 激活 conda 环境 nb (非交互式 shell 下 ~/.bashrc 会提前 return, 必须直接 source conda.sh)
source /public/home/fujiake/miniconda3/etc/profile.d/conda.sh
conda activate nb

WORK_DIR=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}
cd "$WORK_DIR"

NCORES=${SLURM_CPUS_ON_NODE:-64}

export PYTHONUNBUFFERED=1
# C++ 融合核用 OpenMP 吃满全核; numpy 的 BLAS 限单线程以免与进程池读盘抢核
export OMP_NUM_THREADS=${NCORES}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 首次运行自动编译 C++ 核
if [ ! -f clim_kernel.so ]; then
    echo "[$(date)] Building C++ kernel..."
    bash build_kernel.sh
fi

# 清理本节点上历史崩溃进程泄漏的 POSIX 信号量/共享内存段 (独占节点, 只清自己的)。
# 否则进程池创建信号量时可能报 [Errno 28] No space left on device。
echo "[$(date)] /dev/shm before cleanup:"; df -h /dev/shm 2>/dev/null | tail -1
find /dev/shm -maxdepth 1 -user "$USER" \
     \( -name 'sem.*' -o -name 'climdata_*' -o -name 'climmean_*' -o -name 'climp90_*' \
        -o -name 'psm2_*' -o -name '*python*' \) -delete 2>/dev/null || true
echo "[$(date)] /dev/shm after cleanup:";  df -h /dev/shm 2>/dev/null | tail -1

# 可选: 只算某段 doy (双节点切分时由 run_dual.sh 传入)
DOY_START=${DOY_START:-152}
DOY_END=${DOY_END:-243}

echo "[$(date)] Starting climatology (CPU, ${NCORES} cores, doy ${DOY_START}-${DOY_END})..."
time python get_climatology.py \
    --nc-path /public/home/achwjznh4b/Newdata/ \
    --save-path /public/home/fujiake/feng/fast/output/ \
    --doy-start "${DOY_START}" \
    --doy-end "${DOY_END}" \
    --io-workers 32
echo "[$(date)] Done."
