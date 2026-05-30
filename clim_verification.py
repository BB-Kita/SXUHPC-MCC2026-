#!/usr/bin/env python3
"""
气候态 / P90 结果 RMSE 验证 —— CPU 并行版
==========================================
参考源文件: /public/home/fujiake/feng/clim_verification.m

逐日比较参考结果(reference)与待验证结果(contestant)的 Climmean / P90_sst,
计算夏季 6~8 月的逐日 RMSE, 并汇总平均/最大误差。

这是纯 I/O + 轻量算术的任务 (92 天 x 2 文件), 用线程池并行读取即可, 无需 GPU。
RMSE 定义与 MATLAB 一致: sqrt(nanmean((a-b)^2))。
"""

import os
import time
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from netCDF4 import Dataset

# 参考结果 (标准答案)
CLIM_PATH = '/public/home/fujiake/data/'
# 待验证结果 (本优化版的输出)
CONTESTANT_PATH = '/public/home/fujiake/feng/fast/output/'
SAVE_PATH = '/public/home/fujiake/feng/fast/verification/'


def read_var(filepath, var_name):
    try:
        ds = Dataset(filepath, 'r')
        data = np.asarray(ds.variables[var_name][:], dtype=np.float64)
        ds.close()
        return data
    except Exception:
        return None


def rmse(ref, test):
    if ref is None or test is None:
        return np.nan
    # 容错: 若 ref 与 test 维度顺序相反 (Lat,Lon) vs (Lon,Lat), 转置对齐后再比。
    if ref.shape != test.shape and ref.shape == test.T.shape:
        test = test.T
    if ref.shape != test.shape:
        return np.nan
    diff = ref - test
    if not np.any(~np.isnan(diff)):
        return np.nan
    return float(np.sqrt(np.nanmean(diff ** 2)))


def process_day(args):
    month, day = args
    filename = f'{month:02d}{day:02d}.nc'
    ref_file = os.path.join(CLIM_PATH, filename)
    con_file = os.path.join(CONTESTANT_PATH, filename)

    r_clim = read_var(ref_file, 'Climmean')
    r_p90 = read_var(ref_file, 'P90_sst')
    c_clim = read_var(con_file, 'Climmean')
    c_p90 = read_var(con_file, 'P90_sst')

    return rmse(r_clim, c_clim), rmse(r_p90, c_p90)


def main():
    global CLIM_PATH, CONTESTANT_PATH, SAVE_PATH

    parser = argparse.ArgumentParser(description='气候态 RMSE 验证 (CPU 并行)')
    parser.add_argument('--clim-path', type=str, default=CLIM_PATH)
    parser.add_argument('--contestant-path', type=str, default=CONTESTANT_PATH)
    parser.add_argument('--save-path', type=str, default=SAVE_PATH)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    CLIM_PATH = args.clim_path
    CONTESTANT_PATH = args.contestant_path
    SAVE_PATH = args.save_path
    os.makedirs(SAVE_PATH, exist_ok=True)

    print("=" * 60)
    print("  Climatology RMSE Verification (CPU parallel)")
    print(f"  Reference:  {CLIM_PATH}")
    print(f"  Contestant: {CONTESTANT_PATH}")
    print("=" * 60)

    t0 = time.time()

    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    tasks = [(m, d) for m in range(6, 9) for d in range(1, days_in_month[m - 1] + 1)]

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(process_day, tasks))

    rmse_clim = np.array([r[0] for r in results])
    rmse_p90 = np.array([r[1] for r in results])

    R_clim = np.nanmean(rmse_clim)
    R_p90 = np.nanmean(rmse_p90)
    max_clim = np.nanmax(rmse_clim)
    max_p90 = np.nanmax(rmse_p90)

    print('\n' + '=' * 45)
    print('       6~8月夏季 RMSE 统计结果')
    print('=' * 45)
    print(f'气候态 平均误差 : {R_clim:.4f} ℃')
    print(f'气候态 最大误差 : {max_clim:.4f} ℃')
    print('-' * 45)
    print(f'P90分位 平均误差 : {R_p90:.4f} ℃')
    print(f'P90分位 最大误差 : {max_p90:.4f} ℃')
    print('=' * 45)
    print(f'  Time: {time.time() - t0:.2f}s')

    with open(os.path.join(SAVE_PATH, 'RMSE_clim.txt'), 'w') as f:
        f.write(f'{R_clim:.4f}\n')
    with open(os.path.join(SAVE_PATH, 'RMSE_P90.txt'), 'w') as f:
        f.write(f'{R_p90:.4f}\n')


if __name__ == '__main__':
    main()
