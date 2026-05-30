#!/usr/bin/env python3
"""
SST 气候态计算脚本 (高性能 Python 版)
=====================================
基于 1991-2020 年逐日 SST，计算气候态均值与 90% 分位数。

性能优化要点:
1. netCDF4 直读，绕过 xarray 开销
2. 预计算日历与日期字符串，消除重复 datetime 构造
3. ThreadPoolExecutor 并行读文件（单 DOY 内，避免多进程 HDF5 冲突）
4. 预分配 3D 数组，避免 list -> np.array 二次拷贝
5. netCDF4 直写输出
6. 增量均值 + numpy 向量化 nanpercentile (method='linear', 对齐 MATLAB)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from functools import partial
from typing import Dict, List, Optional, Tuple

import netCDF4
import numpy as np

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

LAT_NUM = 721
LON_NUM = 1440


class Config:
    NC_PATH = "/public/home/achwjznh4b/Newdata/"
    SAVE_PATH = "/public/home/achwjznh4b/ERA5/Climatology/"
    START_YEAR = 1991
    END_YEAR = 2020
    VAR_SST = "data"
    VAR_LON = "lon"
    VAR_LAT = "lat"
    DELTA_DAY = 5
    DEFAULT_START_DOY = 152
    DEFAULT_END_DOY = 243


# ---------------------------------------------------------------------------
# 预计算日历（模块级一次性初始化）
# ---------------------------------------------------------------------------
_YEAR_DATE_STR: Dict[int, List[str]] = {}
_DOY_FILENAME: Dict[int, str] = {}


def _build_year_dates(year: int) -> List[str]:
    out: List[str] = []
    current = datetime(year, 1, 1)
    end = datetime(year, 12, 31)
    while current <= end:
        if current.month == 2 and current.day == 29:
            current += timedelta(days=1)
            continue
        out.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return out


def _init_calendar_cache() -> None:
    if _YEAR_DATE_STR:
        return
    for year in range(Config.START_YEAR, Config.END_YEAR + 1):
        _YEAR_DATE_STR[year] = _build_year_dates(year)
    ref = _YEAR_DATE_STR[2020]
    for doy, date_str in enumerate(ref, start=1):
        _DOY_FILENAME[doy] = f"{date_str[4:6]}{date_str[6:8]}.nc"


_init_calendar_cache()


def generate_file_paths(
    nc_path: str,
    start_year: int,
    end_year: int,
    doy: int,
    delta_day: int,
) -> List[str]:
    """生成指定 DOY 的全部输入文件路径（无 datetime 对象分配）。"""
    join = os.path.join
    paths: List[str] = []
    center = doy - 1
    for year in range(start_year, end_year + 1):
        dates = _YEAR_DATE_STR[year]
        n = len(dates)
        lo = max(0, center - delta_day)
        hi = min(n - 1, center + delta_day)
        year_dates = dates[lo : hi + 1]
        paths.extend(join(nc_path, d) for d in year_dates)
    return paths


def read_sst_file(file_path: str, var_sst: str) -> Optional[np.ndarray]:
    """读取单个 SST 文件，失败返回 None。"""
    try:
        with netCDF4.Dataset(file_path, "r") as ds:
            data = ds.variables[var_sst][:]
        if isinstance(data, np.ma.MaskedArray):
            return data.filled(np.nan).astype(np.float64, copy=False)
        return np.ascontiguousarray(data, dtype=np.float64)
    except (OSError, KeyError, netCDF4.lib.NetCDF4Error):
        return None


def _p90_lat_chunk(args: Tuple[np.ndarray, int, int, float]) -> Tuple[int, np.ndarray]:
    """按纬度分块计算 P90（供多进程调用，Linux fork COW 共享 stack）。"""
    stack, lo, hi, q = args
    block = np.nanpercentile(stack[:, lo:hi, :], q, axis=0, method="linear")
    return lo, block


def parallel_nanpercentile(
    stack: np.ndarray,
    q: float,
    workers: int,
) -> np.ndarray:
    """沿时间轴计算逐格 P90，纬度方向多进程并行。"""
    n_lat = stack.shape[1]
    if workers <= 1 or n_lat < workers * 2:
        return np.nanpercentile(stack, q, axis=0, method="linear")

    chunk = max(1, (n_lat + workers - 1) // workers)
    ranges = [(i, min(i + chunk, n_lat)) for i in range(0, n_lat, chunk)]
    tasks = [(stack, lo, hi, q) for lo, hi in ranges]

    out = np.empty((n_lat, stack.shape[2]), dtype=np.float64)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for lo, block in pool.map(_p90_lat_chunk, tasks):
            out[lo : lo + block.shape[0], :] = block
    return out


def compute_climatology(
    file_paths: List[str],
    var_sst: str,
    compute_workers: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """
    计算单个 DOY 的气候态均值与 P90。

    I/O 串行（共享文件系统 HDF5 安全）；P90 按纬度分块多进程并行。
    """
    n_paths = len(file_paths)
    if n_paths == 0:
        return None, None, 0

    running_sum = np.zeros((LAT_NUM, LON_NUM), dtype=np.float64)
    running_count = np.zeros((LAT_NUM, LON_NUM), dtype=np.int64)
    stack = np.empty((n_paths, LAT_NUM, LON_NUM), dtype=np.float64)
    write_idx = 0

    for fp in file_paths:
        sst = read_sst_file(fp, var_sst)
        if sst is None:
            continue
        if sst.shape != (LAT_NUM, LON_NUM):
            if sst.shape == (LON_NUM, LAT_NUM):
                sst = sst.T
            else:
                continue
        valid = ~np.isnan(sst)
        running_sum += np.where(valid, sst, 0.0)
        running_count += valid
        stack[write_idx] = sst
        write_idx += 1

    if write_idx == 0:
        return None, None, 0

    stack = stack[:write_idx]

    with np.errstate(invalid="ignore", divide="ignore"):
        clim = np.full((LAT_NUM, LON_NUM), np.nan, dtype=np.float64)
        np.divide(running_sum, running_count, out=clim, where=running_count > 0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        p90 = parallel_nanpercentile(stack, 90.0, compute_workers)

    return clim, p90, write_idx


def save_climatology_netcdf(
    clim: np.ndarray,
    p90: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    doy: int,
    save_path: str,
) -> None:
    """netCDF4 直写，比 xarray 更快。"""
    filename = _DOY_FILENAME.get(doy)
    if filename is None:
        logger.error("无效 DOY: %d", doy)
        return

    filepath = os.path.join(save_path, filename)
    with netCDF4.Dataset(filepath, "w", format="NETCDF4") as ds:
        ds.createDimension("Lat", LAT_NUM)
        ds.createDimension("Lon", LON_NUM)

        lat_v = ds.createVariable("Lat", "f8", ("Lat",))
        lat_v.units = "degrees_north"
        lat_v[:] = lat

        lon_v = ds.createVariable("Lon", "f8", ("Lon",))
        lon_v.units = "degrees_east"
        lon_v[:] = lon

        clim_v = ds.createVariable("Climmean", "f8", ("Lat", "Lon"))
        clim_v.long_name = "OSTIA SST climatology 1991-2020"
        clim_v.units = "K"
        clim_v[:] = clim

        p90_v = ds.createVariable("P90_sst", "f8", ("Lat", "Lon"))
        p90_v.long_name = "90th percentile of SST"
        p90_v.units = "K"
        p90_v[:] = p90

        doy_v = ds.createVariable("dayofyear", "i4")
        doy_v.long_name = "Day of year (1-365, no 29Feb)"
        doy_v.assignValue(int(doy))

    logger.info("已保存: %s", filepath)


def process_single_doy(
    doy: int,
    nc_path: str,
    var_sst: str,
    start_year: int,
    end_year: int,
    delta_day: int,
    compute_workers: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    t0 = time.perf_counter()
    file_paths = generate_file_paths(nc_path, start_year, end_year, doy, delta_day)
    clim, p90, valid_count = compute_climatology(
        file_paths, var_sst, compute_workers
    )
    elapsed = time.perf_counter() - t0
    if valid_count > 0:
        logger.info(
            "DOY %d 完成: %d/%d 文件, %.1fs",
            doy,
            valid_count,
            len(file_paths),
            elapsed,
        )
    else:
        logger.warning("DOY %d 无有效数据", doy)
    return clim, p90, valid_count


def _process_and_save_doy(
    doy: int,
    nc_path: str,
    save_path: str,
    lat: np.ndarray,
    lon: np.ndarray,
    compute_workers: int,
) -> Tuple[int, int]:
    """处理并保存单个 DOY，供 DOY 级多进程调用。"""
    clim, p90, valid_count = process_single_doy(
        doy,
        nc_path,
        Config.VAR_SST,
        Config.START_YEAR,
        Config.END_YEAR,
        Config.DELTA_DAY,
        compute_workers,
    )
    if clim is not None and p90 is not None and valid_count > 0:
        save_climatology_netcdf(clim, p90, lon, lat, doy, save_path)
    return doy, valid_count


def read_coordinates(nc_path: str) -> Tuple[np.ndarray, np.ndarray]:
    sample = os.path.join(nc_path, "19910101")
    with netCDF4.Dataset(sample, "r") as ds:
        lat = np.asarray(ds.variables[Config.VAR_LAT][:], dtype=np.float64)
        lon = np.asarray(ds.variables[Config.VAR_LON][:], dtype=np.float64)
    return lat, lon


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SST 气候态计算 (高性能 Python 版)")
    parser.add_argument("--start_doy", type=int, default=Config.DEFAULT_START_DOY)
    parser.add_argument("--end_doy", type=int, default=Config.DEFAULT_END_DOY)
    parser.add_argument("--nc_path", type=str, default=Config.NC_PATH)
    parser.add_argument("--save_path", type=str, default=Config.SAVE_PATH)
    parser.add_argument(
        "--compute_workers",
        type=int,
        default=1,
        help="单 DOY 内 P90 分块进程数 (默认: 1，推荐保持 1)",
    )
    parser.add_argument(
        "--doy_workers",
        type=int,
        default=4,
        help="同时处理的 DOY 进程数 (默认: 4，每进程串行读文件)",
    )
    parser.add_argument(
        "--io_workers",
        type=int,
        default=1,
        help=argparse.SUPPRESS,
    )
    # 兼容旧参数
    parser.add_argument("--n_workers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no_dask", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--method", type=str, default="streaming", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    compute_workers = args.compute_workers
    doy_workers = args.doy_workers
    if args.no_dask:
        compute_workers = 1
        doy_workers = 1
    elif args.n_workers is not None:
        doy_workers = max(1, min(args.n_workers, 4))
    if args.io_workers > 1:
        logger.warning("共享文件系统不支持并行 NetCDF 读取，I/O 保持串行")

    Config.NC_PATH = args.nc_path
    Config.SAVE_PATH = args.save_path
    os.makedirs(Config.SAVE_PATH, exist_ok=True)

    sample = os.path.join(Config.NC_PATH, "19910101")
    if not os.path.exists(sample):
        logger.error("样本文件不存在: %s", sample)
        sys.exit(1)

    logger.info("读取经纬度...")
    lat, lon = read_coordinates(Config.NC_PATH)
    logger.info("网格: %d x %d", len(lat), len(lon))
    logger.info("DOY 范围: %d - %d", args.start_doy, args.end_doy)
    logger.info("P90 计算进程数: %d", compute_workers)
    logger.info("DOY 并行进程数: %d", doy_workers)

    doys = list(range(args.start_doy, args.end_doy + 1))
    total_t0 = time.perf_counter()

    if doy_workers <= 1:
        for doy in doys:
            _process_and_save_doy(
                doy, Config.NC_PATH, Config.SAVE_PATH, lat, lon, compute_workers
            )
    else:
        worker = partial(
            _process_and_save_doy,
            nc_path=Config.NC_PATH,
            save_path=Config.SAVE_PATH,
            lat=lat,
            lon=lon,
            compute_workers=compute_workers,
        )
        with ProcessPoolExecutor(max_workers=doy_workers) as pool:
            list(pool.map(worker, doys))

    total_elapsed = time.perf_counter() - total_t0
    n_doy = args.end_doy - args.start_doy + 1
    logger.info(
        "全部完成: %d 天, 总耗时 %.1fs, 平均 %.1fs/天",
        n_doy,
        total_elapsed,
        total_elapsed / max(n_doy, 1),
    )


if __name__ == "__main__":
    main()
