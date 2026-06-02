# MCC26_SXU - 海洋热浪阈值计算 (C++/HIP DCU 加速)

MCC26 超算竞赛初赛参赛作品。基于 1991-2020 年海表温度融合资料，使用海光 K100 DCU 四卡并行计算海洋热浪阈值 (P90) 和气候态海温 (Mean)。

## 赛题概述

在全球气候变暖背景下，海洋热浪频发。本赛题要求基于气候基准期 (1991-2020) 的 SST 数据，计算两个核心指标：

- **气候态海温 (Mean)**：以每日为中心、前后各 5 天形成 11 天窗口，提取 30 年共 330 个样本，取算术平均
- **海洋热浪阈值 (P90)**：同上 330 个样本，按升序排列后取第 90 百分位数

仅计算 6 月 1 日至 8 月 31 日 (DOY 152-243) 共 92 天。

### 精度要求

- 逐日 RMSE < 1 C
- 季度整体平均 RMSE < 2 C

### 硬件约束

- 最多 2 台服务器，单机 32 核 + 4 x K100 DCU
- 运行时间限制 2 小时

## 验证结果

| 指标 | 赛题要求 | 实际结果 |
|------|---------|---------|
| 季度整体 P90 RMSE | < 2.0 C | **0.0127 C** |
| 季度整体 Clim RMSE | < 2.0 C | **0.0000 C** |
| 逐日 P90 RMSE (max) | < 1.0 C | **0.0137 C** |
| 逐日 Clim RMSE (max) | < 1.0 C | **0.000004 C** |

## 性能数据

硬件：2 台服务器, 每台 4x K100_AI DCU (gfx906), 32 核 CPU

### 92 天全量运行

| 配置 | 总时间 (real) | 说明 |
|------|--------------|------|
| 单节点 DTK24 raw pread t32 | **24.7s** | 最优单节点配置 |
| 双节点 t32 (各 46 天) | **24.3s** | 受限于首天全量上传 |

### DCU 计算 kernel

| 操作 | 耗时 | 说明 |
|------|------|------|
| P90/Clim kernel | ~15ms | 4 DCU 并行, rows_per_gpu=181 |
| 全量 upload (330 slots) | ~550ms | 首天 |
| 增量 upload (30 slots) | ~60ms | 后续每天 |
| download 结果 | ~0.35ms | |

### IO 线程数扫描 (2 天 smoke test)

| 线程数 | program_total | 首天 day_total |
|--------|--------------|---------------|
| t1 | 22.1s | 5.82s |
| t8 | 6.98s | 4.83s |
| t32 | 5.17s | 3.49s |

### 对比其他方案

| 版本 | 用时 | 说明 |
|------|------|------|
| MATLAB baseline | >16h | 单核，赛题提供 |
| Python (多进程) | ~3min20s | 双机 64 核 |
| Python (预加载) | ~50s | 全量数据预加载到内存 |
| **C++/HIP DCU (本项目)** | **~24s** | 4 DCU 并行, raw pread, t32 IO |

## 项目结构

```
MCC26_SXU/
├── main.cpp                  # 主程序入口
├── io_handler.cpp/.h         # IO 模块 (NetCDF / raw pread, 滑动窗口)
├── compute_dcu.cpp/.h        # HIP DCU 计算内核 (P90 + Clim)
├── algo_p90.h                # P90 分位数算法
├── config.h                  # 常量定义：维度、路径、IO 线程数等
├── probe_hdf5_offset.cpp     # HDF5 数据偏移探测工具
├── CMakelists.txt            # CMake 构建脚本 (hipcc)
│
├── run_mcc26.slurm           # 基础 Slurm 提交脚本
├── run_mcc26_dtk24.slurm     # DTK24 编译运行
├── run_mcc26_two_node_t32.slurm  # 双节点 t32 运行
├── run_io_threads_sweep.slurm    # IO 线程数扫描实验
├── run_ioopt_2days.slurm     # IO 优化 2 天 smoke test
├── run_inc_2days.slurm       # 增量 IO 2 天测试
├── run_debug_1day.slurm      # 调试用 1 天运行
├── run_probe_*.slurm         # 各种探测脚本 (arch/dtk/flags/simple)
├── run_validate_official_logic.slurm  # 官方逻辑验证
│
├── get_climatology.m/sh      # 赛题基准脚本
├── clim_verification.m/sh    # 赛题验证脚本
├── validate_official_logic.m # 官方逻辑 MATLAB 验证
├── verify.py                 # Python RMSE 验证
│
├── logs/                     # 运行日志
│   ├── probe_arch/           # 架构探测
│   ├── probe_debug/          # 调试日志
│   ├── probe_dtk/            # DTK 版本探测
│   ├── probe_flags/          # 编译选项探测
│   ├── probe_simple/         # 简单 kernel 探测
│   ├── run_2day/             # 2 天 smoke test
│   ├── run_2node/            # 双节点运行
│   ├── run_full/             # 全量 92 天运行
│   ├── run_iosweep/          # IO 线程数扫描
│   └── run_verify/           # 验证运行
│
├── verification_latest/      # 最新 RMSE 验证结果
│   ├── RMSE_P90.txt          # 季度 P90 RMSE
│   ├── RMSE_clim.txt         # 季度 Clim RMSE
│   ├── RMSE_P90_daily.txt    # 逐日 P90 RMSE
│   └── RMSE_clim_daily.txt   # 逐日 Clim RMSE
│
├── BENCHMARK_RESULTS.md      # 详细性能数据
└── .gitignore
```

## 算法设计

### 计算内核 (compute_dcu.cpp)

采用混合直方图算法，避免对 330 个样本做全排序，显著降低寄存器压力：

| Pass | 操作 | 说明 |
|------|------|------|
| 1 | 扫描 | 计算 mean、min、max，过滤 NaN |
| 2 | 粗直方图 | 64-bin 直方图，定位目标 rank 所在 bin |
| 3 | 自适应收集 | 从目标 bin 向两侧扩展，收集 rank 附近的值到 buffer (max 120) |
| 4 | 插入排序 | 对 buffer 排序，精确插值得 P90 |

**数据布局**: `data[day * spatial_points + spatial_idx]`，同一 warp 内 consecutive threads 读 consecutive spatial_idx，实现合并访存。

### 多卡并行

- 4 张 DCU 按 lat 行切分 (721 行 / 4 卡)
- persistent DCU buffers，避免重复分配设备内存
- 使用 HIP Stream 异步调度，kernel 并行执行
- 结果通过 `hipMemcpyAsync` 回传，pin memory 对接 PCIe

### IO 优化

- **增量滑动窗口**：避免每天重读 330 个文件，仅滑动更新 30 个
- **raw HDF5 pread**：绕过 NetCDF 库开销，直接 pread 原始 HDF5 数据
- **多线程 IO (t32)**：并行读取窗口文件，首天初始化加速 40%
- 自动检测数据维度布局 (lat-lon / lon-lat)，必要时转置
- 处理 `_FillValue`、`missing_value`、`scale_factor`、`add_offset`
- 365 天日历 (剔除 2 月 29 日)

### 双节点分片

- 92 天对半分 (DOY 152-197 / 198-243)
- 各节点独立计算，无需节点间通信
- `srun` 分配到不同节点，各跑 46 天

## 构建与运行

### 依赖

- 海光 DTK 24.04 (hipcc 编译器)
- NetCDF-C 库
- OpenMP
- conda `lsd` 环境

### 编译

```bash
source /public/home/fujiake/miniconda3/bin/activate
conda activate lsd

mkdir -p build && cd build
cmake ..
make -j

# 或手动编译
hipcc -std=c++17 -O3 -fopenmp \
    main.cpp io_handler.cpp compute_dcu.cpp \
    -I. -I${CONDA_PREFIX}/include \
    -L${CONDA_PREFIX}/lib -lnetcdf \
    -Wl,-rpath,${CONDA_PREFIX}/lib \
    -o build/mcc_baseline_inc
```

### 提交作业

```bash
# 单节点全量运行
sbatch run_mcc26_dtk24.slurm

# 双节点运行
sbatch run_mcc26_two_node_t32.slurm

# IO 线程数扫描
sbatch run_io_threads_sweep.slurm
```

### 验证结果

```bash
# Python 验证
python verify.py

# MATLAB 官方验证
matlab -nodisplay -nosplash -nodesktop < validate_official_logic.m
```

## 配置参数

在 `config.h` 中修改：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `LON_SIZE` | 1440 | 经度网格数 |
| `LAT_SIZE` | 721 | 纬度网格数 |
| `CLIM_START_YEAR` | 1991 | 气候基准期起始年 |
| `CLIM_END_YEAR` | 2020 | 气候基准期结束年 |
| `CLIM_DELTA_DAY` | 5 | 前后延伸天数 |
| `NC_INPUT_DIR` | `/public/home/achwjznh4b/Newdata/` | 输入数据目录 |
| `IO_THREADS` | 32 | IO 并行线程数 |

## 比赛关键时间

- 提交截止：**2026 年 6 月 15 日 - 6 月 21 日**

## 参考资源

- 赛题网址：https://www.paratera.com/event_detail/1.html
- 海光 DCU 命令：`hy-smi`、`rocm-smi`、`rocminfo`
- 验证数据集：`~/data` 目录下
