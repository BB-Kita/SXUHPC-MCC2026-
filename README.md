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

## 项目结构

```
MCC26_SXU/
├── main.cpp              # 主程序入口：分配锁页内存 -> IO -> DCU 计算
├── io_handler.cpp/.h     # NetCDF 并行读取模块 (OpenMP 多线程)
├── compute_dcu.cpp/.h    # HIP DCU 计算内核 (4 卡异步调度)
├── algo_p90.h            # P90 算法设备端接口 (快速选择，备用)
├── config.h              # 常量定义：维度、路径、HIP 错误检查宏
├── CMakelists.txt        # CMake 构建脚本 (hipcc 编译)
├── run_mcc26.slurm       # Slurm 提交脚本
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
- 每张卡拷贝全量数据到显存，仅计算分配到的空间切片
- 使用 HIP Stream 异步调度，kernel 并行执行
- 结果通过 `hipMemcpyAsync` 回传，pin memory 对接 PCIe

### IO 模块 (io_handler.cpp)

- OpenMP 多线程并行读取 NetCDF 文件
- 自动检测数据维度布局 (lat-lon / lon-lat)，必要时转置
- 处理 `_FillValue`、`missing_value`、`scale_factor`、`add_offset`
- 365 天日历 (剔除 2 月 29 日)

## 构建与运行

### 依赖

- 海光 DTK (hipcc 编译器)
- NetCDF-C 库
- OpenMP
- conda `lsd` 环境 (已在平台上配置)

### 编译

```bash
# 激活环境
source /public/home/fujiake/miniconda3/bin/activate
conda activate lsd

# CMake 构建
mkdir -p build && cd build
cmake ..
make -j

# 或手动编译
hipcc -std=c++14 -O3 -fopenmp \
    main.cpp io_handler.cpp compute_dcu.cpp \
    -I. -I${CONDA_PREFIX}/include \
    -L${CONDA_PREFIX}/lib -lnetcdf \
    -Wl,-rpath,${CONDA_PREFIX}/lib \
    -o build/mcc_baseline
```

### 提交作业

```bash
sbatch run_mcc26.slurm
```

### 查看运行状态

```bash
squeue                    # 查看作业队列
cat logs/mcc26_*.out      # 查看标准输出
hy-smi                    # 查看 DCU 使用情况
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
| `IO_THREADS` | 4 | IO 并行线程数 |

## 性能参考

| 版本 | 用时 | 说明 |
|------|------|------|
| MATLAB baseline | >16h | 单核，赛题提供 |
| Python (多进程) | ~3min20s | 双机 64 核 |
| Python (预加载) | ~50s | 全量数据预加载到内存 |
| C++/HIP (本项目) | TBD | 4 DCU 并行 |

## 待完成

- [ ] 在比赛平台上编译并运行
- [ ] 使用验证脚本 (`clim_verification.m`) 检查精度
- [ ] 性能调优 (profiling、内存带宽优化)
- [ ] 输出结果到指定目录 (`ERA5/Climatology/`)

## 比赛关键时间

- 提交截止：**2026 年 6 月 15 日 - 6 月 21 日**

## 参考资源

- 赛题网址：https://www.paratera.com/event_detail/1.html
- 海光 DCU 命令：`hy-smi`、`rocm-smi`、`rocminfo`
- 验证数据集：`~/data` 目录下
