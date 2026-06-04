# MCC26_SXU GPU Optimization

MCC26 初赛海洋热浪阈值计算优化工程。当前版本基于 C++/HIP，在国产 K100_AI DCU 上计算 1991-2020 气候态海温 `Clim/Mean` 和海洋热浪阈值 `P90`。

目标输出为 6 月 1 日至 8 月 31 日，共 92 天。每个目标日使用前后各 5 天滑动窗口，30 年共 330 个样本。

## 当前最佳结果

运行环境：

- 单节点，4 x K100_AI DCU
- DTK 24.04.3，`gfx906`
- 输入目录：`/public/home/achwjznh4b/Newdata`
- 官方验证参考：`/public/home/achwjznh4b/ERA5/Climatology`

92 天完整运行：

| 版本 | real | program_total | 验证 |
| --- | ---: | ---: | --- |
| raw HDF5 `pread` + sliding window + persistent DCU buffer + local slice upload + `fadvise`/SIMD + `IO_THREADS=32` | **16.529s** | **16.3347s** | PASS |

官方验证精度：

| 指标 | RMSE |
| --- | ---: |
| Clim | **0.0000** |
| P90 | **0.0127** |

## 输入数据特征

输入文件位于 `/public/home/achwjznh4b/Newdata`，共 10980 个文件，覆盖 1991-2020 年逐日数据。

关键特征：

- NetCDF4/HDF5 文件
- 变量名：`data`
- 数据类型：`double`
- 维度：`721 x 1440`
- 单文件约 8.3 MB
- 数据连续存储，无 chunk/compression
- `data` 原始偏移：`23488`

因此当前代码会自动探测 HDF5 数据偏移，在匹配该布局时绕过 NetCDF API，直接用 `pread` 读取原始 double buffer，再转换为 float。

## 已采用优化

1. 滑动窗口 IO

   首日读取 330 个样本，后续每天只读取新增的 30 个文件，覆盖最旧 slot。

2. raw HDF5 `pread`

   对官方输入数据的连续 HDF5 布局直接读取原始 `data` 区域，减少 NetCDF/HDF5 API 开销。

3. 多线程 IO

   默认 `IO_THREADS=32`。完整 92 天 sweep 显示 32 线程最佳：

   | IO 线程数 | real | program_total |
   | ---: | ---: | ---: |
   | 8 | 24.414s | 24.0451s |
   | 16 | 19.530s | 19.296s |
   | 24 | 19.465s | 19.2192s |
   | 32 | **17.534s** | **17.3228s** |

4. DCU persistent buffer + 增量 H2D

   首日上传 330 slots，后续每天只上传 30 slots。设备端 buffer 持久化，避免重复分配。

5. local slice upload

   H2D 只上传每张 DCU 对应纬向切片，减少无用传输。后续增量上传约 17-24 ms。

6. `posix_fadvise` + SIMD conversion

   对 raw `pread` 路径提示顺序读取，并对 double-to-float 转换使用 OpenMP SIMD。当前 best 由该版本产生。

## 已评估但未采用

| 方向 | 结论 |
| --- | --- |
| CPU prefetch 与 kernel overlap | 完整 92 天 `real 22.928s`，受 CPU/文件系统竞争影响，慢于同步滑动读取 |
| `mmap` | microbenchmark 慢于 `pread`，不集成 |
| 每 IO 线程复用 scratch buffer | 完整 92 天 `real 18.508s`，慢于当前 best，已回退 |
| 双节点均分 92 天 | `real` 约 24.3s，受首日全量上传和调度影响，当前不如单节点 best |

## 主要文件

| 文件 | 说明 |
| --- | --- |
| `main.cpp` | 主流程，DOY 范围控制、滑动窗口、DCU 调度、输出 |
| `io_handler.cpp/.h` | NetCDF/raw HDF5 输入、滑动窗口 IO、raw offset 探测 |
| `compute_dcu.cpp/.h` | HIP/DCU kernel 与 4 卡调度 |
| `algo_p90.h` | P90 计算逻辑 |
| `config.h` | 网格尺寸、样本数、输入路径、IO 线程数 |
| `bench_mmap_vs_pread.cpp` | `mmap` 与 `pread` IO microbenchmark |
| `run_mcc26_dtk24.slurm` | 单节点 92 天主运行脚本 |
| `run_ioopt_2days.slurm` | 2 天 smoke test |
| `run_full_io_threads_sweep.slurm` | 92 天 IO 线程数 sweep |
| `run_validate_official_logic.slurm` | 官方验证逻辑封装 |

## 运行

单节点完整运行：

```bash
sbatch --export=ALL,MCC_IO_THREADS=32 run_mcc26_dtk24.slurm
```

2 天 smoke test：

```bash
sbatch --export=ALL,MCC_IO_THREADS=32 run_ioopt_2days.slurm
```

官方逻辑验证：

```bash
sbatch --export=ALL, \
  MCC_REF_CLIM_PATH=/public/home/achwjznh4b/ERA5/Climatology, \
  MCC_CONTESTANT_CLIM_PATH=/path/to/output, \
  MCC_VERIFY_SAVE_PATH=/path/to/verification \
  run_validate_official_logic.slurm
```

## 后续方向

当前主要瓶颈仍在 IO 和 H2D。优先考虑：

- 更细粒度的 H2D 与 kernel overlap，避免 CPU prefetch 与 IO 线程争用
- slot layout 与 GPU 访问模式进一步协同
- 只在有明确收益时再尝试异步双缓冲，避免额外同步和文件系统竞争
