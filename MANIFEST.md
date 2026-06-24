# MCC26 最终提交代码包清单

最终方案：`MC K=30 + source-stream + chunk=30 + stdthread I/O`。

## 入口文件

- `build.sh`：DTK 25.04 编译脚本。
- `mcc_baseline.slurm`：单节点、4 卡最终运行入口，使用 `/usr/bin/time -p` 输出正式计时。
- `run_validate_official_logic.slurm`：92 天官方逻辑验证包装脚本。
- `validate_official_logic.m`：可配置的逐日及季度 RMSE 验证程序。
- `clim_verification.m`、`clim_verification.sh`：官方原始验证脚本参考。

## 主要代码模块

- `main.cpp`：任务调度、source-stream 流水线、92 日循环、结果输出和 profiling。
- `io_handler.cpp`、`io_handler.h`：NetCDF/HDF5 元数据、raw 块读取、O_DIRECT/pread、source chunk 并行 I/O 和 std::thread 后端。
- `compute_dcu.cpp`、`compute_dcu.h`：DCU 数据上传及 Clim/P90 计算。
- `config.h`：数据维度、年份范围、11 日窗口和编译期配置。

## 最终资源与算法口径

- 单节点，32 CPU 核，4 张 K100_AI。
- 1991-2020 年、11 日窗口、6 月 1 日至 8 月 31 日，共 92 天。
- 本届赛题不执行 31 天滑动平均。
- 默认参数：`MCC_SOURCE_MC_K=30`、`MCC_SOURCE_STREAM_CHUNK_DAYS=30`、`MCC_SOURCE_IO_BACKEND=stdthread`、`MCC_RAW_DIRECT=1`。

## 已确认结果

- 三次完整运行平均 shell `real=7.42s`。
- 最快单次 job `115696553`，shell `real=7.27s`。
- 最快输出的全量验证 job `115741330`，覆盖全部 92 个输出文件。
- Clim 平均/逐日最大 RMSE：`0.203868/0.214721 ℃`。
- P90 平均/逐日最大 RMSE：`0.334429/0.355384 ℃`。
- 精度满足逐日小于 `1 ℃`、季度平均小于 `2 ℃` 的要求。

本目录不包含 NetCDF 输出、日志、编译产物、core 文件或其他大型实验数据。
