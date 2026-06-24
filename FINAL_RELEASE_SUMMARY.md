# MCC26 最终版本归档总结

归档日期：2026-06-24

## 最终方案

最终采用单节点、32 个 CPU 核和 4 张 K100_AI：

`MC K=30 + source-stream + chunk=30 + std::thread I/O`

主要运行参数：

```bash
MCC_SOURCE_STREAM=1
MCC_SOURCE_MC_K=30
MCC_SOURCE_STREAM_CHUNK_DAYS=30
MCC_SOURCE_IO_BACKEND=stdthread
MCC_IO_THREADS=32
MCC_SOURCE_REGISTER_UPLOAD=1
MCC_RAW_DIRECT=1
MCC_OUTPUT_CLASSIC=1
```

正式入口为 `mcc_baseline.slurm`，编译产物为 `build/mcc_baseline`。

## 最终性能

固定节点 `f17r4n01` 三次完整运行：

| Job | shell real | program_total |
| --- | ---: | ---: |
| `115696553` | `7.27s` | `7.03717s` |
| `115696556` | `7.55s` | `7.21551s` |
| `115696562` | `7.44s` | `7.16731s` |

- 最快单次 shell `real=7.27s`
- 三次平均 shell `real=7.42s`
- OpenMP I/O 对照三次平均 `real=7.89s`
- std::thread 后端相对 OpenMP I/O 平均提升约 `6.0%`

排名计时以 `/usr/bin/time -p` 输出的 `real` 为准；`program_total` 仅用于程序内部 profiling。

## 最快输出

最快运行 job：`115696553`

输出目录：

```text
/public/home/fujiake/fjk/MCC26_SXU_lsd_repro_20260620/output_stdthread_c30_pair_0620c_r1
```

该目录包含 `0601.nc` 至 `0831.nc`，共 92 个 NetCDF 文件。

## 官方逻辑全量验证

验证 job：`115741330`

验证结果目录：

```text
/public/home/fujiake/fjk/MCC26_SXU_lsd_repro_20260620/verification_fastest_r1_full_0621
```

| 指标 | 平均 RMSE | 逐日最大 RMSE |
| --- | ---: | ---: |
| Clim | `0.203868 ℃` | `0.214721 ℃`，`0831.nc` |
| P90 | `0.334429 ℃` | `0.355384 ℃`，`0803.nc` |

- 92 天 Clim RMSE 大于或等于 `1 ℃` 的天数：`0`
- 92 天 P90 RMSE 大于或等于 `1 ℃` 的天数：`0`
- 季度平均 RMSE 均小于 `2 ℃`

验证满足赛题有效性要求。

`validate_official_logic.m` 保留官方数值逻辑，只增加路径配置、可选文件列表和逐日明细输出。最快版本与默认完整版本的逐日 RMSE 文件 SHA-256 完全一致。

## Profiling 结论

- source-stream 首次读取约 `1.72s`
- GPU kernel 末段约 `3.5-3.7ms/day`
- NetCDF 写出约 `0.002s/day`
- 主要瓶颈已经由 GPU 计算转移到 source chunk 读取、首次读取和异步等待
- 继续单独压缩 kernel 的边际收益有限

## 优化路径摘要

1. 将 Clim/P90 统计迁移到 4 张 K100_AI。
2. 使用滑动 source window，减少每日重复 H2D。
3. 使用 raw HDF5 offset、`pread` 和 `O_DIRECT` 降低读取开销。
4. 使用注册主机内存和增量上传降低 H2D 固定成本。
5. 使用 source-stream 和分块流水控制内存并覆盖部分 I/O。
6. 使用 MC K=30 降低 P90 统计成本，并通过官方精度验证。
7. 将 source I/O 从 OpenMP 调度改为 `std::thread` 原子任务队列。
8. 评估过双节点日期分块；短任务固定成本较高，最终采用单节点。

## 发布目标

同一最终提交将发布到：

- `git@github.com:BB-Kita/MCC26_SXU.git`，分支 `gpu-out-exp-v3`
- `git@github.com:SXU-HPC/MCC26.git`，分支 `main-v2`
