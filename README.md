# MCC26 最终提交与归档版本

本目录为 2026-06-24 确认的最终提交归档，包含提交所需的源代码和脚本，对应方案：

`MC K=30 + source-stream + chunk=30 + stdthread I/O`

源自集群复现实验目录：

`/public/home/fujiake/fjk/MCC26_SXU_lsd_repro_20260620`

目录中不包含 NetCDF 输出、profiling 日志、编译产物或大型中间文件。

最终发布分支：

- `BB-Kita/MCC26_SXU`：`gpu-out-exp-v3`
- `SXU-HPC/MCC26`：`main-v2`

完整归档结论见 `FINAL_RELEASE_SUMMARY.md`。

## 赛题口径

- 计算 6 月 1 日至 8 月 31 日，共 92 天。
- 每日使用前后各 5 天的 11 日窗口和 1991-2020 年 30 年数据。
- 输出气候态均值和海洋热浪 P90 阈值。
- 按本届赛题要求，不计算 31 天滑动平均。
- 最终方案使用单节点、32 个 CPU 核和 4 张 K100_AI 加速卡。

## 编译与运行

```bash
chmod +x build.sh
./build.sh
sbatch mcc_baseline.slurm
```

运行脚本默认启用：

```bash
MCC_SOURCE_STREAM=1
MCC_SOURCE_MC_K=30
MCC_SOURCE_STREAM_CHUNK_DAYS=30
MCC_SOURCE_IO_BACKEND=stdthread
MCC_IO_THREADS=32
OMP_NUM_THREADS=32
MCC_SOURCE_REGISTER_UPLOAD=1
MCC_RAW_DIRECT=1
MCC_OUTPUT_CLASSIC=1
```

输出目录默认为 `./output`。脚本使用 `/usr/bin/time -p`，并把 `real/user/sys` 合并写入 Slurm 标准输出。

可切换到 OpenMP I/O 后端进行回退测试：

```bash
sbatch --export=ALL,MCC_SOURCE_IO_BACKEND=openmp mcc_baseline.slurm
```

## 性能结果

固定节点 `f17r4n01` 上，最终 stdthread 配置的三次完整运行：

- job `115696553`：`real=7.27s`，`program_total=7.03717s`
- job `115696556`：`real=7.55s`，`program_total=7.21551s`
- job `115696562`：`real=7.44s`，`program_total=7.16731s`
- 平均 `real=7.42s`

其中最快单次为 job `115696553`。对应 OpenMP I/O 后端三次平均 `real=7.89s`。排名计时应以 `/usr/bin/time -p` 输出的 shell `real` 为准，程序内部 `program_total` 仅用于 profiling，两者不可混用。

## 官方逻辑全量验证

使用 `/public/home/achwjznh4b/ERA5/Climatology` 为参考数据，对最快 job `115696553` 的 92 个输出文件执行全量验证：

- 最快输出目录：`/public/home/fujiake/fjk/MCC26_SXU_lsd_repro_20260620/output_stdthread_c30_pair_0620c_r1`
- 全量验证 job：`115741330`
- 验证结果目录：`/public/home/fujiake/fjk/MCC26_SXU_lsd_repro_20260620/verification_fastest_r1_full_0621`

- 气候态平均 RMSE：`0.203868 ℃`
- 气候态逐日最大 RMSE：`0.214721 ℃`（`0831.nc`）
- P90 平均 RMSE：`0.334429 ℃`
- P90 逐日最大 RMSE：`0.355384 ℃`（`0803.nc`）
- 气候态与 P90 的逐日 RMSE 大于或等于 `1 ℃` 的天数均为 `0`

所有逐日结果均小于 `1 ℃`，季度平均结果均小于 `2 ℃`，满足赛题有效性要求。

### 组委会复核提示

`validate_official_logic.m` 与官方 `clim_verification.m` 使用相同的数值逻辑：

1. 从参考结果和参赛结果中读取 `Climmean` 与 `P90_sst`。
2. 逐日计算 `sqrt(nanmean((reference(:) - contestant(:)).^2))`。
3. 对 6 月 1 日至 8 月 31 日的 92 个逐日 RMSE 使用 `nanmean` 计算季度统计值，并使用 `max` 统计逐日最大值。

包装版只增加了环境变量路径、可选文件列表、逐日 RMSE 日志和明细文件，不改变官方计算公式。官方脚本本身只输出误差值、不主动执行阈值判定；本项目另外统计了超阈值天数，以便复核。

最快版本与默认完整版本的 `RMSE_clim_daily.txt`、`RMSE_P90_daily.txt` 分别具有完全相同的 SHA-256，说明两次运行的 92 天逐日验证结果一致。

验证命令示例：

```bash
sbatch --export=ALL,\
MCC_REF_CLIM_PATH=/public/home/achwjznh4b/ERA5/Climatology,\
MCC_CONTESTANT_CLIM_PATH="$PWD/output",\
MCC_VERIFY_SAVE_PATH="$PWD/verification_final" \
run_validate_official_logic.slurm
```

`clim_verification.m` 和 `clim_verification.sh` 为官方原始验证脚本参考；可配置、可批量验证的提交包装入口为 `run_validate_official_logic.slurm`。

## Profiling 结论

MC K=30 后，每日 GPU kernel 约为 `3.5 ms`，主要瓶颈已转移到 source chunk 读取、first-read 和异步读取等待。source-stream、分块流水、O_DIRECT/pread、注册主机内存上传和 std::thread 并行读取共同降低了 I/O/H2D 固定成本；继续只优化 GPU kernel 的边际收益有限。
