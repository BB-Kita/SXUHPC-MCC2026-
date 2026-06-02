# MCC26 GPU 优化实验结果摘要

**硬件环境**: 2 台服务器, 每台 4x K100_AI DCU (AMD MI100 gfx906), 32 核 CPU
**编译环境**: DTK 24.04, HIP, C++17
**数据规模**: 720x1440 全球海温格点, 1991-2020 共 30 年, 计算 6/1-8/31 共 92 天

---

## 1. 验证精度 (verification_full92_2node_t32)

| 指标 | 赛题要求 | 实际结果 | 状态 |
|------|---------|---------|------|
| 季度整体 P90 RMSE | < 2.0 °C | **0.0127 °C** | PASS |
| 季度整体 Clim RMSE | < 2.0 °C | **0.0000 °C** | PASS |
| 逐日 P90 RMSE (max) | < 1.0 °C | **0.0137 °C** | PASS |
| 逐日 Clim RMSE (max) | < 1.0 °C | **0.000004 °C** | PASS |

---

## 2. 全量 92 天运行时间

### 2.0 最新: reuse + SIMD + fadvise + t32 (mcc_dtk24_114724079)

| 阶段 | 耗时 | 说明 |
|------|------|------|
| **program_total** | **18.2s** | 最优单节点配置 |
| initialize_window | 0.60s | raw pread |
| 首天 (DOY 152, 全量上传 330 slots) | ~3.5s | - |
| 后续天数 (增量 upload) | ~0.04-0.12s/天 | upload 降至 17-24ms |
| DCU kernel (单次) | ~15ms | - |
| DCU upload (增量, reuse) | **~17-24ms** | 从 60ms 优化到 17ms |
| DCU download | ~0.33ms | - |

### 2.1 raw pread + IO t32 (mcc_dtk24_114706362)

| 阶段 | 耗时 | 占比 |
|------|------|------|
| **program_total** | **22.6s** | - |
| initialize_window | 0.30s | 1.3% |
| 首天 (DOY 152, 全量上传 330 slots) | 3.51s | 15.5% |
| 后续 91 天 (增量 30 slots/天) | ~0.17s/天 | - |
| DCU upload (增量) | ~60ms | - |

### 2.2 双节点 t32 (mcc_2node_114707451, 每节点处理 46 天)

| 节点 | 天数 | program_total |
|------|------|--------------|
| Node 1 (DOY 152-197) | 46 天 | 18.3s |
| Node 2 (DOY 198-243) | 46 天 | 19.5s |
| **整体 (含调度)** | 92 天 | **24.3s (real)** |

**双节点总时间 ≈ 24s, 接近单节点瓶颈 (受限于首天全量上传)**

### 2.3 全量 92 天 IO 线程数扫描 (full_iosweep_114723276)

| IO 线程数 | program_total |
|-----------|--------------|
| t8 | 24.0s |
| t16 | 19.3s |
| t24 | 19.2s |
| **t32** | **17.3s** |

---

## 3. IO 线程数扫描 (2 天 smoke test, DOY 152-153)

| IO 线程数 | program_total | 首天 day_total | 后续 day_total |
|-----------|--------------|---------------|---------------|
| t1 | 22.1s | 5.82s | 0.084s |
| t2 | 11.2s | 5.65s | 0.086s |
| t4 | 8.84s | 5.28s | 0.096s |
| t8 | 6.98s | 4.83s | 0.086s |
| t16 | 5.95s | 4.23s | 0.088s |
| **t32** | **5.17s** | **3.49s** | **0.081s** |

**t32 最优**, 首天从 5.82s 降至 3.49s (窗口初始化加速)

---

## 4. IO 优化迭代对比 (2 天 smoke test)

| 版本 | initialize_window | 首天总时间 | program_total | 备注 |
|------|-------------------|-----------|--------------|------|
| 初始增量 IO | N/A | 0.65s | N/A | baseline |
| IO 优化 v1 | 12.4s | 4.87s | 21.0s | 窗口初始化慢 |
| IO 优化 v2 | 15.1s | 6.25s | 23.9s | 回退 |
| IO 优化 v3 | 4.62s | 4.70s | 12.2s | 改进 |
| IO 优化 v4 | 1.02s | 4.47s | 15.3s | raw pread |
| **raw pread + t32** | **0.30s** | **3.51s** | **5.17s** | **最终方案** |

---

## 5. DCU 计算 kernel 性能

| 操作 | 耗时 | 说明 |
|------|------|------|
| P90 percentile kernel | ~15ms | 4 DCU 并行, rows_per_gpu=181 |
| Clim mean kernel | ~15ms | 同上 |
| 全量 upload (330 slots) | ~550ms | 首天, 1.28GB 数据 |
| 增量 upload (30 slots, 旧) | ~60ms | 无 reuse |
| 增量 upload (30 slots, reuse+simd+fadvise) | **~17-24ms** | 最新优化 |
| download 结果 | ~0.33ms | 极小 |

---

## 6. 关键优化点总结

1. **增量 IO 滑动窗口**: 避免每天重读 330 个文件, 仅滑动更新 30 个
2. **raw HDF5 pread**: 绕过 NetCDF 库开销, 直接 pread 原始数据
3. **多线程 IO (t32)**: 并行读取窗口文件, 首天初始化加速 40%
4. **persistent DCU buffers**: 避免重复分配设备内存
5. **buffer reuse + SIMD + fadvise**: 增量 upload 从 60ms 降至 17-24ms
6. **双节点分片**: 92 天对半分, 各自独立计算

---

## 7. 文件说明

| 文件/目录 | 说明 |
|-----------|------|
| `compute_dcu.cpp/h` | HIP/DCU 计算核心 (P90 + Clim) |
| `io_handler.cpp/h` | IO 模块 (NetCDF/raw pread, 滑动窗口) |
| `main.cpp` | 主程序入口 |
| `config.h` | 编译期配置 |
| `algo_p90.h` | P90 分位数算法 |
| `logs/` | 各实验运行日志 |
| `verification_latest/` | 最新验证 RMSE 结果 |
| `run_*.slurm` | 各种 Slurm 提交脚本 |
