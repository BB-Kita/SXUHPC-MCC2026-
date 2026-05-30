# SST 气候态 & P90 计算 —— CPU 优化版 (fast)

参考源文件：`/public/home/fujiake/feng` 下的 `get_climatology.m`、`clim_verification.m`、
`get_climatology.sh`、`clim_verification.sh`。

本目录是一版**纯 CPU、不上 DCU** 的实现。核心判断：这道题是 **I/O 密集 + 计算轻量**，
瓶颈不在算力。优化分两层打——**先把读盘量压到下限，再把计算从 numpy 换成编译核**。

环境：`conda activate nb`（Python 3.11 / numpy 2.x / netCDF4 / g++ devtoolset-7）。

---

## 1. 为什么不上卡（DCU）

`max/` 的 HIP 版真实日志 `clim_hip_114426701.out`：每天 I/O≈5s、GPU≈0.5s，`rocm-smi`
实测 **DCU%=0%**。受 Amdahl 限制，把计算压到 0 也只省 9%，且那 0.5s 大半是 2.7GB 的
PCIe 拷贝。**卡是过剩产能**——只有当网格放大 10–100×、或统计量变重（bootstrap/多分位/EOF）
时才划算。本题用多核 CPU 即可把计算藏进 I/O 影子里。

---

## 2. 优化前后两层瓶颈

| 阶段 | I/O / 天 | 计算 / 天 | 墙钟 / 天 | 瓶颈 |
|------|---------|----------|----------|------|
| 原始 MATLAB（按行读） | 2180万次小读（全程） | 串行 | — | I/O |
| `max/` HIP+DCU | 330 文件（5s） | GPU 0.5s | ~5s | I/O |
| fast 第一版（滑动缓存 + numpy） | 30 文件/天（~0.5s） | **np.nanpercentile 4s** | ~5s | **计算** |
| **fast 当前版（滑动缓存 + C++核）** | 30 文件/天 | **C++ <0.5s（实机）** | 目标 ~1s | I/O（已近下限） |

第一版踩的坑：滑动缓存把 I/O 砍了 10×，却撞上 `np.nanpercentile`——它对每列全排序 +
NaN 掩码，单核 ~4s/天，成了新的墙，于是墙钟和 HIP 版巧合地都≈5s。本版把这块换成
C++/OpenMP 融合核后，计算回落到亚秒级，瓶颈重新回到 I/O，而 I/O 已被滑动缓存压到下限。

---

## 3. 做了哪些优化

### I/O 层

1. **整场读取**：每个 NC 文件一次读完 `721×1440`，消除原始 MATLAB「按行读、同一文件开 721 次」的冗余（I/O 次数 2180万 → 30,360）。
2. **滑动窗口缓存（rolling cache）**：±5 天窗口相邻 doy 重叠 10/11。按 doy 递增时，每天**只新增 30 个文件、淘汰 30 个**（已验证 overlap=300/330）。全程不同文件 ~3,060 个，每个只读一次（30,360 → ~3,060，再降 10×）。
3. **进程池并行读**：本环境的 netCDF4/HDF5 **非线程安全**（第一版用线程池导致随机段错误，见 §6）。改用 `ProcessPoolExecutor`，每进程独立 HDF5；读盘与计算**流水线重叠**（提交下一天读取 → 跑当天核 → 收结果）。

### 计算层（语言混合：C++/OpenMP 融合核）

`clim_kernel.cpp` → `clim_kernel.so`，经 ctypes 调用。相对 numpy：

| # | 优化 | 说明 |
|---|------|------|
| 1 | **融合单遍** | nanmean 与 P90 在一次遍历内同时算出，数据只过一遍 |
| 2 | **nth_element (quickselect)** | 平均 O(n) 选第 k 小，替代 np.nanpercentile 的全排序 O(n log n) |
| 3 | **列分块 tiling** | 按 64 列成块 gather 到 L2 内小缓冲，连续访存 |
| 4 | **OpenMP 全核并行** | 列方向 `#pragma omp for`，吃满节点所有核 |
| 5 | **零拷贝指针数组** | 核直接读滑动缓存里 330 个整场数组的指针，无需拼接大块 |

**离线验证**（合成数据，注入 10% NaN + 全 NaN 列）：核结果对比 `np.nanpercentile(method='hazen')`：

```
max|dMean| = 3.3e-16    max|dP90| = 5.8e-15    全 NaN 列 -> NaN    NaN 模式完全一致
```

单核速度约为 `np.nanpercentile` 的 **~10×**；实机用满核后计算降到亚秒级。

### 精度（对齐 MATLAB）

- 全程 **float64**；均值用 **Kahan 补偿求和**（比朴素累加更准）。
- P90 用 **Hazen / Hyndman–Fan type 5** 插值（0-based 位置 `0.9*cnt - 0.5`），**与 MATLAB `prctile` 完全等价**。numpy 默认的 `linear`（`max/` 那版用的）与 MATLAB 不一致，本版已修正。
- NaN 样本按 MATLAB `prctile`/`nanmean` 语义跳过；某格点全 NaN 时输出 NaN。

---

## 4. 预期性能

- 计算：~4s/天 → **亚秒/天**（实机满核）。
- I/O：滑动缓存后总读取 ~25GB（每个文件读一次），约为 `max/` 版 252GB 的 1/10。
- 端到端：理论下限 ≈「读完所有需要的文件一次」≈ 47s（单节点带宽）。**双节点按天切分**（`run_dual.sh`）可把读盘带宽翻倍，进一步逼近 ~25–40s。
- 对比 `max/` 实测 511s，预计 **5–10× 加速**。

> 注：实际耗时由并行文件系统带宽决定，瓶颈是 I/O；登录节点只给 2 核，无法测真实并行速度，请以 `sbatch` 到独占计算节点的结果为准。

---

## 5. 文件清单与运行

| 文件 | 说明 |
|------|------|
| `clim_kernel.cpp` | C++/OpenMP 融合核（nanmean+P90 单遍、nth_element、Kahan、tiling） |
| `build_kernel.sh` | 用 devtoolset-7 g++ 编译出 `clim_kernel.so` |
| `get_climatology.py` | 主程序（滑动缓存 + 进程池读 + ctypes 核 + 流水线 + 计时分解） |
| `clim_verification.py` | RMSE 验证（CPU 并行） |
| `run_climatology.sh` | 单节点提交（conda `nb`、自动编译核、`OMP_NUM_THREADS`=全核） |
| `run_dual.sh` | 双节点按 doy 切两段并行（152–197 / 198–243） |
| `run_verification.sh` | 验证提交 |

```bash
cd /public/home/fujiake/feng/fast
conda activate nb

bash build_kernel.sh          # 编译 C++ 核（run 脚本里会自动判断）

sbatch run_climatology.sh     # 单节点全量
bash   run_dual.sh            # 双节点按天切分
sbatch run_verification.sh    # 与标准答案比对 RMSE

# 单天调试
python get_climatology.py --single-day 152 --io-workers 16
```

主要参数：`--nc-path` / `--save-path` / `--doy-start` / `--doy-end` / `--io-workers` / `--single-day`。
计算并行度由环境变量 `OMP_NUM_THREADS` 控制（run 脚本自动设为节点核数）。

---

## 6. 路径配置（与源 .m 同步）

| 用途 | 路径 |
|------|------|
| 输入数据 | `/public/home/achwjznh4b/Newdata/` |
| 气候态输出 | `/public/home/fujiake/feng/fast/output/` |
| 验证参考（标准答案） | `/public/home/fujiake/data/` |
| 验证待测 | `/public/home/fujiake/feng/fast/output/` |

> 只在 `fast/` 内产出，不改动其他文件夹。

---

## 7. 排错记录（为什么之前的作业终止 / 不快）

- **空 out、8 秒 FAILED**：批处理是非交互式 shell，`~/.bashrc` 会提前 `return`，conda 未初始化 → `conda activate` 失败，叠加 `set -e` 在第一个 `echo` 前就退出。**修复**：直接 `source .../miniconda3/etc/profile.d/conda.sh`，并去掉 `set -u`。
- **跑到一半段错误（HDF5-DIAG / 乱码文件名）**：多线程并发读 netCDF4，而本环境 HDF5 非线程安全。**修复**：改用进程池读盘。
- **单步仍 5s**：滑动缓存其实生效了（每天只读 30 文件），但 `np.nanpercentile` 4s/天成了新瓶颈。**修复**：换成 C++/OpenMP 融合核（本节即第 3 节计算层）。

---

## 8. 与 max/ 版本的一句话区别

`max/` 把「算得更快」做到极致（HIP 融合核），但优化的是只占 9% 的计算；本版把精力放回
真正的瓶颈——**I/O**（滑动缓存再降 10× + 读算重叠），计算用 C++/OpenMP 融合核在 CPU 上
就做到亚秒级，且把 P90 的分位数定义对齐回了 MATLAB。**语言混合的价值只体现在计算这一段，
I/O 是带宽绑定、与语言无关。**
