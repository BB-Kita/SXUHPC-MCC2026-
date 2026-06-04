#include "compute_dcu.h"
#include "config.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <limits>
#include <omp.h>

// ============================================================
// Hybrid P90 Kernel: coarse histogram → adaptive collect → exact interp
//
// 每个线程处理 1 个空间点 (lat, lon):
//   Pass 1: 扫描, 计算 mean / min / max
//   Pass 2: 64-bin 粗直方图定位目标 rank 所在 bin
//   Pass 3: 自适应收集 — 从目标 bin 向两侧扩展，直到 buffer
//           包含 floor(rank) 和 ceil(rank) 两个位置的值
//   Pass 4: 插入排序 buffer, 精确插值得 P90
//
// 数据布局: data[day * spatial_points + spatial_idx]
// 同一 warp 内 consecutive threads 读 consecutive spatial_idx → 合并访存
// ============================================================

static constexpr int COARSE_BINS = 64;
static constexpr int COLLECT_MAX = 120;

__global__ void compute_mean_p90_kernel(
    const float* __restrict__ data,
    float* __restrict__ out_mean,
    float* __restrict__ out_p90,
    int spatial_points,
    int days_total,
    int spatial_start,
    int spatial_end,
    int data_spatial_points,
    int data_spatial_start)
{
    int global_idx = spatial_start + (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (global_idx >= spatial_end) return;

    int out_idx = global_idx - spatial_start;
    int data_idx = global_idx - data_spatial_start;

    // ---- Pass 1: scan for mean / min / max ----
    float sum = 0.0f;
    int valid_count = 0;
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int d = 0; d < days_total; ++d) {
        float v = data[d * data_spatial_points + data_idx];
        if (!isnan(v)) {
            sum += v;
            ++valid_count;
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
        }
    }

    if (valid_count == 0) {
        out_mean[out_idx] = NAN;
        out_p90[out_idx]  = NAN;
        return;
    }

    out_mean[out_idx] = sum / (float)valid_count;

    if (valid_count == 1) {
        out_p90[out_idx] = vmin;
        return;
    }
    if (vmin == vmax) {
        out_p90[out_idx] = vmin;
        return;
    }

    // ---- Pass 2: coarse histogram ----
    float hist_f[COARSE_BINS];
    for (int b = 0; b < COARSE_BINS; ++b) hist_f[b] = 0.0f;

    float inv_range = (float)COARSE_BINS / (vmax - vmin);

    for (int d = 0; d < days_total; ++d) {
        float v = data[d * data_spatial_points + data_idx];
        if (!isnan(v)) {
            int bin = (int)((v - vmin) * inv_range);
            if (bin < 0) bin = 0;
            if (bin >= COARSE_BINS) bin = COARSE_BINS - 1;
            hist_f[bin] += 1.0f;
        }
    }

    // ---- Pass 3: locate target bin ----
    float target_rank = 0.9f * (float)(valid_count - 1);
    int rank_lo = (int)target_rank;
    int rank_hi = rank_lo + 1;
    float frac = target_rank - (float)rank_lo;
    bool need_interp = (frac > 1e-6f && rank_hi < valid_count);

    int target_bin = COARSE_BINS - 1;
    float cumulative = 0.0f;

    for (int b = 0; b < COARSE_BINS; ++b) {
        if (cumulative + hist_f[b] > (float)rank_lo) {
            target_bin = b;
            break;
        }
        cumulative += hist_f[b];
    }

    // ---- Pass 4: adaptive collection ----
    float bin_width = (vmax - vmin) / (float)COARSE_BINS;
    int lo_bin = target_bin;
    int hi_bin = target_bin + 1;

    float buf[COLLECT_MAX];
    int buf_count = 0;
    float cum_lo = cumulative;

    bool success = false;
    for (int expand = 0; expand < COARSE_BINS; ++expand) {
        float c_lo = vmin + (float)lo_bin * bin_width;
        float c_hi = vmin + (float)hi_bin * bin_width;
        if (c_lo < vmin) c_lo = vmin;
        if (c_hi > vmax) c_hi = vmax;

        buf_count = 0;
        for (int d = 0; d < days_total && buf_count < COLLECT_MAX; ++d) {
            float v = data[d * data_spatial_points + data_idx];
            if (!isnan(v) && v >= c_lo && v < c_hi) {
                buf[buf_count++] = v;
            }
        }

        if (!need_interp) {
            float local_lo = (float)rank_lo - cum_lo;
            if (local_lo >= 0.0f && local_lo < (float)buf_count) {
                success = true;
                break;
            }
        } else {
            float local_lo = (float)rank_lo - cum_lo;
            float local_hi_f = (float)rank_hi - cum_lo;
            if (local_lo >= 0.0f && local_hi_f < (float)buf_count) {
                success = true;
                break;
            }
        }

        if (lo_bin > 0) {
            --lo_bin;
            cum_lo -= hist_f[lo_bin];
        }
        if (hi_bin < COARSE_BINS) {
            ++hi_bin;
        }
        if (buf_count >= COLLECT_MAX - 10) break;
    }

    if (!success || buf_count == 0) {
        out_p90[out_idx] = vmin + (float)(target_bin + 0.5f) * bin_width;
        return;
    }

    // ---- Pass 5: insertion sort ----
    for (int i = 1; i < buf_count; ++i) {
        float key = buf[i];
        int j = i - 1;
        while (j >= 0 && buf[j] > key) {
            buf[j + 1] = buf[j];
            --j;
        }
        buf[j + 1] = key;
    }

    // ---- Extract P90 ----
    float local_lo = (float)rank_lo - cum_lo;
    int ilo = (int)local_lo;

    if (!need_interp || ilo + 1 >= buf_count) {
        out_p90[out_idx] = buf[ilo];
    } else {
        float local_hi_f = (float)rank_hi - cum_lo;
        int ihi = (int)local_hi_f;
        if (ihi >= buf_count) ihi = buf_count - 1;
        out_p90[out_idx] = buf[ilo] + frac * (buf[ihi] - buf[ilo]);
    }
}

__global__ void compute_season_batch_kernel(
    const float* __restrict__ data,
    float* __restrict__ out_mean_all,
    float* __restrict__ out_p90_all,
    int target_days,
    int season_days,
    int local_spatial_points)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = target_days * local_spatial_points;
    if (global_idx >= total_outputs) return;

    int out_day = global_idx / local_spatial_points;
    int spatial = global_idx - out_day * local_spatial_points;
    int start_day = out_day;

    float sum = 0.0f;
    int valid_count = 0;
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int wd = 0; wd < CLIM_WINDOW_DAYS; ++wd) {
        int season_day = start_day + wd;
        if (season_day < 0 || season_day >= season_days) continue;

        const float* day_base = data +
            static_cast<size_t>(season_day) * CLIM_YEARS * local_spatial_points;
        for (int yr = 0; yr < CLIM_YEARS; ++yr) {
            float v = day_base[static_cast<size_t>(yr) * local_spatial_points + spatial];
            if (!isnan(v)) {
                sum += v;
                ++valid_count;
                if (v < vmin) vmin = v;
                if (v > vmax) vmax = v;
            }
        }
    }

    if (valid_count == 0) {
        out_mean_all[global_idx] = NAN;
        out_p90_all[global_idx] = NAN;
        return;
    }

    out_mean_all[global_idx] = sum / static_cast<float>(valid_count);

    if (valid_count == 1 || vmin == vmax) {
        out_p90_all[global_idx] = vmin;
        return;
    }

    float hist_f[COARSE_BINS];
    for (int b = 0; b < COARSE_BINS; ++b) hist_f[b] = 0.0f;

    float inv_range = static_cast<float>(COARSE_BINS) / (vmax - vmin);
    for (int wd = 0; wd < CLIM_WINDOW_DAYS; ++wd) {
        int season_day = start_day + wd;
        if (season_day < 0 || season_day >= season_days) continue;

        const float* day_base = data +
            static_cast<size_t>(season_day) * CLIM_YEARS * local_spatial_points;
        for (int yr = 0; yr < CLIM_YEARS; ++yr) {
            float v = day_base[static_cast<size_t>(yr) * local_spatial_points + spatial];
            if (!isnan(v)) {
                int bin = static_cast<int>((v - vmin) * inv_range);
                if (bin < 0) bin = 0;
                if (bin >= COARSE_BINS) bin = COARSE_BINS - 1;
                hist_f[bin] += 1.0f;
            }
        }
    }

    float target_rank = 0.9f * static_cast<float>(valid_count - 1);
    int rank_lo = static_cast<int>(target_rank);
    int rank_hi = rank_lo + 1;
    float frac = target_rank - static_cast<float>(rank_lo);
    bool need_interp = (frac > 1e-6f && rank_hi < valid_count);

    int target_bin = COARSE_BINS - 1;
    float cumulative = 0.0f;
    for (int b = 0; b < COARSE_BINS; ++b) {
        if (cumulative + hist_f[b] > static_cast<float>(rank_lo)) {
            target_bin = b;
            break;
        }
        cumulative += hist_f[b];
    }

    float bin_width = (vmax - vmin) / static_cast<float>(COARSE_BINS);
    int lo_bin = target_bin;
    int hi_bin = target_bin + 1;
    float buf[COLLECT_MAX];
    int buf_count = 0;
    float cum_lo = cumulative;
    bool success = false;

    for (int expand = 0; expand < COARSE_BINS; ++expand) {
        float c_lo = vmin + static_cast<float>(lo_bin) * bin_width;
        float c_hi = vmin + static_cast<float>(hi_bin) * bin_width;
        if (c_lo < vmin) c_lo = vmin;
        if (c_hi > vmax) c_hi = vmax;

        buf_count = 0;
        for (int wd = 0; wd < CLIM_WINDOW_DAYS && buf_count < COLLECT_MAX; ++wd) {
            int season_day = start_day + wd;
            if (season_day < 0 || season_day >= season_days) continue;

            const float* day_base = data +
                static_cast<size_t>(season_day) * CLIM_YEARS * local_spatial_points;
            for (int yr = 0; yr < CLIM_YEARS && buf_count < COLLECT_MAX; ++yr) {
                float v = day_base[static_cast<size_t>(yr) * local_spatial_points + spatial];
                if (!isnan(v) && v >= c_lo && v < c_hi) {
                    buf[buf_count++] = v;
                }
            }
        }

        if (!need_interp) {
            float local_lo = static_cast<float>(rank_lo) - cum_lo;
            if (local_lo >= 0.0f && local_lo < static_cast<float>(buf_count)) {
                success = true;
                break;
            }
        } else {
            float local_lo = static_cast<float>(rank_lo) - cum_lo;
            float local_hi_f = static_cast<float>(rank_hi) - cum_lo;
            if (local_lo >= 0.0f && local_hi_f < static_cast<float>(buf_count)) {
                success = true;
                break;
            }
        }

        if (lo_bin > 0) {
            --lo_bin;
            cum_lo -= hist_f[lo_bin];
        }
        if (hi_bin < COARSE_BINS) {
            ++hi_bin;
        }
        if (buf_count >= COLLECT_MAX - 10) break;
    }

    if (!success || buf_count == 0) {
        out_p90_all[global_idx] = vmin + static_cast<float>(target_bin + 0.5f) * bin_width;
        return;
    }

    for (int i = 1; i < buf_count; ++i) {
        float key = buf[i];
        int j = i - 1;
        while (j >= 0 && buf[j] > key) {
            buf[j + 1] = buf[j];
            --j;
        }
        buf[j + 1] = key;
    }

    float local_lo = static_cast<float>(rank_lo) - cum_lo;
    int ilo = static_cast<int>(local_lo);
    if (!need_interp || ilo + 1 >= buf_count) {
        out_p90_all[global_idx] = buf[ilo];
    } else {
        float local_hi_f = static_cast<float>(rank_hi) - cum_lo;
        int ihi = static_cast<int>(local_hi_f);
        if (ihi >= buf_count) ihi = buf_count - 1;
        out_p90_all[global_idx] = buf[ilo] + frac * (buf[ihi] - buf[ilo]);
    }
}

// ============================================================
// 4-DCU 异步调度
// ============================================================

void dispatch_to_4_dcus(float* h_sst_data) {
    using Clock = std::chrono::steady_clock;
    const int days_total = static_cast<int>(DAYS_TOTAL);
    const int spatial_total = static_cast<int>(SPATIAL_POINTS);

    int gpu_count = 0;
    hipGetDeviceCount(&gpu_count);
    const int NUM_GPUS = gpu_count;
    if (NUM_GPUS <= 0) {
        std::cerr << "[DCU 模块] 未发现可用 DCU" << std::endl;
        std::exit(1);
    }

    // 空间网格按行切分：721 行分配给 4 张卡
    int rows_per_gpu = (LAT_SIZE + NUM_GPUS - 1) / NUM_GPUS;

    std::cout << "[DCU 模块] 初始化 " << NUM_GPUS << " 张 DCU 进行空间切片并行..." << std::endl;
    std::cout << "[DCU 模块] 数据布局: [" << days_total << "][" << LAT_SIZE << "][" << LON_SIZE << "]" << std::endl;
    std::cout << "[DCU 模块] 每卡 " << rows_per_gpu << " 行, 每行 " << LON_SIZE << " 点" << std::endl;

    // ---- 分配设备内存 + 异步拷贝全量数据到每张卡 ----
    auto t0 = Clock::now();

    hipStream_t streams[NUM_GPUS];
    float* d_data[NUM_GPUS];
    float* d_mean[NUM_GPUS];
    float* d_p90[NUM_GPUS];
    int lat_start[NUM_GPUS], lat_end[NUM_GPUS], local_rows[NUM_GPUS];

    size_t full_data_bytes = (size_t)days_total * spatial_total * sizeof(float);

    for (int i = 0; i < NUM_GPUS; ++i) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamCreate(&streams[i]));

        lat_start[i] = i * rows_per_gpu;
        lat_end[i] = std::min((i + 1) * rows_per_gpu, (int)LAT_SIZE);
        local_rows[i] = lat_end[i] - lat_start[i];

        if (local_rows[i] <= 0) continue;

        int local_spatial = local_rows[i] * LON_SIZE;

        // 分配设备内存
        HIP_CHECK(hipMalloc(&d_data[i], full_data_bytes));
        HIP_CHECK(hipMalloc(&d_mean[i], local_spatial * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_p90[i],  local_spatial * sizeof(float)));

        // 异步拷贝全量数据到设备（后续滑动窗口复用此缓冲区）
        HIP_CHECK(hipMemcpyAsync(d_data[i], h_sst_data, full_data_bytes,
                                 hipMemcpyHostToDevice, streams[i]));
    }

    // 等待所有数据传输完成
    for (int i = 0; i < NUM_GPUS; ++i) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamSynchronize(streams[i]));
    }

    auto t1 = Clock::now();
    double transfer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[DCU 模块] 数据传输完成: " << transfer_ms << " ms" << std::endl;

    // ---- 并行启动 Kernel ----
    t0 = Clock::now();

    for (int i = 0; i < NUM_GPUS; ++i) {
        if (local_rows[i] <= 0) continue;

        HIP_CHECK(hipSetDevice(i));
        int local_spatial = local_rows[i] * LON_SIZE;
        int spatial_start = lat_start[i] * LON_SIZE;

        int block_size = 256;
        int grid_size = (local_spatial + block_size - 1) / block_size;

        compute_mean_p90_kernel<<<dim3(grid_size), dim3(block_size), 0, streams[i]>>>(
            d_data[i], d_mean[i], d_p90[i],
            spatial_total, days_total,
            spatial_start, spatial_start + local_spatial,
            spatial_total, 0);
    }

    // 等待所有 Kernel 完成
    for (int i = 0; i < NUM_GPUS; ++i) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamSynchronize(streams[i]));
    }

    t1 = Clock::now();
    double kernel_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[DCU 模块] Kernel 计算完成: " << kernel_ms << " ms" << std::endl;

    // ---- 分配输出缓冲区并拉回结果 ----
    float* h_mean = nullptr;
    float* h_p90  = nullptr;
    HIP_CHECK(hipHostMalloc(&h_mean, spatial_total * sizeof(float), hipHostMallocDefault));
    HIP_CHECK(hipHostMalloc(&h_p90,  spatial_total * sizeof(float), hipHostMallocDefault));

    t0 = Clock::now();

    for (int i = 0; i < NUM_GPUS; ++i) {
        if (local_rows[i] <= 0) continue;

        HIP_CHECK(hipSetDevice(i));
        int local_spatial = local_rows[i] * LON_SIZE;
        int host_offset = lat_start[i] * LON_SIZE;

        HIP_CHECK(hipMemcpyAsync(h_mean + host_offset, d_mean[i],
                                 local_spatial * sizeof(float),
                                 hipMemcpyDeviceToHost, streams[i]));
        HIP_CHECK(hipMemcpyAsync(h_p90 + host_offset, d_p90[i],
                                 local_spatial * sizeof(float),
                                 hipMemcpyDeviceToHost, streams[i]));
    }

    for (int i = 0; i < NUM_GPUS; ++i) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamSynchronize(streams[i]));
    }

    t1 = Clock::now();
    double gather_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[DCU 模块] 结果回传完成: " << gather_ms << " ms" << std::endl;

    // ---- 释放设备资源 ----
    for (int i = 0; i < NUM_GPUS; ++i) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamDestroy(streams[i]));
        HIP_CHECK(hipFree(d_data[i]));
        HIP_CHECK(hipFree(d_mean[i]));
        HIP_CHECK(hipFree(d_p90[i]));
    }

    // ---- 快速统计 ----
    int nan_count = 0;
    float mean_min = 1e30f, mean_max = -1e30f;
    float p90_min = 1e30f,  p90_max = -1e30f;
    for (int i = 0; i < spatial_total; ++i) {
        if (std::isnan(h_mean[i])) { nan_count++; continue; }
        if (h_mean[i] < mean_min) mean_min = h_mean[i];
        if (h_mean[i] > mean_max) mean_max = h_mean[i];
        if (h_p90[i] < p90_min) p90_min = h_p90[i];
        if (h_p90[i] > p90_max) p90_max = h_p90[i];
    }
    std::cout << "[DCU 模块] 统计: NaN点=" << nan_count << "/" << spatial_total << std::endl;
    std::cout << "[DCU 模块] Mean范围: [" << mean_min << ", " << mean_max << "]" << std::endl;
    std::cout << "[DCU 模块] P90范围:  [" << p90_min << ", " << p90_max << "]" << std::endl;

    double total_ms = transfer_ms + kernel_ms + gather_ms;
    std::cout << "[DCU 模块] 总计: " << total_ms << " ms"
              << " (传输=" << transfer_ms << " kernel=" << kernel_ms
              << " 回传=" << gather_ms << ")" << std::endl;

    // 释放输出缓冲区
    HIP_CHECK(hipHostFree(h_mean));
    HIP_CHECK(hipHostFree(h_p90));
}

// ============================================================
// 版本：逐次分配+释放（与 dispatch_to_4_dcus 同策略，确保正确性）
// ============================================================

static constexpr int MAX_PERSISTENT_GPUS = 8;
static bool g_persistent_initialized = false;
static int g_persistent_gpu_count = 0;
static int g_persistent_rows_per_gpu = 0;
static int g_persistent_lat_start[MAX_PERSISTENT_GPUS] = {0};
static int g_persistent_lat_end[MAX_PERSISTENT_GPUS] = {0};
static int g_persistent_local_rows[MAX_PERSISTENT_GPUS] = {0};
static hipStream_t g_persistent_streams[MAX_PERSISTENT_GPUS] = {nullptr};
static float* g_persistent_data[MAX_PERSISTENT_GPUS] = {nullptr};
static float* g_persistent_mean[MAX_PERSISTENT_GPUS] = {nullptr};
static float* g_persistent_p90[MAX_PERSISTENT_GPUS] = {nullptr};
static float* g_persistent_h2d_stage[MAX_PERSISTENT_GPUS] = {nullptr};
static size_t g_persistent_h2d_stage_elems[MAX_PERSISTENT_GPUS] = {0};

static bool debug_enabled() {
    const char* env = std::getenv("MCC_DEBUG_VALIDATE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool env_enabled(const char* name) {
    const char* env = std::getenv(name);
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static void ensure_h2d_stage(int gpu_idx, size_t elems) {
    if (g_persistent_h2d_stage_elems[gpu_idx] >= elems) return;
    HIP_CHECK(hipSetDevice(gpu_idx));
    if (g_persistent_h2d_stage[gpu_idx]) {
        HIP_CHECK(hipHostFree(g_persistent_h2d_stage[gpu_idx]));
        g_persistent_h2d_stage[gpu_idx] = nullptr;
        g_persistent_h2d_stage_elems[gpu_idx] = 0;
    }
    HIP_CHECK(hipHostMalloc(&g_persistent_h2d_stage[gpu_idx],
                            elems * sizeof(float),
                            hipHostMallocDefault));
    g_persistent_h2d_stage_elems[gpu_idx] = elems;
}

static void print_field_stats(const char* name, const float* data, int n) {
    long long nan_count = 0;
    long long zero_count = 0;
    long long finite_count = 0;
    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < n; ++i) {
        float v = data[i];
        if (std::isnan(v)) {
            ++nan_count;
            continue;
        }
        ++finite_count;
        if (v == 0.0f) ++zero_count;
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
    }

    std::cout << "[debug] " << name
              << " finite=" << finite_count
              << " nan=" << nan_count
              << " zero=" << zero_count;
    if (finite_count > 0) {
        std::cout << " min=" << min_v << " max=" << max_v;
    }
    std::cout << std::endl;
}

static void init_persistent_dcu_buffers(int spatial_total, int days_total) {
    if (g_persistent_initialized) return;

    int gpu_count = 0;
    hipGetDeviceCount(&gpu_count);
    if (gpu_count <= 0) {
        std::cerr << "[DCU module] no available DCU" << std::endl;
        std::exit(1);
    }
    if (gpu_count > MAX_PERSISTENT_GPUS) {
        std::cerr << "[DCU module] gpu_count=" << gpu_count
                  << " exceeds MAX_PERSISTENT_GPUS=" << MAX_PERSISTENT_GPUS << std::endl;
        std::exit(1);
    }

    g_persistent_gpu_count = gpu_count;
    g_persistent_rows_per_gpu = ((int)LAT_SIZE + g_persistent_gpu_count - 1) /
                                g_persistent_gpu_count;

    auto init_one_gpu = [&](int i) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamCreate(&g_persistent_streams[i]));

        g_persistent_lat_start[i] = i * g_persistent_rows_per_gpu;
        g_persistent_lat_end[i] = std::min((i + 1) * g_persistent_rows_per_gpu, (int)LAT_SIZE);
        g_persistent_local_rows[i] = g_persistent_lat_end[i] - g_persistent_lat_start[i];
        if (g_persistent_local_rows[i] <= 0) return;

        int local_spatial = g_persistent_local_rows[i] * LON_SIZE;
        size_t local_data_bytes = (size_t)days_total * local_spatial * sizeof(float);
        HIP_CHECK(hipMalloc(&g_persistent_data[i], local_data_bytes));
        HIP_CHECK(hipMalloc(&g_persistent_mean[i], local_spatial * sizeof(float)));
        HIP_CHECK(hipMalloc(&g_persistent_p90[i],  local_spatial * sizeof(float)));
    };

    if (env_enabled("MCC_PARALLEL_DCU_INIT")) {
        #pragma omp parallel for num_threads(gpu_count) schedule(static)
        for (int i = 0; i < g_persistent_gpu_count; ++i) {
            init_one_gpu(i);
        }
    } else {
        for (int i = 0; i < g_persistent_gpu_count; ++i) {
            init_one_gpu(i);
        }
    }

    std::cout << "[DCU module] persistent buffers initialized on "
              << g_persistent_gpu_count << " DCU(s); rows_per_gpu="
              << g_persistent_rows_per_gpu << std::endl;
    g_persistent_initialized = true;
}

void prepare_dcu_persistent_buffers() {
    init_persistent_dcu_buffers(static_cast<int>(SPATIAL_POINTS),
                                static_cast<int>(DAYS_TOTAL));
}

static void dispatch_to_4_dcus_with_output_incremental_impl(float* h_sst_data,
                                                            float* h_changed_compact,
                                                            float* h_mean,
                                                            float* h_p90,
                                                            const int* changed_slots,
                                                            int num_changed,
                                                            void (*after_upload)(void*),
                                                            void* after_upload_user) {
    using Clock = std::chrono::steady_clock;
    const int days_total = static_cast<int>(DAYS_TOTAL);
    const int spatial_total = static_cast<int>(SPATIAL_POINTS);

    if (!g_persistent_initialized) {
        init_persistent_dcu_buffers(spatial_total, days_total);
    }

    auto t0 = Clock::now();
    bool full_upload = (changed_slots == nullptr || num_changed <= 0);

    for (int i = 0; i < g_persistent_gpu_count; ++i) {
        if (g_persistent_local_rows[i] <= 0) continue;
        HIP_CHECK(hipSetDevice(i));
        int local_spatial = g_persistent_local_rows[i] * LON_SIZE;
        int spatial_start = g_persistent_lat_start[i] * LON_SIZE;
        size_t local_row_bytes = (size_t)local_spatial * sizeof(float);

        if (full_upload) {
            HIP_CHECK(hipMemcpy2DAsync(g_persistent_data[i], local_row_bytes,
                                       h_sst_data + spatial_start,
                                       (size_t)spatial_total * sizeof(float),
                                       local_row_bytes, days_total,
                                       hipMemcpyHostToDevice,
                                       g_persistent_streams[i]));
        } else {
            bool regular_stride = (num_changed > 0);
            for (int j = 1; j < num_changed; ++j) {
                if (changed_slots[j] - changed_slots[j - 1] != 1) {
                    regular_stride = false;
                    break;
                }
            }

            if (h_changed_compact && env_enabled("MCC_COMPACT_PER_SLOT")) {
                for (int j = 0; j < num_changed; ++j) {
                    int slot = changed_slots[j];
                    HIP_CHECK(hipMemcpyAsync(g_persistent_data[i] + (size_t)slot * local_spatial,
                                             h_changed_compact + (size_t)j * spatial_total + spatial_start,
                                             local_row_bytes,
                                             hipMemcpyHostToDevice,
                                             g_persistent_streams[i]));
                }
            } else if (h_changed_compact && regular_stride) {
                int first_slot = changed_slots[0];
                HIP_CHECK(hipMemcpy2DAsync(g_persistent_data[i] + (size_t)first_slot * local_spatial,
                                           (size_t)local_spatial * sizeof(float),
                                           h_changed_compact + spatial_start,
                                           (size_t)spatial_total * sizeof(float),
                                           local_row_bytes, num_changed,
                                           hipMemcpyHostToDevice,
                                           g_persistent_streams[i]));
            } else if (env_enabled("MCC_H2D_STAGE") && regular_stride) {
                int first_slot = changed_slots[0];
                size_t stage_elems = (size_t)num_changed * local_spatial;
                ensure_h2d_stage(i, stage_elems);
                float* stage = g_persistent_h2d_stage[i];
                for (int j = 0; j < num_changed; ++j) {
                    int slot = changed_slots[j];
                    std::memcpy(stage + (size_t)j * local_spatial,
                                h_sst_data + (size_t)slot * spatial_total + spatial_start,
                                local_row_bytes);
                }
                HIP_CHECK(hipMemcpyAsync(g_persistent_data[i] + (size_t)first_slot * local_spatial,
                                         stage,
                                         stage_elems * sizeof(float),
                                         hipMemcpyHostToDevice,
                                         g_persistent_streams[i]));
            } else if (env_enabled("MCC_H2D_BATCH2D") && regular_stride) {
                int first_slot = changed_slots[0];
                HIP_CHECK(hipMemcpy2DAsync(g_persistent_data[i] + (size_t)first_slot * local_spatial,
                                           (size_t)local_spatial * sizeof(float),
                                           h_sst_data + (size_t)first_slot * spatial_total + spatial_start,
                                           (size_t)spatial_total * sizeof(float),
                                           local_row_bytes, num_changed,
                                           hipMemcpyHostToDevice,
                                           g_persistent_streams[i]));
            } else {
                for (int j = 0; j < num_changed; ++j) {
                    int slot = changed_slots[j];
                    HIP_CHECK(hipMemcpyAsync(g_persistent_data[i] + (size_t)slot * local_spatial,
                                             h_sst_data + (size_t)slot * spatial_total + spatial_start,
                                             local_row_bytes,
                                             hipMemcpyHostToDevice,
                                             g_persistent_streams[i]));
                }
            }
        }
    }

    for (int i = 0; i < g_persistent_gpu_count; ++i) {
        if (g_persistent_local_rows[i] <= 0) continue;
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamSynchronize(g_persistent_streams[i]));
    }

    auto t1 = Clock::now();
    double transfer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (after_upload) {
        after_upload(after_upload_user);
    }

    t0 = Clock::now();
    for (int i = 0; i < g_persistent_gpu_count; ++i) {
        if (g_persistent_local_rows[i] <= 0) continue;

        HIP_CHECK(hipSetDevice(i));
        int local_spatial = g_persistent_local_rows[i] * LON_SIZE;
        int spatial_start = g_persistent_lat_start[i] * LON_SIZE;
        int block_size = 256;
        int grid_size = (local_spatial + block_size - 1) / block_size;

        compute_mean_p90_kernel<<<dim3(grid_size), dim3(block_size), 0, g_persistent_streams[i]>>>(
            g_persistent_data[i], g_persistent_mean[i], g_persistent_p90[i],
            spatial_total, days_total,
            spatial_start, spatial_start + local_spatial,
            local_spatial, spatial_start);
        HIP_CHECK(hipPeekAtLastError());
    }

    for (int i = 0; i < g_persistent_gpu_count; ++i) {
        if (g_persistent_local_rows[i] <= 0) continue;
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamSynchronize(g_persistent_streams[i]));
    }

    t1 = Clock::now();
    double kernel_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = Clock::now();
    for (int i = 0; i < g_persistent_gpu_count; ++i) {
        if (g_persistent_local_rows[i] <= 0) continue;

        HIP_CHECK(hipSetDevice(i));
        int local_spatial = g_persistent_local_rows[i] * LON_SIZE;
        int host_offset = g_persistent_lat_start[i] * LON_SIZE;

        HIP_CHECK(hipMemcpyAsync(h_mean + host_offset, g_persistent_mean[i],
                                 local_spatial * sizeof(float),
                                 hipMemcpyDeviceToHost,
                                 g_persistent_streams[i]));
        HIP_CHECK(hipMemcpyAsync(h_p90 + host_offset, g_persistent_p90[i],
                                 local_spatial * sizeof(float),
                                 hipMemcpyDeviceToHost,
                                 g_persistent_streams[i]));
    }

    for (int i = 0; i < g_persistent_gpu_count; ++i) {
        if (g_persistent_local_rows[i] <= 0) continue;
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamSynchronize(g_persistent_streams[i]));
    }

    t1 = Clock::now();
    double gather_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double total_ms = transfer_ms + kernel_ms + gather_ms;

    std::cout << "[DCU module] upload=" << transfer_ms
              << "ms (" << (full_upload ? "full" : "incremental")
              << ", slots=" << (full_upload ? days_total : num_changed)
              << ") kernel=" << kernel_ms
              << "ms download=" << gather_ms
              << "ms total=" << total_ms << "ms" << std::endl;

    if (debug_enabled()) {
        print_field_stats("gpu_mean_after_d2h", h_mean, spatial_total);
        print_field_stats("gpu_p90_after_d2h", h_p90, spatial_total);
    }
}

void dispatch_to_4_dcus_with_output_incremental_hook(float* h_sst_data, float* h_mean, float* h_p90,
                                                     const int* changed_slots, int num_changed,
                                                     void (*after_upload)(void*),
                                                     void* after_upload_user) {
    dispatch_to_4_dcus_with_output_incremental_impl(
        h_sst_data, nullptr, h_mean, h_p90, changed_slots, num_changed, after_upload, after_upload_user);
}

void dispatch_to_4_dcus_with_output_incremental_compact(float* h_sst_data, float* h_changed_compact,
                                                        float* h_mean, float* h_p90,
                                                        const int* changed_slots, int num_changed,
                                                        void (*after_upload)(void*),
                                                        void* after_upload_user) {
    dispatch_to_4_dcus_with_output_incremental_impl(
        h_sst_data, h_changed_compact, h_mean, h_p90, changed_slots, num_changed,
        after_upload, after_upload_user);
}

void dispatch_to_4_dcus_with_output_incremental(float* h_sst_data, float* h_mean, float* h_p90,
                                                const int* changed_slots, int num_changed) {
    dispatch_to_4_dcus_with_output_incremental_hook(
        h_sst_data, h_mean, h_p90, changed_slots, num_changed, nullptr, nullptr);
}

void cleanup_dcu_persistent_buffers() {
    if (!g_persistent_initialized) return;

    for (int i = 0; i < g_persistent_gpu_count; ++i) {
        HIP_CHECK(hipSetDevice(i));
        if (g_persistent_data[i]) HIP_CHECK(hipFree(g_persistent_data[i]));
        if (g_persistent_mean[i]) HIP_CHECK(hipFree(g_persistent_mean[i]));
        if (g_persistent_p90[i]) HIP_CHECK(hipFree(g_persistent_p90[i]));
        if (g_persistent_h2d_stage[i]) HIP_CHECK(hipHostFree(g_persistent_h2d_stage[i]));
        if (g_persistent_streams[i]) HIP_CHECK(hipStreamDestroy(g_persistent_streams[i]));

        g_persistent_data[i] = nullptr;
        g_persistent_mean[i] = nullptr;
        g_persistent_p90[i] = nullptr;
        g_persistent_h2d_stage[i] = nullptr;
        g_persistent_h2d_stage_elems[i] = 0;
        g_persistent_streams[i] = nullptr;
        g_persistent_local_rows[i] = 0;
    }

    g_persistent_gpu_count = 0;
    g_persistent_rows_per_gpu = 0;
    g_persistent_initialized = false;
}

void dispatch_to_4_dcus_with_output(float* h_sst_data, float* h_mean, float* h_p90,
                                     const int* /*changed_slots*/, int /*num_changed*/) {
    using Clock = std::chrono::steady_clock;
    const int days_total = static_cast<int>(DAYS_TOTAL);
    const int spatial_total = static_cast<int>(SPATIAL_POINTS);

    int gpu_count = 0;
    hipGetDeviceCount(&gpu_count);
    const int NUM_GPUS = gpu_count;
    if (NUM_GPUS <= 0) {
        std::cerr << "[DCU 模块] 未发现可用 DCU" << std::endl;
        std::exit(1);
    }

    int rows_per_gpu = (LAT_SIZE + NUM_GPUS - 1) / NUM_GPUS;

    auto t0 = Clock::now();

    hipStream_t streams[8];
    float* d_data[8];
    float* d_mean[8];
    float* d_p90[8];
    int lat_start[8], lat_end[8], local_rows_arr[8];

    size_t full_data_bytes = (size_t)days_total * spatial_total * sizeof(float);

    for (int i = 0; i < NUM_GPUS; ++i) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamCreate(&streams[i]));

        lat_start[i] = i * rows_per_gpu;
        lat_end[i] = std::min((i + 1) * rows_per_gpu, (int)LAT_SIZE);
        local_rows_arr[i] = lat_end[i] - lat_start[i];

        if (local_rows_arr[i] <= 0) continue;

        int local_spatial = local_rows_arr[i] * LON_SIZE;

        HIP_CHECK(hipMalloc(&d_data[i], full_data_bytes));
        HIP_CHECK(hipMalloc(&d_mean[i], local_spatial * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_p90[i],  local_spatial * sizeof(float)));

        HIP_CHECK(hipMemcpyAsync(d_data[i], h_sst_data, full_data_bytes,
                                 hipMemcpyHostToDevice, streams[i]));
    }

    for (int i = 0; i < NUM_GPUS; ++i) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamSynchronize(streams[i]));
    }

    auto t1 = Clock::now();
    double transfer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---- Kernel 计算 ----
    t0 = Clock::now();

    for (int i = 0; i < NUM_GPUS; ++i) {
        if (local_rows_arr[i] <= 0) continue;

        HIP_CHECK(hipSetDevice(i));
        int local_spatial = local_rows_arr[i] * LON_SIZE;
        int spatial_start = lat_start[i] * LON_SIZE;

        int block_size = 256;
        int grid_size = (local_spatial + block_size - 1) / block_size;

        compute_mean_p90_kernel<<<dim3(grid_size), dim3(block_size), 0, streams[i]>>>(
            d_data[i], d_mean[i], d_p90[i],
            spatial_total, days_total,
            spatial_start, spatial_start + local_spatial,
            spatial_total, 0);
    }

    for (int i = 0; i < NUM_GPUS; ++i) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamSynchronize(streams[i]));
    }

    t1 = Clock::now();
    double kernel_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---- 结果回传 ----
    t0 = Clock::now();

    for (int i = 0; i < NUM_GPUS; ++i) {
        if (local_rows_arr[i] <= 0) continue;

        HIP_CHECK(hipSetDevice(i));
        int local_spatial = local_rows_arr[i] * LON_SIZE;
        int host_offset = lat_start[i] * LON_SIZE;

        HIP_CHECK(hipMemcpy(h_mean + host_offset, d_mean[i],
                            local_spatial * sizeof(float),
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_p90 + host_offset, d_p90[i],
                            local_spatial * sizeof(float),
                            hipMemcpyDeviceToHost));
    }

    t1 = Clock::now();
    double gather_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---- 释放设备资源 ----
    for (int i = 0; i < NUM_GPUS; ++i) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamDestroy(streams[i]));
        HIP_CHECK(hipFree(d_data[i]));
        HIP_CHECK(hipFree(d_mean[i]));
        HIP_CHECK(hipFree(d_p90[i]));
    }

    double total_ms = transfer_ms + kernel_ms + gather_ms;
    std::cout << "[DCU 模块] 传输=" << transfer_ms << "ms kernel=" << kernel_ms
              << "ms 回传=" << gather_ms << "ms 总计=" << total_ms << "ms" << std::endl;
}

void dispatch_season_batch_to_4_dcus(float* h_season_data,
                                     float* h_mean_all,
                                     float* h_p90_all,
                                     int target_days,
                                     int season_days) {
    using Clock = std::chrono::steady_clock;
    const int spatial_total = static_cast<int>(SPATIAL_POINTS);
    const int season_slots = season_days * CLIM_YEARS;

    int gpu_count = 0;
    hipGetDeviceCount(&gpu_count);
    if (gpu_count <= 0) {
        std::cerr << "[DCU season] no available DCU" << std::endl;
        std::exit(1);
    }
    if (gpu_count > MAX_PERSISTENT_GPUS) {
        std::cerr << "[DCU season] gpu_count=" << gpu_count
                  << " exceeds MAX_PERSISTENT_GPUS=" << MAX_PERSISTENT_GPUS << std::endl;
        std::exit(1);
    }

    int rows_per_gpu = (LAT_SIZE + gpu_count - 1) / gpu_count;
    std::cout << "[DCU season] batch target_days=" << target_days
              << " season_days=" << season_days
              << " slots=" << season_slots
              << " gpus=" << gpu_count << std::endl;

    hipStream_t streams[MAX_PERSISTENT_GPUS] = {nullptr};
    float* d_data[MAX_PERSISTENT_GPUS] = {nullptr};
    float* d_mean[MAX_PERSISTENT_GPUS] = {nullptr};
    float* d_p90[MAX_PERSISTENT_GPUS] = {nullptr};
    float* h_packed[MAX_PERSISTENT_GPUS] = {nullptr};
    int lat_start[MAX_PERSISTENT_GPUS] = {0};
    int lat_end[MAX_PERSISTENT_GPUS] = {0};
    int local_rows_arr[MAX_PERSISTENT_GPUS] = {0};
    bool packed_h2d = env_enabled("MCC_SEASON_PACKED_H2D");
    double pack_ms = 0.0;
    if (packed_h2d) {
        std::cout << "[DCU season] MCC_SEASON_PACKED_H2D=1" << std::endl;
    }

    auto t0 = Clock::now();
    for (int i = 0; i < gpu_count; ++i) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamCreate(&streams[i]));

        lat_start[i] = i * rows_per_gpu;
        lat_end[i] = std::min((i + 1) * rows_per_gpu, static_cast<int>(LAT_SIZE));
        local_rows_arr[i] = lat_end[i] - lat_start[i];
        if (local_rows_arr[i] <= 0) continue;

        int local_spatial = local_rows_arr[i] * LON_SIZE;
        int spatial_start = lat_start[i] * LON_SIZE;
        size_t local_row_bytes = static_cast<size_t>(local_spatial) * sizeof(float);
        size_t host_row_bytes = static_cast<size_t>(spatial_total) * sizeof(float);
        size_t data_bytes = static_cast<size_t>(season_slots) * local_row_bytes;
        size_t out_bytes = static_cast<size_t>(target_days) * local_row_bytes;

        HIP_CHECK(hipMalloc(&d_data[i], data_bytes));
        HIP_CHECK(hipMalloc(&d_mean[i], out_bytes));
        HIP_CHECK(hipMalloc(&d_p90[i], out_bytes));

        if (packed_h2d) {
            auto pack_t0 = Clock::now();
            h_packed[i] = static_cast<float*>(std::malloc(data_bytes));
            if (!h_packed[i]) {
                std::cerr << "[DCU season] cannot allocate packed host bytes="
                          << data_bytes << std::endl;
                std::exit(1);
            }
            #pragma omp parallel for schedule(static)
            for (int slot = 0; slot < season_slots; ++slot) {
                std::memcpy(h_packed[i] + static_cast<size_t>(slot) * local_spatial,
                            h_season_data + static_cast<size_t>(slot) * spatial_total + spatial_start,
                            local_row_bytes);
            }
            pack_ms += std::chrono::duration<double, std::milli>(
                Clock::now() - pack_t0).count();
            HIP_CHECK(hipMemcpyAsync(d_data[i], h_packed[i], data_bytes,
                                     hipMemcpyHostToDevice, streams[i]));
        } else {
            HIP_CHECK(hipMemcpy2DAsync(d_data[i], local_row_bytes,
                                       h_season_data + spatial_start, host_row_bytes,
                                       local_row_bytes, season_slots,
                                       hipMemcpyHostToDevice, streams[i]));
        }
    }

    for (int i = 0; i < gpu_count; ++i) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamSynchronize(streams[i]));
    }

    auto t1 = Clock::now();
    double upload_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = Clock::now();
    for (int i = 0; i < gpu_count; ++i) {
        if (local_rows_arr[i] <= 0) continue;
        HIP_CHECK(hipSetDevice(i));
        int local_spatial = local_rows_arr[i] * LON_SIZE;
        int total_outputs = target_days * local_spatial;
        int block_size = 256;
        int grid_size = (total_outputs + block_size - 1) / block_size;

        compute_season_batch_kernel<<<dim3(grid_size), dim3(block_size), 0, streams[i]>>>(
            d_data[i], d_mean[i], d_p90[i], target_days, season_days, local_spatial);
        HIP_CHECK(hipGetLastError());
    }

    for (int i = 0; i < gpu_count; ++i) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamSynchronize(streams[i]));
    }

    t1 = Clock::now();
    double kernel_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = Clock::now();
    for (int i = 0; i < gpu_count; ++i) {
        if (local_rows_arr[i] <= 0) continue;
        HIP_CHECK(hipSetDevice(i));
        int local_spatial = local_rows_arr[i] * LON_SIZE;
        int host_offset = lat_start[i] * LON_SIZE;
        size_t local_row_bytes = static_cast<size_t>(local_spatial) * sizeof(float);
        size_t host_row_bytes = static_cast<size_t>(spatial_total) * sizeof(float);

        HIP_CHECK(hipMemcpy2DAsync(h_mean_all + host_offset, host_row_bytes,
                                   d_mean[i], local_row_bytes,
                                   local_row_bytes, target_days,
                                   hipMemcpyDeviceToHost, streams[i]));
        HIP_CHECK(hipMemcpy2DAsync(h_p90_all + host_offset, host_row_bytes,
                                   d_p90[i], local_row_bytes,
                                   local_row_bytes, target_days,
                                   hipMemcpyDeviceToHost, streams[i]));
    }

    for (int i = 0; i < gpu_count; ++i) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamSynchronize(streams[i]));
    }

    t1 = Clock::now();
    double download_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    for (int i = 0; i < gpu_count; ++i) {
        HIP_CHECK(hipSetDevice(i));
        if (streams[i]) HIP_CHECK(hipStreamDestroy(streams[i]));
        if (d_data[i]) HIP_CHECK(hipFree(d_data[i]));
        if (d_mean[i]) HIP_CHECK(hipFree(d_mean[i]));
        if (d_p90[i]) HIP_CHECK(hipFree(d_p90[i]));
        if (h_packed[i]) std::free(h_packed[i]);
    }

    double total_ms = upload_ms + kernel_ms + download_ms;
    std::cout << "[DCU season] pack=" << pack_ms
              << "ms upload=" << upload_ms
              << "ms kernel=" << kernel_ms
              << "ms download=" << download_ms
              << "ms total=" << total_ms << "ms" << std::endl;
}

void cleanup_dcu_buffers() {
    // 无持久缓冲区，无需清理
}
