#include "compute_dcu.h"
#include "config.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <chrono>

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
    int spatial_end)
{
    int global_idx = spatial_start + (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (global_idx >= spatial_end) return;

    int out_idx = global_idx - spatial_start;

    // ---- Pass 1: scan for mean / min / max ----
    float sum = 0.0f;
    int valid_count = 0;
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int d = 0; d < days_total; ++d) {
        float v = data[d * spatial_points + global_idx];
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
        float v = data[d * spatial_points + global_idx];
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
            float v = data[d * spatial_points + global_idx];
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

// ============================================================
// 4-DCU 异步调度
// ============================================================

void dispatch_to_4_dcus(float* h_sst_data) {
    using Clock = std::chrono::steady_clock;
    const int NUM_GPUS = 4;
    const int days_total = static_cast<int>(DAYS_TOTAL);
    const int spatial_total = static_cast<int>(SPATIAL_POINTS);

    // 查询每张卡的显存，决定切分策略
    int gpu_count = 0;
    hipGetDeviceCount(&gpu_count);
    if (gpu_count < NUM_GPUS) {
        std::cerr << "[DCU 模块] 需要 " << NUM_GPUS << " 张 DCU，仅发现 "
                  << gpu_count << " 张" << std::endl;
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
            spatial_start, spatial_start + local_spatial);
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
