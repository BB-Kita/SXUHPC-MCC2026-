#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <netcdf.h>
#include <hip/hip_runtime.h>
#include "config.h"
#include "io_handler.h"
#include "compute_dcu.h"

// Write one day's output to NetCDF
static void write_output(const std::string& out_path, int doy,
                          const float* mean, const float* p90) {
    int ncid, lat_dimid, lon_dimid, day_dimid;
    int lat_varid, lon_varid, day_varid, clim_varid, p90_varid;

    int status = nc_create(out_path.c_str(), NC_CLOBBER | NC_NETCDF4, &ncid);
    if (status != NC_NOERR) {
        std::cerr << "ERROR creating " << out_path << ": " << nc_strerror(status) << std::endl;
        return;
    }

    nc_def_dim(ncid, "Lat", LAT_SIZE, &lat_dimid);
    nc_def_dim(ncid, "Lon", LON_SIZE, &lon_dimid);
    nc_def_dim(ncid, "Day", 1, &day_dimid);

    nc_def_var(ncid, "dayofyear", NC_DOUBLE, 1, &day_dimid, &day_varid);
    nc_put_att_text(ncid, day_varid, "long_name", 29, "Day of year (1-365, no 29Feb)");

    nc_def_var(ncid, "Lat", NC_DOUBLE, 1, &lat_dimid, &lat_varid);
    nc_def_var(ncid, "Lon", NC_DOUBLE, 1, &lon_dimid, &lon_varid);

    int field_dims[2] = {lat_dimid, lon_dimid};
    nc_def_var(ncid, "Climmean", NC_DOUBLE, 2, field_dims, &clim_varid);
    nc_put_att_text(ncid, clim_varid, "long_name", 30, "OSTIA SST climatology 1991-2020");

    nc_def_var(ncid, "P90_sst", NC_DOUBLE, 2, field_dims, &p90_varid);
    nc_put_att_text(ncid, p90_varid, "long_name", 22, "90th percentile of SST");

    nc_enddef(ncid);

    static std::vector<double> lat;
    static std::vector<double> lon;
    static std::vector<double> mean_d;
    static std::vector<double> p90_d;

    if (lat.empty()) {
        lat.resize(LAT_SIZE);
        lon.resize(LON_SIZE);
        for (size_t i = 0; i < LAT_SIZE; ++i) lat[i] = -90.0 + i * 0.25;
        for (size_t i = 0; i < LON_SIZE; ++i) lon[i] = i * 0.25;
        mean_d.resize(SPATIAL_POINTS);
        p90_d.resize(SPATIAL_POINTS);
    }

    double doy_val = static_cast<double>(doy);
    nc_put_var_double(ncid, day_varid, &doy_val);
    nc_put_var_double(ncid, lat_varid, lat.data());
    nc_put_var_double(ncid, lon_varid, lon.data());
    // Convert float32 -> float64 to match reference output format.
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(SPATIAL_POINTS); ++i) {
        mean_d[i] = static_cast<double>(mean[i]);
        p90_d[i]  = static_cast<double>(p90[i]);
    }
    nc_put_var_double(ncid, clim_varid, mean_d.data());
    nc_put_var_double(ncid, p90_varid, p90_d.data());

    nc_close(ncid);
}

// Convert DOY to mmdd string for output filename
static std::string doy_to_mmdd(int doy) {
    // Use 2020 as reference (leap year), skip Feb 29
    int mdays[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    int m = 0, d = doy;
    while (m < 12 && d > mdays[m]) {
        d -= mdays[m];
        ++m;
        // Skip Feb 29 for leap year
        if (m == 1 && doy > 59) { /* Feb 29 already passed */ }
    }
    char buf[8];
    std::snprintf(buf, sizeof(buf), "%02d%02d", m + 1, d);
    return std::string(buf);
}

static bool debug_enabled() {
    const char* env = std::getenv("MCC_DEBUG_VALIDATE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static void print_buffer_stats(const char* name, const float* data, size_t n) {
    size_t finite_count = 0;
    size_t nan_count = 0;
    size_t zero_count = 0;
    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();

    for (size_t i = 0; i < n; ++i) {
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

static bool cpu_reference_point(const float* h_sst_data, size_t spatial_idx,
                                float& mean, float& p90, int& valid_count) {
    std::vector<float> vals;
    vals.reserve(DAYS_TOTAL);
    double sum = 0.0;

    for (size_t d = 0; d < DAYS_TOTAL; ++d) {
        float v = h_sst_data[d * SPATIAL_POINTS + spatial_idx];
        if (!std::isnan(v)) {
            vals.push_back(v);
            sum += static_cast<double>(v);
        }
    }

    valid_count = static_cast<int>(vals.size());
    if (vals.empty()) {
        mean = std::numeric_limits<float>::quiet_NaN();
        p90 = std::numeric_limits<float>::quiet_NaN();
        return false;
    }

    std::sort(vals.begin(), vals.end());
    mean = static_cast<float>(sum / vals.size());

    if (vals.size() == 1) {
        p90 = vals[0];
    } else {
        float rank = 0.9f * static_cast<float>(vals.size() - 1);
        int lo = static_cast<int>(rank);
        int hi = std::min(lo + 1, static_cast<int>(vals.size()) - 1);
        float frac = rank - static_cast<float>(lo);
        p90 = vals[lo] + frac * (vals[hi] - vals[lo]);
    }
    return true;
}

static void print_cpu_gpu_samples(const float* h_sst_data,
                                  const float* h_mean,
                                  const float* h_p90) {
    std::vector<size_t> sample_indices;
    sample_indices.reserve(8);

    const size_t stride = 97;
    for (size_t idx = 0; idx < SPATIAL_POINTS && sample_indices.size() < 8; idx += stride) {
        for (size_t d = 0; d < DAYS_TOTAL; ++d) {
            float v = h_sst_data[d * SPATIAL_POINTS + idx];
            if (!std::isnan(v)) {
                sample_indices.push_back(idx);
                break;
            }
        }
    }

    std::cout << "[debug] sample CPU reference vs GPU output" << std::endl;
    for (size_t idx : sample_indices) {
        float ref_mean = 0.0f;
        float ref_p90 = 0.0f;
        int valid_count = 0;
        cpu_reference_point(h_sst_data, idx, ref_mean, ref_p90, valid_count);
        size_t lat = idx / LON_SIZE;
        size_t lon = idx % LON_SIZE;
        std::cout << "[debug] idx=" << idx
                  << " lat=" << lat
                  << " lon=" << lon
                  << " valid=" << valid_count
                  << " cpu_mean=" << ref_mean
                  << " gpu_mean=" << h_mean[idx]
                  << " cpu_p90=" << ref_p90
                  << " gpu_p90=" << h_p90[idx]
                  << std::endl;
    }
}

struct PrefetchAfterUpload {
    float* h_sst_data;
    int doy;
    int first_doy;
    int last_doy;
};

static void start_prefetch_after_upload(void* opaque) {
    auto* req = static_cast<PrefetchAfterUpload*>(opaque);
    if (req && req->doy < req->last_doy) {
        prefetch_next_doy_async(req->h_sst_data, req->doy, req->first_doy);
    }
}

int main() {
    using Clock = std::chrono::steady_clock;
    auto program_t0 = Clock::now();
    std::cout << ">>> 启动 MCC 海洋热浪阈值计算引擎 (C++/HIP)" << std::endl;

    std::string output_dir = "output/";
    if (const char* output_env = std::getenv("MCC_OUTPUT_DIR")) {
        output_dir = output_env;
        if (!output_dir.empty() && output_dir[output_dir.size() - 1] != '/') {
            output_dir.push_back('/');
        }
        std::cout << "[实验] MCC_OUTPUT_DIR=" << output_dir << std::endl;
    }

    // 1. 分配主机内存
    float* h_sst_data = nullptr;
    size_t memory_bytes = TOTAL_ELEMENTS * sizeof(float);
    std::cout << "[主线程] 正在分配 " << memory_bytes / (1024.0 * 1024.0 * 1024.0)
              << " GB 的主机内存..." << std::endl;

    h_sst_data = (float*)std::malloc(memory_bytes);
    if (!h_sst_data) {
        std::cerr << "ERROR: 无法分配 " << memory_bytes << " 字节内存" << std::endl;
        return 1;
    }
    std::memset(h_sst_data, 0, memory_bytes);

    // 2. 分配结果缓冲区（必须用 hipHostMalloc 以支持 D2H 传输）
    float* h_mean = nullptr;
    float* h_p90  = nullptr;
    HIP_CHECK(hipHostMalloc(&h_mean, SPATIAL_POINTS * sizeof(float), hipHostMallocDefault));
    HIP_CHECK(hipHostMalloc(&h_p90,  SPATIAL_POINTS * sizeof(float), hipHostMallocDefault));
    std::memset(h_mean, 0, SPATIAL_POINTS * sizeof(float));
    std::memset(h_p90, 0, SPATIAL_POINTS * sizeof(float));

    int target_doy_begin = TARGET_DOY_BEGIN;
    int target_doy_end = TARGET_DOY_END;
    if (const char* begin_env = std::getenv("MCC_TARGET_DOY_BEGIN")) {
        int requested_begin = std::atoi(begin_env);
        if (requested_begin >= TARGET_DOY_BEGIN && requested_begin <= TARGET_DOY_END) {
            target_doy_begin = requested_begin;
        }
    }
    if (const char* end_env = std::getenv("MCC_TARGET_DOY_END")) {
        int requested_end = std::atoi(end_env);
        if (requested_end >= TARGET_DOY_BEGIN && requested_end <= TARGET_DOY_END) {
            target_doy_end = requested_end;
        }
    }
    std::cout << "[experiment] output_dir=" << output_dir << std::endl;
    if (target_doy_begin != TARGET_DOY_BEGIN || target_doy_end != TARGET_DOY_END) {
        std::cout << "[experiment] target DOY range="
                  << target_doy_begin << ".." << target_doy_end << std::endl;
    }

    // 3. 初始化滑动窗口（读取第一个 DOY 的全部 330 个文件）
    auto phase_t0 = Clock::now();
    initialize_window(h_sst_data, target_doy_begin);
    std::cout << "[profile] initialize_window="
              << std::chrono::duration<double>(Clock::now() - phase_t0).count()
              << "s" << std::endl;
    if (debug_enabled()) {
        print_buffer_stats("host_window_after_initialize", h_sst_data, TOTAL_ELEMENTS);
    }

    // 4. 循环处理目标日期
    const int window = 2 * CLIM_DELTA_DAY + 1;
    std::vector<int> changed_slots(CLIM_YEARS);

    for (int doy = target_doy_begin; doy <= target_doy_end; ++doy) {
        auto day_t0 = Clock::now();
        std::cout << "\n>>> 处理 DOY " << doy << " (" << doy_to_mmdd(doy) << ")" << std::endl;

        // DCU 计算：首日全量上传，之后只上传滑动窗口替换的 30 个 slot
        dispatch_to_4_dcus_with_output_incremental(
            h_sst_data, h_mean, h_p90,
            (doy == target_doy_begin) ? nullptr : changed_slots.data(),
            (doy == target_doy_begin) ? 0 : static_cast<int>(changed_slots.size()));

        if (debug_enabled()) {
            print_cpu_gpu_samples(h_sst_data, h_mean, h_p90);
        }

        // 写出结果
        std::string mmdd = doy_to_mmdd(doy);
        std::string out_file = output_dir + mmdd + ".nc";
        phase_t0 = Clock::now();
        write_output(out_file, doy, h_mean, h_p90);
        double write_s = std::chrono::duration<double>(Clock::now() - phase_t0).count();
        std::cout << "[profile] write_output=" << write_s << "s" << std::endl;
        std::cout << "  输出: " << out_file << std::endl;

        // 滑动窗口：读取 30 个新文件覆盖最旧槽位（最后一天不需要）
        double slide_s = 0.0;
        if (doy < target_doy_end) {
            int replace_offset_idx = (doy - target_doy_begin) % window;
            for (int yr_idx = 0; yr_idx < CLIM_YEARS; ++yr_idx) {
                changed_slots[yr_idx] = replace_offset_idx * CLIM_YEARS + yr_idx;
            }
            phase_t0 = Clock::now();
            slide_window_to_next_doy(h_sst_data, doy, target_doy_begin);
            slide_s = std::chrono::duration<double>(Clock::now() - phase_t0).count();
        }
        std::cout << "[profile] day_total="
                  << std::chrono::duration<double>(Clock::now() - day_t0).count()
                  << "s write=" << write_s
                  << "s slide=" << slide_s << "s" << std::endl;
    }

    // 5. 释放资源
    cleanup_dcu_persistent_buffers();
    cleanup_dcu_buffers();
    HIP_CHECK(hipHostFree(h_mean));
    HIP_CHECK(hipHostFree(h_p90));
    std::free(h_sst_data);

    std::cout << "\n>>> 全部 " << (target_doy_end - target_doy_begin + 1)
              << " 天计算完成！" << std::endl;
    std::cout << "[profile] program_total="
              << std::chrono::duration<double>(Clock::now() - program_t0).count()
              << "s" << std::endl;
    return 0;
}
