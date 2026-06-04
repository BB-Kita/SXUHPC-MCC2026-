#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
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
    const char* output_float_env = std::getenv("MCC_OUTPUT_FLOAT");
    bool output_float = output_float_env != nullptr &&
                        output_float_env[0] != '\0' &&
                        output_float_env[0] != '0';
    const char* output_classic_env = std::getenv("MCC_OUTPUT_CLASSIC");
    bool output_classic = output_classic_env != nullptr &&
                          output_classic_env[0] != '\0' &&
                          output_classic_env[0] != '0';
    const char* minimal_output_env = std::getenv("MCC_MINIMAL_OUTPUT");
    bool minimal_output = minimal_output_env != nullptr &&
                          minimal_output_env[0] != '\0' &&
                          minimal_output_env[0] != '0';

    int create_flags = output_classic ? NC_CLOBBER : (NC_CLOBBER | NC_NETCDF4);
    int status = nc_create(out_path.c_str(), create_flags, &ncid);
    if (status != NC_NOERR) {
        std::cerr << "ERROR creating " << out_path << ": " << nc_strerror(status) << std::endl;
        return;
    }

    nc_def_dim(ncid, "Lat", LAT_SIZE, &lat_dimid);
    nc_def_dim(ncid, "Lon", LON_SIZE, &lon_dimid);
    if (!minimal_output) {
        nc_def_dim(ncid, "Day", 1, &day_dimid);

        nc_def_var(ncid, "dayofyear", NC_DOUBLE, 1, &day_dimid, &day_varid);
        nc_put_att_text(ncid, day_varid, "long_name", 29, "Day of year (1-365, no 29Feb)");

        nc_def_var(ncid, "Lat", NC_DOUBLE, 1, &lat_dimid, &lat_varid);
        nc_def_var(ncid, "Lon", NC_DOUBLE, 1, &lon_dimid, &lon_varid);
    }

    int field_dims[2] = {lat_dimid, lon_dimid};
    nc_def_var(ncid, "Climmean", output_float ? NC_FLOAT : NC_DOUBLE, 2, field_dims, &clim_varid);
    if (!minimal_output) {
        nc_put_att_text(ncid, clim_varid, "long_name", 30, "OSTIA SST climatology 1991-2020");
    }

    nc_def_var(ncid, "P90_sst", output_float ? NC_FLOAT : NC_DOUBLE, 2, field_dims, &p90_varid);
    if (!minimal_output) {
        nc_put_att_text(ncid, p90_varid, "long_name", 22, "90th percentile of SST");
    }

    nc_enddef(ncid);

    static std::vector<double> lat;
    static std::vector<double> lon;
    static std::vector<double> mean_d;
    static std::vector<double> p90_d;

    if (!minimal_output && lat.empty()) {
        lat.resize(LAT_SIZE);
        lon.resize(LON_SIZE);
        for (size_t i = 0; i < LAT_SIZE; ++i) lat[i] = -90.0 + i * 0.25;
        for (size_t i = 0; i < LON_SIZE; ++i) lon[i] = i * 0.25;
    }
    if (!output_float && mean_d.empty()) {
        mean_d.resize(SPATIAL_POINTS);
        p90_d.resize(SPATIAL_POINTS);
    }

    if (!minimal_output) {
        double doy_val = static_cast<double>(doy);
        nc_put_var_double(ncid, day_varid, &doy_val);
        nc_put_var_double(ncid, lat_varid, lat.data());
        nc_put_var_double(ncid, lon_varid, lon.data());
    }
    if (output_float) {
        nc_put_var_float(ncid, clim_varid, mean);
        nc_put_var_float(ncid, p90_varid, p90);
    } else {
        // Convert float32 -> float64 to match reference output format.
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(SPATIAL_POINTS); ++i) {
            mean_d[i] = static_cast<double>(mean[i]);
            p90_d[i]  = static_cast<double>(p90[i]);
        }
        nc_put_var_double(ncid, clim_varid, mean_d.data());
        nc_put_var_double(ncid, p90_varid, p90_d.data());
    }

    nc_close(ncid);
}

struct OutputSnapshot {
    std::string out_path;
    int doy = 0;
    std::vector<double> mean;
    std::vector<double> p90;
};

static void prepare_output_snapshot(OutputSnapshot& snapshot,
                                    const std::string& out_path,
                                    int doy,
                                    const float* mean,
                                    const float* p90) {
    snapshot.out_path = out_path;
    snapshot.doy = doy;
    if (snapshot.mean.size() != SPATIAL_POINTS) {
        snapshot.mean.resize(SPATIAL_POINTS);
        snapshot.p90.resize(SPATIAL_POINTS);
    }

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(SPATIAL_POINTS); ++i) {
        snapshot.mean[i] = static_cast<double>(mean[i]);
        snapshot.p90[i] = static_cast<double>(p90[i]);
    }
}

static void write_output_snapshot(const OutputSnapshot& snapshot) {
    int ncid, lat_dimid, lon_dimid, day_dimid;
    int lat_varid, lon_varid, day_varid, clim_varid, p90_varid;

    int status = nc_create(snapshot.out_path.c_str(), NC_CLOBBER | NC_NETCDF4, &ncid);
    if (status != NC_NOERR) {
        std::cerr << "ERROR creating " << snapshot.out_path << ": " << nc_strerror(status) << std::endl;
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
    if (lat.empty()) {
        lat.resize(LAT_SIZE);
        lon.resize(LON_SIZE);
        for (size_t i = 0; i < LAT_SIZE; ++i) lat[i] = -90.0 + i * 0.25;
        for (size_t i = 0; i < LON_SIZE; ++i) lon[i] = i * 0.25;
    }

    double doy_val = static_cast<double>(snapshot.doy);
    nc_put_var_double(ncid, day_varid, &doy_val);
    nc_put_var_double(ncid, lat_varid, lat.data());
    nc_put_var_double(ncid, lon_varid, lon.data());
    nc_put_var_double(ncid, clim_varid, snapshot.mean.data());
    nc_put_var_double(ncid, p90_varid, snapshot.p90.data());

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

static bool env_enabled(const char* name) {
    const char* env = std::getenv(name);
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

static bool overlap_prefetch_enabled() {
    const char* env = std::getenv("MCC_OVERLAP_PREFETCH");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool pinned_window_enabled() {
    const char* env = std::getenv("MCC_PIN_HOST_WINDOW");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool compact_changed_enabled() {
    const char* env = std::getenv("MCC_COMPACT_CHANGED");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool compact_pageable_enabled() {
    const char* env = std::getenv("MCC_COMPACT_PAGEABLE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool omp_warmup_enabled() {
    const char* env = std::getenv("MCC_OMP_WARMUP");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static void warmup_openmp_runtime() {
    #pragma omp parallel num_threads(1)
    {
    }
}

static int preload_copy_threads() {
    if (const char* env = std::getenv("MCC_PRELOAD_COPY_THREADS")) {
        int requested = std::atoi(env);
        if (requested > 0) return requested;
    }
    return 1;
}

static void copy_preload_slots(float* dst, const float* src, int slots, int threads) {
    if (threads <= 1 || slots <= 1) {
        std::memcpy(dst, src,
                    static_cast<size_t>(slots) * SPATIAL_POINTS * sizeof(float));
        return;
    }

    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int slot = 0; slot < slots; ++slot) {
        std::memcpy(dst + static_cast<size_t>(slot) * SPATIAL_POINTS,
                    src + static_cast<size_t>(slot) * SPATIAL_POINTS,
                    SPATIAL_POINTS * sizeof(float));
    }
}

static bool async_write_enabled() {
    const char* env = std::getenv("MCC_ASYNC_WRITE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool async_dcu_init_enabled() {
    const char* env = std::getenv("MCC_ASYNC_DCU_INIT");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool season_batch_enabled() {
    const char* env = std::getenv("MCC_SEASON_BATCH");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool preload_season_enabled() {
    const char* env = std::getenv("MCC_PRELOAD_SEASON");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool preload_direct_h2d_enabled() {
    const char* env = std::getenv("MCC_PRELOAD_DIRECT_H2D");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool season_pageable_output_enabled() {
    const char* env = std::getenv("MCC_SEASON_PAGEABLE_OUTPUT");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
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
    if (target_doy_begin > target_doy_end) {
        std::cerr << "ERROR: invalid target DOY range "
                  << target_doy_begin << ".." << target_doy_end << std::endl;
        return 1;
    }
    std::cout << "[experiment] output_dir=" << output_dir << std::endl;
    if (target_doy_begin != TARGET_DOY_BEGIN || target_doy_end != TARGET_DOY_END) {
        std::cout << "[experiment] target DOY range="
                  << target_doy_begin << ".." << target_doy_end << std::endl;
    }
    if (omp_warmup_enabled()) {
        std::cout << "[experiment] MCC_OMP_WARMUP=1" << std::endl;
        warmup_openmp_runtime();
    }

    if (season_batch_enabled()) {
        std::cout << "[experiment] MCC_SEASON_BATCH=1" << std::endl;
        int target_days = target_doy_end - target_doy_begin + 1;
        int season_start_doy = 0;
        int season_days = 0;

        int expected_season_days = target_days + 2 * CLIM_DELTA_DAY;
        size_t expected_season_slots =
            static_cast<size_t>(expected_season_days) * CLIM_YEARS;
        size_t season_bytes =
            expected_season_slots * SPATIAL_POINTS * sizeof(float);

        std::cout << "[season] allocating input timeline "
                  << season_bytes / (1024.0 * 1024.0 * 1024.0)
                  << " GB" << std::endl;
        float* h_season_data = static_cast<float*>(std::malloc(season_bytes));
        if (!h_season_data) {
            std::cerr << "ERROR: cannot allocate season timeline bytes="
                      << season_bytes << std::endl;
            return 1;
        }

        auto phase_t0 = Clock::now();
        read_season_timeline(h_season_data, target_doy_begin, target_doy_end,
                             &season_start_doy, &season_days);
        std::cout << "[profile] read_season_timeline="
                  << std::chrono::duration<double>(Clock::now() - phase_t0).count()
                  << "s season_start_doy=" << season_start_doy
                  << " season_days=" << season_days << std::endl;

        if (season_days != expected_season_days) {
            std::cerr << "ERROR: unexpected season_days=" << season_days
                      << " expected=" << expected_season_days << std::endl;
            std::free(h_season_data);
            return 1;
        }

        float* h_mean_all = nullptr;
        float* h_p90_all = nullptr;
        size_t output_bytes =
            static_cast<size_t>(target_days) * SPATIAL_POINTS * sizeof(float);
        bool pageable_output = season_pageable_output_enabled();
        if (pageable_output) {
            std::cout << "[season] MCC_SEASON_PAGEABLE_OUTPUT=1" << std::endl;
            h_mean_all = static_cast<float*>(std::malloc(output_bytes));
            h_p90_all = static_cast<float*>(std::malloc(output_bytes));
            if (!h_mean_all || !h_p90_all) {
                std::cerr << "ERROR: cannot allocate season output bytes="
                          << output_bytes << std::endl;
                std::free(h_mean_all);
                std::free(h_p90_all);
                std::free(h_season_data);
                return 1;
            }
        } else {
            HIP_CHECK(hipHostMalloc(&h_mean_all, output_bytes, hipHostMallocDefault));
            HIP_CHECK(hipHostMalloc(&h_p90_all, output_bytes, hipHostMallocDefault));
        }

        phase_t0 = Clock::now();
        dispatch_season_batch_to_4_dcus(h_season_data, h_mean_all, h_p90_all,
                                        target_days, season_days);
        std::cout << "[profile] dispatch_season_batch="
                  << std::chrono::duration<double>(Clock::now() - phase_t0).count()
                  << "s" << std::endl;

        phase_t0 = Clock::now();
        for (int day_idx = 0; day_idx < target_days; ++day_idx) {
            int doy = target_doy_begin + day_idx;
            std::string out_file = output_dir + doy_to_mmdd(doy) + ".nc";
            const float* day_mean =
                h_mean_all + static_cast<size_t>(day_idx) * SPATIAL_POINTS;
            const float* day_p90 =
                h_p90_all + static_cast<size_t>(day_idx) * SPATIAL_POINTS;
            write_output(out_file, doy, day_mean, day_p90);
        }
        std::cout << "[profile] write_all_outputs="
                  << std::chrono::duration<double>(Clock::now() - phase_t0).count()
                  << "s" << std::endl;

        if (pageable_output) {
            std::free(h_mean_all);
            std::free(h_p90_all);
        } else {
            HIP_CHECK(hipHostFree(h_mean_all));
            HIP_CHECK(hipHostFree(h_p90_all));
        }
        std::free(h_season_data);

        std::cout << "\n>>> 全部 " << target_days << " 天计算完成！" << std::endl;
        std::cout << "[profile] program_total="
                  << std::chrono::duration<double>(Clock::now() - program_t0).count()
                  << "s" << std::endl;
        return 0;
    }

    float* h_sst_data = nullptr;
    size_t memory_bytes = TOTAL_ELEMENTS * sizeof(float);
    std::cout << "[主线程] 正在分配 " << memory_bytes / (1024.0 * 1024.0 * 1024.0)
              << " GB 的主机内存..." << std::endl;

    bool pinned_window = pinned_window_enabled();
    if (pinned_window) {
        HIP_CHECK(hipHostMalloc(&h_sst_data, memory_bytes, hipHostMallocDefault));
        std::cout << "[experiment] MCC_PIN_HOST_WINDOW=1" << std::endl;
    } else {
        h_sst_data = (float*)std::malloc(memory_bytes);
    }
    if (!h_sst_data) {
        std::cerr << "ERROR: 无法分配 " << memory_bytes << " 字节内存" << std::endl;
        return 1;
    }
    if (env_enabled("MCC_SKIP_HOST_WINDOW_ZERO")) {
        std::cout << "[experiment] MCC_SKIP_HOST_WINDOW_ZERO=1" << std::endl;
    } else {
        std::memset(h_sst_data, 0, memory_bytes);
    }

    // 2. 分配结果缓冲区（必须用 hipHostMalloc 以支持 D2H 传输）
    float* h_mean = nullptr;
    float* h_p90  = nullptr;
    HIP_CHECK(hipHostMalloc(&h_mean, SPATIAL_POINTS * sizeof(float), hipHostMallocDefault));
    HIP_CHECK(hipHostMalloc(&h_p90,  SPATIAL_POINTS * sizeof(float), hipHostMallocDefault));
    std::memset(h_mean, 0, SPATIAL_POINTS * sizeof(float));
    std::memset(h_p90, 0, SPATIAL_POINTS * sizeof(float));

    bool preload_season = preload_season_enabled();
    if (preload_season) {
        std::cout << "[experiment] MCC_PRELOAD_SEASON=1" << std::endl;
    }
    int preload_copy_thread_count = preload_copy_threads();
    if (preload_season && preload_copy_thread_count > 1) {
        std::cout << "[experiment] MCC_PRELOAD_COPY_THREADS="
                  << preload_copy_thread_count << std::endl;
    }
    bool preload_direct_h2d = preload_direct_h2d_enabled();
    if (preload_direct_h2d) {
        std::cout << "[experiment] MCC_PRELOAD_DIRECT_H2D=1" << std::endl;
    }
    bool preload_source_window = preload_direct_h2d &&
                                 env_enabled("MCC_PRELOAD_SOURCE_WINDOW");
    if (preload_source_window) {
        std::cout << "[experiment] MCC_PRELOAD_SOURCE_WINDOW=1" << std::endl;
    }
    if (preload_direct_h2d && !preload_season) {
        std::cerr << "ERROR: MCC_PRELOAD_DIRECT_H2D requires MCC_PRELOAD_SEASON"
                  << std::endl;
        if (pinned_window) {
            HIP_CHECK(hipHostFree(h_sst_data));
        } else {
            std::free(h_sst_data);
        }
        HIP_CHECK(hipHostFree(h_mean));
        HIP_CHECK(hipHostFree(h_p90));
        return 1;
    }

    bool compact_changed = compact_changed_enabled();
    if (preload_season && compact_changed) {
        std::cerr << "ERROR: MCC_PRELOAD_SEASON and MCC_COMPACT_CHANGED are mutually exclusive"
                  << std::endl;
        if (pinned_window) {
            HIP_CHECK(hipHostFree(h_sst_data));
        } else {
            std::free(h_sst_data);
        }
        HIP_CHECK(hipHostFree(h_mean));
        HIP_CHECK(hipHostFree(h_p90));
        return 1;
    }
    bool compact_pageable = compact_pageable_enabled();
    float* h_changed_compact = nullptr;
    if (compact_changed) {
        std::cout << "[experiment] MCC_COMPACT_CHANGED=1" << std::endl;
        size_t compact_bytes = CLIM_YEARS * SPATIAL_POINTS * sizeof(float);
        if (compact_pageable) {
            h_changed_compact = static_cast<float*>(std::malloc(compact_bytes));
            if (!h_changed_compact) {
                std::cerr << "ERROR: cannot allocate compact buffer bytes="
                          << compact_bytes << std::endl;
                if (pinned_window) {
                    HIP_CHECK(hipHostFree(h_sst_data));
                } else {
                    std::free(h_sst_data);
                }
                HIP_CHECK(hipHostFree(h_mean));
                HIP_CHECK(hipHostFree(h_p90));
                return 1;
            }
            std::cout << "[experiment] MCC_COMPACT_PAGEABLE=1" << std::endl;
        } else {
            HIP_CHECK(hipHostMalloc(&h_changed_compact,
                                    compact_bytes,
                                    hipHostMallocDefault));
        }
    }

    float* h_preload_season = nullptr;
    float* h_preload_changed_source = nullptr;
    int preload_start_doy = 0;
    int preload_days = 0;

    std::future<void> dcu_init_future;
    bool dcu_init_active = false;
    if (async_dcu_init_enabled()) {
        std::cout << "[experiment] MCC_ASYNC_DCU_INIT=1" << std::endl;
        dcu_init_active = true;
        dcu_init_future = std::async(std::launch::async, []() {
            prepare_dcu_persistent_buffers();
        });
    }

    // 3. 初始化滑动窗口（读取第一个 DOY 的全部 330 个文件）
    auto phase_t0 = Clock::now();
    if (preload_season) {
        int target_days = target_doy_end - target_doy_begin + 1;
        int expected_preload_days = target_days + 2 * CLIM_DELTA_DAY;
        size_t preload_slots =
            static_cast<size_t>(expected_preload_days) * CLIM_YEARS;
        size_t preload_bytes = preload_slots * SPATIAL_POINTS * sizeof(float);

        std::cout << "[preload] allocating season timeline "
                  << preload_bytes / (1024.0 * 1024.0 * 1024.0)
                  << " GB" << std::endl;
        h_preload_season = static_cast<float*>(std::malloc(preload_bytes));
        if (!h_preload_season) {
            std::cerr << "ERROR: cannot allocate preload season bytes="
                      << preload_bytes << std::endl;
            if (h_changed_compact) {
                if (compact_pageable) {
                    std::free(h_changed_compact);
                } else {
                    HIP_CHECK(hipHostFree(h_changed_compact));
                }
            }
            HIP_CHECK(hipHostFree(h_mean));
            HIP_CHECK(hipHostFree(h_p90));
            if (pinned_window) {
                HIP_CHECK(hipHostFree(h_sst_data));
            } else {
                std::free(h_sst_data);
            }
            return 1;
        }
        read_season_timeline(h_preload_season, target_doy_begin, target_doy_end,
                             &preload_start_doy, &preload_days);
        if (preload_days != expected_preload_days) {
            std::cerr << "ERROR: unexpected preload_days=" << preload_days
                      << " expected=" << expected_preload_days << std::endl;
            std::free(h_preload_season);
            if (h_changed_compact) {
                if (compact_pageable) {
                    std::free(h_changed_compact);
                } else {
                    HIP_CHECK(hipHostFree(h_changed_compact));
                }
            }
            HIP_CHECK(hipHostFree(h_mean));
            HIP_CHECK(hipHostFree(h_p90));
            if (pinned_window) {
                HIP_CHECK(hipHostFree(h_sst_data));
            } else {
                std::free(h_sst_data);
            }
            return 1;
        }
        if (!preload_source_window) {
            copy_preload_slots(h_sst_data, h_preload_season,
                               static_cast<int>(DAYS_TOTAL),
                               preload_copy_thread_count);
        }
        std::cout << "[profile] preload_initialize_window="
                  << std::chrono::duration<double>(Clock::now() - phase_t0).count()
                  << "s season_start_doy=" << preload_start_doy
                  << " season_days=" << preload_days << std::endl;
    } else {
        initialize_window(h_sst_data, target_doy_begin);
        std::cout << "[profile] initialize_window="
                  << std::chrono::duration<double>(Clock::now() - phase_t0).count()
                  << "s" << std::endl;
    }
    if (debug_enabled()) {
        print_buffer_stats("host_window_after_initialize", h_sst_data, TOTAL_ELEMENTS);
    }
    if (dcu_init_active) {
        phase_t0 = Clock::now();
        dcu_init_future.get();
        std::cout << "[profile] async_dcu_init_wait="
                  << std::chrono::duration<double>(Clock::now() - phase_t0).count()
                  << "s" << std::endl;
    }

    // 4. 循环处理目标日期
    const int window = 2 * CLIM_DELTA_DAY + 1;
    std::vector<int> changed_slots(CLIM_YEARS);
    bool compact_ready = false;
    bool async_write = async_write_enabled();
    if (async_write) {
        std::cout << "[experiment] MCC_ASYNC_WRITE=1" << std::endl;
    }
    OutputSnapshot async_snapshot;
    std::future<void> async_write_future;
    bool async_write_active = false;

    for (int doy = target_doy_begin; doy <= target_doy_end; ++doy) {
        auto day_t0 = Clock::now();
        std::cout << "\n>>> 处理 DOY " << doy << " (" << doy_to_mmdd(doy) << ")" << std::endl;

        // DCU 计算：首日全量上传，之后只上传滑动窗口替换的 30 个 slot
        const int* upload_slots = (doy == target_doy_begin) ? nullptr : changed_slots.data();
        int upload_count = (doy == target_doy_begin) ? 0 : static_cast<int>(changed_slots.size());
        PrefetchAfterUpload prefetch_req{h_sst_data, doy, target_doy_begin, target_doy_end};
        float* dispatch_source =
            (preload_source_window && doy == target_doy_begin && h_preload_season)
                ? h_preload_season
                : h_sst_data;
        if (preload_direct_h2d && doy != target_doy_begin && h_preload_changed_source) {
            dispatch_to_4_dcus_with_output_incremental_compact(
                dispatch_source, h_preload_changed_source, h_mean, h_p90,
                upload_slots, upload_count);
        } else if (compact_changed && doy != target_doy_begin && compact_ready) {
            dispatch_to_4_dcus_with_output_incremental_compact(
                dispatch_source, h_changed_compact, h_mean, h_p90,
                upload_slots, upload_count);
        } else if (overlap_prefetch_enabled()) {
            dispatch_to_4_dcus_with_output_incremental_hook(
                dispatch_source, h_mean, h_p90,
                upload_slots, upload_count,
                start_prefetch_after_upload, &prefetch_req);
        } else {
            dispatch_to_4_dcus_with_output_incremental(
                dispatch_source, h_mean, h_p90,
                upload_slots, upload_count);
        }

        if (debug_enabled()) {
            print_cpu_gpu_samples(h_sst_data, h_mean, h_p90);
        }

        // 写出结果
        std::string mmdd = doy_to_mmdd(doy);
        std::string out_file = output_dir + mmdd + ".nc";
        phase_t0 = Clock::now();
        if (async_write) {
            if (async_write_active) {
                async_write_future.get();
                async_write_active = false;
            }
            prepare_output_snapshot(async_snapshot, out_file, doy, h_mean, h_p90);
            async_write_future = std::async(std::launch::async, [&async_snapshot]() {
                write_output_snapshot(async_snapshot);
            });
            async_write_active = true;
        } else {
            write_output(out_file, doy, h_mean, h_p90);
        }
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
            if (preload_season) {
                int entering_day_idx = (doy + 1 + CLIM_DELTA_DAY) - preload_start_doy;
                if (entering_day_idx < 0 || entering_day_idx >= preload_days) {
                    std::cerr << "ERROR: preload entering_day_idx=" << entering_day_idx
                              << " outside 0.." << (preload_days - 1) << std::endl;
                    return 1;
                }
                float* dst = h_sst_data +
                    static_cast<size_t>(replace_offset_idx) * CLIM_YEARS * SPATIAL_POINTS;
                const float* src = h_preload_season +
                    static_cast<size_t>(entering_day_idx) * CLIM_YEARS * SPATIAL_POINTS;
                if (preload_direct_h2d) {
                    h_preload_changed_source = const_cast<float*>(src);
                } else {
                    copy_preload_slots(dst, src, CLIM_YEARS, preload_copy_thread_count);
                }
            } else if (compact_changed) {
                load_next_doy_compact(h_changed_compact, doy, target_doy_begin);
                compact_ready = true;
            } else {
                slide_window_to_next_doy(h_sst_data, doy, target_doy_begin);
            }
            slide_s = std::chrono::duration<double>(Clock::now() - phase_t0).count();
        }
        std::cout << "[profile] day_total="
                  << std::chrono::duration<double>(Clock::now() - day_t0).count()
                  << "s write=" << write_s
                  << "s slide=" << slide_s << "s" << std::endl;
    }

    // 5. 释放资源
    if (async_write_active) {
        phase_t0 = Clock::now();
        async_write_future.get();
        std::cout << "[profile] async_write_final_wait="
                  << std::chrono::duration<double>(Clock::now() - phase_t0).count()
                  << "s" << std::endl;
    }
    if (env_enabled("MCC_SKIP_FINAL_CLEANUP")) {
        std::cout << "[experiment] MCC_SKIP_FINAL_CLEANUP=1" << std::endl;
        std::cout << "\n>>> 鍏ㄩ儴 " << (target_doy_end - target_doy_begin + 1)
                  << " 澶╄绠楀畬鎴愶紒" << std::endl;
        std::cout << "[profile] program_total="
                  << std::chrono::duration<double>(Clock::now() - program_t0).count()
                  << "s" << std::endl;
        return 0;
    }
    cleanup_dcu_persistent_buffers();
    cleanup_dcu_buffers();
    if (h_changed_compact) {
        if (compact_pageable) {
            std::free(h_changed_compact);
        } else {
            HIP_CHECK(hipHostFree(h_changed_compact));
        }
    }
    if (h_preload_season) {
        std::free(h_preload_season);
    }
    HIP_CHECK(hipHostFree(h_mean));
    HIP_CHECK(hipHostFree(h_p90));
    if (pinned_window) {
        HIP_CHECK(hipHostFree(h_sst_data));
    } else {
        std::free(h_sst_data);
    }

    std::cout << "\n>>> 全部 " << (target_doy_end - target_doy_begin + 1)
              << " 天计算完成！" << std::endl;
    std::cout << "[profile] program_total="
              << std::chrono::duration<double>(Clock::now() - program_t0).count()
              << "s" << std::endl;
    return 0;
}
