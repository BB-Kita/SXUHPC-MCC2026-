#include <sys/stat.h>
#include <unistd.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <cmath>
#include <future>
#include <limits>
#include <numeric>
#include <netcdf.h>
#include <mutex>
#include <thread>
#include <hip/hip_runtime.h>
#include "config.h"
#include "io_handler.h"
#include "compute_dcu.h"

struct OutputTask {
    std::string out_path;
    int doy = 0;
    std::vector<float> data;
};

static std::deque<OutputTask> g_output_queue;
static std::mutex g_output_mutex;
static std::condition_variable g_output_cv;
static std::thread g_output_worker;
static bool g_output_worker_started = false;
static bool g_output_worker_stop = false;
static int g_output_active = 0;

static bool mc_packed_enabled_main() {
    const char* env = std::getenv("MCC_MC_PACKED");
    return env && env[0] != '\0' && env[0] != '0';
}

static bool season_preload_enabled_main() {
    const char* env = std::getenv("MCC_SEASON_PRELOAD");
    return env && env[0] != '\0' && env[0] != '0';
}

static bool output_float_enabled_main() {
    const char* env = std::getenv("MCC_OUTPUT_FLOAT");
    return env && env[0] != '\0' && env[0] != '0';
}

static bool output_classic_enabled_main() {
    const char* env = std::getenv("MCC_OUTPUT_CLASSIC");
    return env && env[0] != '\0' && env[0] != '0';
}

static bool output_light_classic_enabled_main() {
    const char* env = std::getenv("MCC_OUTPUT_LIGHT_CLASSIC");
    return env && env[0] != '\0' && env[0] != '0';
}

static bool output_minimal_enabled_main() {
    return false;
}

static bool source_local_layout_enabled_main() {
    const char* env = std::getenv("MCC_SOURCE_LOCAL_LAYOUT");
    return env && env[0] != '\0' && env[0] != '0';
}

static bool source_local_pinned_enabled_main() {
    const char* env = std::getenv("MCC_SOURCE_LOCAL_PINNED");
    return env == nullptr || env[0] == '\0' || env[0] != '0';
}

static bool source_stream_enabled_main() {
    const char* env = std::getenv("MCC_SOURCE_STREAM");
    return env && env[0] != '\0' && env[0] != '0';
}

static int source_anchor_stride_main() {
    const char* env = std::getenv("MCC_ANCHOR_STRIDE");
    if (!env) return 0;
    int stride = std::atoi(env);
    return stride > 1 ? stride : 0;
}

static int source_anchor_group_size_main() {
    const char* env = std::getenv("MCC_ANCHOR_GROUP_SIZE");
    if (!env) return 2;
    int group_size = std::atoi(env);
    return group_size > 0 ? group_size : 2;
}

static bool source_async_upload_enabled_main() {
    const char* env = std::getenv("MCC_SOURCE_ASYNC_UPLOAD");
    return env && env[0] != '\0' && env[0] != '0';
}

static int source_stream_chunk_days_main() {
    int chunk_days = 16;
    if (const char* env = std::getenv("MCC_SOURCE_STREAM_CHUNK_DAYS")) {
        int requested = std::atoi(env);
        if (requested > 0) chunk_days = requested;
    }
    if (chunk_days < 1) chunk_days = 1;
    return chunk_days;
}

static int mc_k_from_env_main() {
    const char* env = std::getenv("MCC_MC_K");
    if (!env) return 0;
    int requested_k = std::atoi(env);
    if (requested_k <= 0) return 0;
    if (requested_k > 120) requested_k = 120;
    int k_per_year = requested_k / CLIM_YEARS;
    if (k_per_year <= 0) k_per_year = 1;
    return k_per_year * CLIM_YEARS;
}


static void arrays_to_nc(const std::string& nc_path, int doy, const float* mean, const float* p90) {
    int ncid, lat_dim, lon_dim, day_dim, doy_var, lat_var, lon_var, clim_var, p90_var;
    const bool output_float = output_float_enabled_main();
    const bool output_classic = output_classic_enabled_main();
    const bool light_classic = output_classic && output_light_classic_enabled_main();
    int create_mode = NC_CLOBBER;
    if (!output_classic) {
        create_mode |= NC_NETCDF4;
    }
    nc_create(nc_path.c_str(), create_mode, &ncid);
    nc_set_fill(ncid, NC_NOFILL, nullptr);
    nc_def_dim(ncid, "Lat", LAT_SIZE, &lat_dim);
    nc_def_dim(ncid, "Lon", LON_SIZE, &lon_dim);
    int dims[2] = {lat_dim, lon_dim};

    if (output_minimal_enabled_main()) {
        nc_def_var(ncid, "Climmean", output_float ? NC_FLOAT : NC_DOUBLE, 2, dims, &clim_var);
        nc_def_var(ncid, "P90_sst", output_float ? NC_FLOAT : NC_DOUBLE, 2, dims, &p90_var);
        nc_enddef(ncid);
        if (output_float) {
            nc_put_var_float(ncid, clim_var, mean);
            nc_put_var_float(ncid, p90_var, p90);
        } else {
            double* buf = (double*)malloc(SPATIAL_POINTS * sizeof(double));
            for (size_t i = 0; i < SPATIAL_POINTS; ++i) buf[i] = (double)mean[i];
            nc_put_var_double(ncid, clim_var, buf);
            for (size_t i = 0; i < SPATIAL_POINTS; ++i) buf[i] = (double)p90[i];
            nc_put_var_double(ncid, p90_var, buf);
            free(buf);
        }
        nc_close(ncid);
        return;
    }

    nc_def_dim(ncid, "Day", 1, &day_dim);
    nc_def_var(ncid, "dayofyear", NC_DOUBLE, 1, &day_dim, &doy_var);
    if (!light_classic) {
        nc_put_att_text(ncid, doy_var, "long_name", 29, "Day of year (1-365, no 29Feb)");
    }
    nc_def_var(ncid, "Lat", NC_DOUBLE, 1, &lat_dim, &lat_var);
    nc_def_var(ncid, "Lon", NC_DOUBLE, 1, &lon_dim, &lon_var);
    nc_def_var(ncid, "Climmean", output_float ? NC_FLOAT : NC_DOUBLE, 2, dims, &clim_var);
    if (!light_classic) {
        nc_put_att_text(ncid, clim_var, "long_name", 30, "OSTIA SST climatology 1991-2020");
    }
    nc_def_var(ncid, "P90_sst", output_float ? NC_FLOAT : NC_DOUBLE, 2, dims, &p90_var);
    if (!light_classic) {
        nc_put_att_text(ncid, p90_var, "long_name", 22, "90th percentile of SST");
    }
    nc_enddef(ncid);
    double doy_d = (double)doy;
    nc_put_var_double(ncid, doy_var, &doy_d);
    static double* lat_arr = [](){ double* p = (double*)malloc(LAT_SIZE*8); for(size_t i=0;i<LAT_SIZE;++i) p[i]=-90.0+i*0.25; return p; }();
    static double* lon_arr = [](){ double* p = (double*)malloc(LON_SIZE*8); for(size_t i=0;i<LON_SIZE;++i) p[i]=i*0.25; return p; }();
    if (!light_classic) {
        nc_put_var_double(ncid, lat_var, lat_arr);
        nc_put_var_double(ncid, lon_var, lon_arr);
    }
    double* buf = nullptr;
    if (output_float) {
        nc_put_var_float(ncid, clim_var, mean);
        nc_put_var_float(ncid, p90_var, p90);
    } else {
        buf = (double*)malloc(SPATIAL_POINTS * sizeof(double));
        for (size_t i = 0; i < SPATIAL_POINTS; ++i) buf[i] = (double)mean[i];
        nc_put_var_double(ncid, clim_var, buf);
        for (size_t i = 0; i < SPATIAL_POINTS; ++i) buf[i] = (double)p90[i];
        nc_put_var_double(ncid, p90_var, buf);
    }
    nc_close(ncid);
    if (buf) free(buf);
}

static void output_worker_loop() {
    while (true) {
        OutputTask task;
        {
            std::unique_lock<std::mutex> lock(g_output_mutex);
            g_output_cv.wait(lock, [] {
                return g_output_worker_stop || !g_output_queue.empty();
            });
            if (g_output_queue.empty()) {
                if (g_output_worker_stop) break;
                continue;
            }
            task = std::move(g_output_queue.front());
            g_output_queue.pop_front();
            ++g_output_active;
        }

        arrays_to_nc(task.out_path, task.doy,
                     task.data.data(),
                     task.data.data() + SPATIAL_POINTS);

        {
            std::lock_guard<std::mutex> lock(g_output_mutex);
            --g_output_active;
            if (g_output_queue.empty() && g_output_active == 0) {
                g_output_cv.notify_all();
            }
        }
    }
}

static void ensure_output_worker() {
    std::lock_guard<std::mutex> lock(g_output_mutex);
    if (g_output_worker_started) return;
    g_output_worker_stop = false;
    g_output_active = 0;
    g_output_worker = std::thread(output_worker_loop);
    g_output_worker_started = true;
}

static void wait_for_output_writes() {
    if (!g_output_worker_started) return;
    {
        std::unique_lock<std::mutex> lock(g_output_mutex);
        g_output_cv.wait(lock, [] {
            return g_output_queue.empty() && g_output_active == 0;
        });
        g_output_worker_stop = true;
        g_output_cv.notify_all();
    }
    if (g_output_worker.joinable()) {
        g_output_worker.join();
    }
    {
        std::lock_guard<std::mutex> lock(g_output_mutex);
        g_output_worker_started = false;
        g_output_worker_stop = false;
        g_output_active = 0;
    }
}

static void write_output(const std::string& out_path, int doy,
                          const float* mean, const float* p90) {
    OutputTask task;
    task.out_path = out_path;
    task.doy = doy;
    task.data.resize(2 * SPATIAL_POINTS);
    std::memcpy(task.data.data(), mean, SPATIAL_POINTS * sizeof(float));
    std::memcpy(task.data.data() + SPATIAL_POINTS, p90, SPATIAL_POINTS * sizeof(float));

    ensure_output_worker();
    {
        std::lock_guard<std::mutex> lock(g_output_mutex);
        g_output_queue.emplace_back(std::move(task));
    }
    g_output_cv.notify_one();
}
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
// P3: warm HIP runtime BEFORE timer (hipInit ~1s)    hipSetDevice(0);    hipDeviceSynchronize();    { void* d; hipMalloc(&d, 1); hipFree(d); }    std::cout << "[P3] HIP warmed up" << std::endl;
// P3: warm HIP runtime BEFORE timer (hipInit ~1s)    hipSetDevice(0);    hipDeviceSynchronize();    { void* d; hipMalloc(&d, 1); hipFree(d); }    std::cout << "[P3] HIP warmed up" << std::endl;
hipSetDevice(0); hipDeviceSynchronize();    { void* d; hipMalloc(&d, 1); hipFree(d); }
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

    const bool use_mc_packed = mc_packed_enabled_main();
    const bool use_season_preload = season_preload_enabled_main();
    const bool need_window_buffer = use_mc_packed || !use_season_preload;

    float* h_sst_data = nullptr;
    size_t memory_bytes = TOTAL_ELEMENTS * sizeof(float);
    if (need_window_buffer) {
        std::cout << "[主线程] 正在分配 " << memory_bytes / (1024.0 * 1024.0 * 1024.0)
                  << " GB 的主机内存..." << std::endl;

        h_sst_data = (float*)std::malloc(memory_bytes);
        if (!h_sst_data) {
            std::cerr << "ERROR: 无法分配 " << memory_bytes << " 字节内存" << std::endl;
            return 1;
        }
        std::memset(h_sst_data, 0, memory_bytes);
    } else {
        std::cout << "[主线程] source-preload path skips "
                  << memory_bytes / (1024.0 * 1024.0 * 1024.0)
                  << " GB sliding-window host buffer" << std::endl;
    }

    // 2. 分配结果缓冲区（必须用 hipHostMalloc 以支持 D2H 传输）
    float* h_mean = nullptr;
    float* h_p90  = nullptr;
    HIP_CHECK(hipHostMalloc(&h_mean, SPATIAL_POINTS * sizeof(float), hipHostMallocDefault));
    HIP_CHECK(hipHostMalloc(&h_p90,  SPATIAL_POINTS * sizeof(float), hipHostMallocDefault));
    std::memset(h_mean, 0, SPATIAL_POINTS * sizeof(float));
    std::memset(h_p90, 0, SPATIAL_POINTS * sizeof(float));

    // P1: pinned staging for incremental H2D (30 slots, ~124MB)
    const size_t stag_bytes = (size_t)CLIM_YEARS * SPATIAL_POINTS * sizeof(float);
    float* h_staging = nullptr;
    if (need_window_buffer) {
        hipError_t se = hipHostMalloc(&h_staging, stag_bytes, hipHostMallocDefault);
        if (se == hipSuccess) std::cout << "[P1] pinned staging: " << stag_bytes/(1024.0*1024.0) << " MB" << std::endl;
        else std::cerr << "[P1] staging pin failed" << std::endl;
    }

    if (use_mc_packed) {
        int packed_k = mc_k_from_env_main();
        if (packed_k <= 0) {
            std::cerr << "ERROR: MCC_MC_PACKED=1 requires MCC_MC_K > 0" << std::endl;
            cleanup_dcu_persistent_buffers();
            cleanup_dcu_buffers();
            HIP_CHECK(hipHostFree(h_mean));
            HIP_CHECK(hipHostFree(h_p90));
            std::free(h_sst_data);
            if (h_staging) hipHostFree(h_staging);
            return 1;
        }

        std::vector<int> packed_slots(packed_k);
        std::iota(packed_slots.begin(), packed_slots.end(), 0);

        std::cout << "[experiment] MCC_MC_PACKED=1 normalized_K=" << packed_k
                  << " (only slots 0.." << (packed_k - 1) << " are read/uploaded/scanned)"
                  << std::endl;

        for (int doy = target_doy_begin; doy <= target_doy_end; ++doy) {
            auto day_t0 = Clock::now();
            std::cout << "\n>>> 处理 DOY " << doy << " (" << doy_to_mmdd(doy)
                      << ") [MC packed]" << std::endl;

            auto read_t0 = Clock::now();
            read_mc_packed_for_doy(h_sst_data, doy, packed_k);
            double io_s = std::chrono::duration<double>(Clock::now() - read_t0).count();
            std::cout << "[profile] mc_packed_read=" << io_s << "s" << std::endl;

            float* h_src = nullptr;
            dispatch_to_4_dcus_with_output_incremental_hook(
                h_sst_data, h_src, h_mean, h_p90,
                packed_slots.data(), static_cast<int>(packed_slots.size()),
                nullptr, nullptr);

            std::string mmdd = doy_to_mmdd(doy);
            std::string out_file = output_dir + mmdd + ".nc";
            auto write_t0 = Clock::now();
            write_output(out_file, doy, h_mean, h_p90);
            double write_s = std::chrono::duration<double>(Clock::now() - write_t0).count();
            std::cout << "[profile] write_output=" << write_s << "s" << std::endl;
            std::cout << "  输出: " << out_file << std::endl;

            std::cout << "[profile] day_total="
                      << std::chrono::duration<double>(Clock::now() - day_t0).count()
                      << "s read=" << io_s
                      << "s write=" << write_s
                      << "s" << std::endl;
        }

        wait_for_output_writes();
        cleanup_dcu_persistent_buffers();
        cleanup_dcu_buffers();
        HIP_CHECK(hipHostFree(h_mean));
        HIP_CHECK(hipHostFree(h_p90));
        std::free(h_sst_data);
        if (h_staging) hipHostFree(h_staging);

        std::cout << "\n>>> 全部 " << (target_doy_end - target_doy_begin + 1)
                  << " 天计算完成！" << std::endl;
        std::cout << "[profile] program_total="
                  << std::chrono::duration<double>(Clock::now() - program_t0).count()
                  << "s" << std::endl;
        return 0;
    }

    if (use_season_preload) {
        const int source_days =
            (target_doy_end + CLIM_DELTA_DAY) - (target_doy_begin - CLIM_DELTA_DAY) + 1;
        const int source_slots = source_days * CLIM_YEARS;
        const size_t source_elements = (size_t)source_slots * SPATIAL_POINTS;
        const size_t source_bytes = source_elements * sizeof(float);

        std::cout << "[experiment] MCC_SEASON_PRELOAD=1 source_days=" << source_days
                  << " source_slots=" << source_slots
                  << " source_memory=" << source_bytes / (1024.0 * 1024.0 * 1024.0)
                  << " GB output_float=" << (output_float_enabled_main() ? "ON" : "OFF")
                  << std::endl;

        if (source_local_layout_enabled_main()) {
            int gpu_count = 0;
            HIP_CHECK(hipGetDeviceCount(&gpu_count));
            if (gpu_count <= 0) {
                std::cerr << "ERROR: no DCU for local source layout" << std::endl;
                cleanup_dcu_persistent_buffers();
                cleanup_dcu_buffers();
                HIP_CHECK(hipHostFree(h_mean));
                HIP_CHECK(hipHostFree(h_p90));
                std::free(h_sst_data);
                if (h_staging) hipHostFree(h_staging);
                return 1;
            }

            int rows_per_gpu = ((int)LAT_SIZE + gpu_count - 1) / gpu_count;
            std::vector<float*> h_source_local(gpu_count, nullptr);
            std::vector<int> lat_starts(gpu_count, 0);
            std::vector<int> local_rows(gpu_count, 0);
            bool local_pinned = source_local_pinned_enabled_main();

            for (int i = 0; i < gpu_count; ++i) {
                lat_starts[i] = i * rows_per_gpu;
                int lat_end = std::min((i + 1) * rows_per_gpu, (int)LAT_SIZE);
                local_rows[i] = lat_end - lat_starts[i];
                if (local_rows[i] < 0) local_rows[i] = 0;
            }

            if (source_stream_enabled_main()) {
                const int anchor_stride = source_anchor_stride_main();
                if (anchor_stride > 1) {
                    const int window_days = 2 * CLIM_DELTA_DAY + 1;
                    std::vector<int> anchor_doys;
                    for (int doy = target_doy_begin; doy < target_doy_end; doy += anchor_stride) {
                        anchor_doys.push_back(doy);
                    }
                    if (anchor_doys.empty() || anchor_doys.back() != target_doy_end) {
                        anchor_doys.push_back(target_doy_end);
                    }

                    struct AnchorResult {
                        int doy = 0;
                        std::vector<float> mean;
                        std::vector<float> p90;
                    };

                    struct AnchorSourceBlock {
                        int source_begin = 0;
                        int source_days = 0;
                        std::vector<int> anchor_indices;
                    };

                    const int anchor_group_size = source_anchor_group_size_main();
                    std::vector<AnchorSourceBlock> anchor_blocks;
                    int max_block_days = 0;
                    for (int first = 0; first < static_cast<int>(anchor_doys.size());
                         first += anchor_group_size) {
                        const int last = std::min(first + anchor_group_size - 1,
                                                  static_cast<int>(anchor_doys.size()) - 1);
                        AnchorSourceBlock block;
                        block.source_begin = anchor_doys[first] - CLIM_DELTA_DAY;
                        block.source_days = anchor_doys[last] + CLIM_DELTA_DAY - block.source_begin + 1;
                        for (int idx = first; idx <= last; ++idx) {
                            block.anchor_indices.push_back(idx);
                        }
                        max_block_days = std::max(max_block_days, block.source_days);
                        anchor_blocks.emplace_back(std::move(block));
                    }

                    std::vector<float*> anchor_buffers(gpu_count, nullptr);
                    auto free_anchor_buffers = [&]() {
                        for (float*& p : anchor_buffers) {
                            std::free(p);
                            p = nullptr;
                        }
                    };

                    for (int i = 0; i < gpu_count; ++i) {
                        if (local_rows[i] <= 0) continue;
                        const size_t local_spatial = static_cast<size_t>(local_rows[i]) * LON_SIZE;
                        const size_t bytes = static_cast<size_t>(max_block_days) * CLIM_YEARS
                                           * local_spatial * sizeof(float);
                        anchor_buffers[i] = static_cast<float*>(std::malloc(bytes));
                        if (!anchor_buffers[i]) {
                            std::cerr << "ERROR: cannot allocate anchor source buffer part="
                                      << i << " bytes=" << bytes << std::endl;
                            free_anchor_buffers();
                            cleanup_dcu_persistent_buffers();
                            cleanup_dcu_buffers();
                            HIP_CHECK(hipHostFree(h_mean));
                            HIP_CHECK(hipHostFree(h_p90));
                            std::free(h_sst_data);
                            if (h_staging) hipHostFree(h_staging);
                            return 1;
                        }
                    }

                    std::cout << "[experiment] MCC_ANCHOR_STRIDE=" << anchor_stride
                              << " group_size=" << anchor_group_size
                              << " source_blocks=" << anchor_blocks.size()
                              << " window_days=" << window_days
                              << " max_source_days=" << max_block_days
                              << std::endl;

                    init_source_window_device_preloaded(max_block_days);
                    std::vector<AnchorResult> anchors;
                    anchors.reserve(anchor_doys.size());
                    double anchor_read_total_s = 0.0;

                    for (const AnchorSourceBlock& block : anchor_blocks) {
                        std::cout << "\n>>> 读取 anchor source block "
                                  << block.source_begin << ".."
                                  << (block.source_begin + block.source_days - 1)
                                  << " days=" << block.source_days
                                  << " anchors=" << block.anchor_indices.size() << std::endl;
                        auto read_t0 = Clock::now();
                        preload_source_days_local(anchor_buffers.data(), lat_starts.data(),
                                                  local_rows.data(), gpu_count,
                                                  block.source_begin, block.source_days);
                        const double read_s = std::chrono::duration<double>(Clock::now() - read_t0).count();
                        anchor_read_total_s += read_s;
                        std::cout << "[profile] anchor_source_block_read=" << read_s << "s" << std::endl;

                        upload_source_days_preloaded_local(anchor_buffers.data(), 0, block.source_days);
                        for (int anchor_idx : block.anchor_indices) {
                            const int anchor_doy = anchor_doys[anchor_idx];
                            const int source_offset = anchor_doy - CLIM_DELTA_DAY - block.source_begin;
                            std::cout << ">>> 处理 anchor DOY " << anchor_doy
                                      << " (" << doy_to_mmdd(anchor_doy)
                                      << ") source_offset="
                                      << source_offset
                                      << " [exact source-window]" << std::endl;
                            dispatch_source_window_device_preloaded(
                                block.source_days, source_offset, h_mean, h_p90);

                            AnchorResult result;
                            result.doy = anchor_doy;
                            result.mean.resize(SPATIAL_POINTS);
                            result.p90.resize(SPATIAL_POINTS);
                            std::memcpy(result.mean.data(), h_mean, SPATIAL_POINTS * sizeof(float));
                            std::memcpy(result.p90.data(), h_p90, SPATIAL_POINTS * sizeof(float));
                            anchors.emplace_back(std::move(result));
                        }
                    }

                    std::cout << "[profile] anchor_source_read_total="
                              << anchor_read_total_s << "s" << std::endl;

                    size_t right_anchor = 1;
                    for (int doy = target_doy_begin; doy <= target_doy_end; ++doy) {
                        while (right_anchor + 1 < anchors.size() &&
                               anchors[right_anchor].doy < doy) {
                            ++right_anchor;
                        }
                        const size_t left_anchor = right_anchor - 1;
                        const AnchorResult& left = anchors[left_anchor];
                        const AnchorResult& right = anchors[right_anchor];
                        const float t = static_cast<float>(doy - left.doy) /
                                        static_cast<float>(right.doy - left.doy);

                        #pragma omp parallel for schedule(static)
                        for (long long i = 0; i < static_cast<long long>(SPATIAL_POINTS); ++i) {
                            const float mean_l = left.mean[static_cast<size_t>(i)];
                            const float mean_r = right.mean[static_cast<size_t>(i)];
                            const float p90_l = left.p90[static_cast<size_t>(i)];
                            const float p90_r = right.p90[static_cast<size_t>(i)];
                            h_mean[i] = (std::isfinite(mean_l) && std::isfinite(mean_r))
                                            ? mean_l + (mean_r - mean_l) * t
                                            : std::numeric_limits<float>::quiet_NaN();
                            h_p90[i] = (std::isfinite(p90_l) && std::isfinite(p90_r))
                                           ? p90_l + (p90_r - p90_l) * t
                                           : std::numeric_limits<float>::quiet_NaN();
                        }

                        const std::string out_file = output_dir + doy_to_mmdd(doy) + ".nc";
                        auto write_t0 = Clock::now();
                        write_output(out_file, doy, h_mean, h_p90);
                        std::cout << "[profile] anchor_interpolate_write="
                                  << std::chrono::duration<double>(Clock::now() - write_t0).count()
                                  << "s doy=" << doy << std::endl;
                    }

                    wait_for_output_writes();
                    cleanup_dcu_persistent_buffers();
                    cleanup_dcu_buffers();
                    HIP_CHECK(hipHostFree(h_mean));
                    HIP_CHECK(hipHostFree(h_p90));
                    free_anchor_buffers();
                    std::free(h_sst_data);
                    if (h_staging) hipHostFree(h_staging);

                    std::cout << "\n>>> 全部 " << (target_doy_end - target_doy_begin + 1)
                              << " 天计算完成！" << std::endl;
                    std::cout << "[profile] program_total="
                              << std::chrono::duration<double>(Clock::now() - program_t0).count()
                              << "s" << std::endl;
                    return 0;
                }

                const int source_begin = target_doy_begin - CLIM_DELTA_DAY;
                const int window_days = 2 * CLIM_DELTA_DAY + 1;
                const int target_days = target_doy_end - target_doy_begin + 1;
                const int chunk_days = std::min(source_stream_chunk_days_main(), source_days);
                const int chunk_slots = chunk_days * CLIM_YEARS;

                std::vector<std::vector<float*>> stream_buffers(
                    2, std::vector<float*>(gpu_count, nullptr));

                auto free_stream_buffers = [&]() {
                    for (auto& buf : stream_buffers) {
                        for (float*& p : buf) {
                            if (p) std::free(p);
                            p = nullptr;
                        }
                    }
                };

                for (int b = 0; b < 2; ++b) {
                    for (int i = 0; i < gpu_count; ++i) {
                        if (local_rows[i] <= 0) continue;
                        size_t local_spatial = (size_t)local_rows[i] * LON_SIZE;
                        size_t bytes = (size_t)chunk_slots * local_spatial * sizeof(float);
                        stream_buffers[b][i] = (float*)std::malloc(bytes);
                        if (!stream_buffers[b][i]) {
                            std::cerr << "ERROR: cannot allocate stream source buffer part="
                                      << i << " bytes=" << bytes << std::endl;
                            free_stream_buffers();
                            cleanup_dcu_persistent_buffers();
                            cleanup_dcu_buffers();
                            HIP_CHECK(hipHostFree(h_mean));
                            HIP_CHECK(hipHostFree(h_p90));
                            std::free(h_sst_data);
                            if (h_staging) hipHostFree(h_staging);
                            return 1;
                        }
                    }
                }

                const bool use_async_upload = source_async_upload_enabled_main();
                std::cout << "[experiment] MCC_SOURCE_STREAM=1 chunk_days=" << chunk_days
                          << " double_buffer=ON"
                          << " async_upload=" << (use_async_upload ? "ON" : "OFF")
                          << std::endl;

                init_source_window_device_preloaded(source_days);

                auto read_chunk = [&](int buf_idx, int source_day0, int days_this) {
                    preload_source_days_local(stream_buffers[buf_idx].data(),
                                              lat_starts.data(), local_rows.data(),
                                              gpu_count, source_begin + source_day0,
                                              days_this);
                };

                auto first_read_t0 = Clock::now();
                int cur_buf = 0;
                int next_buf = 1;
                int cur_day0 = 0;
                int cur_days = std::min(chunk_days, source_days);
                read_chunk(cur_buf, cur_day0, cur_days);
                std::cout << "[profile] source_stream_first_read="
                          << std::chrono::duration<double>(Clock::now() - first_read_t0).count()
                          << "s" << std::endl;

                int next_target_offset = 0;
                if (!use_async_upload) {
                    while (cur_day0 < source_days) {
                        int next_day0 = cur_day0 + cur_days;
                        int next_days = std::min(chunk_days, source_days - next_day0);
                        bool has_next = next_day0 < source_days;
                        std::future<void> read_future;
                        auto async_read_t0 = Clock::now();
                        if (has_next) {
                            read_future = std::async(std::launch::async, read_chunk,
                                                     next_buf, next_day0, next_days);
                        }

                        upload_source_days_preloaded_local(stream_buffers[cur_buf].data(),
                                                           cur_day0, cur_days);
                        int uploaded_until = cur_day0 + cur_days;

                        while (next_target_offset < target_days &&
                               next_target_offset + window_days <= uploaded_until) {
                            auto day_t0 = Clock::now();
                            int doy = target_doy_begin + next_target_offset;
                            std::cout << "\n>>> 处理 DOY " << doy << " (" << doy_to_mmdd(doy)
                                      << ") [source stream]" << std::endl;

                            dispatch_source_window_device_preloaded(source_days, next_target_offset,
                                                                    h_mean, h_p90);

                            std::string mmdd = doy_to_mmdd(doy);
                            std::string out_file = output_dir + mmdd + ".nc";
                            auto write_t0 = Clock::now();
                            write_output(out_file, doy, h_mean, h_p90);
                            double write_s = std::chrono::duration<double>(Clock::now() - write_t0).count();
                            std::cout << "[profile] write_output=" << write_s << "s" << std::endl;
                            std::cout << "  输出: " << out_file << std::endl;

                            std::cout << "[profile] day_total="
                                      << std::chrono::duration<double>(Clock::now() - day_t0).count()
                                      << "s write=" << write_s << "s" << std::endl;
                            ++next_target_offset;
                        }

                        if (has_next) {
                            read_future.get();
                            std::cout << "[profile] source_stream_async_read_waited total_since_launch="
                                      << std::chrono::duration<double>(Clock::now() - async_read_t0).count()
                                      << "s" << std::endl;
                        } else {
                            break;
                        }

                        cur_buf = next_buf;
                        next_buf = 1 - cur_buf;
                        cur_day0 = next_day0;
                        cur_days = next_days;
                    }
                } else {
                    int uploaded_until = cur_day0 + cur_days;
                    bool current_chunk_upload_launched = false;
                    while (cur_day0 < source_days) {
                        int next_day0 = cur_day0 + cur_days;
                        int next_days = std::min(chunk_days, source_days - next_day0);
                        bool has_next = next_day0 < source_days;
                        std::future<void> read_upload_future;
                        auto async_read_upload_t0 = Clock::now();
                        if (has_next) {
                            int upload_buf = next_buf;
                            int upload_day0 = next_day0;
                            int upload_days = next_days;
                            read_upload_future = std::async(
                                std::launch::async,
                                [&, upload_buf, upload_day0, upload_days]() {
                                    auto read_t0 = Clock::now();
                                    read_chunk(upload_buf, upload_day0, upload_days);
                                    double read_s = std::chrono::duration<double>(Clock::now() - read_t0).count();
                                    upload_source_days_preloaded_local_async(
                                        stream_buffers[upload_buf].data(),
                                        upload_day0,
                                        upload_days);
                                    std::cout << "[profile] source_stream_async_read_upload_launched"
                                              << " source_day0=" << upload_day0
                                              << " days=" << upload_days
                                              << " read=" << read_s << "s" << std::endl;
                                });
                        }

                        if (!current_chunk_upload_launched) {
                            upload_source_days_preloaded_local_async(stream_buffers[cur_buf].data(),
                                                                     cur_day0, cur_days);
                            current_chunk_upload_launched = true;
                        }

                        while (next_target_offset < target_days &&
                               next_target_offset + window_days <= uploaded_until) {
                            auto day_t0 = Clock::now();
                            int doy = target_doy_begin + next_target_offset;
                            std::cout << "\n>>> 处理 DOY " << doy << " (" << doy_to_mmdd(doy)
                                      << ") [source stream]" << std::endl;

                            dispatch_source_window_device_preloaded(source_days, next_target_offset,
                                                                    h_mean, h_p90);

                            std::string mmdd = doy_to_mmdd(doy);
                            std::string out_file = output_dir + mmdd + ".nc";
                            auto write_t0 = Clock::now();
                            write_output(out_file, doy, h_mean, h_p90);
                            double write_s = std::chrono::duration<double>(Clock::now() - write_t0).count();
                            std::cout << "[profile] write_output=" << write_s << "s" << std::endl;
                            std::cout << "  输出: " << out_file << std::endl;

                            std::cout << "[profile] day_total="
                                      << std::chrono::duration<double>(Clock::now() - day_t0).count()
                                      << "s write=" << write_s << "s" << std::endl;
                            ++next_target_offset;
                        }

                        if (has_next) {
                            read_upload_future.get();
                            std::cout << "[profile] source_stream_async_read_upload_waited total_since_launch="
                                      << std::chrono::duration<double>(Clock::now() - async_read_upload_t0).count()
                                      << "s" << std::endl;
                            uploaded_until = next_day0 + next_days;
                            current_chunk_upload_launched = true;
                        } else {
                            break;
                        }

                        cur_buf = next_buf;
                        next_buf = 1 - cur_buf;
                        cur_day0 = next_day0;
                        cur_days = next_days;
                    }
                }

                wait_source_days_preloaded_upload();
                wait_for_output_writes();
                cleanup_dcu_persistent_buffers();
                cleanup_dcu_buffers();
                HIP_CHECK(hipHostFree(h_mean));
                HIP_CHECK(hipHostFree(h_p90));
                free_stream_buffers();
                std::free(h_sst_data);
                if (h_staging) hipHostFree(h_staging);

                std::cout << "\n>>> 全部 " << (target_doy_end - target_doy_begin + 1)
                          << " 天计算完成！" << std::endl;
                std::cout << "[profile] program_total="
                          << std::chrono::duration<double>(Clock::now() - program_t0).count()
                          << "s" << std::endl;
                return 0;
            }

            auto alloc_t0 = Clock::now();
            for (int i = 0; i < gpu_count; ++i) {
                if (local_rows[i] <= 0) continue;
                size_t local_spatial = (size_t)local_rows[i] * LON_SIZE;
                size_t bytes = (size_t)source_slots * local_spatial * sizeof(float);
                if (local_pinned) {
                    HIP_CHECK(hipSetDevice(i));
                    hipError_t err = hipHostMalloc(&h_source_local[i], bytes, hipHostMallocDefault);
                    if (err != hipSuccess) {
                        std::cerr << "[experiment] local pinned source allocation failed on part "
                                  << i << ": " << hipGetErrorString(err)
                                  << "; falling back to malloc" << std::endl;
                        local_pinned = false;
                        for (float*& p : h_source_local) {
                            if (p) {
                                hipHostFree(p);
                                p = nullptr;
                            }
                        }
                        break;
                    }
                } else {
                    h_source_local[i] = nullptr;
                }
            }

            if (!local_pinned) {
                for (int i = 0; i < gpu_count; ++i) {
                    if (local_rows[i] <= 0) continue;
                    size_t local_spatial = (size_t)local_rows[i] * LON_SIZE;
                    size_t bytes = (size_t)source_slots * local_spatial * sizeof(float);
                    h_source_local[i] = (float*)std::malloc(bytes);
                    if (!h_source_local[i]) {
                        std::cerr << "ERROR: cannot allocate local source buffer part="
                                  << i << " bytes=" << bytes << std::endl;
                        for (float*& p : h_source_local) {
                            if (p) std::free(p);
                            p = nullptr;
                        }
                        cleanup_dcu_persistent_buffers();
                        cleanup_dcu_buffers();
                        HIP_CHECK(hipHostFree(h_mean));
                        HIP_CHECK(hipHostFree(h_p90));
                        std::free(h_sst_data);
                        if (h_staging) hipHostFree(h_staging);
                        return 1;
                    }
                }
            }

            std::cout << "[experiment] MCC_SOURCE_LOCAL_LAYOUT=1 parts=" << gpu_count
                      << " local_pinned=" << (local_pinned ? "ON" : "OFF")
                      << " alloc="
                      << std::chrono::duration<double>(Clock::now() - alloc_t0).count()
                      << "s" << std::endl;

            auto preload_t0 = Clock::now();
            preload_season_source_local(h_source_local.data(), lat_starts.data(), local_rows.data(),
                                        gpu_count, target_doy_begin, target_doy_end);
            double preload_s = std::chrono::duration<double>(Clock::now() - preload_t0).count();
            std::cout << "[profile] season_preload_local_read=" << preload_s << "s" << std::endl;

            for (int doy = target_doy_begin; doy <= target_doy_end; ++doy) {
                auto day_t0 = Clock::now();
                int target_offset = doy - target_doy_begin;
                std::cout << "\n>>> 处理 DOY " << doy << " (" << doy_to_mmdd(doy)
                          << ") [season source-window local]" << std::endl;

                dispatch_source_window_preloaded_local(h_source_local.data(), source_days,
                                                       target_offset, h_mean, h_p90);

                std::string mmdd = doy_to_mmdd(doy);
                std::string out_file = output_dir + mmdd + ".nc";
                auto write_t0 = Clock::now();
                write_output(out_file, doy, h_mean, h_p90);
                double write_s = std::chrono::duration<double>(Clock::now() - write_t0).count();
                std::cout << "[profile] write_output=" << write_s << "s" << std::endl;
                std::cout << "  输出: " << out_file << std::endl;

                std::cout << "[profile] day_total="
                          << std::chrono::duration<double>(Clock::now() - day_t0).count()
                          << "s write=" << write_s << "s" << std::endl;
            }

            wait_for_output_writes();
            cleanup_dcu_persistent_buffers();
            cleanup_dcu_buffers();
            HIP_CHECK(hipHostFree(h_mean));
            HIP_CHECK(hipHostFree(h_p90));
            for (float*& p : h_source_local) {
                if (!p) continue;
                if (local_pinned) {
                    hipHostFree(p);
                } else {
                    std::free(p);
                }
                p = nullptr;
            }
            std::free(h_sst_data);
            if (h_staging) hipHostFree(h_staging);

            std::cout << "\n>>> 全部 " << (target_doy_end - target_doy_begin + 1)
                      << " 天计算完成！" << std::endl;
            std::cout << "[profile] program_total="
                      << std::chrono::duration<double>(Clock::now() - program_t0).count()
                      << "s" << std::endl;
            return 0;
        }

        auto alloc_t0 = Clock::now();
        float* h_source_data = (float*)std::malloc(source_bytes);
        if (!h_source_data) {
            std::cerr << "ERROR: cannot allocate season source buffer bytes="
                      << source_bytes << std::endl;
            cleanup_dcu_persistent_buffers();
            cleanup_dcu_buffers();
            HIP_CHECK(hipHostFree(h_mean));
            HIP_CHECK(hipHostFree(h_p90));
            std::free(h_sst_data);
            if (h_staging) hipHostFree(h_staging);
            return 1;
        }
        std::cout << "[profile] source_alloc="
                  << std::chrono::duration<double>(Clock::now() - alloc_t0).count()
                  << "s" << std::endl;

        auto preload_t0 = Clock::now();
        preload_season_source(h_source_data, target_doy_begin, target_doy_end);
        double preload_s = std::chrono::duration<double>(Clock::now() - preload_t0).count();
        std::cout << "[profile] season_preload_read=" << preload_s << "s" << std::endl;

        for (int doy = target_doy_begin; doy <= target_doy_end; ++doy) {
            auto day_t0 = Clock::now();
            int target_offset = doy - target_doy_begin;
            std::cout << "\n>>> 处理 DOY " << doy << " (" << doy_to_mmdd(doy)
                      << ") [season source-window]" << std::endl;

            dispatch_source_window_preloaded(h_source_data, source_days, target_offset, h_mean, h_p90);

            std::string mmdd = doy_to_mmdd(doy);
            std::string out_file = output_dir + mmdd + ".nc";
            auto write_t0 = Clock::now();
            write_output(out_file, doy, h_mean, h_p90);
            double write_s = std::chrono::duration<double>(Clock::now() - write_t0).count();
            std::cout << "[profile] write_output=" << write_s << "s" << std::endl;
            std::cout << "  输出: " << out_file << std::endl;

            std::cout << "[profile] day_total="
                      << std::chrono::duration<double>(Clock::now() - day_t0).count()
                      << "s write=" << write_s << "s" << std::endl;
        }

        wait_for_output_writes();
        cleanup_dcu_persistent_buffers();
        cleanup_dcu_buffers();
        HIP_CHECK(hipHostFree(h_mean));
        HIP_CHECK(hipHostFree(h_p90));
        std::free(h_source_data);
        std::free(h_sst_data);
        if (h_staging) hipHostFree(h_staging);

        std::cout << "\n>>> 全部 " << (target_doy_end - target_doy_begin + 1)
                  << " 天计算完成！" << std::endl;
        std::cout << "[profile] program_total="
                  << std::chrono::duration<double>(Clock::now() - program_t0).count()
                  << "s" << std::endl;
        return 0;
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

    // ============================================================
    // 4. 双缓冲异步预取：IO 与 DCU 计算重叠
    //    核心思路: H2D 上传完成后立刻启动后台线程预读下一天的 30 个文件，
    //    让 IO 与 DCU kernel 计算 + D2H 下载 + NetCDF 写出完全并行。
    // ============================================================
    for (int doy = target_doy_begin; doy <= target_doy_end; ++doy) {
        auto day_t0 = Clock::now();
        std::cout << "\n>>> 处理 DOY " << doy << " (" << doy_to_mmdd(doy) << ")" << std::endl;

        // 等待前一天的异步预取完成（首日跳过）
        double prefetch_wait_s = 0.0;
        if (doy > target_doy_begin) {
            auto wait_t0 = Clock::now();
            wait_for_prefetch_next_doy();
            prefetch_wait_s = std::chrono::duration<double>(Clock::now() - wait_t0).count();
            std::cout << "[profile] prefetch_wait=" << prefetch_wait_s << "s (等待后台IO完成)" << std::endl;
        }

        // 准备预取请求：after_upload hook 用它启动下一轮异步 IO
        PrefetchAfterUpload prefetch_req;
        prefetch_req.h_sst_data = h_sst_data;
        prefetch_req.doy = doy;
        prefetch_req.first_doy = target_doy_begin;
        prefetch_req.last_doy = target_doy_end;

        // DCU 计算：H2D 上传完成后 hook 触发异步预取
        //   start_prefetch_after_upload 在后台 std::async 中读取 30 个文件
        //   与 DCU kernel + D2H 下载 + NetCDF 写出完全并行
        std::cerr << "[DBG] day=" << doy << " BEFORE_dispatch prog=" << std::chrono::duration<double>(Clock::now() - program_t0).count() << "s" << std::endl;
        float* h_src = nullptr;
        dispatch_to_4_dcus_with_output_incremental_hook(
            h_sst_data, h_src, h_mean, h_p90,
            (doy == target_doy_begin) ? nullptr : changed_slots.data(),
            (doy == target_doy_begin) ? 0 : static_cast<int>(changed_slots.size()),
            (doy < target_doy_end) ? start_prefetch_after_upload : nullptr,
            (doy < target_doy_end) ? &prefetch_req : nullptr);

        auto _dbg_t = Clock::now();
        std::cerr << "[DBG] day=" << doy << " AFTER_dispatch prog=" << std::chrono::duration<double>(Clock::now() - program_t0).count() << "s" << std::endl;
        if (debug_enabled()) {
            print_cpu_gpu_samples(h_sst_data, h_mean, h_p90);
        }

        // 写出结果（异步预取在后台并行运行中）
        std::string mmdd = doy_to_mmdd(doy);
        std::string out_file = output_dir + mmdd + ".nc";
        phase_t0 = Clock::now();
        write_output(out_file, doy, h_mean, h_p90);
        double write_s = std::chrono::duration<double>(Clock::now() - phase_t0).count();
        std::cout << "[profile] write_output=" << write_s << "s" << std::endl;
        std::cout << "  输出: " << out_file << std::endl;

        // 更新 changed_slots（为下一轮增量 H2D 上传标记变更槽位）
        if (doy < target_doy_end) {
            int replace_offset_idx = (doy - target_doy_begin) % window;
            for (int yr_idx = 0; yr_idx < CLIM_YEARS; ++yr_idx) {
                changed_slots[yr_idx] = replace_offset_idx * CLIM_YEARS + yr_idx;
            }
        }

        std::cout << "[profile] day_total="
                  << std::chrono::duration<double>(Clock::now() - day_t0).count()
                  << "s write=" << write_s
                  << "s prefetch_wait=" << prefetch_wait_s
                  << "s (async IO overlapped with compute)" << std::endl;
    }

    wait_for_output_writes();
    // 5. 释放资源
    cleanup_dcu_persistent_buffers();
    cleanup_dcu_buffers();
    HIP_CHECK(hipHostFree(h_mean));
    HIP_CHECK(hipHostFree(h_p90));
    std::free(h_sst_data);
    if (h_staging) hipHostFree(h_staging);

    std::cout << "\n>>> 全部 " << (target_doy_end - target_doy_begin + 1)
              << " 天计算完成！" << std::endl;
    std::cout << "[profile] program_total="
              << std::chrono::duration<double>(Clock::now() - program_t0).count()
              << "s" << std::endl;
    return 0;
}
