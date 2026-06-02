#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

    // Write lat/lon arrays (need to generate them)
    std::vector<double> lat(LAT_SIZE), lon(LON_SIZE);
    for (size_t i = 0; i < LAT_SIZE; ++i) lat[i] = -90.0 + i * 0.25;
    for (size_t i = 0; i < LON_SIZE; ++i) lon[i] = i * 0.25;

    double doy_val = static_cast<double>(doy);
    nc_put_var_double(ncid, day_varid, &doy_val);
    nc_put_var_double(ncid, lat_varid, lat.data());
    nc_put_var_double(ncid, lon_varid, lon.data());
    // Convert float32 -> float64 to match reference output format
    std::vector<double> mean_d(LAT_SIZE * LON_SIZE), p90_d(LAT_SIZE * LON_SIZE);
    for (size_t i = 0; i < LAT_SIZE * LON_SIZE; ++i) {
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

int main() {
    std::cout << ">>> 启动 MCC 海洋热浪阈值计算引擎 (C++/HIP)" << std::endl;

    const std::string output_dir = "/public/home/fujiake/fjk/MCC26_SXU/output/";

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

    // 3. 初始化滑动窗口（读取第一个 DOY 的全部 330 个文件）
    initialize_window(h_sst_data, TARGET_DOY_BEGIN);

    // 4. 循环处理 92 天
    const int window = 2 * CLIM_DELTA_DAY + 1;
    std::vector<int> changed_slots(CLIM_YEARS);

    for (int doy = TARGET_DOY_BEGIN; doy <= TARGET_DOY_END; ++doy) {
        std::cout << "\n>>> 处理 DOY " << doy << " (" << doy_to_mmdd(doy) << ")" << std::endl;

        // DCU 计算（全量传输，待增量传输调试完成后再启用）
        dispatch_to_4_dcus_with_output(h_sst_data, h_mean, h_p90);

        // 写出结果
        std::string mmdd = doy_to_mmdd(doy);
        std::string out_file = output_dir + mmdd + ".nc";
        write_output(out_file, doy, h_mean, h_p90);
        std::cout << "  输出: " << out_file << std::endl;

        // 滑动窗口：读取 30 个新文件覆盖最旧槽位（最后一天不需要）
        if (doy < TARGET_DOY_END) {
            int replace_offset_idx = (doy - TARGET_DOY_BEGIN) % window;
            for (int yr_idx = 0; yr_idx < CLIM_YEARS; ++yr_idx) {
                changed_slots[yr_idx] = yr_idx * window + replace_offset_idx;
            }
            slide_window_to_next_doy(h_sst_data, doy, TARGET_DOY_BEGIN);
        }
    }

    // 5. 释放资源
    cleanup_dcu_buffers();
    HIP_CHECK(hipHostFree(h_mean));
    HIP_CHECK(hipHostFree(h_p90));
    std::free(h_sst_data);

    std::cout << "\n>>> 全部 " << TARGET_DAYS_TOTAL << " 天计算完成！" << std::endl;
    return 0;
}