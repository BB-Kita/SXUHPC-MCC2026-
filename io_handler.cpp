#include "io_handler.h"
#include "config.h"

#include <netcdf.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Date {
    int y;
    int m;
    int d;
};

enum DataLayout {
    DATA_LAT_LON = 0,
    DATA_LON_LAT = 1
};

struct SSTReadInfo {
    DataLayout layout;

    bool has_fill;
    double fill_value;

    bool has_missing;
    double missing_value;

    bool has_scale;
    double scale_factor;

    bool has_offset;
    double add_offset;
};

bool is_leap(int y) {
    return (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
}

int days_in_month(int y, int m) {
    static const int mdays[] = {
        31, 28, 31, 30, 31, 30,
        31, 31, 30, 31, 30, 31
    };
    return (m == 2 && is_leap(y)) ? 29 : mdays[m - 1];
}

Date add_days(Date date, int offset) {
    while (offset > 0) {
        const int dim = days_in_month(date.y, date.m);
        if (date.d < dim) {
            ++date.d;
        } else {
            date.d = 1;
            if (date.m == 12) {
                date.m = 1;
                ++date.y;
            } else {
                ++date.m;
            }
        }
        --offset;
    }

    while (offset < 0) {
        if (date.d > 1) {
            --date.d;
        } else {
            if (date.m == 1) {
                date.m = 12;
                --date.y;
            } else {
                --date.m;
            }
            date.d = days_in_month(date.y, date.m);
        }
        ++offset;
    }

    return date;
}

// 365 天日历：闰年剔除 2 月 29 日，与原 MATLAB 脚本保持一致。
Date no_feb29_doy_to_date(int year, int doy) {
    if (doy < 1 || doy > 365) {
        throw std::runtime_error("doy must be in [1,365]");
    }

    Date date{year, 1, 1};
    int remaining = doy - 1;
    while (remaining > 0) {
        date = add_days(date, 1);
        if (is_leap(date.y) && date.m == 2 && date.d == 29) {
            date = add_days(date, 1);
        }
        --remaining;
    }
    return date;
}

std::string yyyymmdd(Date date) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%04d%02d%02d", date.y, date.m, date.d);
    return std::string(buf);
}

std::string with_trailing_slash(std::string p) {
    if (!p.empty() && p[p.size() - 1] != '/') {
        p.push_back('/');
    }
    return p;
}

bool get_att_double(int ncid, int varid, const char* name, double& value) {
    const int status = nc_get_att_double(ncid, varid, name, &value);
    return status == NC_NOERR;
}

void set_var_chunk_cache_if_possible(int ncid, int varid) {
    // 对 NetCDF-4/HDF5 chunked/compressed 文件可能有帮助；失败不影响正确性。
    const size_t cache_size = 64ULL * 1024ULL * 1024ULL;
    const size_t cache_nelems = 1000003;
    const float preemption = 0.75f;
    nc_set_var_chunk_cache(ncid, varid, cache_size, cache_nelems, preemption);
}

SSTReadInfo inspect_sst_file(const std::string& filepath) {
    SSTReadInfo info{};
    info.layout = DATA_LAT_LON;
    info.has_fill = false;
    info.fill_value = std::numeric_limits<double>::quiet_NaN();
    info.has_missing = false;
    info.missing_value = std::numeric_limits<double>::quiet_NaN();
    info.has_scale = false;
    info.scale_factor = 1.0;
    info.has_offset = false;
    info.add_offset = 0.0;

    int ncid = -1;
    int varid = -1;

    int status = nc_open(filepath.c_str(), NC_NOWRITE, &ncid);
    if (status != NC_NOERR) {
        throw std::runtime_error("Cannot open sample NetCDF file: " + filepath);
    }

    status = nc_inq_varid(ncid, "data", &varid);
    if (status != NC_NOERR) {
        nc_close(ncid);
        throw std::runtime_error("Cannot find variable 'data' in: " + filepath);
    }

    int ndims = 0;
    int dimids[NC_MAX_VAR_DIMS] = {0};
    status = nc_inq_var(ncid, varid, nullptr, nullptr, &ndims, dimids, nullptr);
    if (status != NC_NOERR || ndims != 2) {
        nc_close(ncid);
        throw std::runtime_error("Variable 'data' must be 2-D in: " + filepath);
    }

    size_t dim0 = 0;
    size_t dim1 = 0;
    nc_inq_dimlen(ncid, dimids[0], &dim0);
    nc_inq_dimlen(ncid, dimids[1], &dim1);

    if (dim0 == LAT_SIZE && dim1 == LON_SIZE) {
        info.layout = DATA_LAT_LON;
    } else if (dim0 == LON_SIZE && dim1 == LAT_SIZE) {
        info.layout = DATA_LON_LAT;
    } else {
        nc_close(ncid);
        std::ostringstream oss;
        oss << "data dimension mismatch: data=(" << dim0 << "," << dim1
            << "), expected (" << LAT_SIZE << "," << LON_SIZE
            << ") or (" << LON_SIZE << "," << LAT_SIZE << ")";
        throw std::runtime_error(oss.str());
    }

    info.has_fill = get_att_double(ncid, varid, "_FillValue", info.fill_value);
    info.has_missing = get_att_double(ncid, varid, "missing_value", info.missing_value);
    info.has_scale = get_att_double(ncid, varid, "scale_factor", info.scale_factor);
    info.has_offset = get_att_double(ncid, varid, "add_offset", info.add_offset);

    nc_close(ncid);
    return info;
}

void normalize_buffer(float* data, size_t n, const SSTReadInfo& info) {
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        const size_t idx = static_cast<size_t>(i);
        const float v = data[idx];

        bool missing = false;
        if (info.has_fill && static_cast<double>(v) == info.fill_value) {
            missing = true;
        }
        if (info.has_missing && static_cast<double>(v) == info.missing_value) {
            missing = true;
        }

        if (missing) {
            data[idx] = std::numeric_limits<float>::quiet_NaN();
            continue;
        }

#if APPLY_SCALE_OFFSET
        if (info.has_scale || info.has_offset) {
            data[idx] = static_cast<float>(
                static_cast<double>(v) * info.scale_factor + info.add_offset
            );
        }
#endif
    }
}

std::vector<std::string> build_sample_dates_for_doy(int doy) {
    std::vector<std::string> dates;
    dates.reserve(CLIM_SAMPLE_COUNT);

    for (int yr = CLIM_START_YEAR; yr <= CLIM_END_YEAR; ++yr) {
        const Date center = no_feb29_doy_to_date(yr, doy);
        for (int off = -CLIM_DELTA_DAY; off <= CLIM_DELTA_DAY; ++off) {
            dates.push_back(yyyymmdd(add_days(center, off)));
        }
    }

    return dates;
}

void fill_slot_nan(float* h_sst_data, int slot) {
    const size_t offset = static_cast<size_t>(slot) * SPATIAL_POINTS;

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(SPATIAL_POINTS); ++i) {
        h_sst_data[offset + static_cast<size_t>(i)] = std::numeric_limits<float>::quiet_NaN();
    }
}

bool read_one_file_into_slot(const std::string& filepath,
                             int slot,
                             const SSTReadInfo& info,
                             float* h_sst_data) {
    int ncid = -1;
    int varid = -1;

    int status = nc_open(filepath.c_str(), NC_NOWRITE, &ncid);
    if (status != NC_NOERR) {
        fill_slot_nan(h_sst_data, slot);
        return false;
    }

    status = nc_inq_varid(ncid, "data", &varid);
    if (status != NC_NOERR) {
        nc_close(ncid);
        fill_slot_nan(h_sst_data, slot);
        return false;
    }

    set_var_chunk_cache_if_possible(ncid, varid);

    const size_t slot_offset = static_cast<size_t>(slot) * SPATIAL_POINTS;

    if (info.layout == DATA_LAT_LON) {
        size_t start[2] = {0, 0};
        size_t count[2] = {LAT_SIZE, LON_SIZE};

        status = nc_get_vara_float(ncid, varid, start, count, &h_sst_data[slot_offset]);
        nc_close(ncid);

        if (status != NC_NOERR) {
            fill_slot_nan(h_sst_data, slot);
            return false;
        }

        normalize_buffer(&h_sst_data[slot_offset], SPATIAL_POINTS, info);
        return true;
    }

    // data(lon, lat) 时先读临时缓冲，再转置为 [lat][lon]。
    std::vector<float> tmp(SPATIAL_POINTS);

    size_t start[2] = {0, 0};
    size_t count[2] = {LON_SIZE, LAT_SIZE};

    status = nc_get_vara_float(ncid, varid, start, count, tmp.data());
    nc_close(ncid);

    if (status != NC_NOERR) {
        fill_slot_nan(h_sst_data, slot);
        return false;
    }

    normalize_buffer(tmp.data(), SPATIAL_POINTS, info);

    #pragma omp parallel for schedule(static)
    for (long long x_ll = 0; x_ll < static_cast<long long>(LON_SIZE); ++x_ll) {
        const size_t x = static_cast<size_t>(x_ll);
        for (size_t y = 0; y < LAT_SIZE; ++y) {
            h_sst_data[slot_offset + y * LON_SIZE + x] = tmp[x * LAT_SIZE + y];
        }
    }

    return true;
}

} // namespace

void read_netcdf_parallel_for_doy(float* h_sst_data, int target_doy) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);

    if (target_doy < 1 || target_doy > 365) {
        throw std::runtime_error("target_doy must be in [1,365]");
    }

    const std::string input_dir = with_trailing_slash(NC_INPUT_DIR);
    const std::vector<std::string> dates = build_sample_dates_for_doy(target_doy);

    if (dates.size() != DAYS_TOTAL) {
        std::cerr << "[I/O 模块] DAYS_TOTAL 与样本窗口不一致：DAYS_TOTAL="
                  << DAYS_TOTAL << ", dates.size()=" << dates.size() << '\n';
    }

    const std::string inspect_file = input_dir + "19910101";
    const SSTReadInfo info = inspect_sst_file(inspect_file);

    int io_threads = IO_THREADS;
#ifdef _OPENMP
    const int max_threads = omp_get_max_threads();
    if (io_threads <= 0) {
        io_threads = 1;
    }
    io_threads = std::min(io_threads, max_threads);
#endif

    std::cout << "[I/O 模块] 读取 doy=" << target_doy
              << " 的 30年×11天样本窗口，samples=" << dates.size()
              << ", io_threads=" << io_threads << '\n';

    int failed_count = 0;

    // 注意：NetCDF/HDF5 多线程并发读在部分环境下不稳定。
    // 如遇随机崩溃或 HDF5 报错，将 config.h 中 IO_THREADS 改为 1 或 2。
    #pragma omp parallel for schedule(dynamic, 1) num_threads(io_threads) reduction(+:failed_count)
    for (int d = 0; d < static_cast<int>(dates.size()); ++d) {
        const std::string filepath = input_dir + dates[static_cast<size_t>(d)];
        const bool ok = read_one_file_into_slot(filepath, d, info, h_sst_data);
        if (!ok) {
            ++failed_count;
        }
    }

    std::cout << "[I/O 模块] 数据读取并拼接完毕。failed=" << failed_count
              << "/" << dates.size() << '\n';
}

void read_netcdf_parallel(float* h_sst_data) {
    read_netcdf_parallel_for_doy(h_sst_data, DEFAULT_TARGET_DOY);
}
