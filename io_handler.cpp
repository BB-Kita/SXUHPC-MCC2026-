#include "io_handler.h"
#include "config.h"

#include <netcdf.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NC_INPUT_DIR
#define NC_INPUT_DIR "/public/home/achwjznh4b/Newdata/"
#endif

#ifndef CLIM_START_YEAR
#define CLIM_START_YEAR 1991
#endif

#ifndef CLIM_END_YEAR
#define CLIM_END_YEAR 2020
#endif

#ifndef CLIM_DELTA_DAY
#define CLIM_DELTA_DAY 5
#endif

// 当前需要读取哪个 day-of-year 的 30年×11天窗口。
// 如果你的主程序按 doy 循环，建议调用 read_netcdf_parallel_for_doy(h_sst_data, doy)。
#ifndef TARGET_DOY
#define TARGET_DOY 152
#endif

// NetCDF 并发读不要盲目打满 32 核，建议先用 4 或 8。
// 如果确认文件系统和 NetCDF/HDF5 环境稳定，再加大。
#ifndef IO_THREADS
#define IO_THREADS 4
#endif

// 如果 ncdump -h 中有 scale_factor/add_offset 且参考结果需要物理值，保持 true。
// 如果输入 data 已经是物理值，可以改成 false。
#ifndef APPLY_SCALE_OFFSET
#define APPLY_SCALE_OFFSET 1
#endif

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

static bool is_leap(int y) {
    return (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
}

static int days_in_month(int y, int m) {
    static const int mdays[] = {
        31, 28, 31, 30, 31, 30,
        31, 31, 30, 31, 30, 31
    };
    return (m == 2 && is_leap(y)) ? 29 : mdays[m - 1];
}

static Date add_days(Date date, int offset) {
    while (offset > 0) {
        int dim = days_in_month(date.y, date.m);
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

// 365 天日历，闰年剔除 2 月 29 日
static Date no_feb29_doy_to_date(int year, int doy) {
    if (doy < 1 || doy > 365) {
        throw std::runtime_error("doy must be in [1,365]");
    }

    Date date;
    date.y = year;
    date.m = 1;
    date.d = 1;

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

static std::string yyyymmdd(Date date) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%04d%02d%02d", date.y, date.m, date.d);
    return std::string(buf);
}

static std::string with_trailing_slash(std::string p) {
    if (!p.empty() && p[p.size() - 1] != '/') {
        p.push_back('/');
    }
    return p;
}

static bool get_att_double(int ncid, int varid, const char* name, double& value) {
    int status = nc_get_att_double(ncid, varid, name, &value);
    return status == NC_NOERR;
}

static void set_var_chunk_cache_if_possible(int ncid, int varid) {
    // 对 NetCDF-4/HDF5 chunked/compressed 文件可能有帮助。
    // 对非 chunked 文件通常没有明显副作用。
    const size_t cache_size = 64ULL * 1024ULL * 1024ULL;  // 64 MB
    const size_t cache_nelems = 1000003;
    const float preemption = 0.75f;

    // 如果失败，不中断程序。
    nc_set_var_chunk_cache(ncid, varid, cache_size, cache_nelems, preemption);
}

static SSTReadInfo inspect_sst_file(const std::string& filepath) {
    SSTReadInfo info;
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

static void normalize_buffer(float* data, size_t n, const SSTReadInfo& info) {
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        float v = data[i];

        bool is_missing = false;
        if (info.has_fill && static_cast<double>(v) == info.fill_value) {
            is_missing = true;
        }
        if (info.has_missing && static_cast<double>(v) == info.missing_value) {
            is_missing = true;
        }

        if (is_missing) {
            data[i] = std::numeric_limits<float>::quiet_NaN();
            continue;
        }

#if APPLY_SCALE_OFFSET
        if (info.has_scale || info.has_offset) {
            data[i] = static_cast<float>(
                static_cast<double>(v) * info.scale_factor + info.add_offset
            );
        }
#endif
    }
}

static std::vector<std::string> build_sample_dates_for_doy(int doy) {
    std::vector<std::string> dates;
    dates.reserve((CLIM_END_YEAR - CLIM_START_YEAR + 1) * (2 * CLIM_DELTA_DAY + 1));

    for (int yr = CLIM_START_YEAR; yr <= CLIM_END_YEAR; ++yr) {
        Date center = no_feb29_doy_to_date(yr, doy);

        for (int off = -CLIM_DELTA_DAY; off <= CLIM_DELTA_DAY; ++off) {
            Date target = add_days(center, off);
            dates.push_back(yyyymmdd(target));
        }
    }

    return dates;
}

static void fill_slot_nan(float* h_sst_data, int slot) {
    size_t offset = static_cast<size_t>(slot) * SPATIAL_POINTS;

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(SPATIAL_POINTS); ++i) {
        h_sst_data[offset + static_cast<size_t>(i)] =
            std::numeric_limits<float>::quiet_NaN();
    }
}

static bool read_one_file_into_slot(const std::string& filepath,
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

    size_t slot_offset = static_cast<size_t>(slot) * SPATIAL_POINTS;

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

    // data(lon, lat) 时需要读入临时缓冲并转置成 [lat][lon]
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
        size_t x = static_cast<size_t>(x_ll);
        for (size_t y = 0; y < LAT_SIZE; ++y) {
            h_sst_data[slot_offset + y * LON_SIZE + x] =
                tmp[x * LAT_SIZE + y];
        }
    }

    return true;
}

void read_netcdf_parallel_for_doy(float* h_sst_data, int target_doy) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);

    std::string input_dir = with_trailing_slash(NC_INPUT_DIR);

    std::vector<std::string> dates = build_sample_dates_for_doy(target_doy);

    if (static_cast<int>(dates.size()) != DAYS_TOTAL) {
        std::cerr << "[I/O 模块] DAYS_TOTAL 与实际样本数不一致：DAYS_TOTAL="
                  << DAYS_TOTAL << ", dates.size()=" << dates.size() << '\n';
        std::cerr << "[I/O 模块] 请确认 config.h 中 DAYS_TOTAL 是否等于 30*11=330。\n";
    }

    std::string inspect_file = input_dir + "19910101";
    SSTReadInfo info = inspect_sst_file(inspect_file);

    int io_threads = IO_THREADS;
#ifdef _OPENMP
    int max_threads = omp_get_max_threads();
    if (io_threads <= 0) {
        io_threads = 1;
    }
    io_threads = std::min(io_threads, max_threads);
#endif

    std::cout << "[I/O 模块] 开始读取 doy=" << target_doy
              << "，样本数=" << dates.size()
              << "，I/O threads=" << io_threads << '\n';

    int failed_count = 0;

    // 注意：NetCDF/HDF5 多线程并发读在部分环境下不一定稳定。
    // 如果运行时出现随机崩溃，把 IO_THREADS 改成 1 或 2。
    #pragma omp parallel for schedule(dynamic, 1) num_threads(io_threads) reduction(+:failed_count)
    for (int d = 0; d < static_cast<int>(dates.size()); ++d) {
        std::string filepath = input_dir + dates[d];

        bool ok = read_one_file_into_slot(filepath, d, info, h_sst_data);
        if (!ok) {
            ++failed_count;
        }
    }

    std::cout << "[I/O 模块] 数据读取完成。failed=" << failed_count
              << "/" << dates.size() << '\n';
}

// 保留原来的接口：默认读取 TARGET_DOY 对应的 30年×11天窗口
void read_netcdf_parallel(float* h_sst_data) {
    read_netcdf_parallel_for_doy(h_sst_data, TARGET_DOY);
}