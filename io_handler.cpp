#include "io_handler.h"
#include "config.h"

#include <netcdf.h>
#include <hdf5.h>
#include <omp.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <future>
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

// NetCDF 并发
#ifndef IO_THREADS
#define IO_THREADS 32
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

    bool raw_data_read;
    std::size_t raw_data_offset;
};

static int runtime_io_threads() {
    int io_threads = IO_THREADS;
    if (const char* env = std::getenv("MCC_IO_THREADS")) {
        int requested = std::atoi(env);
        if (requested > 0) {
            io_threads = requested;
        }
    }
#ifdef _OPENMP
    int max_threads = omp_get_max_threads();
    if (io_threads <= 0) {
        io_threads = 1;
    }
    io_threads = std::min(io_threads, max_threads);
#else
    io_threads = 1;
#endif
    return io_threads;
}

static int runtime_prefetch_io_threads() {
    int io_threads = runtime_io_threads();
    if (const char* env = std::getenv("MCC_PREFETCH_IO_THREADS")) {
        int requested = std::atoi(env);
        if (requested > 0) {
            io_threads = requested;
        }
    }
#ifdef _OPENMP
    io_threads = std::max(1, std::min(io_threads, omp_get_max_threads()));
#else
    io_threads = 1;
#endif
    return io_threads;
}

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

static bool inspect_raw_hdf5_data(const std::string& filepath, std::size_t& data_offset) {
    hid_t file_id = H5Fopen(filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) return false;

    hid_t dset_id = H5Dopen2(file_id, "data", H5P_DEFAULT);
    if (dset_id < 0) {
        H5Fclose(file_id);
        return false;
    }

    bool ok = false;
    hid_t dcpl_id = H5Dget_create_plist(dset_id);
    hid_t space_id = H5Dget_space(dset_id);
    hid_t type_id = H5Dget_type(dset_id);

    hsize_t dims[2] = {0, 0};
    int ndims = H5Sget_simple_extent_ndims(space_id);
    if (ndims == 2) {
        H5Sget_simple_extent_dims(space_id, dims, nullptr);
    }

    H5D_layout_t layout = H5Pget_layout(dcpl_id);
    haddr_t offset = H5Dget_offset(dset_id);
    bool is_float64_le = H5Tequal(type_id, H5T_IEEE_F64LE) > 0;

    if (layout == H5D_CONTIGUOUS &&
        offset != HADDR_UNDEF &&
        ndims == 2 &&
        dims[0] == LAT_SIZE &&
        dims[1] == LON_SIZE &&
        is_float64_le) {
        data_offset = static_cast<std::size_t>(offset);
        ok = true;
    }

    H5Tclose(type_id);
    H5Sclose(space_id);
    H5Pclose(dcpl_id);
    H5Dclose(dset_id);
    H5Fclose(file_id);
    return ok;
}

static void set_var_chunk_cache_if_possible(int /*ncid*/, int /*varid*/) {
    // Disabled: nc_set_var_chunk_cache causes HDF5 threading issues
    // with OpenMP parallel reads on this platform.
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
    info.raw_data_read = false;
    info.raw_data_offset = 0;

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

    std::size_t raw_offset = 0;
    info.raw_data_read = (info.layout == DATA_LAT_LON &&
                          !info.has_fill &&
                          !info.has_missing &&
                          !info.has_scale &&
                          !info.has_offset &&
                          inspect_raw_hdf5_data(filepath, raw_offset));
    info.raw_data_offset = raw_offset;

    std::cout << "[I/O module] raw HDF5 data path: "
              << (info.raw_data_read ? "enabled" : "disabled")
              << ", offset=" << info.raw_data_offset << std::endl;
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

static bool read_full_at(int fd, void* buffer, std::size_t bytes, std::size_t offset) {
    char* out = static_cast<char*>(buffer);
    std::size_t done = 0;
    while (done < bytes) {
        ssize_t n = pread(fd, out + done, bytes - done,
                          static_cast<off_t>(offset + done));
        if (n == 0) return false;
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        done += static_cast<std::size_t>(n);
    }
    return true;
}

static bool read_one_file_into_slot(const std::string& filepath,
                                    int slot,
                                    const SSTReadInfo& info,
                                    float* h_sst_data) {
    size_t slot_offset = static_cast<size_t>(slot) * SPATIAL_POINTS;

    if (info.raw_data_read) {
        int fd = open(filepath.c_str(), O_RDONLY);
        if (fd < 0) {
            fill_slot_nan(h_sst_data, slot);
            return false;
        }

        std::vector<double> tmp_double(SPATIAL_POINTS);
        const std::size_t data_bytes = SPATIAL_POINTS * sizeof(double);
#ifdef POSIX_FADV_SEQUENTIAL
        posix_fadvise(fd, static_cast<off_t>(info.raw_data_offset), data_bytes, POSIX_FADV_SEQUENTIAL);
#endif
        bool ok = read_full_at(fd, tmp_double.data(), data_bytes, info.raw_data_offset);
        close(fd);

        if (!ok) {
            fill_slot_nan(h_sst_data, slot);
            return false;
        }

        #pragma omp parallel for simd schedule(static)
        for (long long i = 0; i < static_cast<long long>(SPATIAL_POINTS); ++i) {
            h_sst_data[slot_offset + i] = static_cast<float>(tmp_double[i]);
        }
        return true;
    }

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

    // Read as double first (data files are float64), then convert to float.
    std::vector<double> tmp_double(SPATIAL_POINTS);

    if (info.layout == DATA_LAT_LON) {
        size_t start[2] = {0, 0};
        size_t count[2] = {LAT_SIZE, LON_SIZE};

        status = nc_get_vara_double(ncid, varid, start, count, tmp_double.data());
        nc_close(ncid);

        if (status != NC_NOERR) {
            fill_slot_nan(h_sst_data, slot);
            return false;
        }

        // Convert double → float, apply scale/offset, handle missing values
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(SPATIAL_POINTS); ++i) {
            double v = tmp_double[i];
            bool is_missing = false;
            if (info.has_fill && v == info.fill_value) is_missing = true;
            if (info.has_missing && v == info.missing_value) is_missing = true;
            if (is_missing) {
                h_sst_data[slot_offset + i] = std::numeric_limits<float>::quiet_NaN();
            } else {
#if APPLY_SCALE_OFFSET
                if (info.has_scale || info.has_offset) {
                    v = v * info.scale_factor + info.add_offset;
                }
#endif
                h_sst_data[slot_offset + i] = static_cast<float>(v);
            }
        }
        return true;
    }

    // data(lon, lat) layout: read then transpose to [lat][lon]
    size_t start[2] = {0, 0};
    size_t count[2] = {LON_SIZE, LAT_SIZE};

    status = nc_get_vara_double(ncid, varid, start, count, tmp_double.data());
    nc_close(ncid);

    if (status != NC_NOERR) {
        fill_slot_nan(h_sst_data, slot);
        return false;
    }

    #pragma omp parallel for schedule(static)
    for (long long x_ll = 0; x_ll < static_cast<long long>(LON_SIZE); ++x_ll) {
        size_t x = static_cast<size_t>(x_ll);
        for (size_t y = 0; y < LAT_SIZE; ++y) {
            double v = tmp_double[x * LAT_SIZE + y];
            bool is_missing = false;
            if (info.has_fill && v == info.fill_value) is_missing = true;
            if (info.has_missing && v == info.missing_value) is_missing = true;
            if (is_missing) {
                h_sst_data[slot_offset + y * LON_SIZE + x] = std::numeric_limits<float>::quiet_NaN();
            } else {
#if APPLY_SCALE_OFFSET
                if (info.has_scale || info.has_offset) {
                    v = v * info.scale_factor + info.add_offset;
                }
#endif
                h_sst_data[slot_offset + y * LON_SIZE + x] = static_cast<float>(v);
            }
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

    int io_threads = runtime_io_threads();

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

// ============================================================
// 滑动窗口 IO：相邻两天共享 328/330 样本，每天只读 30 个新文件
// ============================================================

static SSTReadInfo g_cached_info;
static bool g_info_cached = false;

static std::string date_for_year_doy_offset(int year, int doy, int offset) {
    Date center = no_feb29_doy_to_date(year, doy);
    Date target = add_days(center, offset);
    return yyyymmdd(target);
}

void initialize_window(float* h_sst_data, int first_doy) {
    auto t0 = std::chrono::steady_clock::now();
    std::string input_dir = with_trailing_slash(NC_INPUT_DIR);

    // 缓存文件元信息（只需检查一次）
    if (!g_info_cached) {
        std::string inspect_file = input_dir + "19910101";
        g_cached_info = inspect_sst_file(inspect_file);
        g_info_cached = true;
    }

    int io_threads = runtime_io_threads();

    const int window = 2 * CLIM_DELTA_DAY + 1;
    std::cout << "[I/O 模块] 初始化滑动窗口: doy=" << first_doy
              << "，读取 " << DAYS_TOTAL << " 个文件" << std::endl;

    int failed_count = 0;

    #pragma omp parallel for schedule(dynamic, 1) num_threads(io_threads) reduction(+:failed_count)
    for (int yr_idx = 0; yr_idx < CLIM_YEARS; ++yr_idx) {
        int year = CLIM_START_YEAR + yr_idx;
        for (int off = -CLIM_DELTA_DAY; off <= CLIM_DELTA_DAY; ++off) {
            int slot = (off + CLIM_DELTA_DAY) * CLIM_YEARS + yr_idx;
            std::string date_str = date_for_year_doy_offset(year, first_doy, off);
            std::string filepath = input_dir + date_str;
            bool ok = read_one_file_into_slot(filepath, slot, g_cached_info, h_sst_data);
            if (!ok) ++failed_count;
        }
    }

    std::cout << "[I/O 模块] 窗口初始化完成。failed=" << failed_count
              << "/" << DAYS_TOTAL
              << " elapsed="
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
              << "s" << std::endl;
}

static void load_next_doy_slots(float* h_sst_data, int current_doy, int first_doy, const char* tag) {
    auto t0 = std::chrono::steady_clock::now();
    std::string input_dir = with_trailing_slash(NC_INPUT_DIR);
    const int window = 2 * CLIM_DELTA_DAY + 1;
    const int next_doy = current_doy + 1;

    // 循环槽位替换：因为 mean/P90 与样本顺序无关
    int replace_offset_idx = (current_doy - first_doy) % window;

    int io_threads = (std::strcmp(tag, "prefetch") == 0)
                         ? runtime_prefetch_io_threads()
                         : runtime_io_threads();

    int failed_count = 0;

    // 每年读 1 个新文件（进入样本：next_doy + delta_day），覆盖最旧槽位
    #pragma omp parallel for schedule(dynamic, 1) num_threads(io_threads) reduction(+:failed_count)
    for (int yr_idx = 0; yr_idx < CLIM_YEARS; ++yr_idx) {
        int year = CLIM_START_YEAR + yr_idx;
        int slot = replace_offset_idx * CLIM_YEARS + yr_idx;
        std::string date_str = date_for_year_doy_offset(year, next_doy, CLIM_DELTA_DAY);
        std::string filepath = input_dir + date_str;
        bool ok = read_one_file_into_slot(filepath, slot, g_cached_info, h_sst_data);
        if (!ok) ++failed_count;
    }

    std::cout << "[I/O 模块] 滑动 doy=" << next_doy
              << "，failed=" << failed_count << "/" << CLIM_YEARS
              << " elapsed="
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
              << "s" << std::endl;
}

void load_next_doy_compact(float* h_compact_data, int current_doy, int first_doy) {
    auto t0 = std::chrono::steady_clock::now();
    std::string input_dir = with_trailing_slash(NC_INPUT_DIR);
    const int next_doy = current_doy + 1;

    int io_threads = runtime_io_threads();
    int failed_count = 0;

    #pragma omp parallel for schedule(dynamic, 1) num_threads(io_threads) reduction(+:failed_count)
    for (int yr_idx = 0; yr_idx < CLIM_YEARS; ++yr_idx) {
        int year = CLIM_START_YEAR + yr_idx;
        std::string date_str = date_for_year_doy_offset(year, next_doy, CLIM_DELTA_DAY);
        std::string filepath = input_dir + date_str;
        bool ok = read_one_file_into_slot(filepath, yr_idx, g_cached_info, h_compact_data);
        if (!ok) ++failed_count;
    }

    std::cout << "[I/O module] compact next doy=" << next_doy
              << ", failed=" << failed_count << "/" << CLIM_YEARS
              << " elapsed="
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
              << "s" << std::endl;
}

void read_season_timeline(float* h_season_data,
                          int target_doy_begin,
                          int target_doy_end,
                          int* season_start_doy,
                          int* season_days) {
    auto t0 = std::chrono::steady_clock::now();
    std::string input_dir = with_trailing_slash(NC_INPUT_DIR);

    if (!g_info_cached) {
        std::string inspect_file = input_dir + "19910101";
        g_cached_info = inspect_sst_file(inspect_file);
        g_info_cached = true;
    }

    int start_doy = target_doy_begin - CLIM_DELTA_DAY;
    int end_doy = target_doy_end + CLIM_DELTA_DAY;
    if (start_doy < 1 || end_doy > 365) {
        throw std::runtime_error("season timeline prototype expects target range inside one no-Feb29 year");
    }

    int days = end_doy - start_doy + 1;
    if (season_start_doy) *season_start_doy = start_doy;
    if (season_days) *season_days = days;

    int io_threads = runtime_io_threads();
    int failed_count = 0;
    int total_files = days * CLIM_YEARS;

    std::cout << "[I/O module] reading season timeline: target="
              << target_doy_begin << ".." << target_doy_end
              << ", season=" << start_doy << ".." << end_doy
              << ", files=" << total_files
              << ", I/O threads=" << io_threads << std::endl;

    #pragma omp parallel for schedule(dynamic, 1) num_threads(io_threads) reduction(+:failed_count)
    for (int idx = 0; idx < total_files; ++idx) {
        int day_idx = idx / CLIM_YEARS;
        int yr_idx = idx % CLIM_YEARS;
        int year = CLIM_START_YEAR + yr_idx;
        int doy = start_doy + day_idx;
        int slot = day_idx * CLIM_YEARS + yr_idx;

        std::string date_str = date_for_year_doy_offset(year, doy, 0);
        std::string filepath = input_dir + date_str;
        bool ok = read_one_file_into_slot(filepath, slot, g_cached_info, h_season_data);
        if (!ok) ++failed_count;
    }

    std::cout << "[I/O module] season timeline complete. failed=" << failed_count
              << "/" << total_files
              << " elapsed="
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
              << "s" << std::endl;
}

// 保留原来的接口：默认读取 TARGET_DOY 对应的 30年×11天窗口
static std::future<void> g_prefetch_future;
static bool g_prefetch_active = false;
static bool g_prefetch_ready = false;
static int g_prefetch_current_doy = -1;
static int g_prefetch_first_doy = -1;

void wait_for_prefetch_next_doy() {
    if (!g_prefetch_active) return;
    g_prefetch_future.get();
    g_prefetch_active = false;
    g_prefetch_ready = true;
}

void prefetch_next_doy_async(float* h_sst_data, int current_doy, int first_doy) {
    wait_for_prefetch_next_doy();
    g_prefetch_active = true;
    g_prefetch_ready = false;
    g_prefetch_current_doy = current_doy;
    g_prefetch_first_doy = first_doy;
    g_prefetch_future = std::async(std::launch::async, [h_sst_data, current_doy, first_doy]() {
        load_next_doy_slots(h_sst_data, current_doy, first_doy, "prefetch");
    });
}

void slide_window_to_next_doy(float* h_sst_data, int current_doy, int first_doy) {
    wait_for_prefetch_next_doy();
    if (g_prefetch_ready &&
        g_prefetch_current_doy == current_doy &&
        g_prefetch_first_doy == first_doy) {
        g_prefetch_ready = false;
        return;
    }
    load_next_doy_slots(h_sst_data, current_doy, first_doy, "slide");
}

void read_netcdf_parallel(float* h_sst_data) {
    read_netcdf_parallel_for_doy(h_sst_data, TARGET_DOY);
}
