#include "io_handler.h"
#include "config.h"

#include <netcdf.h>
#include <hdf5.h>
#include <omp.h>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <atomic>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
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

static SSTReadInfo g_cached_info;
static bool g_info_cached = false;

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

static bool source_date_order_enabled() {
    const char* env = std::getenv("MCC_SOURCE_IO_DATE_ORDER");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static std::vector<int> source_file_read_order(const std::vector<std::string>& filepaths) {
    std::vector<int> order(filepaths.size());
    std::iota(order.begin(), order.end(), 0);
    if (source_date_order_enabled()) {
        std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
            return filepaths[lhs] < filepaths[rhs];
        });
    }
    return order;
}

static bool source_io_stdthread_enabled() {
    const char* env = std::getenv("MCC_SOURCE_IO_BACKEND");
    if (!env || env[0] == '\0') return false;
    std::string backend(env);
    std::transform(backend.begin(), backend.end(), backend.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return backend == "thread" || backend == "stdthread" || backend == "pthread";
}

template <typename Fn>
static int run_source_io_stdthread(int tasks, int io_threads, Fn&& fn) {
    if (tasks <= 0) return 0;
    int nthreads = std::max(1, std::min(io_threads, tasks));
    std::atomic<int> next(0);
    std::atomic<int> failed(0);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(nthreads));
    for (int t = 0; t < nthreads; ++t) {
        workers.emplace_back([&]() {
            while (true) {
                int idx = next.fetch_add(1, std::memory_order_relaxed);
                if (idx >= tasks) break;
                if (!fn(idx)) {
                    failed.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
    return failed.load(std::memory_order_relaxed);
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

static bool raw_hdf5_enabled() {
    const char* env = std::getenv("MCC_RAW_HDF5");
    return env == nullptr || (env[0] != 0 && env[0] != '0');
}

static bool raw_direct_enabled() {
    const char* env = std::getenv("MCC_RAW_DIRECT");
    return env == nullptr || (env[0] != 0 && env[0] != '0');
}

static bool raw_mmap_enabled() {
    const char* env = std::getenv("MCC_RAW_MMAP");
    return env && env[0] != 0 && env[0] != '0';
}

static std::size_t raw_direct_alignment() {
    std::size_t align = 512;
    if (const char* env = std::getenv("MCC_RAW_DIRECT_ALIGN")) {
        int requested = std::atoi(env);
        if (requested >= 512 && (requested & (requested - 1)) == 0) {
            align = static_cast<std::size_t>(requested);
        }
    }
    return align;
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
    info.raw_data_read = (raw_hdf5_enabled() &&
                          info.layout == DATA_LAT_LON &&
                          !info.has_fill &&
                          !info.has_missing &&
                          !info.has_scale &&
                          !info.has_offset &&
                          inspect_raw_hdf5_data(filepath, raw_offset));
    info.raw_data_offset = raw_offset;

    std::cout << "[I/O module] raw HDF5 data path: "
              << (info.raw_data_read ? "enabled" : "disabled")
              << ", offset=" << info.raw_data_offset
              << ", mmap=" << (raw_mmap_enabled() ? "ON" : "OFF")
              << ", direct=" << (raw_direct_enabled() ? "ON" : "OFF")
              << ", direct_align=" << raw_direct_alignment()
              << std::endl;
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

struct PackedSample {
    std::string date;
    int slot;
};

struct HostXorShift128State {
    uint64_t s0;
    uint64_t s1;
};

static uint64_t splitmix64_step(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void host_xorshift_seed(HostXorShift128State& st, int year_idx, int target_doy) {
    uint64_t seed = 0xD1B54A32D192ED03ULL
                  ^ (static_cast<uint64_t>(target_doy) * 0x9E3779B97F4A7C15ULL)
                  ^ (static_cast<uint64_t>(year_idx + 1) * 0xBF58476D1CE4E5B9ULL);
    st.s0 = splitmix64_step(seed);
    st.s1 = splitmix64_step(seed);
    if ((st.s0 | st.s1) == 0) st.s1 = 1;
}

static uint64_t host_xorshift128_next(HostXorShift128State& st) {
    uint64_t s1 = st.s0;
    const uint64_t s0 = st.s1;
    st.s0 = s0;
    s1 ^= s1 << 23;
    st.s1 = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    return st.s1 + s0;
}

static int host_xorshift_rand_int(HostXorShift128State& st, int n) {
    return static_cast<int>(host_xorshift128_next(st) % static_cast<uint64_t>(n));
}

static std::vector<PackedSample> build_mc_packed_samples_for_doy(int doy, int sample_count) {
    std::vector<PackedSample> samples;
    if (sample_count <= 0) return samples;

    const int window = 2 * CLIM_DELTA_DAY + 1;
    int k_per_year = sample_count / CLIM_YEARS;
    if (k_per_year <= 0) k_per_year = 1;
    if (k_per_year > window) k_per_year = window;

    samples.reserve(CLIM_YEARS * k_per_year);
    int slot = 0;
    for (int yr_idx = 0; yr_idx < CLIM_YEARS; ++yr_idx) {
        int year = CLIM_START_YEAR + yr_idx;
        HostXorShift128State rng;
        host_xorshift_seed(rng, yr_idx, doy);

        bool used[2 * CLIM_DELTA_DAY + 1] = {};
        for (int k = 0; k < k_per_year; ++k) {
            int off_idx = host_xorshift_rand_int(rng, window);
            for (int tries = 0; used[off_idx] && tries < window; ++tries) {
                off_idx = (off_idx + 1) % window;
            }
            used[off_idx] = true;
            int off = off_idx - CLIM_DELTA_DAY;
            Date center = no_feb29_doy_to_date(year, doy);
            Date target = add_days(center, off);
            samples.push_back(PackedSample{yyyymmdd(target), slot++});
        }
    }
    return samples;
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

struct DirectScratch {
    char* ptr = nullptr;
    std::size_t capacity = 0;
    std::size_t alignment = 0;

    ~DirectScratch() {
        std::free(ptr);
    }

    char* get(std::size_t bytes, std::size_t align) {
        if (ptr && capacity >= bytes && alignment == align) {
            return ptr;
        }
        std::free(ptr);
        ptr = nullptr;
        capacity = 0;
        alignment = align;
        void* p = nullptr;
        if (posix_memalign(&p, align, bytes) != 0) {
            return nullptr;
        }
        ptr = static_cast<char*>(p);
        capacity = bytes;
        return ptr;
    }
};

static thread_local DirectScratch g_direct_scratch;

struct MappedRawRegion {
    void* base = MAP_FAILED;
    std::size_t length = 0;
    const double* data = nullptr;

    void unmap() {
        if (base != MAP_FAILED) {
            munmap(base, length);
            base = MAP_FAILED;
            length = 0;
            data = nullptr;
        }
    }
};

static std::size_t align_down_size(std::size_t x, std::size_t align) {
    return x & ~(align - 1);
}

static std::size_t align_up_size(std::size_t x, std::size_t align) {
    return (x + align - 1) & ~(align - 1);
}

static bool read_raw_region_direct(const std::string& filepath,
                                   std::size_t offset,
                                   std::size_t bytes,
                                   const double*& out) {
#ifndef O_DIRECT
    (void)filepath;
    (void)offset;
    (void)bytes;
    (void)out;
    return false;
#else
    if (!raw_direct_enabled()) return false;

    const std::size_t align = raw_direct_alignment();
    const std::size_t aligned_offset = align_down_size(offset, align);
    const std::size_t prefix = offset - aligned_offset;
    const std::size_t needed = prefix + bytes;
    const std::size_t direct_bytes = align_down_size(needed, align);
    const std::size_t tail_bytes = needed - direct_bytes;

    char* scratch = g_direct_scratch.get(needed, 4096);
    if (!scratch) return false;

    int fd = open(filepath.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) return false;

    bool ok = true;
    if (direct_bytes > 0) {
        ok = read_full_at(fd, scratch, direct_bytes, aligned_offset);
    }
    close(fd);

    if (!ok) return false;

    if (tail_bytes > 0) {
        int tail_fd = open(filepath.c_str(), O_RDONLY);
        if (tail_fd < 0) return false;
        ok = read_full_at(tail_fd, scratch + direct_bytes, tail_bytes,
                          aligned_offset + direct_bytes);
        close(tail_fd);
        if (!ok) return false;
    }

    out = reinterpret_cast<const double*>(scratch + prefix);
    return true;
#endif
}

static bool map_raw_region_double(const std::string& filepath,
                                  std::size_t offset,
                                  std::size_t elements,
                                  MappedRawRegion& mapped) {
    if (!raw_mmap_enabled()) return false;

    const long page_long = sysconf(_SC_PAGESIZE);
    const std::size_t page_size = page_long > 0 ? static_cast<std::size_t>(page_long) : 4096;
    const std::size_t bytes = elements * sizeof(double);
    const std::size_t aligned_offset = align_down_size(offset, page_size);
    const std::size_t prefix = offset - aligned_offset;
    const std::size_t map_length = prefix + bytes;

    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd < 0) return false;

    void* base = mmap(nullptr, map_length, PROT_READ, MAP_PRIVATE,
                      fd, static_cast<off_t>(aligned_offset));
    close(fd);
    if (base == MAP_FAILED) return false;

#ifdef MADV_SEQUENTIAL
    madvise(base, map_length, MADV_SEQUENTIAL);
#endif
    mapped.base = base;
    mapped.length = map_length;
    mapped.data = reinterpret_cast<const double*>(static_cast<const char*>(base) + prefix);
    return true;
}

static bool read_raw_region_normal(const std::string& filepath,
                                   std::size_t offset,
                                   std::size_t elements,
                                   std::vector<double>& tmp,
                                   const double*& out) {
    if (tmp.size() < elements) tmp.resize(elements);
    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd < 0) return false;
    bool ok = read_full_at(fd, tmp.data(), elements * sizeof(double), offset);
    close(fd);
    if (!ok) return false;
    out = tmp.data();
    return true;
}

static bool read_raw_region_double(const std::string& filepath,
                                   std::size_t offset,
                                   std::size_t elements,
                                   std::vector<double>& tmp,
                                   const double*& out) {
    const std::size_t bytes = elements * sizeof(double);
    if (read_raw_region_direct(filepath, offset, bytes, out)) {
        return true;
    }
    return read_raw_region_normal(filepath, offset, elements, tmp, out);
}

static bool read_one_file_into_slot(const std::string& filepath,
                                    int slot,
                                    const SSTReadInfo& info,
                                    float* h_sst_data) {
    size_t slot_offset = static_cast<size_t>(slot) * SPATIAL_POINTS;

    if (info.raw_data_read) {
        MappedRawRegion mapped;
        if (map_raw_region_double(filepath, info.raw_data_offset, SPATIAL_POINTS, mapped)) {
            const double* src = mapped.data;
            #pragma omp simd
            for (long long i = 0; i < static_cast<long long>(SPATIAL_POINTS); ++i) {
                h_sst_data[slot_offset + i] = static_cast<float>(src[i]);
            }
            mapped.unmap();
            return true;
        }

        static thread_local std::vector<double> tl_tmp;
        const double* src = nullptr;
        bool ok = read_raw_region_double(filepath, info.raw_data_offset,
                                         SPATIAL_POINTS, tl_tmp, src);
        if (!ok) {
            fill_slot_nan(h_sst_data, slot);
            return false;
        }

        #pragma omp simd
        for (long long i = 0; i < static_cast<long long>(SPATIAL_POINTS); ++i) {
            h_sst_data[slot_offset + i] = static_cast<float>(src[i]);
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
    static thread_local std::vector<double> tl_tmp; if (tl_tmp.size() < SPATIAL_POINTS) tl_tmp.resize(SPATIAL_POINTS);

    if (info.layout == DATA_LAT_LON) {
        size_t start[2] = {0, 0};
        size_t count[2] = {LAT_SIZE, LON_SIZE};

        status = nc_get_vara_double(ncid, varid, start, count, tl_tmp.data());
        nc_close(ncid);

        if (status != NC_NOERR) {
            fill_slot_nan(h_sst_data, slot);
            return false;
        }

        // Convert double → float, apply scale/offset, handle missing values
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(SPATIAL_POINTS); ++i) {
            double v = tl_tmp[i];
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

    status = nc_get_vara_double(ncid, varid, start, count, tl_tmp.data());
    nc_close(ncid);

    if (status != NC_NOERR) {
        fill_slot_nan(h_sst_data, slot);
        return false;
    }

    #pragma omp parallel for schedule(static)
    for (long long x_ll = 0; x_ll < static_cast<long long>(LON_SIZE); ++x_ll) {
        size_t x = static_cast<size_t>(x_ll);
        for (size_t y = 0; y < LAT_SIZE; ++y) {
            double v = tl_tmp[x * LAT_SIZE + y];
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

static bool read_one_file_into_local_parts(const std::string& filepath,
                                           int slot,
                                           const SSTReadInfo& info,
                                           float** h_source_local,
                                           const int* lat_starts,
                                           const int* local_rows,
                                           int num_parts) {
    if (info.raw_data_read) {
        MappedRawRegion mapped;
        if (map_raw_region_double(filepath, info.raw_data_offset, SPATIAL_POINTS, mapped)) {
            const double* src = mapped.data;
            for (int part = 0; part < num_parts; ++part) {
                if (!h_source_local[part] || local_rows[part] <= 0) continue;
                const int local_spatial = local_rows[part] * static_cast<int>(LON_SIZE);
                const int spatial_start = lat_starts[part] * static_cast<int>(LON_SIZE);
                float* dst = h_source_local[part] + static_cast<std::size_t>(slot) * local_spatial;
                const double* part_src = src + spatial_start;
                #pragma omp simd
                for (int i = 0; i < local_spatial; ++i) {
                    dst[i] = static_cast<float>(part_src[i]);
                }
            }
            mapped.unmap();
            return true;
        }

        static thread_local std::vector<double> tl_tmp;
        const double* src = nullptr;
        bool ok = read_raw_region_double(filepath, info.raw_data_offset,
                                         SPATIAL_POINTS, tl_tmp, src);
        if (!ok) {
            for (int part = 0; part < num_parts; ++part) {
                if (!h_source_local[part] || local_rows[part] <= 0) continue;
                const int local_spatial = local_rows[part] * static_cast<int>(LON_SIZE);
                float* dst = h_source_local[part] + static_cast<std::size_t>(slot) * local_spatial;
                std::fill(dst, dst + local_spatial, std::numeric_limits<float>::quiet_NaN());
            }
            return false;
        }

        for (int part = 0; part < num_parts; ++part) {
            if (!h_source_local[part] || local_rows[part] <= 0) continue;
            const int local_spatial = local_rows[part] * static_cast<int>(LON_SIZE);
            const int spatial_start = lat_starts[part] * static_cast<int>(LON_SIZE);
            float* dst = h_source_local[part] + static_cast<std::size_t>(slot) * local_spatial;
            const double* part_src = src + spatial_start;
            #pragma omp simd
            for (int i = 0; i < local_spatial; ++i) {
                dst[i] = static_cast<float>(part_src[i]);
            }
        }
        return true;
    }

    static thread_local std::vector<float> tl_full;
    if (tl_full.size() < SPATIAL_POINTS) tl_full.resize(SPATIAL_POINTS);
    bool ok = read_one_file_into_slot(filepath, 0, info, tl_full.data());
    for (int part = 0; part < num_parts; ++part) {
        if (!h_source_local[part] || local_rows[part] <= 0) continue;
        const int local_spatial = local_rows[part] * static_cast<int>(LON_SIZE);
        const int spatial_start = lat_starts[part] * static_cast<int>(LON_SIZE);
        std::memcpy(h_source_local[part] + static_cast<std::size_t>(slot) * local_spatial,
                    tl_full.data() + spatial_start,
                    static_cast<std::size_t>(local_spatial) * sizeof(float));
    }
    return ok;
}


static std::string resolve_input_dir() {
    const char* env = std::getenv("MCC_INPUT_DIR");
    if (env && env[0] != '\0') {
        std::string d(env);
        if (!d.empty() && d[d.size()-1] != '/') d.push_back('/');
        std::cout << "[I/O module] input override: " << d << std::endl;
        return d;
    }
    return with_trailing_slash(NC_INPUT_DIR);
}

static std::string date_for_year_doy_offset(int year, int doy, int offset);

void read_netcdf_parallel_for_doy(float* h_sst_data, int target_doy) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);

    std::string input_dir = resolve_input_dir();

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

void read_mc_packed_for_doy(float* h_sst_data, int target_doy, int sample_count) {
    auto t0 = std::chrono::steady_clock::now();
    std::string input_dir = resolve_input_dir();

    if (!g_info_cached) {
        std::string inspect_file = input_dir + "19910101";
        g_cached_info = inspect_sst_file(inspect_file);
        g_info_cached = true;
    }

    std::vector<PackedSample> samples =
        build_mc_packed_samples_for_doy(target_doy, sample_count);

    int io_threads = runtime_io_threads();
    int failed_count = 0;

    std::cout << "[I/O 模块] MC packed doy=" << target_doy
              << "，K=" << samples.size()
              << "，I/O threads=" << io_threads << std::endl;

    #pragma omp parallel for schedule(dynamic, 1) num_threads(io_threads) reduction(+:failed_count)
    for (int i = 0; i < static_cast<int>(samples.size()); ++i) {
        const PackedSample& sample = samples[i];
        std::string filepath = input_dir + sample.date;
        bool ok = read_one_file_into_slot(filepath, sample.slot, g_cached_info, h_sst_data);
        if (!ok) ++failed_count;
    }

    std::cout << "[I/O 模块] MC packed 读取完成。failed=" << failed_count
              << "/" << samples.size()
              << " elapsed="
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
              << "s" << std::endl;
}

void preload_season_source(float* h_source_data, int target_doy_begin, int target_doy_end) {
    auto t0 = std::chrono::steady_clock::now();
    std::string input_dir = resolve_input_dir();

    if (!g_info_cached) {
        std::string inspect_file = input_dir + "19910101";
        g_cached_info = inspect_sst_file(inspect_file);
        g_info_cached = true;
    }

    const int source_begin = target_doy_begin - CLIM_DELTA_DAY;
    const int source_end = target_doy_end + CLIM_DELTA_DAY;
    const int source_days = source_end - source_begin + 1;
    const int source_slots = source_days * CLIM_YEARS;
    int io_threads = runtime_io_threads();
    int failed_count = 0;

    std::cout << "[I/O 模块] 整季预加载 source-window: target="
              << target_doy_begin << ".." << target_doy_end
              << " source=" << source_begin << ".." << source_end
              << " slots=" << source_slots
              << "，I/O threads=" << io_threads << std::endl;

    #pragma omp parallel for collapse(2) schedule(dynamic, 1) num_threads(io_threads) reduction(+:failed_count)
    for (int source_idx = 0; source_idx < source_days; ++source_idx) {
        for (int yr_idx = 0; yr_idx < CLIM_YEARS; ++yr_idx) {
            int year = CLIM_START_YEAR + yr_idx;
            int doy = source_begin + source_idx;
            int slot = source_idx * CLIM_YEARS + yr_idx;
            std::string date_str = date_for_year_doy_offset(year, doy, 0);
            std::string filepath = input_dir + date_str;
            bool ok = read_one_file_into_slot(filepath, slot, g_cached_info, h_source_data);
            if (!ok) ++failed_count;
        }
    }

    std::cout << "[I/O 模块] 整季预加载完成。failed=" << failed_count
              << "/" << source_slots
              << " elapsed="
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
              << "s" << std::endl;
}

void preload_season_source_local(float** h_source_local,
                                 const int* lat_starts,
                                 const int* local_rows,
                                 int num_parts,
                                 int target_doy_begin,
                                 int target_doy_end) {
    auto t0 = std::chrono::steady_clock::now();
    std::string input_dir = resolve_input_dir();

    if (!g_info_cached) {
        std::string inspect_file = input_dir + "19910101";
        g_cached_info = inspect_sst_file(inspect_file);
        g_info_cached = true;
    }

    const int source_begin = target_doy_begin - CLIM_DELTA_DAY;
    const int source_end = target_doy_end + CLIM_DELTA_DAY;
    const int source_days = source_end - source_begin + 1;
    const int source_slots = source_days * CLIM_YEARS;
    int io_threads = runtime_io_threads();
    int failed_count = 0;

    std::cout << "[I/O 模块] 整季预加载 GPU-local source-window: target="
              << target_doy_begin << ".." << target_doy_end
              << " source=" << source_begin << ".." << source_end
              << " slots=" << source_slots
              << " parts=" << num_parts
              << "，I/O threads=" << io_threads << std::endl;

    std::vector<std::string> filepaths(source_slots);
    for (int source_idx = 0; source_idx < source_days; ++source_idx) {
        for (int yr_idx = 0; yr_idx < CLIM_YEARS; ++yr_idx) {
            int year = CLIM_START_YEAR + yr_idx;
            int doy = source_begin + source_idx;
            int slot = source_idx * CLIM_YEARS + yr_idx;
            std::string date_str = date_for_year_doy_offset(year, doy, 0);
            filepaths[slot] = input_dir + date_str;
        }
    }

    const std::vector<int> read_order = source_file_read_order(filepaths);

    #pragma omp parallel for schedule(static) num_threads(io_threads) reduction(+:failed_count)
    for (int order_idx = 0; order_idx < source_slots; ++order_idx) {
        const int slot = read_order[order_idx];
        bool ok = read_one_file_into_local_parts(filepaths[slot], slot, g_cached_info,
                                                 h_source_local, lat_starts,
                                                 local_rows, num_parts);
        if (!ok) {
            ++failed_count;
        }
    }

    std::cout << "[I/O 模块] GPU-local 整季预加载完成。failed=" << failed_count
              << "/" << source_slots
              << " order=" << (source_date_order_enabled() ? "date" : "slot")
              << " elapsed="
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
              << "s" << std::endl;
}

void preload_source_days_local(float** h_source_local,
                               const int* lat_starts,
                               const int* local_rows,
                               int num_parts,
                               int source_doy_begin,
                               int source_day_count) {
    auto t0 = std::chrono::steady_clock::now();
    std::string input_dir = resolve_input_dir();

    if (!g_info_cached) {
        std::string inspect_file = input_dir + "19910101";
        g_cached_info = inspect_sst_file(inspect_file);
        g_info_cached = true;
    }

    const int source_slots = source_day_count * CLIM_YEARS;
    int io_threads = runtime_io_threads();
    int failed_count = 0;

    std::vector<std::string> filepaths(source_slots);
    for (int day_idx = 0; day_idx < source_day_count; ++day_idx) {
        for (int yr_idx = 0; yr_idx < CLIM_YEARS; ++yr_idx) {
            int year = CLIM_START_YEAR + yr_idx;
            int doy = source_doy_begin + day_idx;
            int slot = day_idx * CLIM_YEARS + yr_idx;
            filepaths[slot] = input_dir + date_for_year_doy_offset(year, doy, 0);
        }
    }

    const std::vector<int> read_order = source_file_read_order(filepaths);

    if (source_io_stdthread_enabled()) {
        failed_count = run_source_io_stdthread(
            source_slots, io_threads,
            [&](int order_idx) {
                const int slot = read_order[order_idx];
                return read_one_file_into_local_parts(filepaths[slot], slot, g_cached_info,
                                                      h_source_local, lat_starts,
                                                      local_rows, num_parts);
            });
    } else {
        #pragma omp parallel for schedule(static) num_threads(io_threads) reduction(+:failed_count)
        for (int order_idx = 0; order_idx < source_slots; ++order_idx) {
            const int slot = read_order[order_idx];
            bool ok = read_one_file_into_local_parts(filepaths[slot], slot, g_cached_info,
                                                     h_source_local, lat_starts,
                                                     local_rows, num_parts);
            if (!ok) ++failed_count;
        }
    }

    std::cout << "[I/O 模块] source chunk 读取完成: source_doy="
              << source_doy_begin << ".." << (source_doy_begin + source_day_count - 1)
              << " slots=" << source_slots
              << " failed=" << failed_count << "/" << source_slots
              << " order=" << (source_date_order_enabled() ? "date" : "slot")
              << " backend=" << (source_io_stdthread_enabled() ? "stdthread" : "openmp")
              << " elapsed="
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
              << "s" << std::endl;
}

// ============================================================
// 滑动窗口 IO：相邻两天共享 328/330 样本，每天只读 30 个新文件
// ============================================================

static std::string date_for_year_doy_offset(int year, int doy, int offset) {
    Date center = no_feb29_doy_to_date(year, doy);
    Date target = add_days(center, offset);
    return yyyymmdd(target);
}

void initialize_window(float* h_sst_data, int first_doy) {
    auto t0 = std::chrono::steady_clock::now();
    std::string input_dir = resolve_input_dir();

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
    std::string input_dir = resolve_input_dir();
    const int window = 2 * CLIM_DELTA_DAY + 1;
    const int next_doy = current_doy + 1;

    // 循环槽位替换：因为 mean/P90 与样本顺序无关
    int replace_offset_idx = (current_doy - first_doy) % window;

    int io_threads = runtime_io_threads();

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

// 保留原来的接口：默认读取 TARGET_DOY 对应的 30年×11天窗口
static std::future<void> g_prefetch_future;
static bool g_prefetch_active = false;

void wait_for_prefetch_next_doy() {
    if (!g_prefetch_active) return;
    g_prefetch_future.get();
    g_prefetch_active = false;
}

void prefetch_next_doy_async(float* h_sst_data, int current_doy, int first_doy) {
    wait_for_prefetch_next_doy();
    g_prefetch_active = true;
    g_prefetch_future = std::async(std::launch::async, [h_sst_data, current_doy, first_doy]() {
        load_next_doy_slots(h_sst_data, current_doy, first_doy, "prefetch");
    });
}

void slide_window_to_next_doy(float* h_sst_data, int current_doy, int first_doy) {
    wait_for_prefetch_next_doy();
    load_next_doy_slots(h_sst_data, current_doy, first_doy, "slide");
}

void read_netcdf_parallel(float* h_sst_data) {
    read_netcdf_parallel_for_doy(h_sst_data, TARGET_DOY);
}
