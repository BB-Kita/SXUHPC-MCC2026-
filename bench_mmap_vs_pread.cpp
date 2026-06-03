#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <omp.h>

#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static constexpr int LON_SIZE = 1440;
static constexpr int LAT_SIZE = 721;
static constexpr std::size_t SPATIAL_POINTS = static_cast<std::size_t>(LON_SIZE) * LAT_SIZE;
static constexpr std::size_t DATA_BYTES = SPATIAL_POINTS * sizeof(double);
static constexpr off_t RAW_OFFSET = 23488;
static constexpr int CLIM_START_YEAR = 1991;
static constexpr int CLIM_YEARS = 30;
static constexpr int CLIM_DELTA_DAY = 5;

struct Date {
    int y, m, d;
};

static bool is_leap(int y) {
    return (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
}

static int days_in_month(int y, int m) {
    static const int mdays[] = {31,28,31,30,31,30,31,31,30,31,30,31};
    return (m == 2 && is_leap(y)) ? 29 : mdays[m - 1];
}

static Date add_days(Date date, int offset) {
    while (offset > 0) {
        int dim = days_in_month(date.y, date.m);
        if (date.d < dim) ++date.d;
        else {
            date.d = 1;
            if (date.m == 12) { date.m = 1; ++date.y; }
            else ++date.m;
        }
        --offset;
    }
    while (offset < 0) {
        if (date.d > 1) --date.d;
        else {
            if (date.m == 1) { date.m = 12; --date.y; }
            else --date.m;
            date.d = days_in_month(date.y, date.m);
        }
        ++offset;
    }
    return date;
}

static Date no_feb29_doy_to_date(int year, int doy) {
    static const int mdays[] = {31,28,31,30,31,30,31,31,30,31,30,31};
    Date date{year, 1, doy};
    for (int m = 1; m <= 12; ++m) {
        if (date.d <= mdays[m - 1]) {
            date.m = m;
            return date;
        }
        date.d -= mdays[m - 1];
    }
    return Date{year, 12, 31};
}

static std::string yyyymmdd(Date date) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%04d%02d%02d", date.y, date.m, date.d);
    return std::string(buf);
}

static std::vector<std::string> build_slide_files(int current_doy) {
    std::vector<std::string> files;
    files.reserve(CLIM_YEARS);
    int next_doy = current_doy + 1;
    for (int yr = 0; yr < CLIM_YEARS; ++yr) {
        int year = CLIM_START_YEAR + yr;
        Date center = no_feb29_doy_to_date(year, next_doy);
        Date target = add_days(center, CLIM_DELTA_DAY);
        files.push_back("/public/home/achwjznh4b/Newdata/" + yyyymmdd(target));
    }
    return files;
}

static bool read_full_at(int fd, void* buffer, std::size_t bytes, off_t offset) {
    char* out = static_cast<char*>(buffer);
    std::size_t done = 0;
    while (done < bytes) {
        ssize_t n = pread(fd, out + done, bytes - done, offset + static_cast<off_t>(done));
        if (n == 0) return false;
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        done += static_cast<std::size_t>(n);
    }
    return true;
}

static double bench_pread(const std::vector<std::string>& files, int threads) {
    std::vector<float> slots(files.size() * SPATIAL_POINTS);
    auto t0 = std::chrono::steady_clock::now();
    int failed = 0;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads) reduction(+:failed)
    for (int f = 0; f < static_cast<int>(files.size()); ++f) {
        int fd = open(files[f].c_str(), O_RDONLY);
        if (fd < 0) { ++failed; continue; }
        std::vector<double> tmp(SPATIAL_POINTS);
        bool ok = read_full_at(fd, tmp.data(), DATA_BYTES, RAW_OFFSET);
        close(fd);
        if (!ok) { ++failed; continue; }
        float* dst = slots.data() + static_cast<std::size_t>(f) * SPATIAL_POINTS;
        for (std::size_t i = 0; i < SPATIAL_POINTS; ++i) dst[i] = static_cast<float>(tmp[i]);
    }
    if (failed) std::cerr << "pread failed=" << failed << "\n";
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static double bench_pread_fadvise(const std::vector<std::string>& files, int threads) {
    std::vector<float> slots(files.size() * SPATIAL_POINTS);
    auto t0 = std::chrono::steady_clock::now();
    int failed = 0;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads) reduction(+:failed)
    for (int f = 0; f < static_cast<int>(files.size()); ++f) {
        int fd = open(files[f].c_str(), O_RDONLY);
        if (fd < 0) { ++failed; continue; }
#ifdef POSIX_FADV_SEQUENTIAL
        posix_fadvise(fd, RAW_OFFSET, DATA_BYTES, POSIX_FADV_SEQUENTIAL);
#endif
        std::vector<double> tmp(SPATIAL_POINTS);
        bool ok = read_full_at(fd, tmp.data(), DATA_BYTES, RAW_OFFSET);
        close(fd);
        if (!ok) { ++failed; continue; }
        float* dst = slots.data() + static_cast<std::size_t>(f) * SPATIAL_POINTS;
#pragma omp simd
        for (std::size_t i = 0; i < SPATIAL_POINTS; ++i) dst[i] = static_cast<float>(tmp[i]);
    }
    if (failed) std::cerr << "pread_fadvise failed=" << failed << "\n";
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static double bench_pread_reuse_simd(const std::vector<std::string>& files, int threads) {
    std::vector<float> slots(files.size() * SPATIAL_POINTS);
    auto t0 = std::chrono::steady_clock::now();
    int failed = 0;
#pragma omp parallel num_threads(threads) reduction(+:failed)
    {
        std::vector<double> tmp(SPATIAL_POINTS);
#pragma omp for schedule(dynamic, 1)
        for (int f = 0; f < static_cast<int>(files.size()); ++f) {
            int fd = open(files[f].c_str(), O_RDONLY);
            if (fd < 0) { ++failed; continue; }
            bool ok = read_full_at(fd, tmp.data(), DATA_BYTES, RAW_OFFSET);
            close(fd);
            if (!ok) { ++failed; continue; }
            float* dst = slots.data() + static_cast<std::size_t>(f) * SPATIAL_POINTS;
#pragma omp simd
            for (std::size_t i = 0; i < SPATIAL_POINTS; ++i) dst[i] = static_cast<float>(tmp[i]);
        }
    }
    if (failed) std::cerr << "pread_reuse_simd failed=" << failed << "\n";
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static double bench_pread_reuse_simd_fadvise(const std::vector<std::string>& files, int threads) {
    std::vector<float> slots(files.size() * SPATIAL_POINTS);
    auto t0 = std::chrono::steady_clock::now();
    int failed = 0;
#pragma omp parallel num_threads(threads) reduction(+:failed)
    {
        std::vector<double> tmp(SPATIAL_POINTS);
#pragma omp for schedule(dynamic, 1)
        for (int f = 0; f < static_cast<int>(files.size()); ++f) {
            int fd = open(files[f].c_str(), O_RDONLY);
            if (fd < 0) { ++failed; continue; }
#ifdef POSIX_FADV_SEQUENTIAL
            posix_fadvise(fd, RAW_OFFSET, DATA_BYTES, POSIX_FADV_SEQUENTIAL);
#endif
            bool ok = read_full_at(fd, tmp.data(), DATA_BYTES, RAW_OFFSET);
            close(fd);
            if (!ok) { ++failed; continue; }
            float* dst = slots.data() + static_cast<std::size_t>(f) * SPATIAL_POINTS;
#pragma omp simd
            for (std::size_t i = 0; i < SPATIAL_POINTS; ++i) dst[i] = static_cast<float>(tmp[i]);
        }
    }
    if (failed) std::cerr << "pread_reuse_simd_fadvise failed=" << failed << "\n";
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static double bench_mmap(const std::vector<std::string>& files, int threads) {
    std::vector<float> slots(files.size() * SPATIAL_POINTS);
    auto t0 = std::chrono::steady_clock::now();
    int failed = 0;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads) reduction(+:failed)
    for (int f = 0; f < static_cast<int>(files.size()); ++f) {
        int fd = open(files[f].c_str(), O_RDONLY);
        if (fd < 0) { ++failed; continue; }
        void* mapped = mmap(nullptr, RAW_OFFSET + DATA_BYTES, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) {
            close(fd);
            ++failed;
            continue;
        }
#ifdef MADV_SEQUENTIAL
        madvise(mapped, RAW_OFFSET + DATA_BYTES, MADV_SEQUENTIAL);
#endif
        const double* src = reinterpret_cast<const double*>(
            static_cast<const char*>(mapped) + RAW_OFFSET);
        float* dst = slots.data() + static_cast<std::size_t>(f) * SPATIAL_POINTS;
        for (std::size_t i = 0; i < SPATIAL_POINTS; ++i) dst[i] = static_cast<float>(src[i]);
        munmap(mapped, RAW_OFFSET + DATA_BYTES);
        close(fd);
    }
    if (failed) std::cerr << "mmap failed=" << failed << "\n";
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char** argv) {
    int threads = argc > 1 ? std::atoi(argv[1]) : 32;
    int reps = argc > 2 ? std::atoi(argv[2]) : 5;
    std::vector<int> doys = {153, 160, 190, 220, 240};
    for (int rep = 0; rep < reps; ++rep) {
        for (int doy : doys) {
            auto files = build_slide_files(doy);
            double pread_s = bench_pread(files, threads);
            double fadvise_s = bench_pread_fadvise(files, threads);
            double reuse_s = bench_pread_reuse_simd(files, threads);
            double reuse_fadvise_s = bench_pread_reuse_simd_fadvise(files, threads);
            double mmap_s = bench_mmap(files, threads);
            std::cout << "rep=" << rep
                      << " doy=" << doy
                      << " pread_s=" << pread_s
                      << " fadvise_s=" << fadvise_s
                      << " reuse_simd_s=" << reuse_s
                      << " reuse_simd_fadvise_s=" << reuse_fadvise_s
                      << " mmap_s=" << mmap_s
                      << " ratio_mmap_over_pread=" << (mmap_s / pread_s)
                      << std::endl;
        }
    }
    return 0;
}
