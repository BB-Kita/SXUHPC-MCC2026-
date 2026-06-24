#include "compute_dcu.h"
#include "config.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <limits>
#include <cstring>
#include <string>
#include <mutex>

// ============================================================
// Hybrid P90 Kernel: coarse histogram → adaptive collect → exact interp
// (UNCHANGED — works on float* data exclusively)
// ============================================================

static constexpr int COARSE_BINS = 64;
static constexpr int COLLECT_MAX = 120;
static constexpr int FAST_HIST_BINS = 512;
static constexpr int FIXED_COLLECT_MAX = 256;
static constexpr float FAST_HIST_MIN = -3.0f;
static constexpr float FAST_HIST_MAX = 45.0f;

__device__ inline float select_kth_float(float* vals, int n, int k) {
    if (k <= 0) k = 0;
    if (k >= n) k = n - 1;
    int left = 0;
    int right = n - 1;
    while (true) {
        if (left >= right) return vals[left];
        float pivot = vals[(left + right) >> 1];
        int lt = left;
        int i = left;
        int gt = right;

        while (i <= gt) {
            float v = vals[i];
            if (v < pivot) {
                float tmp = vals[lt];
                vals[lt] = vals[i];
                vals[i] = tmp;
                ++lt;
                ++i;
            } else if (v > pivot) {
                float tmp = vals[i];
                vals[i] = vals[gt];
                vals[gt] = tmp;
                --gt;
            } else {
                ++i;
            }
        }

        if (k < lt) {
            right = lt - 1;
        } else if (k > gt) {
            left = gt + 1;
        } else {
            return pivot;
        }
    }
}

__global__ void compute_mean_p90_kernel(
    const float* __restrict__ data,
    float* __restrict__ out_mean,
    float* __restrict__ out_p90,
    int spatial_points,
    int days_total,
    int spatial_start,
    int spatial_end,
    int data_spatial_points,
    int data_spatial_start)
{
    int global_idx = spatial_start + (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (global_idx >= spatial_end) return;

    int out_idx = global_idx - spatial_start;
    int data_idx = global_idx - data_spatial_start;

    float sum = 0.0f;
    int valid_count = 0;
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int d = 0; d < days_total; ++d) {
        float v = data[d * data_spatial_points + data_idx];
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

    float hist_f[COARSE_BINS];
    for (int b = 0; b < COARSE_BINS; ++b) hist_f[b] = 0.0f;

    float inv_range = (float)COARSE_BINS / (vmax - vmin);

    for (int d = 0; d < days_total; ++d) {
        float v = data[d * data_spatial_points + data_idx];
        if (!isnan(v)) {
            int bin = (int)((v - vmin) * inv_range);
            if (bin < 0) bin = 0;
            if (bin >= COARSE_BINS) bin = COARSE_BINS - 1;
            hist_f[bin] += 1.0f;
        }
    }

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
            float v = data[d * data_spatial_points + data_idx];
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

    for (int i = 1; i < buf_count; ++i) {
        float key = buf[i];
        int j = i - 1;
        while (j >= 0 && buf[j] > key) {
            buf[j + 1] = buf[j];
            --j;
        }
        buf[j + 1] = key;
    }

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

__global__ void compute_source_window_kernel(
    const float* __restrict__ data,
    float* __restrict__ out_mean,
    float* __restrict__ out_p90,
    int target_offset,
    int clim_years,
    int window_days,
    int spatial_points,
    int spatial_start,
    int spatial_end,
    int data_spatial_points,
    bool skip_zero,
    bool select_p90)
{
    int global_idx = spatial_start + (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (global_idx >= spatial_end) return;

    int out_idx = global_idx - spatial_start;
    int data_idx = global_idx - spatial_start;

    if (skip_zero) {
        float first = data[(size_t)target_offset * clim_years * data_spatial_points + data_idx];
        if (first == 0.0f) {
            out_mean[out_idx] = 0.0f;
            out_p90[out_idx] = 0.0f;
            return;
        }
    }

    float sum = 0.0f;
    int valid_count = 0;
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int w = 0; w < window_days; ++w) {
        int source_idx = target_offset + w;
        for (int y = 0; y < clim_years; ++y) {
            int slot = source_idx * clim_years + y;
            float v = data[(size_t)slot * data_spatial_points + data_idx];
            if (!isnan(v)) {
                sum += v;
                ++valid_count;
                if (v < vmin) vmin = v;
                if (v > vmax) vmax = v;
            }
        }
    }

    if (valid_count == 0) {
        out_mean[out_idx] = NAN;
        out_p90[out_idx] = NAN;
        return;
    }

    out_mean[out_idx] = sum / (float)valid_count;

    if (valid_count == 1 || vmin == vmax) {
        out_p90[out_idx] = vmin;
        return;
    }

    unsigned short hist[COARSE_BINS];
    for (int b = 0; b < COARSE_BINS; ++b) hist[b] = 0;

    float inv_range = (float)COARSE_BINS / (vmax - vmin);
    for (int w = 0; w < window_days; ++w) {
        int source_idx = target_offset + w;
        for (int y = 0; y < clim_years; ++y) {
            int slot = source_idx * clim_years + y;
            float v = data[(size_t)slot * data_spatial_points + data_idx];
            if (!isnan(v)) {
                int bin = (int)((v - vmin) * inv_range);
                if (bin < 0) bin = 0;
                if (bin >= COARSE_BINS) bin = COARSE_BINS - 1;
                ++hist[bin];
            }
        }
    }

    float target_rank = 0.9f * (float)(valid_count - 1);
    int rank_lo = (int)target_rank;
    int rank_hi = rank_lo + 1;
    float frac = target_rank - (float)rank_lo;
    bool need_interp = (frac > 1e-6f && rank_hi < valid_count);

    int target_bin = COARSE_BINS - 1;
    int cumulative = 0;
    for (int b = 0; b < COARSE_BINS; ++b) {
        if (cumulative + (int)hist[b] > rank_lo) {
            target_bin = b;
            break;
        }
        cumulative += (int)hist[b];
    }

    float bin_width = (vmax - vmin) / (float)COARSE_BINS;
    int lo_bin = target_bin;
    int hi_bin = target_bin + 1;
    float buf[COLLECT_MAX];
    int buf_count = 0;
    int cum_lo = cumulative;
    bool success = false;

    for (int expand = 0; expand < COARSE_BINS; ++expand) {
        float c_lo = vmin + (float)lo_bin * bin_width;
        float c_hi = vmin + (float)hi_bin * bin_width;
        if (c_lo < vmin) c_lo = vmin;
        if (c_hi > vmax) c_hi = vmax;

        buf_count = 0;
        for (int w = 0; w < window_days && buf_count < COLLECT_MAX; ++w) {
            int source_idx = target_offset + w;
            for (int y = 0; y < clim_years && buf_count < COLLECT_MAX; ++y) {
                int slot = source_idx * clim_years + y;
                float v = data[(size_t)slot * data_spatial_points + data_idx];
                if (!isnan(v) && v >= c_lo && v < c_hi) {
                    buf[buf_count++] = v;
                }
            }
        }

        if (!need_interp) {
            int local_lo = rank_lo - cum_lo;
            if (local_lo >= 0 && local_lo < buf_count) {
                success = true;
                break;
            }
        } else {
            int local_lo = rank_lo - cum_lo;
            int local_hi = rank_hi - cum_lo;
            if (local_lo >= 0 && local_hi < buf_count) {
                success = true;
                break;
            }
        }

        if (lo_bin > 0) {
            --lo_bin;
            cum_lo -= (int)hist[lo_bin];
        }
        if (hi_bin < COARSE_BINS) ++hi_bin;
        if (buf_count >= COLLECT_MAX - 10) break;
    }

    if (!success || buf_count == 0) {
        out_p90[out_idx] = vmin + (float)(target_bin + 0.5f) * bin_width;
        return;
    }

    int ilo = rank_lo - cum_lo;
    if (ilo < 0) ilo = 0;
    if (ilo >= buf_count) ilo = buf_count - 1;

    if (select_p90) {
        float v_lo = select_kth_float(buf, buf_count, ilo);
        if (!need_interp || ilo + 1 >= buf_count) {
            out_p90[out_idx] = v_lo;
        } else {
            int ihi = rank_hi - cum_lo;
            if (ihi < 0) ihi = 0;
            if (ihi >= buf_count) ihi = buf_count - 1;
            float v_hi = select_kth_float(buf, buf_count, ihi);
            out_p90[out_idx] = v_lo + frac * (v_hi - v_lo);
        }
        return;
    }

    for (int i = 1; i < buf_count; ++i) {
        float key = buf[i];
        int j = i - 1;
        while (j >= 0 && buf[j] > key) {
            buf[j + 1] = buf[j];
            --j;
        }
        buf[j + 1] = key;
    }

    if (!need_interp || ilo + 1 >= buf_count) {
        out_p90[out_idx] = buf[ilo];
    } else {
        int ihi = rank_hi - cum_lo;
        if (ihi < 0) ihi = 0;
        if (ihi >= buf_count) ihi = buf_count - 1;
        out_p90[out_idx] = buf[ilo] + frac * (buf[ihi] - buf[ilo]);
    }
}

__global__ void compute_source_window_fast_hist_kernel(
    const float* __restrict__ data,
    float* __restrict__ out_mean,
    float* __restrict__ out_p90,
    int target_offset,
    int clim_years,
    int window_days,
    int spatial_points,
    int spatial_start,
    int spatial_end,
    int data_spatial_points,
    bool skip_zero)
{
    int global_idx = spatial_start + (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (global_idx >= spatial_end) return;

    int out_idx = global_idx - spatial_start;
    int data_idx = global_idx - spatial_start;

    if (skip_zero) {
        float first = data[(size_t)target_offset * clim_years * data_spatial_points + data_idx];
        if (first == 0.0f) {
            out_mean[out_idx] = 0.0f;
            out_p90[out_idx] = 0.0f;
            return;
        }
    }

    unsigned short hist[FAST_HIST_BINS];
    for (int b = 0; b < FAST_HIST_BINS; ++b) hist[b] = 0;

    float sum = 0.0f;
    int valid_count = 0;
    const float inv_width = (float)FAST_HIST_BINS / (FAST_HIST_MAX - FAST_HIST_MIN);

    for (int w = 0; w < window_days; ++w) {
        int source_idx = target_offset + w;
        for (int y = 0; y < clim_years; ++y) {
            int slot = source_idx * clim_years + y;
            float v = data[(size_t)slot * data_spatial_points + data_idx];
            if (!isnan(v)) {
                sum += v;
                ++valid_count;
                int bin = (int)((v - FAST_HIST_MIN) * inv_width);
                if (bin < 0) bin = 0;
                if (bin >= FAST_HIST_BINS) bin = FAST_HIST_BINS - 1;
                ++hist[bin];
            }
        }
    }

    if (valid_count == 0) {
        out_mean[out_idx] = NAN;
        out_p90[out_idx] = NAN;
        return;
    }

    out_mean[out_idx] = sum / (float)valid_count;

    if (valid_count == 1) {
        for (int b = 0; b < FAST_HIST_BINS; ++b) {
            if (hist[b]) {
                out_p90[out_idx] = FAST_HIST_MIN + ((float)b + 0.5f) *
                    ((FAST_HIST_MAX - FAST_HIST_MIN) / (float)FAST_HIST_BINS);
                return;
            }
        }
    }

    float rank = 0.9f * (float)(valid_count - 1);
    int rank_i = (int)rank;
    int cumulative = 0;
    const float width = (FAST_HIST_MAX - FAST_HIST_MIN) / (float)FAST_HIST_BINS;

    for (int b = 0; b < FAST_HIST_BINS; ++b) {
        int count = (int)hist[b];
        if (count <= 0) continue;
        if (cumulative + count > rank_i) {
            float in_bin = (rank - (float)cumulative) / (float)count;
            if (in_bin < 0.0f) in_bin = 0.0f;
            if (in_bin > 1.0f) in_bin = 1.0f;
            out_p90[out_idx] = FAST_HIST_MIN + ((float)b + in_bin) * width;
            return;
        }
        cumulative += count;
    }

    out_p90[out_idx] = FAST_HIST_MAX;
}

__global__ void compute_source_window_fixed_collect_kernel(
    const float* __restrict__ data,
    float* __restrict__ out_mean,
    float* __restrict__ out_p90,
    int target_offset,
    int clim_years,
    int window_days,
    int spatial_points,
    int spatial_start,
    int spatial_end,
    int data_spatial_points,
    bool skip_zero)
{
    int global_idx = spatial_start + (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (global_idx >= spatial_end) return;

    int out_idx = global_idx - spatial_start;
    int data_idx = global_idx - spatial_start;

    if (skip_zero) {
        float first = data[(size_t)target_offset * clim_years * data_spatial_points + data_idx];
        if (first == 0.0f) {
            out_mean[out_idx] = 0.0f;
            out_p90[out_idx] = 0.0f;
            return;
        }
    }

    unsigned short hist[FAST_HIST_BINS];
    for (int b = 0; b < FAST_HIST_BINS; ++b) hist[b] = 0;

    float sum = 0.0f;
    float first_valid = NAN;
    int valid_count = 0;
    const float width = (FAST_HIST_MAX - FAST_HIST_MIN) / (float)FAST_HIST_BINS;
    const float inv_width = (float)FAST_HIST_BINS / (FAST_HIST_MAX - FAST_HIST_MIN);

    for (int w = 0; w < window_days; ++w) {
        int source_idx = target_offset + w;
        for (int y = 0; y < clim_years; ++y) {
            int slot = source_idx * clim_years + y;
            float v = data[(size_t)slot * data_spatial_points + data_idx];
            if (!isnan(v)) {
                if (valid_count == 0) first_valid = v;
                sum += v;
                ++valid_count;
                int bin = (int)((v - FAST_HIST_MIN) * inv_width);
                if (bin < 0) bin = 0;
                if (bin >= FAST_HIST_BINS) bin = FAST_HIST_BINS - 1;
                ++hist[bin];
            }
        }
    }

    if (valid_count == 0) {
        out_mean[out_idx] = NAN;
        out_p90[out_idx] = NAN;
        return;
    }

    out_mean[out_idx] = sum / (float)valid_count;
    if (valid_count == 1) {
        out_p90[out_idx] = first_valid;
        return;
    }

    float target_rank = 0.9f * (float)(valid_count - 1);
    int rank_lo = (int)target_rank;
    int rank_hi = rank_lo + 1;
    float frac = target_rank - (float)rank_lo;
    bool need_interp = (frac > 1e-6f && rank_hi < valid_count);
    int wanted_hi = need_interp ? rank_hi : rank_lo;

    int lo_bin = FAST_HIST_BINS - 1;
    int hi_bin = FAST_HIST_BINS - 1;
    int cumulative = 0;
    int cum_before_lo = 0;
    for (int b = 0; b < FAST_HIST_BINS; ++b) {
        int next = cumulative + (int)hist[b];
        if (next > rank_lo && lo_bin == FAST_HIST_BINS - 1) {
            lo_bin = b;
            cum_before_lo = cumulative;
        }
        if (next > wanted_hi) {
            hi_bin = b;
            break;
        }
        cumulative = next;
    }

    int selected_count = 0;
    for (int b = lo_bin; b <= hi_bin; ++b) selected_count += (int)hist[b];
    if (selected_count <= 0) {
        out_p90[out_idx] = NAN;
        return;
    }

    if (selected_count > FIXED_COLLECT_MAX) {
        out_p90[out_idx] = FAST_HIST_MIN + ((float)lo_bin + 0.5f) * width;
        return;
    }

    float c_lo = FAST_HIST_MIN + (float)lo_bin * width;
    float c_hi = FAST_HIST_MIN + (float)(hi_bin + 1) * width;
    float buf[FIXED_COLLECT_MAX];
    int buf_count = 0;

    for (int w = 0; w < window_days; ++w) {
        int source_idx = target_offset + w;
        for (int y = 0; y < clim_years; ++y) {
            int slot = source_idx * clim_years + y;
            float v = data[(size_t)slot * data_spatial_points + data_idx];
            if (isnan(v)) continue;
            bool in_range = false;
            if (lo_bin == 0 && v < c_hi) {
                in_range = true;
            } else if (hi_bin == FAST_HIST_BINS - 1 && v >= c_lo) {
                in_range = true;
            } else if (v >= c_lo && v < c_hi) {
                in_range = true;
            }
            if (in_range && buf_count < FIXED_COLLECT_MAX) {
                buf[buf_count++] = v;
            }
        }
    }

    for (int i = 1; i < buf_count; ++i) {
        float key = buf[i];
        int j = i - 1;
        while (j >= 0 && buf[j] > key) {
            buf[j + 1] = buf[j];
            --j;
        }
        buf[j + 1] = key;
    }

    int ilo = rank_lo - cum_before_lo;
    if (ilo < 0) ilo = 0;
    if (ilo >= buf_count) ilo = buf_count - 1;

    if (!need_interp) {
        out_p90[out_idx] = buf[ilo];
    } else {
        int ihi = rank_hi - cum_before_lo;
        if (ihi < 0) ihi = 0;
        if (ihi >= buf_count) ihi = buf_count - 1;
        out_p90[out_idx] = buf[ilo] + frac * (buf[ihi] - buf[ilo]);
    }
}

// ============================================================
// NEW: fp16 ↔ fp32 conversion kernels
// ============================================================

// float → half (used for full upload DCU-side conversion)
__global__ void float_to_half_kernel(
    const float* __restrict__ src,
    __half* __restrict__ dst,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

// half → float (used for incremental upload conversion)
__global__ void fp16_to_fp32_2d_kernel(
    const __half* __restrict__ src,
    float* __restrict__ dst,
    int rows,
    int src_stride,
    int dst_stride)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = global_idx / src_stride;
    int col = global_idx % src_stride;
    if (row >= rows) return;
    int idx = row * src_stride + col;
    dst[row * dst_stride + col] = __half2float(src[idx]);
}

// ============================================================
// NEW: Monte Carlo P90 Kernel (xorshift128+ + consistent sampling)
//
// Two modes:
//   CONSISTENT: all spatial points use same K pre-computed day indices
//               → H2D: only K days need uploading (5-10x savings)
//   INDEPENDENT: each point samples its own K random days via xorshift
//               → compute savings only (K reads vs 3×330 reads)
//
// Controlled by MCC_MC_K env var. K=0 means exact (original kernel).
// ============================================================

// XorShift128+ state (each thread has its own, seeded by spatial index)
struct XorShift128State {
    uint64_t s0, s1;
};

__device__ inline uint64_t xorshift128_next(XorShift128State* st) {
    uint64_t s1 = st->s0;
    const uint64_t s0 = st->s1;
    st->s0 = s0;
    s1 ^= s1 << 23;
    st->s1 = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    return st->s1 + s0;
}

__device__ inline int xorshift_rand_int(XorShift128State* st, int n) {
    // Keep this strictly bounded. The previous 64-bit multiply-high variant
    // truncated the product before shifting and could return values >= n.
    return (int)(xorshift128_next(st) % (uint64_t)n);
}

__device__ inline void xorshift_seed(XorShift128State* st, int spatial_idx, int doy) {
    // SplitMix64-style seeding
    uint64_t z = (uint64_t)(spatial_idx + 1) * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    st->s0 = z;
    st->s1 = z ^ ((uint64_t)doy * 0x9E3779B97F4A7C15ULL);
    // Warm up
    for (int i = 0; i < 4; ++i) xorshift128_next(st);
}

// Consistent MC kernel: all points use same pre-computed day indices
__global__ void mc_p90_consistent_kernel(
    const float* __restrict__ data,
    const int*   __restrict__ indices,  // K day indices (same for all threads)
    int K,
    float* __restrict__ out_mean,
    float* __restrict__ out_p90,
    int spatial_points,
    int spatial_start,
    int spatial_end,
    int data_spatial_points)
{
    int global_idx = spatial_start + (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (global_idx >= spatial_end) return;

    int out_idx = global_idx - spatial_start;
    int data_idx = global_idx - spatial_start;  // local index within GPU chunk

    float vals[120];  // K ≤ 120
    int valid = 0;
    double sum = 0.0;

    for (int k = 0; k < K; ++k) {
        int day = indices[k];
        float v = data[day * data_spatial_points + data_idx];
        if (!isnan(v)) {
            vals[valid++] = v;
            sum += (double)v;
        }
    }

    if (valid == 0) {
        out_mean[out_idx] = NAN;
        out_p90[out_idx] = NAN;
        return;
    }

    out_mean[out_idx] = (float)(sum / (double)valid);

    if (valid == 1) {
        out_p90[out_idx] = vals[0];
        return;
    }

    // nth_element: find the ceil(0.9*valid)-th smallest value
    int target = (int)(0.9 * (valid - 1));
    int lo = 0, hi = valid - 1;
    while (lo < hi) {
        float pivot = vals[(lo + hi) / 2];
        int i = lo, j = hi;
        while (i <= j) {
            while (vals[i] < pivot) ++i;
            while (vals[j] > pivot) --j;
            if (i <= j) {
                float tmp = vals[i]; vals[i] = vals[j]; vals[j] = tmp;
                ++i; --j;
            }
        }
        if (target <= j) hi = j;
        else if (target >= i) lo = i;
        else break;
    }

    // Linear interpolation for fractional rank
    float v_lo = vals[lo];
    double rank = 0.9 * (valid - 1);
    int rank_lo = (int)rank;
    float frac = (float)(rank - rank_lo);

    if (frac < 1e-6f || rank_lo + 1 >= valid) {
        out_p90[out_idx] = v_lo;
    } else {
        // Find next larger value
        float v_hi = 1e30f;
        for (int k = 0; k < valid; ++k) {
            if (vals[k] > v_lo && vals[k] < v_hi) v_hi = vals[k];
        }
        out_p90[out_idx] = v_lo + frac * (v_hi - v_lo);
    }
}

// Independent MC kernel: each point has its own random indices (xorshift)
__global__ void mc_p90_independent_kernel(
    const float* __restrict__ data,
    int K,
    int doy,
    int days_total,
    float* __restrict__ out_mean,
    float* __restrict__ out_p90,
    int spatial_points,
    int spatial_start,
    int spatial_end,
    int data_spatial_points)
{
    int global_idx = spatial_start + (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (global_idx >= spatial_end) return;

    int out_idx = global_idx - spatial_start;
    int data_idx = global_idx - spatial_start;  // local index

    // Initialize xorshift128+ per spatial point
    XorShift128State rng;
    xorshift_seed(&rng, global_idx, doy);

    float vals[120];
    int valid = 0;
    double sum = 0.0;

    for (int k = 0; k < K && k < days_total; ++k) {
        int day = xorshift_rand_int(&rng, days_total);
        float v = data[day * data_spatial_points + data_idx];
        if (!isnan(v)) {
            vals[valid++] = v;
            sum += (double)v;
        }
    }

    if (valid == 0) {
        out_mean[out_idx] = NAN;
        out_p90[out_idx] = NAN;
        return;
    }

    out_mean[out_idx] = (float)(sum / (double)valid);

    if (valid == 1) {
        out_p90[out_idx] = vals[0];
        return;
    }

    // nth_element + interp (same as consistent kernel)
    int target = (int)(0.9 * (valid - 1));
    int lo = 0, hi = valid - 1;
    while (lo < hi) {
        float pivot = vals[(lo + hi) / 2];
        int i = lo, j = hi;
        while (i <= j) {
            while (vals[i] < pivot) ++i;
            while (vals[j] > pivot) --j;
            if (i <= j) {
                float tmp = vals[i]; vals[i] = vals[j]; vals[j] = tmp;
                ++i; --j;
            }
        }
        if (target <= j) hi = j;
        else if (target >= i) lo = i;
        else break;
    }

    float v_lo = vals[lo];
    double rank = 0.9 * (valid - 1);
    int rank_lo = (int)rank;
    float frac = (float)(rank - rank_lo);

    if (frac < 1e-6f || rank_lo + 1 >= valid) {
        out_p90[out_idx] = v_lo;
    } else {
        float v_hi = 1e30f;
        for (int k = 0; k < valid; ++k) {
            if (vals[k] > v_lo && vals[k] < v_hi) v_hi = vals[k];
        }
        out_p90[out_idx] = v_lo + frac * (v_hi - v_lo);
    }
}

// Packed MC kernel: the host has already sampled files and packed them into
// slots [0, K). The kernel therefore scans only those K slots directly.
__global__ void mc_p90_packed_kernel(
    const float* __restrict__ data,
    int K,
    float* __restrict__ out_mean,
    float* __restrict__ out_p90,
    int spatial_points,
    int spatial_start,
    int spatial_end,
    int data_spatial_points)
{
    int global_idx = spatial_start + (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (global_idx >= spatial_end) return;

    int out_idx = global_idx - spatial_start;
    int data_idx = global_idx - spatial_start;

    float vals[120];
    int valid = 0;
    double sum = 0.0;

    for (int k = 0; k < K; ++k) {
        float v = data[k * data_spatial_points + data_idx];
        if (!isnan(v)) {
            vals[valid++] = v;
            sum += (double)v;
        }
    }

    if (valid == 0) {
        out_mean[out_idx] = NAN;
        out_p90[out_idx] = NAN;
        return;
    }

    out_mean[out_idx] = (float)(sum / (double)valid);

    if (valid == 1) {
        out_p90[out_idx] = vals[0];
        return;
    }

    int target = (int)(0.9 * (valid - 1));
    int lo = 0, hi = valid - 1;
    while (lo < hi) {
        float pivot = vals[(lo + hi) / 2];
        int i = lo, j = hi;
        while (i <= j) {
            while (vals[i] < pivot) ++i;
            while (vals[j] > pivot) --j;
            if (i <= j) {
                float tmp = vals[i]; vals[i] = vals[j]; vals[j] = tmp;
                ++i; --j;
            }
        }
        if (target <= j) hi = j;
        else if (target >= i) lo = i;
        else break;
    }

    float v_lo = vals[lo];
    double rank = 0.9 * (valid - 1);
    int rank_lo = (int)rank;
    float frac = (float)(rank - rank_lo);

    if (frac < 1e-6f || rank_lo + 1 >= valid) {
        out_p90[out_idx] = v_lo;
    } else {
        float v_hi = 1e30f;
        for (int k = 0; k < valid; ++k) {
            if (vals[k] > v_lo && vals[k] < v_hi) v_hi = vals[k];
        }
        out_p90[out_idx] = v_lo + frac * (v_hi - v_lo);
    }
}

// Block-shared MC kernel: one xorshift-generated stratified sample set per block.
// This keeps random-number overhead low while avoiding one global sample set for
// the whole field. K is normalized on host to CLIM_YEARS * k_per_year and <= 120.
__global__ void mc_p90_block_xorshift_kernel(
    const float* __restrict__ data,
    int K,
    int doy,
    int days_total,
    int clim_years,
    int window_days,
    float* __restrict__ out_mean,
    float* __restrict__ out_p90,
    int spatial_points,
    int spatial_start,
    int spatial_end,
    int data_spatial_points)
{
    __shared__ int s_indices[120];

    int k_per_year = K / clim_years;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        int year = k / k_per_year;
        int rep = k - year * k_per_year;
        XorShift128State rng;
        int seed_idx = spatial_start + (int)blockIdx.x * blockDim.x + rep * 65537 + year * 131071;
        xorshift_seed(&rng, seed_idx, doy);
        int off = xorshift_rand_int(&rng, window_days);
        s_indices[k] = year * window_days + off;
    }
    __syncthreads();

    int global_idx = spatial_start + (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (global_idx >= spatial_end) return;

    int out_idx = global_idx - spatial_start;
    int data_idx = global_idx - spatial_start;

    float vals[120];
    int valid = 0;
    double sum = 0.0;

    for (int k = 0; k < K && k < days_total; ++k) {
        int day = s_indices[k];
        float v = data[day * data_spatial_points + data_idx];
        if (!isnan(v)) {
            vals[valid++] = v;
            sum += (double)v;
        }
    }

    if (valid == 0) {
        out_mean[out_idx] = NAN;
        out_p90[out_idx] = NAN;
        return;
    }

    out_mean[out_idx] = (float)(sum / (double)valid);

    if (valid == 1) {
        out_p90[out_idx] = vals[0];
        return;
    }

    int target = (int)(0.9 * (valid - 1));
    int lo = 0, hi = valid - 1;
    while (lo < hi) {
        float pivot = vals[(lo + hi) / 2];
        int i = lo, j = hi;
        while (i <= j) {
            while (vals[i] < pivot) ++i;
            while (vals[j] > pivot) --j;
            if (i <= j) {
                float tmp = vals[i]; vals[i] = vals[j]; vals[j] = tmp;
                ++i; --j;
            }
        }
        if (target <= j) hi = j;
        else if (target >= i) lo = i;
        else break;
    }

    float v_lo = vals[lo];
    double rank = 0.9 * (valid - 1);
    int rank_lo = (int)rank;
    float frac = (float)(rank - rank_lo);

    if (frac < 1e-6f || rank_lo + 1 >= valid) {
        out_p90[out_idx] = v_lo;
    } else {
        float v_hi = 1e30f;
        for (int k = 0; k < valid; ++k) {
            if (vals[k] > v_lo && vals[k] < v_hi) v_hi = vals[k];
        }
        out_p90[out_idx] = v_lo + frac * (v_hi - v_lo);
    }
}

__device__ inline int source_mc_stratified_offset(XorShift128State* rng,
                                                 int rep,
                                                 int k_per_year,
                                                 int window_days) {
    int seg0 = (rep * window_days) / k_per_year;
    int seg1 = ((rep + 1) * window_days) / k_per_year;
    if (seg1 <= seg0) seg1 = seg0 + 1;
    if (seg1 > window_days) seg1 = window_days;
    int span = seg1 - seg0;
    return seg0 + xorshift_rand_int(rng, span);
}

__device__ inline int source_mc_flat_stratified_slot(XorShift128State* rng,
                                                    int rep,
                                                    int sample_count,
                                                    int total_window_slots) {
    int seg0 = (rep * total_window_slots) / sample_count;
    int seg1 = ((rep + 1) * total_window_slots) / sample_count;
    if (seg1 <= seg0) seg1 = seg0 + 1;
    if (seg1 > total_window_slots) seg1 = total_window_slots;
    int span = seg1 - seg0;
    return seg0 + xorshift_rand_int(rng, span);
}

// Source-window MC kernel: keep Climmean exact, approximate only P90 with
// stratified xorshift samples from the 11-day x 30-year source window.
__global__ void compute_source_window_mc_kernel(
    const float* __restrict__ data,
    float* __restrict__ out_mean,
    float* __restrict__ out_p90,
    int target_offset,
    int clim_years,
    int window_days,
    int spatial_points,
    int spatial_start,
    int spatial_end,
    int data_spatial_points,
    bool skip_zero,
    int sample_k,
    bool independent_samples,
    bool exact_mean)
{
    __shared__ int s_slots[120];

    int k_per_year = sample_k / clim_years;
    if (k_per_year <= 0) k_per_year = 1;
    int normalized_k = k_per_year * clim_years;
    if (normalized_k > 120) normalized_k = 120;

    if (!independent_samples) {
        for (int k = threadIdx.x; k < normalized_k; k += blockDim.x) {
            int year = k / k_per_year;
            int rep = k - year * k_per_year;
            XorShift128State rng;
            int seed_idx = spatial_start + (int)blockIdx.x * blockDim.x
                         + year * 131071 + rep * 65537;
            xorshift_seed(&rng, seed_idx, target_offset + 4099);
            int off = source_mc_stratified_offset(&rng, rep, k_per_year, window_days);
            s_slots[k] = (target_offset + off) * clim_years + year;
        }
        __syncthreads();
    }

    int global_idx = spatial_start + (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (global_idx >= spatial_end) return;

    int out_idx = global_idx - spatial_start;
    int data_idx = global_idx - spatial_start;

    if (skip_zero) {
        float first = data[(size_t)target_offset * clim_years * data_spatial_points + data_idx];
        if (first == 0.0f) {
            out_mean[out_idx] = 0.0f;
            out_p90[out_idx] = 0.0f;
            return;
        }
    }

    if (exact_mean) {
        float sum = 0.0f;
        int valid_count = 0;
        for (int w = 0; w < window_days; ++w) {
            int source_idx = target_offset + w;
            for (int y = 0; y < clim_years; ++y) {
                int slot = source_idx * clim_years + y;
                float v = data[(size_t)slot * data_spatial_points + data_idx];
                if (!isnan(v)) {
                    sum += v;
                    ++valid_count;
                }
            }
        }

        if (valid_count == 0) {
            out_mean[out_idx] = NAN;
            out_p90[out_idx] = NAN;
            return;
        }
        out_mean[out_idx] = sum / (float)valid_count;
    }

    float vals[120];
    int sample_valid = 0;
    float sample_sum = 0.0f;

    if (independent_samples) {
        for (int y = 0; y < clim_years; ++y) {
            for (int rep = 0; rep < k_per_year && sample_valid < 120; ++rep) {
                XorShift128State rng;
                int seed_idx = global_idx + y * 131071 + rep * 65537;
                xorshift_seed(&rng, seed_idx, target_offset + 4099);
                int off = source_mc_stratified_offset(&rng, rep, k_per_year, window_days);
                int slot = (target_offset + off) * clim_years + y;
                float v = data[(size_t)slot * data_spatial_points + data_idx];
                if (!isnan(v)) {
                    vals[sample_valid++] = v;
                    sample_sum += v;
                }
            }
        }
    } else {
        for (int k = 0; k < normalized_k; ++k) {
            int slot = s_slots[k];
            float v = data[(size_t)slot * data_spatial_points + data_idx];
            if (!isnan(v)) {
                vals[sample_valid++] = v;
                sample_sum += v;
            }
        }
    }

    if (sample_valid == 0) {
        if (!exact_mean) out_mean[out_idx] = NAN;
        out_p90[out_idx] = NAN;
        return;
    }
    if (!exact_mean) {
        out_mean[out_idx] = sample_sum / (float)sample_valid;
    }
    if (sample_valid == 1) {
        out_p90[out_idx] = vals[0];
        return;
    }

    float target_rank = 0.9f * (float)(sample_valid - 1);
    int rank_lo = (int)target_rank;
    int rank_hi = rank_lo + 1;
    float frac = target_rank - (float)rank_lo;
    float v_lo = select_kth_float(vals, sample_valid, rank_lo);
    if (frac <= 1e-6f || rank_hi >= sample_valid) {
        out_p90[out_idx] = v_lo;
    } else {
        float v_hi = select_kth_float(vals, sample_valid, rank_hi);
        out_p90[out_idx] = v_lo + frac * (v_hi - v_lo);
    }
}

__global__ void compute_source_window_mc_sample_kernel(
    const float* __restrict__ data,
    float* __restrict__ out_mean,
    float* __restrict__ out_p90,
    int target_offset,
    int clim_years,
    int window_days,
    int spatial_points,
    int spatial_start,
    int spatial_end,
    int data_spatial_points,
    bool skip_zero,
    int sample_k,
    bool independent_samples,
    bool flat_samples)
{
    __shared__ int s_slots[120];

    int sample_count = sample_k;
    if (sample_count < 1) sample_count = 1;
    if (sample_count > 120) sample_count = 120;
    bool use_flat = flat_samples || sample_count < clim_years;
    int k_per_year = sample_count / clim_years;
    if (k_per_year <= 0) k_per_year = 1;
    int normalized_k = use_flat ? sample_count : k_per_year * clim_years;
    if (normalized_k > 120) normalized_k = 120;

    if (!independent_samples) {
        for (int k = threadIdx.x; k < normalized_k; k += blockDim.x) {
            if (use_flat) {
                XorShift128State rng;
                int seed_idx = spatial_start + (int)blockIdx.x * blockDim.x + k * 65537;
                xorshift_seed(&rng, seed_idx, target_offset + 4099);
                int flat = source_mc_flat_stratified_slot(&rng, k, normalized_k,
                                                          window_days * clim_years);
                s_slots[k] = target_offset * clim_years + flat;
            } else {
                int year = k / k_per_year;
                int rep = k - year * k_per_year;
                XorShift128State rng;
                int seed_idx = spatial_start + (int)blockIdx.x * blockDim.x
                             + year * 131071 + rep * 65537;
                xorshift_seed(&rng, seed_idx, target_offset + 4099);
                int off = source_mc_stratified_offset(&rng, rep, k_per_year, window_days);
                s_slots[k] = (target_offset + off) * clim_years + year;
            }
        }
        __syncthreads();
    }

    int global_idx = spatial_start + (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (global_idx >= spatial_end) return;

    int out_idx = global_idx - spatial_start;
    int data_idx = global_idx - spatial_start;

    if (skip_zero) {
        float first = data[(size_t)target_offset * clim_years * data_spatial_points + data_idx];
        if (first == 0.0f) {
            out_mean[out_idx] = 0.0f;
            out_p90[out_idx] = 0.0f;
            return;
        }
    }

    float vals[120];
    int sample_valid = 0;
    float sample_sum = 0.0f;

    if (independent_samples && use_flat) {
        for (int k = 0; k < normalized_k && sample_valid < 120; ++k) {
            XorShift128State rng;
            int seed_idx = global_idx + k * 65537;
            xorshift_seed(&rng, seed_idx, target_offset + 4099);
            int flat = source_mc_flat_stratified_slot(&rng, k, normalized_k,
                                                      window_days * clim_years);
            int slot = target_offset * clim_years + flat;
            float v = data[(size_t)slot * data_spatial_points + data_idx];
            if (!isnan(v)) {
                vals[sample_valid++] = v;
                sample_sum += v;
            }
        }
    } else if (independent_samples) {
        for (int y = 0; y < clim_years; ++y) {
            for (int rep = 0; rep < k_per_year && sample_valid < 120; ++rep) {
                XorShift128State rng;
                int seed_idx = global_idx + y * 131071 + rep * 65537;
                xorshift_seed(&rng, seed_idx, target_offset + 4099);
                int off = source_mc_stratified_offset(&rng, rep, k_per_year, window_days);
                int slot = (target_offset + off) * clim_years + y;
                float v = data[(size_t)slot * data_spatial_points + data_idx];
                if (!isnan(v)) {
                    vals[sample_valid++] = v;
                    sample_sum += v;
                }
            }
        }
    } else {
        for (int k = 0; k < normalized_k; ++k) {
            int slot = s_slots[k];
            float v = data[(size_t)slot * data_spatial_points + data_idx];
            if (!isnan(v)) {
                vals[sample_valid++] = v;
                sample_sum += v;
            }
        }
    }

    if (sample_valid == 0) {
        out_mean[out_idx] = NAN;
        out_p90[out_idx] = NAN;
        return;
    }

    out_mean[out_idx] = sample_sum / (float)sample_valid;
    if (sample_valid == 1) {
        out_p90[out_idx] = vals[0];
        return;
    }

    float target_rank = 0.9f * (float)(sample_valid - 1);
    int rank_lo = (int)target_rank;
    int rank_hi = rank_lo + 1;
    float frac = target_rank - (float)rank_lo;
    float v_lo = select_kth_float(vals, sample_valid, rank_lo);
    if (frac <= 1e-6f || rank_hi >= sample_valid) {
        out_p90[out_idx] = v_lo;
    } else {
        float v_hi = select_kth_float(vals, sample_valid, rank_hi);
        out_p90[out_idx] = v_lo + frac * (v_hi - v_lo);
    }
}

// CPU-side: generate stratified random indices (same for all spatial points)
static void generate_mc_indices(int n_years, int window_days, int k_per_year,
                                  std::vector<int>& indices) {
    indices.clear();
    if (k_per_year <= 0) return;

    // Simple LCG for reproducibility
    uint64_t seed = 0x9E3779B97F4A7C15ULL;
    auto lcg = [&]() -> uint64_t {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        return seed;
    };

    for (int y = 0; y < n_years; ++y) {
        int year_start = y * window_days;
        for (int k = 0; k < k_per_year; ++k) {
            int d = (int)(lcg() % window_days);
            indices.push_back(year_start + d);
        }
    }
}

// Env-controlled MC mode
static int mc_k_from_env() {
    const char* env = std::getenv("MCC_MC_K");
    if (!env) return 0;
    int k = std::atoi(env);
    return (k > 0 && k <= 300) ? k : 0;
}

static bool mc_consistent_mode() {
    const char* env = std::getenv("MCC_MC_CONSISTENT");
    return env && env[0] != '0';
}

static bool mc_block_mode() {
    const char* env = std::getenv("MCC_MC_BLOCK");
    return env && env[0] != '0';
}

static bool mc_packed_mode() {
    const char* env = std::getenv("MCC_MC_PACKED");
    return env && env[0] != '0';
}

static int mc_base_doy_from_env() {
    const char* env = std::getenv("MCC_TARGET_DOY_BEGIN");
    if (!env) return TARGET_DOY_BEGIN;
    int doy = std::atoi(env);
    if (doy < TARGET_DOY_BEGIN || doy > TARGET_DOY_END) return TARGET_DOY_BEGIN;
    return doy;
}

static int normalize_mc_k(int requested_k) {
    if (requested_k <= 0) return 0;
    if (requested_k > 120) requested_k = 120;
    int k_per_year = requested_k / CLIM_YEARS;
    if (k_per_year <= 0) k_per_year = 1;
    return k_per_year * CLIM_YEARS;
}

// ============================================================
// Legacy: 4-DCU 异步调度 (single-shot, kept for backward compat)
// ============================================================

void dispatch_to_4_dcus(float* h_sst_data) {
    using Clock = std::chrono::steady_clock;
    const int days_total = static_cast<int>(DAYS_TOTAL);
    const int spatial_total = static_cast<int>(SPATIAL_POINTS);

    int gpu_count = 0;
    hipGetDeviceCount(&gpu_count);
    const int NUM_GPUS = gpu_count;
    if (NUM_GPUS <= 0) {
        std::cerr << "[DCU 模块] 未发现可用 DCU" << std::endl;
        std::exit(1);
    }

    int rows_per_gpu = (LAT_SIZE + NUM_GPUS - 1) / NUM_GPUS;

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

        HIP_CHECK(hipMalloc(&d_data[i], full_data_bytes));
        HIP_CHECK(hipMalloc(&d_mean[i], local_spatial * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_p90[i],  local_spatial * sizeof(float)));

        HIP_CHECK(hipMemcpyAsync(d_data[i], h_sst_data, full_data_bytes,
                                 hipMemcpyHostToDevice, streams[i]));
    }

    for (int i = 0; i < NUM_GPUS; ++i) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamSynchronize(streams[i]));
    }

    auto t1 = Clock::now();
    double transfer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

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
            spatial_start, spatial_start + local_spatial,
            spatial_total, 0);
    }

    for (int i = 0; i < NUM_GPUS; ++i) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamSynchronize(streams[i]));
    }

    t1 = Clock::now();
    double kernel_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

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

    for (int i = 0; i < NUM_GPUS; ++i) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamDestroy(streams[i]));
        HIP_CHECK(hipFree(d_data[i]));
        HIP_CHECK(hipFree(d_mean[i]));
        HIP_CHECK(hipFree(d_p90[i]));
    }

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

    double total_ms = transfer_ms + kernel_ms + gather_ms;
    std::cout << "[DCU 模块] 总计: " << total_ms << " ms"
              << " (传输=" << transfer_ms << " kernel=" << kernel_ms
              << " 回传=" << gather_ms << ")" << std::endl;

    HIP_CHECK(hipHostFree(h_mean));
    HIP_CHECK(hipHostFree(h_p90));
}

// ============================================================
// PERSISTENT DCU BUFFERS — with fp16 + dual-stream support
// ============================================================

static constexpr int MAX_PERSISTENT_GPUS = 8;

// --- global state ---
static bool g_persistent_initialized = false;
static bool g_use_fp16 = false;
static int g_persistent_gpu_count = 0;
static int g_persistent_rows_per_gpu = 0;
static int g_persistent_lat_start[MAX_PERSISTENT_GPUS] = {0};
static int g_persistent_lat_end[MAX_PERSISTENT_GPUS]   = {0};
static int g_persistent_local_rows[MAX_PERSISTENT_GPUS] = {0};

// Dual stream per GPU: h2d (upload+convert) + compute (kernel+download)
static hipStream_t g_h2d_stream[MAX_PERSISTENT_GPUS]     = {nullptr};
static hipStream_t g_compute_stream[MAX_PERSISTENT_GPUS] = {nullptr};

// Events for cross-stream sync: h2d_done signals upload complete
static hipEvent_t g_h2d_done[MAX_PERSISTENT_GPUS] = {nullptr};
static hipEvent_t g_source_upload_done[MAX_PERSISTENT_GPUS] = {nullptr};

static std::mutex g_source_upload_mutex;
static bool g_source_upload_pending = false;
static int g_source_upload_day0 = 0;
static int g_source_upload_day_count = 0;
static int g_source_upload_generation = 0;
static int g_source_upload_compute_wait_generation = -1;
static bool g_source_upload_keep_registered = false;
static bool g_source_upload_registered[MAX_PERSISTENT_GPUS] = {false};
static float* g_source_upload_registered_ptr[MAX_PERSISTENT_GPUS] = {nullptr};

static constexpr int MAX_SOURCE_UPLOAD_REGISTRATIONS = MAX_PERSISTENT_GPUS * 4;
static float* g_source_kept_registered_ptr[MAX_SOURCE_UPLOAD_REGISTRATIONS] = {nullptr};
static size_t g_source_kept_registered_bytes[MAX_SOURCE_UPLOAD_REGISTRATIONS] = {0};
static int g_source_kept_registered_count = 0;

// Persistent data buffers
//   float path: g_data[i] stores float data for p90 kernel
//   fp16 path:  g_data_fp16[i] stores __half (half VRAM), g_data[i] is conversion target
static float*  g_data[MAX_PERSISTENT_GPUS]      = {nullptr};
static __half* g_data_fp16[MAX_PERSISTENT_GPUS] = {nullptr};

// Double-buffered output: 2 sets of mean/p90 per GPU for D2H/H2D overlap
static float* g_mean_buf[2][MAX_PERSISTENT_GPUS] = {{nullptr}};
static float* g_p90_buf[2][MAX_PERSISTENT_GPUS]  = {{nullptr}};

// Ping-pong state: which buffer set is "current"
static int g_output_pingpong = 0;

// Host-side fp16 staging: pinned memory for float→half conversion before H2D
static __half* g_half_staging[MAX_PERSISTENT_GPUS] = {nullptr};
static size_t  g_half_staging_sz[MAX_PERSISTENT_GPUS] = {0};

// MC state: consistent sampling indices + mode
static int  g_mc_K = 0;
static bool g_mc_consistent = false;
static bool g_mc_block = false;
static bool g_mc_packed = false;
static int  g_mc_current_doy = TARGET_DOY_BEGIN;
static int* g_mc_indices[MAX_PERSISTENT_GPUS] = {nullptr};  // K ints per GPU
static std::vector<int> g_mc_indices_cpu;  // CPU-side master copy

// ============================================================
// Helpers
// ============================================================

static bool debug_enabled() {
    const char* env = std::getenv("MCC_DEBUG_VALIDATE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool fp16_enabled() {
    const char* env = std::getenv("MCC_USE_FP16");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool skip_zero_enabled() {
    const char* env = std::getenv("MCC_SKIP_ZERO");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool source_pinned_upload_enabled() {
    const char* env = std::getenv("MCC_SOURCE_PINNED_UPLOAD");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool source_register_upload_enabled() {
    const char* env = std::getenv("MCC_SOURCE_REGISTER_UPLOAD");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool source_keep_registered_enabled() {
    const char* env = std::getenv("MCC_SOURCE_KEEP_REGISTERED");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool source_fast_hist_enabled() {
    const char* env = std::getenv("MCC_SOURCE_FAST_HIST");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool source_fixed_collect_enabled() {
    const char* env = std::getenv("MCC_SOURCE_FIXED_COLLECT");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool source_select_p90_enabled() {
    const char* env = std::getenv("MCC_SOURCE_SELECT_P90");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static int source_mc_k_from_env() {
    const char* env = std::getenv("MCC_SOURCE_MC_K");
    if (!env) return 0;
    int k = std::atoi(env);
    if (k <= 0) return 0;
    if (k > 120) k = 120;
    if (k < CLIM_YEARS) return k;
    return normalize_mc_k(k);
}

static bool source_mc_independent_enabled() {
    const char* env = std::getenv("MCC_SOURCE_MC_INDEPENDENT");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool source_mc_flat_enabled() {
    const char* env = std::getenv("MCC_SOURCE_MC_FLAT");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool source_mc_exact_mean_enabled() {
    const char* env = std::getenv("MCC_SOURCE_MC_EXACT_MEAN");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static int source_upload_stage_slots() {
    int slots = 512;
    if (const char* env = std::getenv("MCC_UPLOAD_STAGE_SLOTS")) {
        int requested = std::atoi(env);
        if (requested > 0) slots = requested;
    }
    if (slots < 16) slots = 16;
    return slots;
}

static bool register_source_upload_host_ptr(float* ptr, size_t bytes, bool keep_registered) {
    if (!source_register_upload_enabled() || ptr == nullptr || bytes == 0) return false;

    if (!keep_registered) {
        return hipHostRegister(ptr, bytes, hipHostRegisterDefault) == hipSuccess;
    }

    {
        std::lock_guard<std::mutex> lock(g_source_upload_mutex);
        for (int i = 0; i < g_source_kept_registered_count; ++i) {
            if (g_source_kept_registered_ptr[i] == ptr) {
                return bytes <= g_source_kept_registered_bytes[i];
            }
        }
    }

    hipError_t reg_err = hipHostRegister(ptr, bytes, hipHostRegisterDefault);
    if (reg_err != hipSuccess) return false;

    {
        std::lock_guard<std::mutex> lock(g_source_upload_mutex);
        if (g_source_kept_registered_count >= MAX_SOURCE_UPLOAD_REGISTRATIONS) {
            HIP_CHECK(hipHostUnregister(ptr));
            return false;
        }
        int idx = g_source_kept_registered_count++;
        g_source_kept_registered_ptr[idx] = ptr;
        g_source_kept_registered_bytes[idx] = bytes;
    }
    return true;
}

static void unregister_kept_source_uploads() {
    float* ptrs[MAX_SOURCE_UPLOAD_REGISTRATIONS] = {nullptr};
    int count = 0;

    {
        std::lock_guard<std::mutex> lock(g_source_upload_mutex);
        count = g_source_kept_registered_count;
        for (int i = 0; i < count; ++i) {
            ptrs[i] = g_source_kept_registered_ptr[i];
            g_source_kept_registered_ptr[i] = nullptr;
            g_source_kept_registered_bytes[i] = 0;
        }
        g_source_kept_registered_count = 0;
    }

    for (int i = 0; i < count; ++i) {
        if (ptrs[i]) HIP_CHECK(hipHostUnregister(ptrs[i]));
    }
    if (count > 0) {
        std::cout << "[DCU module] source stream kept host registrations released="
                  << count << std::endl;
    }
}

static void print_field_stats(const char* name, const float* data, int n) {
    long long nan_count = 0, zero_count = 0, finite_count = 0;
    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < n; ++i) {
        float v = data[i];
        if (std::isnan(v)) { ++nan_count; continue; }
        ++finite_count;
        if (v == 0.0f) ++zero_count;
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
    }

    std::cout << "[debug] " << name
              << " finite=" << finite_count
              << " nan=" << nan_count
              << " zero=" << zero_count;
    if (finite_count > 0)
        std::cout << " min=" << min_v << " max=" << max_v;
    std::cout << std::endl;
}

// Convert float slice to __half on CPU (for fp16 H2D staging)
static void float_to_half_staging(
    __half* __restrict__ dst, const float* __restrict__ src,
    int rows, int src_stride, int dst_stride)
{
    for (int r = 0; r < rows; ++r) {
        const float* src_row = src + (size_t)r * src_stride;
        __half* dst_row = dst + (size_t)r * dst_stride;
        for (int c = 0; c < dst_stride; ++c) {
            dst_row[c] = __float2half(src_row[c]);
        }
    }
}

// ============================================================
// Persistent buffer init (supports fp16)
// ============================================================

static void init_persistent_dcu_buffers(int spatial_total, int days_total) {
    int gpu_count = 0;
    hipGetDeviceCount(&gpu_count);
    if (gpu_count <= 0) {
        std::cerr << "[DCU module] no available DCU" << std::endl;
        std::exit(1);
    }
    if (gpu_count > MAX_PERSISTENT_GPUS) {
        std::cerr << "[DCU module] gpu_count=" << gpu_count
                  << " exceeds MAX_PERSISTENT_GPUS=" << MAX_PERSISTENT_GPUS << std::endl;
        std::exit(1);
    }

    g_use_fp16 = fp16_enabled();
    g_persistent_gpu_count = gpu_count;
    g_persistent_rows_per_gpu = ((int)LAT_SIZE + g_persistent_gpu_count - 1) /
                                g_persistent_gpu_count;

    for (int i = 0; i < g_persistent_gpu_count; ++i) {
        HIP_CHECK(hipSetDevice(i));

        // Create dual streams
        HIP_CHECK(hipStreamCreate(&g_h2d_stream[i]));
        HIP_CHECK(hipStreamCreate(&g_compute_stream[i]));
        HIP_CHECK(hipEventCreate(&g_h2d_done[i]));
        HIP_CHECK(hipEventCreate(&g_source_upload_done[i]));

        g_persistent_lat_start[i] = i * g_persistent_rows_per_gpu;
        g_persistent_lat_end[i]   = std::min((i + 1) * g_persistent_rows_per_gpu, (int)LAT_SIZE);
        g_persistent_local_rows[i] = g_persistent_lat_end[i] - g_persistent_lat_start[i];
        if (g_persistent_local_rows[i] <= 0) continue;

        int local_spatial = g_persistent_local_rows[i] * LON_SIZE;

        if (g_use_fp16) {
            // fp16 path: DCU stores __half (half VRAM), convert to float in-place before kernel
            //   g_data_fp16[i]: persistent __half storage (330 days, half VRAM)
            //   g_data[i]:      float work buffer (330 days, same size as before)
            //   g_half_staging[i]: pinned host buffer for float→half conversion
            size_t data_half_bytes = (size_t)days_total * local_spatial * sizeof(__half);
            size_t data_float_bytes = (size_t)days_total * local_spatial * sizeof(float);

            HIP_CHECK(hipMalloc(&g_data_fp16[i], data_half_bytes));
            HIP_CHECK(hipMalloc(&g_data[i], data_float_bytes));

            // Host-side fp16 staging: only need CLIM_YEARS (30) slots for incremental
            // NOT days_total (330) — saves 11x pinned memory
            size_t stag_bytes = (size_t)CLIM_YEARS * local_spatial * sizeof(__half);
            g_half_staging_sz[i] = stag_bytes;
            HIP_CHECK(hipHostMalloc(&g_half_staging[i], stag_bytes, hipHostMallocDefault));

            std::cout << "[DCU module] fp16 enabled on DCU " << i
                      << " (VRAM data: " << (data_half_bytes >> 20) << " MB __half"
                      << " + " << (data_float_bytes >> 20) << " MB float)" << std::endl;
        } else {
            // Original float path
            size_t data_float_bytes = (size_t)days_total * local_spatial * sizeof(float);
            HIP_CHECK(hipMalloc(&g_data[i], data_float_bytes));
            g_data_fp16[i] = nullptr;
            g_half_staging[i] = nullptr;
            g_half_staging_sz[i] = 0;
        }

        // Double-buffered output (2 sets for D2H/H2D overlap)
        for (int b = 0; b < 2; ++b) {
            HIP_CHECK(hipMalloc(&g_mean_buf[b][i], local_spatial * sizeof(float)));
            HIP_CHECK(hipMalloc(&g_p90_buf[b][i],  local_spatial * sizeof(float)));
        }
    }

    // MC mode: allocate indices buffer on each GPU + generate indices on CPU
    g_mc_K = normalize_mc_k(mc_k_from_env());
    g_mc_consistent = mc_consistent_mode();
    g_mc_block = mc_block_mode();
    g_mc_packed = mc_packed_mode();
    g_mc_current_doy = mc_base_doy_from_env();
    if (g_mc_K > 0 && g_mc_consistent && !g_mc_packed) {
        generate_mc_indices(CLIM_YEARS, 2 * CLIM_DELTA_DAY + 1,
                            g_mc_K / CLIM_YEARS, g_mc_indices_cpu);
        g_mc_K = (int)g_mc_indices_cpu.size();  // actual K (may differ from requested)
        for (int i = 0; i < g_persistent_gpu_count; ++i) {
            HIP_CHECK(hipSetDevice(i));
            HIP_CHECK(hipMalloc(&g_mc_indices[i], g_mc_K * sizeof(int)));
            HIP_CHECK(hipMemcpy(g_mc_indices[i], g_mc_indices_cpu.data(),
                                g_mc_K * sizeof(int), hipMemcpyHostToDevice));
        }
    }

    std::cout << "[DCU module] persistent buffers initialized on "
              << g_persistent_gpu_count << " DCU(s); rows_per_gpu="
              << g_persistent_rows_per_gpu
              << " fp16=" << (g_use_fp16 ? "ON" : "OFF")
              << " mc_K=" << g_mc_K << " mc_consistent=" << (g_mc_consistent ? "ON" : "OFF")
              << " mc_block=" << (g_mc_block ? "ON" : "OFF")
              << " mc_packed=" << (g_mc_packed ? "ON" : "OFF")
              << " dual_stream=ON double_buffer=ON"
              << std::endl;
    g_persistent_initialized = true;
}

// ============================================================
// Main dispatch: fp16 H2D + dual-stream pipeline
// ============================================================

void dispatch_to_4_dcus_with_output_incremental_hook(
    float* h_sst_data, float* h_src_data,
    float* h_mean, float* h_p90,
    const int* changed_slots, int num_changed,
    void (*after_upload)(void*),
    void* after_upload_user)
{
    using Clock = std::chrono::steady_clock;
    const int days_total   = static_cast<int>(DAYS_TOTAL);
    const int spatial_total = static_cast<int>(SPATIAL_POINTS);

    if (!g_persistent_initialized) {
        init_persistent_dcu_buffers(spatial_total, days_total);
    }

    auto total_t0 = Clock::now();
    bool full_upload = (changed_slots == nullptr || num_changed <= 0);

    // Toggle ping-pong output buffer
    int out_cur = g_output_pingpong;        // current output target
    int out_prev = 1 - g_output_pingpong;   // previous (safe to reuse)

    // ---- STEP 1: H2D upload on h2d_stream (all GPUs in parallel) ----
    auto upload_t0 = Clock::now();

    for (int i = 0; i < g_persistent_gpu_count; ++i) {
        if (g_persistent_local_rows[i] <= 0) continue;
        HIP_CHECK(hipSetDevice(i));

        int local_spatial = g_persistent_local_rows[i] * LON_SIZE;
        int spatial_start = g_persistent_lat_start[i] * LON_SIZE;

        // ---- FP16 PATH (incremental only — full upload uses float for speed) ----
        if (g_use_fp16 && g_data_fp16[i] && !full_upload && num_changed <= CLIM_YEARS) {
            // Batch all changed_slots: single OpenMP region → single H2D block
            const float* h_src = h_src_data ? h_src_data : h_sst_data;
            int total_cells = num_changed * local_spatial;

            // Single OpenMP parallel float→half conversion (all slots at once)
            #pragma omp parallel for schedule(static)
            for (int c = 0; c < total_cells; ++c) {
                int slot_idx = c / local_spatial;
                int cell     = c % local_spatial;
                int slot     = changed_slots[slot_idx];
                g_half_staging[i][c] = __float2half(
                    h_src[(size_t)slot * spatial_total + spatial_start + cell]);
            }

            // Single bulk H2D upload: all changed slots as half
            // width = one row (local_spatial elements × 2 bytes), height = num_changed slots
            size_t half_row_bytes = (size_t)local_spatial * sizeof(__half);
            HIP_CHECK(hipMemcpy2DAsync(
                g_data_fp16[i] + (size_t)changed_slots[0] * local_spatial,
                half_row_bytes,
                g_half_staging[i],
                half_row_bytes,
                half_row_bytes, num_changed,
                hipMemcpyHostToDevice, g_h2d_stream[i]));
        } else if (g_use_fp16 && g_data_fp16[i] && full_upload) {
            // Full upload (Day 152): use float path for speed
            // float→half conversion of 330 slots on CPU is too slow
            size_t local_row_bytes = (size_t)local_spatial * sizeof(float);
            HIP_CHECK(hipMemcpy2DAsync(g_data[i], local_row_bytes,
                                       h_sst_data + spatial_start,
                                       (size_t)spatial_total * sizeof(float),
                                       local_row_bytes, days_total,
                                       hipMemcpyHostToDevice, g_h2d_stream[i]));
            // Convert float→half on DCU (single kernel, ~1ms)
            {
                int total = days_total * local_spatial;
                int block = 256;
                int grid  = (total + block - 1) / block;
                float_to_half_kernel<<<dim3(grid), dim3(block), 0, g_h2d_stream[i]>>>(
                    g_data[i], g_data_fp16[i], total);
            }
            HIP_CHECK(hipPeekAtLastError());
            // Now g_data_fp16 has the half data; g_data will be repopulated
            // from g_data_fp16 in the conversion step below
        } else {
            // ---- FLOAT PATH (original, or fp16 disabled) ----
            size_t local_row_bytes = (size_t)local_spatial * sizeof(float);

            if (full_upload) {
                HIP_CHECK(hipMemcpy2DAsync(g_data[i], local_row_bytes,
                                           h_sst_data + spatial_start,
                                           (size_t)spatial_total * sizeof(float),
                                           local_row_bytes, days_total,
                                           hipMemcpyHostToDevice, g_h2d_stream[i]));
            } else {
                bool regular_stride = (num_changed > 0);
                for (int j = 1; j < num_changed; ++j) {
                    if (changed_slots[j] - changed_slots[j - 1] != 1) {
                        regular_stride = false; break;
                    }
                }

                if (regular_stride) {
                    int first_slot = changed_slots[0];
                    const float* src = h_src_data
                        ? h_src_data + spatial_start
                        : h_sst_data + (size_t)first_slot * spatial_total + spatial_start;
                    HIP_CHECK(hipMemcpy2DAsync(
                        g_data[i] + (size_t)first_slot * local_spatial,
                        local_row_bytes, src,
                        (size_t)spatial_total * sizeof(float),
                        local_row_bytes, num_changed,
                        hipMemcpyHostToDevice, g_h2d_stream[i]));
                } else {
                    const float* h_src = h_src_data ? h_src_data : h_sst_data;
                    for (int j = 0; j < num_changed; ++j) {
                        int slot = changed_slots[j];
                        HIP_CHECK(hipMemcpyAsync(
                            g_data[i] + (size_t)slot * local_spatial,
                            h_src + (size_t)slot * spatial_total + spatial_start,
                            local_row_bytes,
                            hipMemcpyHostToDevice, g_h2d_stream[i]));
                    }
                }
            }
        }
    }

    // ---- STEP 1b: fp16 conversion (h2d_stream, after upload) ----
    // For full upload: g_data already has float from the upload, skip conversion
    // For incremental: convert half→float for the changed slots
    if (g_use_fp16 && !full_upload && num_changed <= CLIM_YEARS) {
        for (int i = 0; i < g_persistent_gpu_count; ++i) {
            if (g_persistent_local_rows[i] <= 0) continue;
            HIP_CHECK(hipSetDevice(i));

            int local_spatial = g_persistent_local_rows[i] * LON_SIZE;

            // Single bulk kernel: convert all changed slots from __half to float
            int block = 256;
            int total = num_changed * local_spatial;
            int grid  = (total + block - 1) / block;
            fp16_to_fp32_2d_kernel<<<dim3(grid), dim3(block), 0, g_h2d_stream[i]>>>(
                g_data_fp16[i] + (size_t)changed_slots[0] * local_spatial,
                g_data[i] + (size_t)changed_slots[0] * local_spatial,
                num_changed, local_spatial, local_spatial);
            HIP_CHECK(hipPeekAtLastError());
        }
    }

    // Record h2d_done event on h2d_stream (compute_stream will wait)
    for (int i = 0; i < g_persistent_gpu_count; ++i) {
        if (g_persistent_local_rows[i] <= 0) continue;
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipEventRecord(g_h2d_done[i], g_h2d_stream[i]));
    }

    auto upload_t1 = Clock::now();
    double upload_ms = std::chrono::duration<double, std::milli>(upload_t1 - upload_t0).count();
    double elapsed_since_start = std::chrono::duration<double, std::milli>(upload_t1 - total_t0).count();
    if (g_use_fp16)
        std::cerr << "[DBG] after_upload_launch total_elapsed=" << elapsed_since_start
                  << "ms upload=" << upload_ms << "ms" << std::endl;

    // ---- after_upload hook (triggers async prefetch for next day) ----
    if (after_upload) {
        after_upload(after_upload_user);
    }
    auto after_hook_t = Clock::now();
    if (g_use_fp16)
        std::cerr << "[DBG] after_hook total_elapsed="
                  << std::chrono::duration<double, std::milli>(after_hook_t - total_t0).count()
                  << "ms" << std::endl;

    // ---- STEP 2: Compute on compute_stream (waits for h2d_done) ----
    auto kernel_t0 = Clock::now();

    for (int i = 0; i < g_persistent_gpu_count; ++i) {
        if (g_persistent_local_rows[i] <= 0) continue;
        HIP_CHECK(hipSetDevice(i));

        // compute_stream must wait for h2d_stream to finish
        HIP_CHECK(hipStreamWaitEvent(g_compute_stream[i], g_h2d_done[i], 0));

        int local_spatial = g_persistent_local_rows[i] * LON_SIZE;
        int spatial_start = g_persistent_lat_start[i] * LON_SIZE;
        int block_size = 256;
        int grid_size  = (local_spatial + block_size - 1) / block_size;

        if (g_mc_K > 0) {
            // === Monte Carlo P90 kernel ===
            if (g_mc_packed) {
                mc_p90_packed_kernel<<<dim3(grid_size), dim3(block_size), 0, g_compute_stream[i]>>>(
                    g_data[i], g_mc_K,
                    g_mean_buf[out_cur][i], g_p90_buf[out_cur][i],
                    spatial_total,
                    spatial_start, spatial_start + local_spatial,
                    local_spatial);
            } else if (g_mc_block) {
                mc_p90_block_xorshift_kernel<<<dim3(grid_size), dim3(block_size), 0, g_compute_stream[i]>>>(
                    g_data[i], g_mc_K, g_mc_current_doy, days_total,
                    CLIM_YEARS, 2 * CLIM_DELTA_DAY + 1,
                    g_mean_buf[out_cur][i], g_p90_buf[out_cur][i],
                    spatial_total,
                    spatial_start, spatial_start + local_spatial,
                    local_spatial);
            } else if (g_mc_consistent) {
                mc_p90_consistent_kernel<<<dim3(grid_size), dim3(block_size), 0, g_compute_stream[i]>>>(
                    g_data[i], g_mc_indices[i], g_mc_K,
                    g_mean_buf[out_cur][i], g_p90_buf[out_cur][i],
                    spatial_total,
                    spatial_start, spatial_start + local_spatial,
                    local_spatial);
            } else {
                // Independent sampling: each point has own random indices
                mc_p90_independent_kernel<<<dim3(grid_size), dim3(block_size), 0, g_compute_stream[i]>>>(
                    g_data[i], g_mc_K, g_mc_current_doy, days_total,
                    g_mean_buf[out_cur][i], g_p90_buf[out_cur][i],
                    spatial_total,
                    spatial_start, spatial_start + local_spatial,
                    local_spatial);
            }
        } else {
            // === Exact P90 kernel (original) ===
            compute_mean_p90_kernel<<<dim3(grid_size), dim3(block_size), 0, g_compute_stream[i]>>>(
                g_data[i], g_mean_buf[out_cur][i], g_p90_buf[out_cur][i],
                spatial_total, days_total,
                spatial_start, spatial_start + local_spatial,
                local_spatial, spatial_start);
        }
        HIP_CHECK(hipPeekAtLastError());
    }

    // ---- STEP 3: D2H download on compute_stream ----
    for (int i = 0; i < g_persistent_gpu_count; ++i) {
        if (g_persistent_local_rows[i] <= 0) continue;
        HIP_CHECK(hipSetDevice(i));

        int local_spatial = g_persistent_local_rows[i] * LON_SIZE;
        int host_offset   = g_persistent_lat_start[i] * LON_SIZE;

        HIP_CHECK(hipMemcpyAsync(h_mean + host_offset, g_mean_buf[out_cur][i],
                                 local_spatial * sizeof(float),
                                 hipMemcpyDeviceToHost, g_compute_stream[i]));
        HIP_CHECK(hipMemcpyAsync(h_p90 + host_offset, g_p90_buf[out_cur][i],
                                 local_spatial * sizeof(float),
                                 hipMemcpyDeviceToHost, g_compute_stream[i]));
    }

    // Sync compute_stream across all GPUs
    for (int i = 0; i < g_persistent_gpu_count; ++i) {
        if (g_persistent_local_rows[i] <= 0) continue;
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamSynchronize(g_compute_stream[i]));
    }

    auto kernel_t1 = Clock::now();
    double compute_ms = std::chrono::duration<double, std::milli>(kernel_t1 - kernel_t0).count();

    // Toggle ping-pong for next call
    g_output_pingpong = 1 - g_output_pingpong;

    auto total_t1 = Clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(total_t1 - total_t0).count();

    // Log: h2d and compute are now measured separately and may overlap
    // across consecutive calls via the double-buffered output
    std::cout << "[DCU module] upload=" << upload_ms
              << "ms (" << (full_upload ? "full" : "incremental")
              << ", slots=" << (full_upload ? days_total : num_changed)
              << ", fp16=" << (g_use_fp16 ? "ON" : "OFF")
              << ", mc=" << (g_mc_K > 0 ? std::to_string(g_mc_K) : "OFF")
              << ", mc_mode=" << (g_mc_K <= 0 ? "OFF" : (g_mc_packed ? "PACKED" : (g_mc_block ? "BLOCK" : (g_mc_consistent ? "CONSISTENT" : "INDEPENDENT"))))
              << ", mc_doy=" << (g_mc_K > 0 ? std::to_string(g_mc_current_doy) : "-")
              << ") compute=" << compute_ms
              << "ms total=" << total_ms << "ms" << std::endl;

    if (debug_enabled()) {
        print_field_stats("gpu_mean_after_d2h", h_mean, spatial_total);
        print_field_stats("gpu_p90_after_d2h", h_p90, spatial_total);
    }

    if (g_mc_K > 0) {
        ++g_mc_current_doy;
    }
}

void dispatch_to_4_dcus_with_output_incremental(float* h_sst_data, float* h_mean, float* h_p90,
                                                const int* changed_slots, int num_changed) {
    dispatch_to_4_dcus_with_output_incremental_hook(
        h_sst_data, nullptr, h_mean, h_p90, changed_slots, num_changed, nullptr, nullptr);
}

void dispatch_source_window_preloaded(float* h_source_data, int source_days, int target_offset,
                                      float* h_mean, float* h_p90) {
    using Clock = std::chrono::steady_clock;
    const int spatial_total = static_cast<int>(SPATIAL_POINTS);
    const int source_slots = source_days * CLIM_YEARS;
    const bool skip_zero = skip_zero_enabled();
    const bool fast_hist = source_fast_hist_enabled();
    const bool fixed_collect = source_fixed_collect_enabled();
    const bool select_p90 = source_select_p90_enabled();
    const int source_mc_k = source_mc_k_from_env();
    const bool source_mc_independent = source_mc_independent_enabled();
    const bool source_mc_flat = source_mc_flat_enabled() || (source_mc_k > 0 && source_mc_k < CLIM_YEARS);
    const bool source_mc_exact_mean = source_mc_exact_mean_enabled();

    if (!g_persistent_initialized) {
        init_persistent_dcu_buffers(spatial_total, source_slots);
    }

    auto total_t0 = Clock::now();
    double upload_ms = 0.0;

    if (target_offset == 0) {
        auto upload_t0 = Clock::now();
        bool used_pinned_stage = false;

        if (source_pinned_upload_enabled()) {
            int stage_slots = std::min(source_upload_stage_slots(), source_slots);
            std::vector<float*> h_stage(g_persistent_gpu_count, nullptr);
            bool stage_ready = false;

            while (stage_slots >= 16 && !stage_ready) {
                stage_ready = true;
                for (int i = 0; i < g_persistent_gpu_count; ++i) {
                    if (g_persistent_local_rows[i] <= 0) continue;
                    int local_spatial = g_persistent_local_rows[i] * LON_SIZE;
                    size_t stage_bytes = (size_t)stage_slots * local_spatial * sizeof(float);
                    HIP_CHECK(hipSetDevice(i));
                    hipError_t err = hipHostMalloc(&h_stage[i], stage_bytes, hipHostMallocDefault);
                    if (err != hipSuccess) {
                        stage_ready = false;
                        for (float*& p : h_stage) {
                            if (p) {
                                hipHostFree(p);
                                p = nullptr;
                            }
                        }
                        stage_slots /= 2;
                        break;
                    }
                }
            }

            if (stage_ready) {
                used_pinned_stage = true;
                std::cout << "[DCU module] source upload uses pinned staging, stage_slots="
                          << stage_slots << std::endl;

                for (int slot0 = 0; slot0 < source_slots; slot0 += stage_slots) {
                    int slots_this = std::min(stage_slots, source_slots - slot0);

                    #pragma omp parallel for collapse(2) schedule(static)
                    for (int i = 0; i < g_persistent_gpu_count; ++i) {
                        for (int s = 0; s < slots_this; ++s) {
                            if (g_persistent_local_rows[i] <= 0) continue;
                            int local_spatial = g_persistent_local_rows[i] * LON_SIZE;
                            int spatial_start = g_persistent_lat_start[i] * LON_SIZE;
                            const float* src = h_source_data
                                + (size_t)(slot0 + s) * spatial_total + spatial_start;
                            float* dst = h_stage[i] + (size_t)s * local_spatial;
                            std::memcpy(dst, src, (size_t)local_spatial * sizeof(float));
                        }
                    }

                    for (int i = 0; i < g_persistent_gpu_count; ++i) {
                        if (g_persistent_local_rows[i] <= 0) continue;
                        HIP_CHECK(hipSetDevice(i));
                        int local_spatial = g_persistent_local_rows[i] * LON_SIZE;
                        size_t bytes = (size_t)slots_this * local_spatial * sizeof(float);
                        HIP_CHECK(hipMemcpyAsync(
                            g_data[i] + (size_t)slot0 * local_spatial,
                            h_stage[i], bytes,
                            hipMemcpyHostToDevice, g_h2d_stream[i]));
                    }
                    for (int i = 0; i < g_persistent_gpu_count; ++i) {
                        if (g_persistent_local_rows[i] <= 0) continue;
                        HIP_CHECK(hipSetDevice(i));
                        HIP_CHECK(hipStreamSynchronize(g_h2d_stream[i]));
                    }
                }

                for (float*& p : h_stage) {
                    if (p) {
                        HIP_CHECK(hipHostFree(p));
                        p = nullptr;
                    }
                }
            } else {
                std::cerr << "[DCU module] pinned staging allocation failed; fallback to pageable 2D upload"
                          << std::endl;
            }
        }

        if (!used_pinned_stage) {
            bool registered_source = false;
            if (source_register_upload_enabled()) {
                size_t source_bytes = (size_t)source_slots * spatial_total * sizeof(float);
                hipError_t reg_err = hipHostRegister(h_source_data, source_bytes, hipHostRegisterDefault);
                if (reg_err == hipSuccess) {
                    registered_source = true;
                    std::cout << "[DCU module] source upload registered host buffer, bytes="
                              << source_bytes << std::endl;
                } else {
                    std::cerr << "[DCU module] hipHostRegister source buffer failed: "
                              << hipGetErrorString(reg_err)
                              << "; fallback to pageable 2D upload" << std::endl;
                }
            }

            for (int i = 0; i < g_persistent_gpu_count; ++i) {
                if (g_persistent_local_rows[i] <= 0) continue;
                HIP_CHECK(hipSetDevice(i));

                int local_spatial = g_persistent_local_rows[i] * LON_SIZE;
                int spatial_start = g_persistent_lat_start[i] * LON_SIZE;
                size_t local_row_bytes = (size_t)local_spatial * sizeof(float);

                HIP_CHECK(hipMemcpy2DAsync(g_data[i], local_row_bytes,
                                           h_source_data + spatial_start,
                                           (size_t)spatial_total * sizeof(float),
                                           local_row_bytes, source_slots,
                                           hipMemcpyHostToDevice, g_h2d_stream[i]));
            }
            for (int i = 0; i < g_persistent_gpu_count; ++i) {
                if (g_persistent_local_rows[i] <= 0) continue;
                HIP_CHECK(hipSetDevice(i));
                HIP_CHECK(hipStreamSynchronize(g_h2d_stream[i]));
            }
            if (registered_source) {
                HIP_CHECK(hipHostUnregister(h_source_data));
            }
        }
        upload_ms = std::chrono::duration<double, std::milli>(Clock::now() - upload_t0).count();
    }

    int out_cur = g_output_pingpong;
    auto kernel_t0 = Clock::now();

    for (int i = 0; i < g_persistent_gpu_count; ++i) {
        if (g_persistent_local_rows[i] <= 0) continue;
        HIP_CHECK(hipSetDevice(i));

        int local_spatial = g_persistent_local_rows[i] * LON_SIZE;
        int spatial_start = g_persistent_lat_start[i] * LON_SIZE;
        int block_size = 256;
        int grid_size = (local_spatial + block_size - 1) / block_size;

        if (source_mc_k > 0) {
            if (source_mc_exact_mean) {
                compute_source_window_mc_kernel<<<dim3(grid_size), dim3(block_size), 0, g_compute_stream[i]>>>(
                    g_data[i],
                    g_mean_buf[out_cur][i], g_p90_buf[out_cur][i],
                    target_offset, CLIM_YEARS, 2 * CLIM_DELTA_DAY + 1,
                    spatial_total, spatial_start, spatial_start + local_spatial,
                    local_spatial, skip_zero, source_mc_k, source_mc_independent,
                    source_mc_exact_mean);
            } else {
                compute_source_window_mc_sample_kernel<<<dim3(grid_size), dim3(block_size), 0, g_compute_stream[i]>>>(
                    g_data[i],
                    g_mean_buf[out_cur][i], g_p90_buf[out_cur][i],
                    target_offset, CLIM_YEARS, 2 * CLIM_DELTA_DAY + 1,
                    spatial_total, spatial_start, spatial_start + local_spatial,
                    local_spatial, skip_zero, source_mc_k, source_mc_independent,
                    source_mc_flat);
            }
        } else if (fixed_collect) {
            compute_source_window_fixed_collect_kernel<<<dim3(grid_size), dim3(block_size), 0, g_compute_stream[i]>>>(
                g_data[i],
                g_mean_buf[out_cur][i], g_p90_buf[out_cur][i],
                target_offset, CLIM_YEARS, 2 * CLIM_DELTA_DAY + 1,
                spatial_total, spatial_start, spatial_start + local_spatial,
                local_spatial, skip_zero);
        } else if (fast_hist) {
            compute_source_window_fast_hist_kernel<<<dim3(grid_size), dim3(block_size), 0, g_compute_stream[i]>>>(
                g_data[i],
                g_mean_buf[out_cur][i], g_p90_buf[out_cur][i],
                target_offset, CLIM_YEARS, 2 * CLIM_DELTA_DAY + 1,
                spatial_total, spatial_start, spatial_start + local_spatial,
                local_spatial, skip_zero);
        } else {
            compute_source_window_kernel<<<dim3(grid_size), dim3(block_size), 0, g_compute_stream[i]>>>(
                g_data[i],
                g_mean_buf[out_cur][i], g_p90_buf[out_cur][i],
                target_offset, CLIM_YEARS, 2 * CLIM_DELTA_DAY + 1,
                spatial_total, spatial_start, spatial_start + local_spatial,
                local_spatial, skip_zero, select_p90);
        }
        HIP_CHECK(hipPeekAtLastError());
    }

    for (int i = 0; i < g_persistent_gpu_count; ++i) {
        if (g_persistent_local_rows[i] <= 0) continue;
        HIP_CHECK(hipSetDevice(i));

        int local_spatial = g_persistent_local_rows[i] * LON_SIZE;
        int host_offset = g_persistent_lat_start[i] * LON_SIZE;

        HIP_CHECK(hipMemcpyAsync(h_mean + host_offset, g_mean_buf[out_cur][i],
                                 local_spatial * sizeof(float),
                                 hipMemcpyDeviceToHost, g_compute_stream[i]));
        HIP_CHECK(hipMemcpyAsync(h_p90 + host_offset, g_p90_buf[out_cur][i],
                                 local_spatial * sizeof(float),
                                 hipMemcpyDeviceToHost, g_compute_stream[i]));
    }

    for (int i = 0; i < g_persistent_gpu_count; ++i) {
        if (g_persistent_local_rows[i] <= 0) continue;
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamSynchronize(g_compute_stream[i]));
    }

    double compute_ms = std::chrono::duration<double, std::milli>(Clock::now() - kernel_t0).count();
    g_output_pingpong = 1 - g_output_pingpong;

    double total_ms = std::chrono::duration<double, std::milli>(Clock::now() - total_t0).count();
    std::cout << "[DCU module] source_window upload=" << upload_ms
              << "ms (source_slots=" << source_slots
              << ", target_offset=" << target_offset
              << ", skip_zero=" << (skip_zero ? "ON" : "OFF")
              << ", fast_hist=" << (fast_hist ? "ON" : "OFF")
              << ", fixed_collect=" << (fixed_collect ? "ON" : "OFF")
              << ", select_p90=" << (select_p90 ? "ON" : "OFF")
              << ", source_mc=" << (source_mc_k > 0 ? std::to_string(source_mc_k) : "OFF")
              << ", source_mc_mode=" << (source_mc_k > 0 ? (source_mc_independent ? "INDEPENDENT" : "BLOCK") : "OFF")
              << ", source_mc_sample=" << (source_mc_k > 0 ? (source_mc_flat ? "FLAT" : "PER_YEAR") : "OFF")
              << ", source_mc_mean=" << (source_mc_k > 0 ? (source_mc_exact_mean ? "EXACT" : "SAMPLE") : "OFF")
              << ") compute=" << compute_ms
              << "ms total=" << total_ms << "ms" << std::endl;
}

void dispatch_source_window_preloaded_local(float** h_source_local, int source_days, int target_offset,
                                            float* h_mean, float* h_p90) {
    using Clock = std::chrono::steady_clock;
    const int spatial_total = static_cast<int>(SPATIAL_POINTS);
    const int source_slots = source_days * CLIM_YEARS;
    const bool skip_zero = skip_zero_enabled();
    const bool fast_hist = source_fast_hist_enabled();
    const bool fixed_collect = source_fixed_collect_enabled();
    const bool select_p90 = source_select_p90_enabled();
    const int source_mc_k = source_mc_k_from_env();
    const bool source_mc_independent = source_mc_independent_enabled();
    const bool source_mc_flat = source_mc_flat_enabled() || (source_mc_k > 0 && source_mc_k < CLIM_YEARS);
    const bool source_mc_exact_mean = source_mc_exact_mean_enabled();

    if (!g_persistent_initialized) {
        init_persistent_dcu_buffers(spatial_total, source_slots);
    }

    auto total_t0 = Clock::now();
    double upload_ms = 0.0;

    if (target_offset == 0) {
        auto upload_t0 = Clock::now();
        std::vector<bool> registered(g_persistent_gpu_count, false);

        if (source_register_upload_enabled()) {
            for (int i = 0; i < g_persistent_gpu_count; ++i) {
                if (g_persistent_local_rows[i] <= 0) continue;
                int local_spatial = g_persistent_local_rows[i] * LON_SIZE;
                size_t bytes = static_cast<size_t>(source_slots) * local_spatial * sizeof(float);
                hipError_t reg_err = hipHostRegister(h_source_local[i], bytes, hipHostRegisterDefault);
                if (reg_err == hipSuccess) {
                    registered[i] = true;
                } else {
                    std::cerr << "[DCU module] local hipHostRegister failed on part "
                              << i << ": " << hipGetErrorString(reg_err)
                              << "; upload will use pageable host memory" << std::endl;
                }
            }
            std::cout << "[DCU module] source local upload registered parts="
                      << std::count(registered.begin(), registered.end(), true)
                      << "/" << g_persistent_gpu_count << std::endl;
        }

        for (int i = 0; i < g_persistent_gpu_count; ++i) {
            if (g_persistent_local_rows[i] <= 0) continue;
            HIP_CHECK(hipSetDevice(i));

            int local_spatial = g_persistent_local_rows[i] * LON_SIZE;
            size_t bytes = static_cast<size_t>(source_slots) * local_spatial * sizeof(float);
            HIP_CHECK(hipMemcpyAsync(g_data[i], h_source_local[i], bytes,
                                     hipMemcpyHostToDevice, g_h2d_stream[i]));
        }
        for (int i = 0; i < g_persistent_gpu_count; ++i) {
            if (g_persistent_local_rows[i] <= 0) continue;
            HIP_CHECK(hipSetDevice(i));
            HIP_CHECK(hipStreamSynchronize(g_h2d_stream[i]));
        }
        for (int i = 0; i < g_persistent_gpu_count; ++i) {
            if (registered[i]) {
                HIP_CHECK(hipHostUnregister(h_source_local[i]));
            }
        }
        upload_ms = std::chrono::duration<double, std::milli>(Clock::now() - upload_t0).count();
    }

    int out_cur = g_output_pingpong;
    auto kernel_t0 = Clock::now();

    for (int i = 0; i < g_persistent_gpu_count; ++i) {
        if (g_persistent_local_rows[i] <= 0) continue;
        HIP_CHECK(hipSetDevice(i));

        int local_spatial = g_persistent_local_rows[i] * LON_SIZE;
        int spatial_start = g_persistent_lat_start[i] * LON_SIZE;
        int block_size = 256;
        int grid_size = (local_spatial + block_size - 1) / block_size;

        if (source_mc_k > 0) {
            if (source_mc_exact_mean) {
                compute_source_window_mc_kernel<<<dim3(grid_size), dim3(block_size), 0, g_compute_stream[i]>>>(
                    g_data[i],
                    g_mean_buf[out_cur][i], g_p90_buf[out_cur][i],
                    target_offset, CLIM_YEARS, 2 * CLIM_DELTA_DAY + 1,
                    spatial_total, spatial_start, spatial_start + local_spatial,
                    local_spatial, skip_zero, source_mc_k, source_mc_independent,
                    source_mc_exact_mean);
            } else {
                compute_source_window_mc_sample_kernel<<<dim3(grid_size), dim3(block_size), 0, g_compute_stream[i]>>>(
                    g_data[i],
                    g_mean_buf[out_cur][i], g_p90_buf[out_cur][i],
                    target_offset, CLIM_YEARS, 2 * CLIM_DELTA_DAY + 1,
                    spatial_total, spatial_start, spatial_start + local_spatial,
                    local_spatial, skip_zero, source_mc_k, source_mc_independent,
                    source_mc_flat);
            }
        } else if (fixed_collect) {
            compute_source_window_fixed_collect_kernel<<<dim3(grid_size), dim3(block_size), 0, g_compute_stream[i]>>>(
                g_data[i],
                g_mean_buf[out_cur][i], g_p90_buf[out_cur][i],
                target_offset, CLIM_YEARS, 2 * CLIM_DELTA_DAY + 1,
                spatial_total, spatial_start, spatial_start + local_spatial,
                local_spatial, skip_zero);
        } else if (fast_hist) {
            compute_source_window_fast_hist_kernel<<<dim3(grid_size), dim3(block_size), 0, g_compute_stream[i]>>>(
                g_data[i],
                g_mean_buf[out_cur][i], g_p90_buf[out_cur][i],
                target_offset, CLIM_YEARS, 2 * CLIM_DELTA_DAY + 1,
                spatial_total, spatial_start, spatial_start + local_spatial,
                local_spatial, skip_zero);
        } else {
            compute_source_window_kernel<<<dim3(grid_size), dim3(block_size), 0, g_compute_stream[i]>>>(
                g_data[i],
                g_mean_buf[out_cur][i], g_p90_buf[out_cur][i],
                target_offset, CLIM_YEARS, 2 * CLIM_DELTA_DAY + 1,
                spatial_total, spatial_start, spatial_start + local_spatial,
                local_spatial, skip_zero, select_p90);
        }
        HIP_CHECK(hipPeekAtLastError());
    }

    for (int i = 0; i < g_persistent_gpu_count; ++i) {
        if (g_persistent_local_rows[i] <= 0) continue;
        HIP_CHECK(hipSetDevice(i));

        int local_spatial = g_persistent_local_rows[i] * LON_SIZE;
        int host_offset = g_persistent_lat_start[i] * LON_SIZE;

        HIP_CHECK(hipMemcpyAsync(h_mean + host_offset, g_mean_buf[out_cur][i],
                                 local_spatial * sizeof(float),
                                 hipMemcpyDeviceToHost, g_compute_stream[i]));
        HIP_CHECK(hipMemcpyAsync(h_p90 + host_offset, g_p90_buf[out_cur][i],
                                 local_spatial * sizeof(float),
                                 hipMemcpyDeviceToHost, g_compute_stream[i]));
    }

    for (int i = 0; i < g_persistent_gpu_count; ++i) {
        if (g_persistent_local_rows[i] <= 0) continue;
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamSynchronize(g_compute_stream[i]));
    }

    double compute_ms = std::chrono::duration<double, std::milli>(Clock::now() - kernel_t0).count();
    g_output_pingpong = 1 - g_output_pingpong;

    double total_ms = std::chrono::duration<double, std::milli>(Clock::now() - total_t0).count();
    std::cout << "[DCU module] source_window_local upload=" << upload_ms
              << "ms (source_slots=" << source_slots
              << ", target_offset=" << target_offset
              << ", skip_zero=" << (skip_zero ? "ON" : "OFF")
              << ", fast_hist=" << (fast_hist ? "ON" : "OFF")
              << ", fixed_collect=" << (fixed_collect ? "ON" : "OFF")
              << ", select_p90=" << (select_p90 ? "ON" : "OFF")
              << ", source_mc=" << (source_mc_k > 0 ? std::to_string(source_mc_k) : "OFF")
              << ", source_mc_mode=" << (source_mc_k > 0 ? (source_mc_independent ? "INDEPENDENT" : "BLOCK") : "OFF")
              << ", source_mc_sample=" << (source_mc_k > 0 ? (source_mc_flat ? "FLAT" : "PER_YEAR") : "OFF")
              << ", source_mc_mean=" << (source_mc_k > 0 ? (source_mc_exact_mean ? "EXACT" : "SAMPLE") : "OFF")
              << ") compute=" << compute_ms
              << "ms total=" << total_ms << "ms" << std::endl;
}

void init_source_window_device_preloaded(int source_days) {
    const int spatial_total = static_cast<int>(SPATIAL_POINTS);
    const int source_slots = source_days * CLIM_YEARS;
    if (!g_persistent_initialized) {
        init_persistent_dcu_buffers(spatial_total, source_slots);
    }
}

static void make_source_upload_compute_wait_if_needed(int target_offset,
                                                      int window_days) {
    std::lock_guard<std::mutex> lock(g_source_upload_mutex);
    if (!g_source_upload_pending) return;

    int need_begin = target_offset;
    int need_end = target_offset + window_days;
    int upload_begin = g_source_upload_day0;
    int upload_end = g_source_upload_day0 + g_source_upload_day_count;
    if (need_end <= upload_begin || need_begin >= upload_end) return;
    if (g_source_upload_compute_wait_generation == g_source_upload_generation) return;

    for (int i = 0; i < g_persistent_gpu_count; ++i) {
        if (g_persistent_local_rows[i] <= 0) continue;
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamWaitEvent(g_compute_stream[i], g_source_upload_done[i], 0));
    }

    g_source_upload_compute_wait_generation = g_source_upload_generation;
    std::cout << "[DCU module] source stream upload_event_wait"
              << " target_offset=" << target_offset
              << " upload_day0=" << upload_begin
              << " upload_days=" << g_source_upload_day_count
              << " generation=" << g_source_upload_generation
              << std::endl;
}

void wait_source_days_preloaded_upload() {
    using Clock = std::chrono::steady_clock;

    int gpu_count = 0;
    int generation = 0;
    int source_day0 = 0;
    int source_day_count = 0;
    bool keep_registered = false;
    bool registered[MAX_PERSISTENT_GPUS] = {false};
    float* registered_ptr[MAX_PERSISTENT_GPUS] = {nullptr};

    {
        std::lock_guard<std::mutex> lock(g_source_upload_mutex);
        if (!g_source_upload_pending) return;
        gpu_count = g_persistent_gpu_count;
        generation = g_source_upload_generation;
        source_day0 = g_source_upload_day0;
        source_day_count = g_source_upload_day_count;
        keep_registered = g_source_upload_keep_registered;
        for (int i = 0; i < gpu_count; ++i) {
            registered[i] = g_source_upload_registered[i];
            registered_ptr[i] = g_source_upload_registered_ptr[i];
        }
    }

    auto t0 = Clock::now();
    for (int i = 0; i < gpu_count; ++i) {
        if (g_persistent_local_rows[i] <= 0) continue;
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipEventSynchronize(g_source_upload_done[i]));
    }

    int unregister_count = 0;
    if (!keep_registered) {
        for (int i = 0; i < gpu_count; ++i) {
            if (registered[i] && registered_ptr[i]) {
                HIP_CHECK(hipHostUnregister(registered_ptr[i]));
                ++unregister_count;
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(g_source_upload_mutex);
        if (g_source_upload_pending && g_source_upload_generation == generation) {
            g_source_upload_pending = false;
            g_source_upload_day0 = 0;
            g_source_upload_day_count = 0;
            g_source_upload_keep_registered = false;
            g_source_upload_compute_wait_generation = -1;
            for (int i = 0; i < MAX_PERSISTENT_GPUS; ++i) {
                g_source_upload_registered[i] = false;
                g_source_upload_registered_ptr[i] = nullptr;
            }
        }
    }

    double wait_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    std::cout << "[DCU module] source stream upload_async_wait="
              << wait_ms << "ms (source_day0=" << source_day0
              << ", source_days=" << source_day_count
              << ", generation=" << generation
              << ", unregistered=" << unregister_count
              << "/" << gpu_count
              << ", keep_registered=" << (keep_registered ? "ON" : "OFF")
              << ")" << std::endl;
}

void upload_source_days_preloaded_local_async(float** h_source_local,
                                              int source_day0,
                                              int source_day_count) {
    using Clock = std::chrono::steady_clock;
    wait_source_days_preloaded_upload();

    const int slots_this = source_day_count * CLIM_YEARS;
    const int slot0 = source_day0 * CLIM_YEARS;
    const bool keep_registered = source_keep_registered_enabled();
    auto t0 = Clock::now();

    bool registered[MAX_PERSISTENT_GPUS] = {false};
    if (source_register_upload_enabled()) {
        for (int i = 0; i < g_persistent_gpu_count; ++i) {
            if (g_persistent_local_rows[i] <= 0) continue;
            int local_spatial = g_persistent_local_rows[i] * LON_SIZE;
            size_t bytes = static_cast<size_t>(slots_this) * local_spatial * sizeof(float);
            registered[i] = register_source_upload_host_ptr(h_source_local[i], bytes, keep_registered);
        }
    }

    for (int i = 0; i < g_persistent_gpu_count; ++i) {
        if (g_persistent_local_rows[i] <= 0) continue;
        HIP_CHECK(hipSetDevice(i));
        int local_spatial = g_persistent_local_rows[i] * LON_SIZE;
        size_t bytes = static_cast<size_t>(slots_this) * local_spatial * sizeof(float);
        HIP_CHECK(hipMemcpyAsync(g_data[i] + static_cast<size_t>(slot0) * local_spatial,
                                 h_source_local[i], bytes,
                                 hipMemcpyHostToDevice, g_h2d_stream[i]));
        HIP_CHECK(hipEventRecord(g_source_upload_done[i], g_h2d_stream[i]));
    }

    int generation = 0;
    {
        std::lock_guard<std::mutex> lock(g_source_upload_mutex);
        g_source_upload_pending = true;
        g_source_upload_day0 = source_day0;
        g_source_upload_day_count = source_day_count;
        g_source_upload_keep_registered = keep_registered;
        ++g_source_upload_generation;
        generation = g_source_upload_generation;
        g_source_upload_compute_wait_generation = -1;
        for (int i = 0; i < MAX_PERSISTENT_GPUS; ++i) {
            g_source_upload_registered[i] = false;
            g_source_upload_registered_ptr[i] = nullptr;
        }
        for (int i = 0; i < g_persistent_gpu_count; ++i) {
            g_source_upload_registered[i] = registered[i];
            g_source_upload_registered_ptr[i] = registered[i] ? h_source_local[i] : nullptr;
        }
    }

    int registered_count = 0;
    for (int i = 0; i < g_persistent_gpu_count; ++i) {
        if (registered[i]) ++registered_count;
    }
    double launch_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    std::cout << "[DCU module] source stream upload_async_launch="
              << launch_ms << "ms (source_day0=" << source_day0
              << ", source_days=" << source_day_count
              << ", slots=" << slots_this
              << ", generation=" << generation
              << ", registered=" << registered_count
              << "/" << g_persistent_gpu_count
              << ", keep_registered=" << (keep_registered ? "ON" : "OFF")
              << ")" << std::endl;
}

void upload_source_days_preloaded_local(float** h_source_local,
                                        int source_day0,
                                        int source_day_count) {
    upload_source_days_preloaded_local_async(h_source_local, source_day0, source_day_count);
    wait_source_days_preloaded_upload();
}

void dispatch_source_window_device_preloaded(int source_days, int target_offset,
                                             float* h_mean, float* h_p90) {
    using Clock = std::chrono::steady_clock;
    const int spatial_total = static_cast<int>(SPATIAL_POINTS);
    const int source_slots = source_days * CLIM_YEARS;
    const bool skip_zero = skip_zero_enabled();
    const bool fast_hist = source_fast_hist_enabled();
    const bool fixed_collect = source_fixed_collect_enabled();
    const bool select_p90 = source_select_p90_enabled();
    const int source_mc_k = source_mc_k_from_env();
    const bool source_mc_independent = source_mc_independent_enabled();
    const bool source_mc_flat = source_mc_flat_enabled() || (source_mc_k > 0 && source_mc_k < CLIM_YEARS);
    const bool source_mc_exact_mean = source_mc_exact_mean_enabled();

    if (!g_persistent_initialized) {
        init_persistent_dcu_buffers(spatial_total, source_slots);
    }

    make_source_upload_compute_wait_if_needed(target_offset, 2 * CLIM_DELTA_DAY + 1);

    int out_cur = g_output_pingpong;
    auto kernel_t0 = Clock::now();

    for (int i = 0; i < g_persistent_gpu_count; ++i) {
        if (g_persistent_local_rows[i] <= 0) continue;
        HIP_CHECK(hipSetDevice(i));

        int local_spatial = g_persistent_local_rows[i] * LON_SIZE;
        int spatial_start = g_persistent_lat_start[i] * LON_SIZE;
        int block_size = 256;
        int grid_size = (local_spatial + block_size - 1) / block_size;

        if (source_mc_k > 0) {
            if (source_mc_exact_mean) {
                compute_source_window_mc_kernel<<<dim3(grid_size), dim3(block_size), 0, g_compute_stream[i]>>>(
                    g_data[i],
                    g_mean_buf[out_cur][i], g_p90_buf[out_cur][i],
                    target_offset, CLIM_YEARS, 2 * CLIM_DELTA_DAY + 1,
                    spatial_total, spatial_start, spatial_start + local_spatial,
                    local_spatial, skip_zero, source_mc_k, source_mc_independent,
                    source_mc_exact_mean);
            } else {
                compute_source_window_mc_sample_kernel<<<dim3(grid_size), dim3(block_size), 0, g_compute_stream[i]>>>(
                    g_data[i],
                    g_mean_buf[out_cur][i], g_p90_buf[out_cur][i],
                    target_offset, CLIM_YEARS, 2 * CLIM_DELTA_DAY + 1,
                    spatial_total, spatial_start, spatial_start + local_spatial,
                    local_spatial, skip_zero, source_mc_k, source_mc_independent,
                    source_mc_flat);
            }
        } else if (fixed_collect) {
            compute_source_window_fixed_collect_kernel<<<dim3(grid_size), dim3(block_size), 0, g_compute_stream[i]>>>(
                g_data[i],
                g_mean_buf[out_cur][i], g_p90_buf[out_cur][i],
                target_offset, CLIM_YEARS, 2 * CLIM_DELTA_DAY + 1,
                spatial_total, spatial_start, spatial_start + local_spatial,
                local_spatial, skip_zero);
        } else if (fast_hist) {
            compute_source_window_fast_hist_kernel<<<dim3(grid_size), dim3(block_size), 0, g_compute_stream[i]>>>(
                g_data[i],
                g_mean_buf[out_cur][i], g_p90_buf[out_cur][i],
                target_offset, CLIM_YEARS, 2 * CLIM_DELTA_DAY + 1,
                spatial_total, spatial_start, spatial_start + local_spatial,
                local_spatial, skip_zero);
        } else {
            compute_source_window_kernel<<<dim3(grid_size), dim3(block_size), 0, g_compute_stream[i]>>>(
                g_data[i],
                g_mean_buf[out_cur][i], g_p90_buf[out_cur][i],
                target_offset, CLIM_YEARS, 2 * CLIM_DELTA_DAY + 1,
                spatial_total, spatial_start, spatial_start + local_spatial,
                local_spatial, skip_zero, select_p90);
        }
        HIP_CHECK(hipPeekAtLastError());
    }

    for (int i = 0; i < g_persistent_gpu_count; ++i) {
        if (g_persistent_local_rows[i] <= 0) continue;
        HIP_CHECK(hipSetDevice(i));

        int local_spatial = g_persistent_local_rows[i] * LON_SIZE;
        int host_offset = g_persistent_lat_start[i] * LON_SIZE;

        HIP_CHECK(hipMemcpyAsync(h_mean + host_offset, g_mean_buf[out_cur][i],
                                 local_spatial * sizeof(float),
                                 hipMemcpyDeviceToHost, g_compute_stream[i]));
        HIP_CHECK(hipMemcpyAsync(h_p90 + host_offset, g_p90_buf[out_cur][i],
                                 local_spatial * sizeof(float),
                                 hipMemcpyDeviceToHost, g_compute_stream[i]));
    }

    for (int i = 0; i < g_persistent_gpu_count; ++i) {
        if (g_persistent_local_rows[i] <= 0) continue;
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamSynchronize(g_compute_stream[i]));
    }

    double compute_ms = std::chrono::duration<double, std::milli>(Clock::now() - kernel_t0).count();
    g_output_pingpong = 1 - g_output_pingpong;

    std::cout << "[DCU module] source_window_device target_offset=" << target_offset
              << " (source_slots=" << source_slots
              << ", skip_zero=" << (skip_zero ? "ON" : "OFF")
              << ", fast_hist=" << (fast_hist ? "ON" : "OFF")
              << ", fixed_collect=" << (fixed_collect ? "ON" : "OFF")
              << ", select_p90=" << (select_p90 ? "ON" : "OFF")
              << ", source_mc=" << (source_mc_k > 0 ? std::to_string(source_mc_k) : "OFF")
              << ", source_mc_mode=" << (source_mc_k > 0 ? (source_mc_independent ? "INDEPENDENT" : "BLOCK") : "OFF")
              << ", source_mc_sample=" << (source_mc_k > 0 ? (source_mc_flat ? "FLAT" : "PER_YEAR") : "OFF")
              << ", source_mc_mean=" << (source_mc_k > 0 ? (source_mc_exact_mean ? "EXACT" : "SAMPLE") : "OFF")
              << ") compute=" << compute_ms << "ms" << std::endl;
}

// ============================================================
// Cleanup (with fp16 + dual-stream resources)
// ============================================================

void cleanup_dcu_persistent_buffers() {
    if (!g_persistent_initialized) return;
    wait_source_days_preloaded_upload();
    unregister_kept_source_uploads();

    for (int i = 0; i < g_persistent_gpu_count; ++i) {
        HIP_CHECK(hipSetDevice(i));

        if (g_data[i])     HIP_CHECK(hipFree(g_data[i]));
        if (g_data_fp16[i]) HIP_CHECK(hipFree(g_data_fp16[i]));

        for (int b = 0; b < 2; ++b) {
            if (g_mean_buf[b][i]) HIP_CHECK(hipFree(g_mean_buf[b][i]));
            if (g_p90_buf[b][i])  HIP_CHECK(hipFree(g_p90_buf[b][i]));
        }

        if (g_h2d_stream[i])     HIP_CHECK(hipStreamDestroy(g_h2d_stream[i]));
        if (g_compute_stream[i]) HIP_CHECK(hipStreamDestroy(g_compute_stream[i]));
        if (g_h2d_done[i])       HIP_CHECK(hipEventDestroy(g_h2d_done[i]));
        if (g_source_upload_done[i]) HIP_CHECK(hipEventDestroy(g_source_upload_done[i]));

        if (g_half_staging[i]) HIP_CHECK(hipHostFree(g_half_staging[i]));

        // MC indices cleanup
        if (g_mc_indices[i]) HIP_CHECK(hipFree(g_mc_indices[i]));
        g_mc_indices[i] = nullptr;

        g_data[i]      = nullptr;
        g_data_fp16[i] = nullptr;
        g_mean_buf[0][i] = g_mean_buf[1][i] = nullptr;
        g_p90_buf[0][i]  = g_p90_buf[1][i]  = nullptr;
        g_h2d_stream[i]     = nullptr;
        g_compute_stream[i] = nullptr;
        g_h2d_done[i]       = nullptr;
        g_source_upload_done[i] = nullptr;
        g_half_staging[i]   = nullptr;
        g_half_staging_sz[i] = 0;
        g_persistent_local_rows[i] = 0;
    }

    g_use_fp16 = false;
    g_persistent_gpu_count = 0;
    g_persistent_rows_per_gpu = 0;
    g_output_pingpong = 0;
    g_mc_K = 0;
    g_mc_consistent = false;
    g_mc_block = false;
    g_mc_packed = false;
    g_mc_current_doy = TARGET_DOY_BEGIN;
    g_mc_indices_cpu.clear();
    g_source_upload_pending = false;
    g_source_upload_day0 = 0;
    g_source_upload_day_count = 0;
    g_source_upload_keep_registered = false;
    g_source_upload_compute_wait_generation = -1;
    for (int i = 0; i < MAX_PERSISTENT_GPUS; ++i) {
        g_source_upload_registered[i] = false;
        g_source_upload_registered_ptr[i] = nullptr;
    }
    g_source_kept_registered_count = 0;
    g_persistent_initialized = false;
}

void dispatch_to_4_dcus_with_output(float* h_sst_data, float* h_mean, float* h_p90,
                                     const int* /*changed_slots*/, int /*num_changed*/) {
    using Clock = std::chrono::steady_clock;
    const int days_total   = static_cast<int>(DAYS_TOTAL);
    const int spatial_total = static_cast<int>(SPATIAL_POINTS);

    int gpu_count = 0;
    hipGetDeviceCount(&gpu_count);
    const int NUM_GPUS = gpu_count;
    if (NUM_GPUS <= 0) {
        std::cerr << "[DCU 模块] 未发现可用 DCU" << std::endl;
        std::exit(1);
    }

    int rows_per_gpu = (LAT_SIZE + NUM_GPUS - 1) / NUM_GPUS;

    auto t0 = Clock::now();

    hipStream_t streams[8];
    float* d_data[8];
    float* d_mean[8];
    float* d_p90[8];
    int lat_start[8], lat_end[8], local_rows_arr[8];

    size_t full_data_bytes = (size_t)days_total * spatial_total * sizeof(float);

    for (int i = 0; i < NUM_GPUS; ++i) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamCreate(&streams[i]));

        lat_start[i] = i * rows_per_gpu;
        lat_end[i]   = std::min((i + 1) * rows_per_gpu, (int)LAT_SIZE);
        local_rows_arr[i] = lat_end[i] - lat_start[i];

        if (local_rows_arr[i] <= 0) continue;

        int local_spatial = local_rows_arr[i] * LON_SIZE;

        HIP_CHECK(hipMalloc(&d_data[i], full_data_bytes));
        HIP_CHECK(hipMalloc(&d_mean[i], local_spatial * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_p90[i],  local_spatial * sizeof(float)));

        HIP_CHECK(hipMemcpyAsync(d_data[i], h_sst_data, full_data_bytes,
                                 hipMemcpyHostToDevice, streams[i]));
    }

    for (int i = 0; i < NUM_GPUS; ++i) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamSynchronize(streams[i]));
    }

    auto t1 = Clock::now();
    double transfer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = Clock::now();

    for (int i = 0; i < NUM_GPUS; ++i) {
        if (local_rows_arr[i] <= 0) continue;
        HIP_CHECK(hipSetDevice(i));
        int local_spatial = local_rows_arr[i] * LON_SIZE;
        int spatial_start = lat_start[i] * LON_SIZE;
        int block_size = 256;
        int grid_size  = (local_spatial + block_size - 1) / block_size;

        compute_mean_p90_kernel<<<dim3(grid_size), dim3(block_size), 0, streams[i]>>>(
            d_data[i], d_mean[i], d_p90[i],
            spatial_total, days_total,
            spatial_start, spatial_start + local_spatial,
            spatial_total, 0);
    }

    for (int i = 0; i < NUM_GPUS; ++i) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamSynchronize(streams[i]));
    }

    t1 = Clock::now();
    double kernel_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = Clock::now();

    for (int i = 0; i < NUM_GPUS; ++i) {
        if (local_rows_arr[i] <= 0) continue;
        HIP_CHECK(hipSetDevice(i));
        int local_spatial = local_rows_arr[i] * LON_SIZE;
        int host_offset   = lat_start[i] * LON_SIZE;

        HIP_CHECK(hipMemcpy(h_mean + host_offset, d_mean[i],
                            local_spatial * sizeof(float),
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_p90 + host_offset, d_p90[i],
                            local_spatial * sizeof(float),
                            hipMemcpyDeviceToHost));
    }

    t1 = Clock::now();
    double gather_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    for (int i = 0; i < NUM_GPUS; ++i) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamDestroy(streams[i]));
        HIP_CHECK(hipFree(d_data[i]));
        HIP_CHECK(hipFree(d_mean[i]));
        HIP_CHECK(hipFree(d_p90[i]));
    }

    double total_ms = transfer_ms + kernel_ms + gather_ms;
    std::cout << "[DCU 模块] 传输=" << transfer_ms << "ms kernel=" << kernel_ms
              << "ms 回传=" << gather_ms << "ms 总计=" << total_ms << "ms" << std::endl;
}

void cleanup_dcu_buffers() {
    // no persistent buffers to clean up
}
