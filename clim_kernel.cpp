/**
 * clim_kernel.cpp —— SST 气候态 nanmean + P90 融合核 (CPU / OpenMP)
 * ==================================================================
 * 替代 numpy 的 np.nanmean + np.nanpercentile。后者对每一列做全排序 + NaN 掩码,
 * 在 (330, 104万) 规模下要 ~4s/天, 是 CPU 版的真正瓶颈。本核做三件事:
 *
 *   1. 融合: 单次遍历同时算 nanmean 和 P90, 数据只过一遍。
 *   2. nth_element (quickselect, 平均 O(n)) 代替全排序 O(n log n) 求分位数。
 *   3. 列分块 (tiling) + OpenMP: 把 330 个样本数组按列块 gather 到 L2 内的小缓冲,
 *      获得连续访存; 列方向用所有 CPU 核并行。
 *
 * 精度:
 *   - 全程 double (float64)。
 *   - 均值用 Kahan 补偿求和 (比朴素累加更准)。
 *   - P90 用 Hazen / Hyndman-Fan type 5 插值, 与 MATLAB prctile 完全等价:
 *       0-based 位置 pos = 0.9 * cnt - 0.5, 在 [0, cnt-1] 内线性插值。
 *   - NaN 样本被跳过 (与 MATLAB prctile / numpy nanpercentile 一致);
 *     某格点全为 NaN 时输出 NaN。
 *
 * 输入采用"指针数组"零拷贝设计: sample_ptrs[s] 指向第 s 个样本的 (n_lat*n_lon)
 * 连续 double 数组 (即某一天的整场 SST)。这样滑动缓存里的数组无需拼接成大块。
 *
 * 编译: 见 build_kernel.sh
 */

#include <algorithm>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

static const int TILE = 64;   // 每个列块的列数 (64*330*8 ≈ 169KB, 落在 L2)

extern "C" {

/**
 * @param sample_ptrs  长度 n_samples 的指针数组, 每个指向 ncols 个 double
 * @param n_samples    样本数 (=30年*11天=330)
 * @param ncols        格点数 (=n_lat*n_lon)
 * @param mean_out     输出 [ncols]  气候态均值
 * @param p90_out      输出 [ncols]  P90 分位数
 */
int compute_stats(
    const double* const* sample_ptrs,
    int n_samples,
    long ncols,
    double* mean_out,
    double* p90_out)
{
    const double NaN = std::nan("");

    #pragma omp parallel
    {
        std::vector<double> tile((size_t)n_samples * TILE);  // gather 缓冲
        std::vector<double> col(n_samples);                  // 单列有效值

        #pragma omp for schedule(dynamic)
        for (long jt = 0; jt < ncols; jt += TILE) {
            int w = (int)std::min((long)TILE, ncols - jt);

            // ---- gather: 连续读 w 个 double, 每个样本一行 ----
            for (int s = 0; s < n_samples; ++s) {
                const double* src = sample_ptrs[s] + jt;
                std::memcpy(&tile[(size_t)s * w], src, (size_t)w * sizeof(double));
            }

            // ---- 逐列计算 (tile 已在 cache 内) ----
            for (int jj = 0; jj < w; ++jj) {
                int cnt = 0;
                double sum = 0.0, comp = 0.0;   // Kahan
                for (int s = 0; s < n_samples; ++s) {
                    double v = tile[(size_t)s * w + jj];
                    if (v == v) {               // 非 NaN
                        double y = v - comp;
                        double t = sum + y;
                        comp = (t - sum) - y;
                        sum = t;
                        col[cnt++] = v;
                    }
                }

                long j = jt + jj;
                if (cnt == 0) { mean_out[j] = NaN; p90_out[j] = NaN; continue; }

                mean_out[j] = sum / (double)cnt;

                // ---- P90: Hazen 插值 (= MATLAB prctile) ----
                double pos = 0.9 * (double)cnt - 0.5;   // 0-based 虚拟位置
                if (pos <= 0.0) {
                    p90_out[j] = *std::min_element(col.begin(), col.begin() + cnt);
                } else if (pos >= (double)(cnt - 1)) {
                    p90_out[j] = *std::max_element(col.begin(), col.begin() + cnt);
                } else {
                    int lo = (int)pos;
                    double frac = pos - (double)lo;
                    // 第 lo 小就位, 其右侧最小值即第 lo+1 小
                    std::nth_element(col.begin(), col.begin() + lo, col.begin() + cnt);
                    double vlo = col[lo];
                    double vhi = *std::min_element(col.begin() + lo + 1, col.begin() + cnt);
                    p90_out[j] = vlo + frac * (vhi - vlo);
                }
            }
        }
    }
    return 0;
}

int kernel_num_threads() {
#ifdef _OPENMP
    int n = 0;
    #pragma omp parallel
    { 
        #pragma omp single
        n = omp_get_num_threads();
    }
    return n;
#else
    return 1;
#endif
}

}  // extern "C"
