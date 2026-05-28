// algo_p90.h
#pragma once
#include <hip/hip_runtime.h>
#include <math.h>

// 设备端内联交换函数
__device__ __forceinline__ void swap_val(float& a, float& b) {
    float tmp = a;
    a = b;
    b = tmp;
}

// 经典的快速选择算法 (求升序数组中的第 k 个元素)
// 完全在 GPU 线程内部执行
__device__ inline float quick_select(float* arr, int n, int k) {
    int left = 0;
    int right = n - 1;
    while (left <= right) {
        float pivot = arr[right];
        int i = left;
        for (int j = left; j < right; ++j) {
            if (arr[j] <= pivot) {
                swap_val(arr[i], arr[j]);
                i++;
            }
        }
        swap_val(arr[i], arr[right]);
        
        if (i == k) return arr[i];
        else if (i < k) left = i + 1;
        else right = i - 1;
    }
    return nanf(""); // 异常分支
}

// 核心计算接口：供 Kernel 调用
// global_data: 全局 45GB 锁页内存指针
// offset: 当前线程负责的空间点偏移量
// SPATIAL_POINTS: 空间总点数 (用于跨步读取)
// DAYS_TOTAL: 时间窗口 (通常为 330)
__device__ inline void compute_mean_and_p90(const float* global_data, int offset, int SPATIAL_POINTS, int DAYS_TOTAL, float& out_mean, float& out_p90) {
    // ⚠️ 警告算法队员：这里直接开辟 330 大小的数组会触发 Register Spilling (寄存器溢出到局部内存)
    // 这是最需要他进行优化的性能毒点（比如改用直方图算法规避数组排序）
    float temp[330]; 
    
    int valid_count = 0;
    float sum = 0.0f;
    
    // 1. 从全局内存读取数据（完美合并访存）并过滤 NaN
    for (int d = 0; d < DAYS_TOTAL; ++d) {
        float val = global_data[d * SPATIAL_POINTS + offset];
        if (!isnan(val)) {
            temp[valid_count] = val;
            sum += val;
            valid_count++;
        }
    }
    
    // 2. 处理全为 NaN 的陆地区域
    if (valid_count == 0) {
        out_mean = nanf("");
        out_p90 = nanf("");
        return;
    }
    
    // 3. 计算均值
    out_mean = sum / valid_count;
    
    // 4. 计算 90% 分位数
    // PyTorch 的 nanpercentile 逻辑：索引 = (N-1) * 0.9
    // 这里采用四舍五入或向下取整的近似位置
    int k = (int)(valid_count * 0.90f); 
    if (k >= valid_count) k = valid_count - 1;
    
    out_p90 = quick_select(temp, valid_count, k);
}