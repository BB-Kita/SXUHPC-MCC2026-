#pragma once

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <hip/hip_runtime.h>

// 空间维度
static constexpr std::size_t LON_SIZE = 1440;
static constexpr std::size_t LAT_SIZE = 721;
static constexpr std::size_t SPATIAL_POINTS = LON_SIZE * LAT_SIZE;

// 气候态统计参数：1991-2020，目标日前后各 5 天，共 30 * 11 = 330 个样本
static constexpr int CLIM_START_YEAR = 1991;
static constexpr int CLIM_END_YEAR = 2020;
static constexpr int CLIM_DELTA_DAY = 5;
static constexpr int CLIM_YEARS = CLIM_END_YEAR - CLIM_START_YEAR + 1;
static constexpr int CLIM_WINDOW_DAYS = 2 * CLIM_DELTA_DAY + 1;
static constexpr std::size_t CLIM_SAMPLE_COUNT = static_cast<std::size_t>(CLIM_YEARS * CLIM_WINDOW_DAYS);

// 赛题目标输出日期：6.1-8.31，对应 day-of-year 152-243，共 92 天
static constexpr int TARGET_DOY_BEGIN = 152;
static constexpr int TARGET_DOY_END = 243;
static constexpr std::size_t TARGET_DAYS_TOTAL = static_cast<std::size_t>(TARGET_DOY_END - TARGET_DOY_BEGIN + 1);

// 为兼容当前 compute_dcu/algo_p90 接口，DAYS_TOTAL 表示“单个目标日”的输入样本数，不是 92 个输出日
static constexpr std::size_t DAYS_TOTAL = CLIM_SAMPLE_COUNT;
static constexpr std::size_t TOTAL_ELEMENTS = SPATIAL_POINTS * DAYS_TOTAL;

// I/O 配置
#define NC_INPUT_DIR "/public/home/achwjznh4b/Newdata/"
#define DEFAULT_TARGET_DOY TARGET_DOY_BEGIN
#define IO_THREADS 32
#define APPLY_SCALE_OFFSET 1

// 统一的 HIP 错误检查宏
#define HIP_CHECK(command) {                                    \
    hipError_t status = command;                                \
    if (status != hipSuccess) {                                 \
        std::cerr << "HIP Error: " << hipGetErrorString(status) \
                  << " at line " << __LINE__ << std::endl;      \
        std::exit(1);                                           \
    }                                                           \
}
