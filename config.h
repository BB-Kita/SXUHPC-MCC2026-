#pragma once
#include <cstddef>

// 空间维度
const size_t LON_SIZE = 1440;
const size_t LAT_SIZE = 721;
const size_t SPATIAL_POINTS = LON_SIZE * LAT_SIZE;

// 时间维度：赛题要求仅计算 6.1-8.31 共 92 天
const size_t DAYS_TOTAL = 92; 
const size_t TOTAL_ELEMENTS = SPATIAL_POINTS * DAYS_TOTAL;

// 统一的 HIP 错误检查宏
#define HIP_CHECK(command) {                                    \
    hipError_t status = command;                                \
    if (status != hipSuccess) {                                 \
        std::cerr << "HIP Error: " << hipGetErrorString(status) \
                  << " at line " << __LINE__ << std::endl;      \
        exit(1);                                                \
    }                                                           \
}