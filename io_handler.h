#pragma once

// 读取默认目标日 DEFAULT_TARGET_DOY 对应的 30年×11天 SST 样本窗口。
// h_sst_data 布局为 [sample][lat][lon]，长度为 DAYS_TOTAL * SPATIAL_POINTS。
void read_netcdf_parallel(float* h_sst_data);

// 读取指定 day-of-year 的 30年×11天 SST 样本窗口。
// target_doy 使用无 2 月 29 日的 365 天日历，6月1日通常为 152。
void read_netcdf_parallel_for_doy(float* h_sst_data, int target_doy);
