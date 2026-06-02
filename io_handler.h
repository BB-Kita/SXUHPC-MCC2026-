#ifndef IO_HANDLER_H
#define IO_HANDLER_H

#pragma once

// 读取默认目标日对应的 30年×11天 SST 样本窗口。
// h_sst_data 布局为 [sample][lat][lon]。
void read_netcdf_parallel(float* h_sst_data);

// 读取指定 day-of-year 的 30年×11天 SST 样本窗口。
// target_doy 使用剔除 2 月 29 日后的 365 天日历。
void read_netcdf_parallel_for_doy(float* h_sst_data, int target_doy);

// 滑动窗口：初始化 — 读取 first_doy 的全部 330 个文件
void initialize_window(float* h_sst_data, int first_doy);

// 滑动窗口：滑动 — 读取 30 个新文件，覆盖最旧的循环槽位
void slide_window_to_next_doy(float* h_sst_data, int current_doy, int first_doy);

#endif