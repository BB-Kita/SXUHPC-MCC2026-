#ifndef IO_HANDLER_H
#define IO_HANDLER_H

#pragma once

// 读取默认目标日对应的 30年×11天 SST 样本窗口。
// h_sst_data 布局为 [sample][lat][lon]。
void read_netcdf_parallel(float* h_sst_data);

// 读取指定 day-of-year 的 30年×11天 SST 样本窗口。
// target_doy 使用剔除 2 月 29 日后的 365 天日历。
void read_netcdf_parallel_for_doy(float* h_sst_data, int target_doy);

// Monte Carlo packed I/O：每个目标日只读取 sample_count 个采样文件，
// 并打包到 h_sst_data 的前 sample_count 个 slot。
void read_mc_packed_for_doy(float* h_sst_data, int target_doy, int sample_count);

// 整季预加载：读取 [target_begin-CLIM_DELTA_DAY, target_end+CLIM_DELTA_DAY]
// 的所有源日，每个源日包含 30 年，布局为 [source_day][year][lat][lon]。
void preload_season_source(float* h_source_data, int target_doy_begin, int target_doy_end);

// 整季预加载到 GPU-local host buffers：每个 buffer 布局为
// [source_day][year][local_lat][lon]，避免后续 H2D 上传时的 2D stride。
void preload_season_source_local(float** h_source_local,
                                 const int* lat_starts,
                                 const int* local_rows,
                                 int num_parts,
                                 int target_doy_begin,
                                 int target_doy_end);

// 读取连续 source-day 分块到 GPU-local host buffers。buffer 布局为
// [source_day_in_chunk][year][local_lat][lon]。
void preload_source_days_local(float** h_source_local,
                               const int* lat_starts,
                               const int* local_rows,
                               int num_parts,
                               int source_doy_begin,
                               int source_day_count);

// 滑动窗口：初始化 — 读取 first_doy 的全部 330 个文件
void initialize_window(float* h_sst_data, int first_doy);

// 滑动窗口：滑动 — 读取 30 个新文件，覆盖最旧的循环槽位
void slide_window_to_next_doy(float* h_sst_data, int current_doy, int first_doy);

void prefetch_next_doy_async(float* h_sst_data, int current_doy, int first_doy);
void wait_for_prefetch_next_doy();

#endif
