#pragma once

// 多卡计算调度函数，接收锁页内存指针
// h_sst_data 布局: [DAYS_TOTAL][LAT_SIZE][LON_SIZE] (row-major)
void dispatch_to_4_dcus(float* h_sst_data);

// 版本：DCU 缓冲区持久化 + 增量传输
// changed_slots: 变化的槽位索引数组，nullptr 表示全量传输（首次调用）
// num_changed: 变化槽位数，0 表示全量传输
void dispatch_to_4_dcus_with_output(float* h_sst_data, float* h_mean, float* h_p90,
                                     const int* changed_slots = nullptr, int num_changed = 0);

// 释放 DCU 设备缓冲区
void cleanup_dcu_buffers();
