#pragma once

// 多卡计算调度函数，接收锁页内存指针
// h_sst_data 布局: [DAYS_TOTAL][LAT_SIZE][LON_SIZE] (row-major)
void dispatch_to_4_dcus(float* h_sst_data);
