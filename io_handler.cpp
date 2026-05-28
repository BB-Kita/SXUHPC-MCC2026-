#include "io_handler.h"
#include "config.h"
#include <netcdf.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <string>

void read_netcdf_parallel(float* h_sst_data) {
    std::cout << "[I/O 模块] 启动 OpenMP 多线程数据读取..." << std::endl;

    // TODO: 获取赛题所需的真实日期字符串列表
    std::vector<std::string> dummy_dates(DAYS_TOTAL, "19910601"); 

    // 使用 OpenMP 打满 CPU 核心并发读取文件
    #pragma omp parallel for
    for (int d = 0; d < DAYS_TOTAL; ++d) {
        std::string filepath = "/public/home/achwjznh4b/Newdata/" + dummy_dates[d];
        
        int ncid, varid;
        // 如果文件打不开，直接 continue 跳过（注意：实际需处理缺测为 NaN）
        if (nc_open(filepath.c_str(), NC_NOWRITE, &ncid) != NC_NOERR) {
            continue; 
        }

        nc_inq_varid(ncid, "data", &varid);

        size_t start[] = {0, 0}; 
        size_t count[] = {LAT_SIZE, LON_SIZE}; 

        // 计算该天数据在一维大数组 [day][lat][lon] 中的绝对偏移量
        size_t offset = d * SPATIAL_POINTS;

        // 直接通过零拷贝方式，将 NetCDF 数据直写入分配好的锁页内存
        nc_get_vara_float(ncid, varid, start, count, &h_sst_data[offset]);

        nc_close(ncid);
    }
    std::cout << "[I/O 模块] 数据读取并拼接完毕。" << std::endl;
}