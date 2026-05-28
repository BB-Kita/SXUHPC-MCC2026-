#include <iostream>
#include <hip/hip_runtime.h>
#include "config.h"
#include "io_handler.h"
#include "compute_dcu.h"

int main() {
    std::cout << ">>> 启动 MCC 海洋热浪阈值计算引擎 (C++/HIP)" << std::endl;

    // 1. 分配锁页内存 (Pinned Memory) 以极速对接 PCIe
    float* h_sst_data = nullptr;
    size_t memory_bytes = TOTAL_ELEMENTS * sizeof(float);
    std::cout << "[主线程] 正在分配 " << memory_bytes / (1024.0 * 1024.0 * 1024.0) 
              << " GB 的主机锁页内存..." << std::endl;
    
    HIP_CHECK(hipHostMalloc((void**)&h_sst_data, memory_bytes, hipHostMallocDefault));

    // 2. 调用 I/O 模块：多线程填充数据
    std::cout << "[主线程] 转交控制权至 I/O 模块..." << std::endl;
    read_netcdf_parallel(h_sst_data);

    // 3. 调用 DCU 模块：四卡异步计算
    std::cout << "[主线程] 转交控制权至 DCU 计算模块..." << std::endl;
    dispatch_to_4_dcus(h_sst_data);

    // 4. 安全释放内存
    HIP_CHECK(hipHostFree(h_sst_data));
    std::cout << ">>> 全部任务执行完毕！" << std::endl;
    return 0;
}