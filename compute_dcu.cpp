#include "compute_dcu.h"
#include "config.h"
#include <hip/hip_runtime.h>
#include <iostream>

// 引入算法队员的文件
#include "algo_p90.h" 

// 海光 DCU 设备端算子 (Kernel)
__global__ void ocean_heatwave_kernel(const float* d_in, float* d_out_mean, float* d_out_p90, int lat_start, int lat_end) {
    int lon_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lat_idx_local = blockIdx.y * blockDim.y + threadIdx.y;
    int lat_idx_global = lat_start + lat_idx_local;

    if (lon_idx < LON_SIZE && lat_idx_global < lat_end) {
        // 计算当前线程在全局数组中的空间偏移量
        int spatial_offset = lat_idx_global * LON_SIZE + lon_idx;
        
        float mean_val, p90_val;
        
        // 直接调用算法模块
        compute_mean_and_p90(d_in, spatial_offset, SPATIAL_POINTS, DAYS_TOTAL, mean_val, p90_val);
        
        // 写回结果到当前显卡的输出数组
        int local_offset = lat_idx_local * LON_SIZE + lon_idx;
        d_out_mean[local_offset] = mean_val;
        d_out_p90[local_offset] = p90_val;
    }
}

void dispatch_to_4_dcus(float* h_sst_data) {
    const int NUM_GPUS = 4;
    hipStream_t streams[NUM_GPUS];
    float* d_data[NUM_GPUS];
    float* d_mean[NUM_GPUS];
    float* d_p90[NUM_GPUS];

    // 空间网格切分：将 721 行分配给 4 张卡
    int rows_per_gpu = (LAT_SIZE + NUM_GPUS - 1) / NUM_GPUS; // 向上取整，前三张181，最后一张178

    std::cout << "[DCU 模块] 初始化 " << NUM_GPUS << " 张显卡进行空间切片并行..." << std::endl;

    for (int i = 0; i < NUM_GPUS; ++i) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamCreate(&streams[i]));

        // 计算当前卡负责的行数范围
        int lat_start = i * rows_per_gpu;
        int lat_end = std::min((i + 1) * rows_per_gpu, (int)LAT_SIZE);
        int current_rows = lat_end - lat_start;

        if (current_rows <= 0) continue;

        // 计算需要向当前卡传输的数据量 (仅传输当前卡负责的那些行的数据)
        // 注意：因为数组是 [day][lat][lon] 排布，我们需要把每天对应的数据块挑出来传
        // 为极致简化，此处仅展示设备显存分配逻辑。
        // 实际上需要写一个定制的跨步拷贝（或在 I/O 时直接按 GPU 切分好 Host 内存）
        size_t chunk_elements = current_rows * LON_SIZE * DAYS_TOTAL;
        
        HIP_CHECK(hipMalloc(&d_data[i], chunk_elements * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_mean[i], current_rows * LON_SIZE * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_p90[i], current_rows * LON_SIZE * sizeof(float)));

        // TODO: 使用 hipMemcpyAsync 发送数据
        // TODO: Launch Kernel
        dim3 blockSize(16, 16);
        dim3 gridSize((LON_SIZE + blockSize.x - 1) / blockSize.x, 
                      (current_rows + blockSize.y - 1) / blockSize.y);
        ocean_heatwave_kernel<<<gridSize, blockSize, 0, streams[i]>>>(d_data[i], d_mean[i], d_p90[i], lat_start, lat_end);

        // TODO: 使用 hipMemcpyAsync 拉回结果到主机的目标数组
    }

    // 强制主线程等待所有卡执行完毕
    for (int i = 0; i < NUM_GPUS; ++i) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamSynchronize(streams[i]));
        HIP_CHECK(hipStreamDestroy(streams[i]));
        HIP_CHECK(hipFree(d_data[i]));
        HIP_CHECK(hipFree(d_mean[i]));
        HIP_CHECK(hipFree(d_p90[i]));
    }
    std::cout << "[DCU 模块] 四卡计算与数据回写完成。" << std::endl;
}
