#!/usr/bin/env bash
#SBATCH -J mcc26_simple
#SBATCH -p kshdmcc2026
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --exclusive
#SBATCH --gres=dcu:4
#SBATCH --time=00:05:00
#SBATCH -o /public/home/fujiake/fjk/MCC26_SXU_gpuopt_exp_20260601/logs/simple_%j.out
#SBATCH -e /public/home/fujiake/fjk/MCC26_SXU_gpuopt_exp_20260601/logs/simple_%j.err

set -euo pipefail

source /public/home/fujiake/miniconda3/bin/activate
conda activate lsd
set +u
source /public/software/compiler/rocm/dtk-22.10.1/env.sh
set -u

cat > /tmp/mcc26_simple_kernel_${SLURM_JOB_ID}.cpp <<'CPP'
#include <hip/hip_runtime.h>
#include <iostream>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    std::cerr << "HIP " << hipGetErrorString(e) << " at line " << __LINE__ << "\n"; return 1; } } while (0)

__global__ void fill_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 123.0f + i;
}

int main() {
    int ndev = 0;
    CHECK(hipGetDeviceCount(&ndev));
    std::cout << "device_count=" << ndev << "\n";
    for (int dev = 0; dev < ndev; ++dev) {
        CHECK(hipSetDevice(dev));
        hipDeviceProp_t p{};
        CHECK(hipGetDeviceProperties(&p, dev));
        std::cout << "dev=" << dev << " name=" << p.name
                  << " arch=" << p.gcnArchName << "\n";
        float* d = nullptr;
        float h[4] = {};
        CHECK(hipMalloc(&d, 4 * sizeof(float)));
        fill_kernel<<<1, 64>>>(d, 4);
        CHECK(hipPeekAtLastError());
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(h, d, 4 * sizeof(float), hipMemcpyDeviceToHost));
        CHECK(hipFree(d));
        std::cout << "dev=" << dev << " values="
                  << h[0] << "," << h[1] << "," << h[2] << "," << h[3] << "\n";
    }
    return 0;
}
CPP

hipcc -O2 --offload-arch=gfx906 /tmp/mcc26_simple_kernel_${SLURM_JOB_ID}.cpp \
  -o /tmp/mcc26_simple_kernel_${SLURM_JOB_ID}
/tmp/mcc26_simple_kernel_${SLURM_JOB_ID}
