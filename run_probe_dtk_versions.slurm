#!/usr/bin/env bash
#SBATCH -J mcc26_dtk
#SBATCH -p kshdmcc2026
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --exclusive
#SBATCH --gres=dcu:4
#SBATCH --time=00:15:00
#SBATCH -o /public/home/fujiake/fjk/MCC26_SXU_gpuopt_exp_20260601/logs/dtk_%j.out
#SBATCH -e /public/home/fujiake/fjk/MCC26_SXU_gpuopt_exp_20260601/logs/dtk_%j.err

set -uo pipefail

SRC=/tmp/mcc26_dtk_${SLURM_JOB_ID}.cpp
cat > "${SRC}" <<'CPP'
#include <hip/hip_runtime.h>
#include <iostream>

__global__ void fill_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 321.0f + i;
}

int main() {
    int ndev = 0;
    hipError_t e = hipGetDeviceCount(&ndev);
    if (e != hipSuccess) { std::cerr << "count " << hipGetErrorString(e) << "\n"; return 10; }
    std::cout << "device_count=" << ndev << "\n";
    hipDeviceProp_t p{};
    hipGetDeviceProperties(&p, 0);
    std::cout << "dev0 name=" << p.name << " arch=" << p.gcnArchName << "\n";
    hipSetDevice(0);
    float* d = nullptr;
    float h[4] = {};
    e = hipMalloc(&d, 4 * sizeof(float));
    if (e != hipSuccess) { std::cerr << "malloc " << hipGetErrorString(e) << "\n"; return 11; }
    fill_kernel<<<1, 64>>>(d, 4);
    e = hipPeekAtLastError();
    if (e != hipSuccess) { std::cerr << "launch " << hipGetErrorString(e) << "\n"; return 12; }
    e = hipDeviceSynchronize();
    if (e != hipSuccess) { std::cerr << "sync " << hipGetErrorString(e) << "\n"; return 13; }
    e = hipMemcpy(h, d, 4 * sizeof(float), hipMemcpyDeviceToHost);
    if (e != hipSuccess) { std::cerr << "copy " << hipGetErrorString(e) << "\n"; return 14; }
    hipFree(d);
    std::cout << "values=" << h[0] << "," << h[1] << "," << h[2] << "," << h[3] << "\n";
    return 0;
}
CPP

try_dtk() {
  root="$1"
  name="$(basename "${root}")"
  echo "===== ${name} ====="
  if [ ! -f "${root}/env.sh" ]; then
    echo "missing env.sh"
    return
  fi
  (
    set +u
    source "${root}/env.sh"
    set -u
    echo "hipcc=$(which hipcc)"
    hipcc --version | head -3
    bin="/tmp/mcc26_dtk_${SLURM_JOB_ID}_${name}"
    hipcc -O2 --offload-arch=gfx906 "${SRC}" -o "${bin}"
    echo "compile_rc=$?"
    "${bin}"
    echo "run_rc=$?"
  )
}

try_dtk /public/software/compiler/rocm/dtk-22.10.1
try_dtk /public/software/compiler/rocm/dtk-23.10
try_dtk /public/software/compiler/rocm/dtk-24.04.3
try_dtk /public/software/compiler/rocm/dtk-25.04.4
try_dtk /public/software/compiler/rocm/dtk-26.04
