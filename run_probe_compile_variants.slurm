#!/usr/bin/env bash
#SBATCH -J mcc26_flags
#SBATCH -p kshdmcc2026
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --exclusive
#SBATCH --gres=dcu:4
#SBATCH --time=00:10:00
#SBATCH -o /public/home/fujiake/fjk/MCC26_SXU_gpuopt_exp_20260601/logs/flags_%j.out
#SBATCH -e /public/home/fujiake/fjk/MCC26_SXU_gpuopt_exp_20260601/logs/flags_%j.err

set -uo pipefail

source /public/home/fujiake/miniconda3/bin/activate
conda activate lsd
set +u
source /public/software/compiler/rocm/dtk-22.10.1/env.sh
set -u

SRC=/tmp/mcc26_flags_${SLURM_JOB_ID}.cpp
cat > "${SRC}" <<'CPP'
#include <hip/hip_runtime.h>
#include <iostream>

__global__ void fill_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 123.0f + i;
}

int main() {
    int ndev = 0;
    hipError_t e = hipGetDeviceCount(&ndev);
    if (e != hipSuccess) {
        std::cerr << "hipGetDeviceCount " << hipGetErrorString(e) << "\n";
        return 10;
    }
    std::cout << "device_count=" << ndev << "\n";
    hipDeviceProp_t p{};
    hipGetDeviceProperties(&p, 0);
    std::cout << "dev0 name=" << p.name << " arch=" << p.gcnArchName << "\n";
    hipSetDevice(0);
    float* d = nullptr;
    float h[4] = {};
    e = hipMalloc(&d, 4 * sizeof(float));
    if (e != hipSuccess) { std::cerr << "hipMalloc " << hipGetErrorString(e) << "\n"; return 11; }
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

try_variant() {
  name="$1"
  shift
  bin="/tmp/mcc26_flags_${SLURM_JOB_ID}_${name}"
  echo "===== ${name}: $* ====="
  hipcc -O2 "$@" "${SRC}" -o "${bin}"
  rc_compile=$?
  echo "compile_rc=${rc_compile}"
  if [ "${rc_compile}" -eq 0 ]; then
    "${bin}"
    echo "run_rc=$?"
  fi
}

echo "HIPCC=$(which hipcc)"
hipcc --version | head -4

try_variant default
try_variant offload_gfx906 --offload-arch=gfx906
try_variant amdgpu_target_gfx906 --amdgpu-target=gfx906
try_variant offload_gfx906_cov3 --offload-arch=gfx906 -mcode-object-version=3
try_variant amdgpu_gfx906_cov3 --amdgpu-target=gfx906 -mcode-object-version=3
try_variant mcpu_gfx906 -mcpu=gfx906
try_variant mcpu_gfx906_cov3 -mcpu=gfx906 -mcode-object-version=3
try_variant offload_gfx906_xnack_off --offload-arch=gfx906:xnack-
try_variant offload_gfx906_sramecc_off_xnack_off --offload-arch=gfx906:sramecc-:xnack-
try_variant offload_gfx906_sramecc_on_xnack_off --offload-arch=gfx906:sramecc+:xnack-
try_variant amdgpu_gfx906_xnack_off --amdgpu-target=gfx906:xnack-
try_variant offload_gfx900 --offload-arch=gfx900
try_variant offload_gfx926 --offload-arch=gfx926
