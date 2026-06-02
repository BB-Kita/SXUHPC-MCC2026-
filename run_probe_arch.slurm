#!/usr/bin/env bash
#SBATCH -J mcc26_arch
#SBATCH -p kshdmcc2026
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=dcu:1
#SBATCH --time=00:05:00
#SBATCH -o /public/home/fujiake/fjk/MCC26_SXU_gpuopt_exp_20260601/logs/arch_%j.out
#SBATCH -e /public/home/fujiake/fjk/MCC26_SXU_gpuopt_exp_20260601/logs/arch_%j.err

set -euo pipefail

source /public/home/fujiake/miniconda3/bin/activate
conda activate lsd

echo "NODE $(hostname)"
echo "HIPCC $(which hipcc)"
hipcc --version | head -5

echo "ROCMINFO $(which rocminfo || true)"
if command -v rocminfo >/dev/null 2>&1; then
  rocminfo 2>/dev/null | grep -E 'Name: *gfx|Marketing Name|Uuid' | head -40 || true
fi

cat > /tmp/mcc26_hip_props_${SLURM_JOB_ID}.cpp <<'CPP'
#include <hip/hip_runtime.h>
#include <iostream>
int main() {
    int n = 0;
    hipGetDeviceCount(&n);
    std::cout << "device_count=" << n << "\n";
    for (int i = 0; i < n; ++i) {
        hipDeviceProp_t p{};
        hipGetDeviceProperties(&p, i);
        std::cout << "device=" << i
                  << " name=" << p.name
                  << " gcnArchName=" << p.gcnArchName
                  << " major=" << p.major
                  << " minor=" << p.minor
                  << "\n";
    }
}
CPP

hipcc /tmp/mcc26_hip_props_${SLURM_JOB_ID}.cpp -o /tmp/mcc26_hip_props_${SLURM_JOB_ID}
/tmp/mcc26_hip_props_${SLURM_JOB_ID}
