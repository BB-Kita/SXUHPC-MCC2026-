#!/usr/bin/env bash
#SBATCH -J mcc_baseline
#SBATCH -p kshdmcc2026
#SBATCH -N 1 -n 4 --exclusive
#SBATCH --exclude=f16r4n03
#SBATCH --gres=dcu:4
#SBATCH --time=01:00:00
#SBATCH -o logs/mcc_baseline_%j.out
#SBATCH -e logs/mcc_baseline_%j.err

set -eo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "${SCRIPT_DIR}"
mkdir -p logs

export LD_LIBRARY_PATH=""
source /public/software/compiler/rocm/dtk-25.04/env.sh
export LD_LIBRARY_PATH="/public/home/fujiake/miniconda3/envs/lsd/lib:${LD_LIBRARY_PATH}"

: "${MCC_SOURCE_MC_K:=30}"
: "${MCC_SOURCE_MC_INDEPENDENT:=0}"
: "${MCC_SOURCE_MC_FLAT:=0}"
: "${MCC_SOURCE_STREAM_CHUNK_DAYS:=30}"
: "${MCC_SOURCE_IO_BACKEND:=stdthread}"
: "${MCC_SOURCE_REGISTER_UPLOAD:=1}"
: "${MCC_SOURCE_ASYNC_UPLOAD:=0}"
: "${MCC_OUTPUT_LIGHT_CLASSIC:=0}"

export MCC_IO_THREADS=32
export OMP_NUM_THREADS=32
export OMP_PROC_BIND=close
export OMP_PLACES=cores
: "${MCC_TARGET_DOY_BEGIN:=152}"
: "${MCC_TARGET_DOY_END:=243}"
export MCC_TARGET_DOY_BEGIN
export MCC_TARGET_DOY_END
export MCC_SEASON_PRELOAD=1
export MCC_SOURCE_LOCAL_LAYOUT=1
export MCC_SOURCE_STREAM=1
export MCC_SOURCE_STREAM_CHUNK_DAYS
export MCC_SOURCE_IO_BACKEND
export MCC_SOURCE_LOCAL_PINNED=0
export MCC_SOURCE_REGISTER_UPLOAD
unset MCC_SOURCE_KEEP_REGISTERED
if [ "${MCC_SOURCE_ASYNC_UPLOAD}" = "0" ]; then
  unset MCC_SOURCE_ASYNC_UPLOAD
else
  export MCC_SOURCE_ASYNC_UPLOAD
fi
export MCC_SOURCE_MC_K
unset MCC_SOURCE_MC_EXACT_MEAN
if [ "${MCC_SOURCE_MC_FLAT}" = "0" ]; then
  unset MCC_SOURCE_MC_FLAT
else
  export MCC_SOURCE_MC_FLAT
fi
if [ "${MCC_SOURCE_MC_INDEPENDENT}" = "0" ]; then
  unset MCC_SOURCE_MC_INDEPENDENT
else
  export MCC_SOURCE_MC_INDEPENDENT
fi
unset MCC_SOURCE_SELECT_P90
unset MCC_SOURCE_FAST_HIST
unset MCC_SOURCE_FIXED_COLLECT
export MCC_SKIP_ZERO=1
export MCC_OUTPUT_FLOAT=1
export MCC_OUTPUT_CLASSIC=1
if [ "${MCC_OUTPUT_LIGHT_CLASSIC}" = "0" ]; then
  unset MCC_OUTPUT_LIGHT_CLASSIC
else
  export MCC_OUTPUT_LIGHT_CLASSIC
fi
export MCC_RAW_DIRECT=1
export MCC_RAW_DIRECT_ALIGN=512
unset MCC_RAW_MMAP
unset MCC_MC_K
unset MCC_MC_PACKED
unset MCC_MC_BLOCK
unset MCC_MC_CONSISTENT
unset MCC_USE_FP16

mode="block"
if [ -n "${MCC_SOURCE_MC_INDEPENDENT:-}" ]; then
  mode="independent"
fi
reg_mode="reg"
if [ "${MCC_SOURCE_REGISTER_UPLOAD}" = "0" ]; then
  reg_mode="noreg"
fi
async_mode="syncup"
if [ -n "${MCC_SOURCE_ASYNC_UPLOAD:-}" ]; then
  async_mode="asyncup"
fi
light_mode="classic"
if [ -n "${MCC_OUTPUT_LIGHT_CLASSIC:-}" ]; then
  light_mode="lightclassic"
fi
sample_mode="peryear"
if [ "${MCC_SOURCE_MC_K}" -lt 30 ] || [ -n "${MCC_SOURCE_MC_FLAT:-}" ]; then
  sample_mode="flat"
fi
export MCC_OUTPUT_DIR="${MCC_OUTPUT_DIR:-${SCRIPT_DIR}/output}"
mkdir -p "${MCC_OUTPUT_DIR}"

echo "========================================="
echo "  MCC26 mcc_baseline source-window MC run"
echo "  Node: $(hostname)"
echo "  Date: $(date)"
echo "  Output: ${MCC_OUTPUT_DIR}"
echo "  MCC_SOURCE_MC_K=${MCC_SOURCE_MC_K}"
echo "  target_doy=${MCC_TARGET_DOY_BEGIN}..${MCC_TARGET_DOY_END}"
echo "  source_mc_mode=${mode}"
echo "  source_mc_sample=${sample_mode}"
echo "  source_mc_mean=sample"
echo "  chunk_days=${MCC_SOURCE_STREAM_CHUNK_DAYS}"
echo "  source_io_backend=${MCC_SOURCE_IO_BACKEND}"
echo "  register_upload=${MCC_SOURCE_REGISTER_UPLOAD}"
echo "  async_upload=${async_mode}"
echo "  output_mode=${light_mode}"
echo "========================================="

/usr/bin/time -p ./build/mcc_baseline 2>&1

echo "Output files: $(ls "${MCC_OUTPUT_DIR}" | wc -l)"
du -sh "${MCC_OUTPUT_DIR}" || true
echo "Done at $(date)"
