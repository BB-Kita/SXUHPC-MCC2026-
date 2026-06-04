#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="${MCC_WORK_DIR:-/public/home/fujiake/fjk/MCC26_SXU_two_node_exp_20260603}"
OUTPUT_DIR="${MCC_OUTPUT_DIR:-${WORK_DIR}/output_two_node_preload_direct}"
EXCLUDE_NODES="${MCC_EXCLUDE_NODES:-f16r4n03}"

mkdir -p "${WORK_DIR}/logs" "${OUTPUT_DIR}"

common_export="ALL,MCC_WORK_DIR=${WORK_DIR},MCC_OUTPUT_DIR=${OUTPUT_DIR},MCC_PRELOAD_SEASON=1,MCC_PRELOAD_DIRECT_H2D=1,MCC_COMPACT_PER_SLOT=1,MCC_PRELOAD_COPY_THREADS=8,MCC_ASYNC_DCU_INIT=1,MCC_IO_THREADS=32"

job_a=$(sbatch --parsable --exclude="${EXCLUDE_NODES}" \
  --export="${common_export},MCC_SHARD_NAME=A,MCC_TARGET_DOY_BEGIN=152,MCC_TARGET_DOY_END=197" \
  run_mcc26_shard_dtk24.slurm)

job_b=$(sbatch --parsable --exclude="${EXCLUDE_NODES}" \
  --export="${common_export},MCC_SHARD_NAME=B,MCC_TARGET_DOY_BEGIN=198,MCC_TARGET_DOY_END=243" \
  run_mcc26_shard_dtk24.slurm)

echo "JOB_A=${job_a}"
echo "JOB_B=${job_b}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
