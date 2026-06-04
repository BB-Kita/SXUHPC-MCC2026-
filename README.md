# MCC26 GPU Output Optimization

This repository contains the current C++/HIP implementation used to optimize the
MCC26 climatology task on K100_AI DCUs.

Scope:

- Input: `/public/home/achwjznh4b/Newdata`
- Reference verification data: `/public/home/achwjznh4b/ERA5/Climatology`
- Target outputs: daily `Climmean` and `P90_sst` for `0601..0831` (`92` files)
- Window shape: `11` days x `30` years = `330` samples per target day

## Current best result

Best adopted two-node configuration:

- Job: `114789362`
- Script: `run_mcc26_2nodes_dtk24.slurm`
- Output: `/public/home/fujiake/fjk/MCC26_SXU_two_node_exp_20260603/output_2nodes_float_skipcleanup_0603f`
- Main `real`: `11.613s`
- Slower shard `real`: `10.458s`
- Output size: `731M`

Main runtime switches:

- `MCC_PRELOAD_SEASON=1`
- `MCC_PRELOAD_DIRECT_H2D=1`
- `MCC_COMPACT_PER_SLOT=1`
- `MCC_PRELOAD_COPY_THREADS=8`
- `MCC_ASYNC_DCU_INIT=1`
- `MCC_IO_THREADS=32`
- `MCC_OUTPUT_FLOAT=1`
- `MCC_SKIP_FINAL_CLEANUP=1`

Validation sample (`0602`, `0815`):

- Clim RMSE: `0.000004`, `0.000004`
- P90 RMSE: `0.011536`, `0.012882`

Single-node best baseline for comparison:

- Job: `114788132`
- Main `real`: `16.319s`
- `program_total`: `15.6449s`

## Fast profile summary

Per-shard profile for the fastest version:

| Metric | Shard A | Shard B |
| --- | ---: | ---: |
| DOY range | `152..197` | `198..243` |
| Subprocess `real` | `10.356s` | `10.458s` |
| `program_total` | `10.1442s` | `10.2935s` |
| Season read | `3.24575s` | `2.96961s` |
| Window init | `3.39859s` | `3.09762s` |
| Async DCU init wait | `2.4071s` | `2.93493s` |
| Day-loop sum | `2.81287s` | `2.776631s` |
| Write total | `0.356014s` | `0.362242s` |
| Incremental H2D avg | `25.899ms` | `25.239ms` |
| Kernel avg | `16.137ms` | `15.790ms` |
| D2H avg | `0.413ms` | `0.379ms` |

Interpretation:

- The per-day compute loop is already small, about `2.8s` per shard over `46` days.
- Fixed costs dominate: season preload, initial window setup, and DCU init wait.
- Output is no longer a primary bottleneck after `MCC_OUTPUT_FLOAT=1`.
- The main process still pays about `1.155s` of wrapper overhead beyond the slower shard.

Detailed profiling notes are recorded in:

- `FASTEST_PROFILE_114789362.md`
- `TWO_NODE_PROFILE_20260603.md`

## Key implementation points

- Raw HDF5 data path via `pread` when the input layout matches the expected `data`
  offset (`23488` bytes in the provided files)
- Sliding window update: first day uploads `330` slots, later days upload only `30`
  changed slots
- Persistent DCU buffers across days
- Local-slice H2D upload per GPU instead of full-domain copies
- OpenMP-parallel file reads and float conversion
- Official-logic verification wrapper kept in-repo

## Main files

- `main.cpp`: top-level control flow, runtime switches, output path, DOY range
- `io_handler.cpp`, `io_handler.h`: raw HDF5/NetCDF input, season preload, sliding reads
- `compute_dcu.cpp`, `compute_dcu.h`: HIP kernels and multi-DCU scheduling
- `config.h`: grid sizes, constants, default paths
- `run_mcc26_2nodes_dtk24.slurm`: current adopted two-node script
- `run_mcc26_shard_dtk24.slurm`: single-shard helper
- `run_validate_official_logic.slurm`: official verification wrapper
- `validate_official_logic.m`: official-logic verification script

## Two-node run

```bash
sbatch --export=ALL run_mcc26_2nodes_dtk24.slurm
```

## Official verification

```bash
sbatch --export=ALL, \
  MCC_REF_CLIM_PATH=/public/home/achwjznh4b/ERA5/Climatology, \
  MCC_CONTESTANT_CLIM_PATH=/path/to/output, \
  MCC_VERIFY_SAVE_PATH=/path/to/verification, \
  MCC_VALIDATE_FILES=0602,0815 \
  run_validate_official_logic.slurm
```

## Rejected directions

- Parallel DCU init: slower, likely allocation/runtime contention
- NetCDF classic output: much slower writes
- Minimal NetCDF variable set: did not improve wall time
- Single-`srun` two-task launch: worse than background dual `srun`
- Source-window first upload: unstable and slower in practice
- SSH launch to second allocated node: worse than `srun`
