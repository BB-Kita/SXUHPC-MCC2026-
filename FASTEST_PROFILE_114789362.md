# Fastest version profile: job 114789362

Remote output:

`/public/home/fujiake/fjk/MCC26_SXU_two_node_exp_20260603/output_2nodes_float_skipcleanup_0603f`

Config:

- `MCC_PRELOAD_SEASON=1`
- `MCC_PRELOAD_DIRECT_H2D=1`
- `MCC_COMPACT_PER_SLOT=1`
- `MCC_PRELOAD_COPY_THREADS=8`
- `MCC_ASYNC_DCU_INIT=1`
- `MCC_IO_THREADS=32`
- `MCC_OUTPUT_FLOAT=1`
- `MCC_SKIP_FINAL_CLEANUP=1`

Overall:

- Main `real`: `11.613s`
- Max shard `real`: `10.458s`
- Slurm/srun wrapper overhead: about `1.155s`
- Output files: `92`
- Output size: `731M`
- Validation sample `0602,0815`: Clim daily RMSE `0.000004, 0.000004`; P90 daily RMSE `0.011536, 0.012882`

## Shard profile

| Metric | Shard A | Shard B |
| --- | ---: | ---: |
| DOY range | `152..197` | `198..243` |
| Days | `46` | `46` |
| Subprocess `real` | `10.356s` | `10.458s` |
| `program_total` | `10.1442s` | `10.2935s` |
| Season timeline read | `3.24575s` | `2.96961s` |
| Preload initialize window | `3.39859s` | `3.09762s` |
| Async DCU init wait | `2.4071s` | `2.93493s` |
| Day-loop sum | `2.81287s` | `2.776631s` |
| Write total | `0.356014s` | `0.362242s` |
| Write average | `7.739ms` | `7.875ms` |
| Write max | `19.743ms` | `25.540ms` |

## DCU transfer/kernel profile

| Metric | Shard A | Shard B |
| --- | ---: | ---: |
| First full upload | `526.293ms` | `531.910ms` |
| First full kernel | `18.969ms` | `19.569ms` |
| First full D2H | `4.371ms` | `3.347ms` |
| First full DCU total | `549.633ms` | `554.827ms` |
| Incremental upload avg | `25.899ms` | `25.239ms` |
| Incremental upload p50 | `25.314ms` | `23.985ms` |
| Incremental upload p90 | `33.337ms` | `33.122ms` |
| Incremental upload max | `34.822ms` | `34.344ms` |
| Kernel avg | `16.137ms` | `15.790ms` |
| Kernel max | `18.969ms` | `19.569ms` |
| D2H avg | `0.413ms` | `0.379ms` |
| DCU total avg | `53.327ms` | `52.423ms` |

## Interpretation

- The 46-day compute loop is already small: about `2.8s` per shard.
- Per-shard fixed cost dominates:
  season read plus initial window setup is about `3.1-3.4s`, and waiting for async DCU init is `2.4-2.9s`.
- Output is no longer a primary bottleneck after `MCC_OUTPUT_FLOAT=1`: about `0.36s` per shard.
- The main process pays about `1.16s` beyond the slower shard, mostly launch/wait/srun overhead.
- Further large wins need to reduce fixed costs, especially DCU init wait and season timeline read/init, rather than shaving per-day kernel or write time.
