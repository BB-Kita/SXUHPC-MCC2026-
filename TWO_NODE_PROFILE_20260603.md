# Two-node sharding profile, 2026-06-03

Remote project:

`/public/home/fujiake/fjk/MCC26_SXU_two_node_exp_20260603`

## Best single-node baseline

- Output: `/public/home/fujiake/fjk/MCC26_SXU_gpuopt_exp_20260601/output_full_preload_direct_asyncinit_t32`
- Job: `114788132`
- Config: `MCC_PRELOAD_SEASON=1`, `MCC_PRELOAD_DIRECT_H2D=1`, `MCC_COMPACT_PER_SLOT=1`,
  `MCC_PRELOAD_COPY_THREADS=8`, `MCC_ASYNC_DCU_INIT=1`, `MCC_IO_THREADS=32`
- `real`: `0m16.319s`
- `program_total`: `15.6449s`
- Validation sample `0602`: Clim RMSE `0.000004`, P90 RMSE `0.011536`

## Two separate Slurm jobs

- Jobs: `114788572` and `114788573`
- Nodes: `f16r4n13`, `f16r4n14`
- Shard A: DOY `152..197`
- Shard B: DOY `198..243`
- Output: `/public/home/fujiake/fjk/MCC26_SXU_two_node_exp_20260603/output_two_node_preload_direct_0603a`
- Output files: `92`
- Shard real: A `0m12.000s`, B `0m10.915s`
- Effective parallel real: about `12.000s`
- Validation `0602,0815`: Clim mean RMSE `0.0000`, P90 mean RMSE `0.0122`

## Single Slurm job with two nodes

- Job: `114788698`
- Nodes: `f17r4n00`, `f17r4n01`
- Script: `run_mcc26_2nodes_dtk24.slurm`
- Output: `/public/home/fujiake/fjk/MCC26_SXU_two_node_exp_20260603/output_2nodes_injob_preload_direct_0603b`
- Output files: `92`
- Main `time` real around the parallel section: `0m12.225s`
- Shard A real: `0m11.099s`, `program_total=10.6069s`
- Shard B real: `0m10.869s`, `program_total=10.3351s`
- Validation `0602,0815`: Clim daily RMSE `0.000004, 0.000004`; P90 daily RMSE `0.011536, 0.012882`

Key profile for job `114788698`:

| Shard | Days | Preload read | Init window | Async DCU wait | Upload avg | Upload max | Kernel avg | D2H avg | Write total | Day-loop sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 46 | `2.7975s` | `2.89785s` | `2.65055s` | `39.172ms` | `538.648ms` | `16.365ms` | `0.488ms` | `0.490s` | `3.091s` |
| B | 46 | `2.78122s` | `2.92801s` | `2.60282s` | `38.756ms` | `524.317ms` | `15.906ms` | `0.412ms` | `0.463s` | `3.003s` |

## Takeaways

- Two-node date sharding is a real win under the "at most 2 servers" rule: `16.319s -> 12.225s`.
- The gain is smaller than a perfect half split because each node pays fixed costs again:
  season preload, initial 330-slot H2D, and DCU buffer initialization.
- The day loop itself is now only about `3.0s` per shard; further wins need to attack fixed setup cost or first-day upload.
- The current output path is not the dominant limiter in the two-node result: writing is about `0.46-0.49s` per shard.

## Follow-up fixed-cost tests

Best adopted combination:

- Script: `run_mcc26_2nodes_dtk24.slurm`
- Job: `114789362`
- Output: `/public/home/fujiake/fjk/MCC26_SXU_two_node_exp_20260603/output_2nodes_float_skipcleanup_0603f`
- Config: `MCC_OUTPUT_FLOAT=1`, `MCC_SKIP_FINAL_CLEANUP=1`, `MCC_PARALLEL_DCU_INIT=0`,
  `MCC_OUTPUT_CLASSIC=0`, `MCC_MINIMAL_OUTPUT=0`
- Output files: `92`
- Output size: `731M`
- Main `real`: `0m11.613s`
- Shard real: A `0m10.356s`, B `0m10.458s`
- Validation `0602,0815`: Clim daily RMSE `0.000004, 0.000004`; P90 daily RMSE `0.011536, 0.012882`

Rejected variants:

| Variant | Job | Main real | Result |
| --- | ---: | ---: | --- |
| Parallel DCU init + skip cleanup | `114789198` | `0m12.974s` | slower, likely ROCm allocation contention |
| Skip cleanup only | `114789300` | `0m11.820s` | useful but weaker than float+skip |
| NetCDF classic + float + skip | `114789502` | `0m14.373s` | classic writes were much slower |
| Minimal NetCDF vars + float + skip | `114789638` | `0m12.514s` | smaller metadata did not help wall time |
| Single `srun` with 2 tasks | `114789883` | `0m12.168s` | launch/slow shard overhead higher |
| `ROCR_VISIBLE_DEVICES=0,1` | `114790037` | `0m11.687s` | did not actually reduce visible DCUs |
| Source-window first upload | `114790421` | `0m15.310s` | unstable/worse; keep opt-in only |
| SSH launch to second allocated node | `114820552` | `0m13.163s` | compute-node ssh launch overhead worse than `srun` |
