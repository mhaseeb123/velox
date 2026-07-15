# Wave–cuDF synthetic comparison

The two backend executables run the same deterministic, null-free, fixed-width
workloads and validate their materialized results against CPU Velox:

- `velox_wave_synthetic_benchmark`
- `velox_cudf_synthetic_benchmark`

Use the comparison runner to select the least busy GPU, run each backend in a
separate process, and write merged JSON and CSV. Unless `--gpu` is specified,
the runner queries GPU utilization again before every backend/workload run:

```bash
python3 velox/experimental/gpu/benchmarks/compare_synthetic.py \
  --build-dir _build/release \
  --rows 1000000 \
  --warmups 2 \
  --repetitions 10 \
  --output-prefix synthetic_gpu_results
```

Available workloads are `scan`, `filter`, `project`, `aggregate`, `group_by`,
`join_aggregate`, `q1`, `q6`, and `q3`. Use `--filter-percent` and
`--group-cardinality` to vary filter selectivity and group count.

Wave scan measurements use the in-memory `wavemock` reader; they do not include
Parquet or ORC I/O. Wave cannot currently fuse its generated hash-join kernel
with the requested post-join aggregation reliably, so both backends materialize
the join output and execute the aggregation as a second GPU task. Q3 filters are
applied at that second stage; this is equivalent for the inner join used here.
Join inputs use one batch because Wave currently produces incorrect results
when a `Values`-backed hash join receives multiple probe or build batches.
