# Wave–cuDF synthetic comparison

The two backend executables run the same deterministic, null-free, fixed-width
workloads and validate their materialized results against CPU Velox:

- `velox_wave_synthetic_benchmark`
- `velox_cudf_synthetic_benchmark`

Use the comparison runner to run each backend in a separate process and write
merged JSON and CSV. It runs on the device given by `--gpu` (0 by default) and
never picks a device itself. Every measured run takes an exclusive `flock` on
`--gpu-lock` and waits for the device to go idle before starting, so several
agents can share one GPU without corrupting each other's timings:

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

## Parquet input mode

`--input parquet` gives both backends the same logical input read from a
Parquet file instead of vectors built in memory, so the comparison includes a
scan phase. Execution is deliberately two-stage: the CPU Hive scan runs to
completion *before* either backend registers its driver adapter, because Wave
and cuDF both install global adapters and Wave has no Parquet split reader.
The host vectors then feed a `Values`-rooted plan on the GPU.

```bash
python3 velox/experimental/gpu/benchmarks/compare_synthetic.py \
  --build-dir _build/release \
  --input parquet \
  --parquet-dir /tmp/velox_synthetic_parquet \
  --rows 10000000 \
  --output-prefix parquet_10m
```

Three timings are reported per backend: `scan_*_ms` for the CPU Parquet read,
`cold_ms`/`warm_median_ms` for the GPU stage including the host-to-device
upload (this is what the pre-Parquet results measured), and `total_*_ms` for
the sum. The runner asserts that the total equals the sum of the other two. The
scan is measured `--scan-repetitions` times; the first read is the cold number
and the rest are warm, with only the last read kept in memory.

Wave compiles its kernels with NVRTC during the first measured workload, which
shows up as one outlier sample and a p95 far above the median. Give it extra
warmups with `--wave-warmups` to keep compilation out of the samples.

### Writing fixtures

Fixtures are written on demand and cached on disk under `--parquet-dir`, in a
directory named after the row count, group cardinality, and file shape. Pass
`--write-fixture` to overwrite one. Generation streams 100K rows at a time, so
writing a 1B-row fixture does not need 1B rows of host memory. A 1B-row fact
table is roughly 80 GB of plain, uncompressed Parquet, so delete fixtures you
no longer need.

### Batch size

Batch size is a per-backend setting, because the two backends want opposite
shapes: Wave is only reliable at 100K rows per batch, while cuDF wants as few
batches as possible. `--wave-batch-rows` and `--cudf-batch-rows` both take a
list, so a backend can be swept over several sizes in one matrix, and `0` means
"the whole input in one batch". Every output row records the batch size, the
row-group size, and the batch sizes the reader actually produced, so a run's
configuration is never ambiguous.

The requested batch size drives three things:

- `max_output_batch_rows` and `preferred_output_batch_bytes`, which cap what
  the scan operator asks for.
- The fixture's row-group size, which is `min(batch rows,
  --max-row-group-rows)`, because a reader batch never crosses a row-group
  boundary.
- Stitching after the scan. A row group has to fit in one sink write, since the
  local file sink writes through `fwrite` and that tops out just below 2 GB —
  with the ten BIGINT columns used here that caps a row group near 13M rows. A
  batch larger than a row group is assembled by concatenating reader batches,
  and that copy is charged to the scan phase.

The join workloads (`join_aggregate`, `q3`) need their whole input in one batch
and are stitched the same way.

### Fixture tuning and the parity control

By default each backend reads a fixture tuned for it — Wave 100K-row row
groups, cuDF 10M-row row groups, both plain, uncompressed, 1 MB pages. The
logical input and the result checksums are identical either way, but the two
backends are then reading *different files*, so their scan timings are not
comparable to each other. Report the scan per backend rather than as a shared
constant.

`--shared-row-group-rows N` pins one layout for every backend, which makes the
scan a real control: same file, same encoding, same pages, only the requested
batch size differs. 10M is the useful setting, because it is the only single
layout that hands both backends their requested batch sizes without stitching.

In Parquet mode Wave uses the shared `Values` plan for every workload; the
`wavemock` reader path is what `--input generated` still exercises, and it
remains the reference for what an in-memory Wave scan costs. cuDF stops
pre-uploading its input with `toGpu`, so `CudfFromVelox` lands inside the
compute timer.

Wave scan measurements use the in-memory `wavemock` reader; they do not include
Parquet or ORC I/O. Wave cannot currently fuse its generated hash-join kernel
with the requested post-join aggregation reliably, so both backends materialize
the join output and execute the aggregation as a second GPU task. Q3 filters are
applied at that second stage; this is equivalent for the inner join used here.
Join inputs use one batch because Wave currently produces incorrect results
when a `Values`-backed hash join receives multiple probe or build batches.
