#!/usr/bin/env python3

"""Runs matched Wave and cuDF synthetic benchmarks on one GPU."""

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


WORKLOADS = [
    "scan",
    "filter",
    "project",
    "aggregate",
    "group_by",
    "join_aggregate",
    "q1",
    "q6",
    "q3",
]


def settle_script(gpu, max_memory_mib=1024, attempts=60):
    """Shell snippet that waits for the GPU to go idle. It runs after the lock
    is taken, because a process that has just exited can still hold memory."""
    return (
        f"for _ in $(seq 1 {attempts}); do "
        f"u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i {gpu}); "
        f"m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {gpu}); "
        f'if [ "$u" -le 2 ] && [ "$m" -le {max_memory_mib} ]; then break; fi; '
        "sleep 2; done"
    )


def parse_result(output):
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "backend" in value:
            return value
    raise RuntimeError(f"Benchmark produced no JSON result:\n{output}")


# Ten BIGINT columns. Used to turn a nominal row target into the byte budget
# that is the only batch-size knob the cuDF GPU reader has.
BYTES_PER_ROW = 10 * 8


def backend_scan_modes(backend, args):
    """Wave always reads on the CPU; only cuDF has a native GPU reader."""
    return ["cpu"] if backend == "wave" else args.cudf_scan


def backend_batch_rows(backend, args, rows):
    """Batch sizes the two backends are run at. Wave is only reliable at small
    batches; cuDF wants the input in as few batches as possible, so it is swept.
    Zero means 'the whole input in one batch'."""
    sizes = args.wave_batch_rows if backend == "wave" else args.cudf_batch_rows
    return [rows if size == 0 else size for size in sizes]


def row_group_rows(batch_rows, args):
    """Row groups bound the reader's batch size, because a Parquet batch never
    crosses a row-group boundary. The writer caps how large a row group can be,
    so batches above that size are stitched together after the scan instead.

    Tuning the row group per backend means the two backends read different
    files, which makes their scan cost incomparable. --shared-row-group-rows
    pins one layout for everyone so the scan becomes a real control."""
    if args.shared_row_group_rows:
        return args.shared_row_group_rows
    return min(batch_rows, args.max_row_group_rows)


def backend_warmups(backend, args):
    """Wave compiles kernels with NVRTC on its first measured workload, so one
    warmup leaves compilation inside the samples and inflates p95."""
    if backend == "wave" and args.wave_warmups is not None:
        return args.wave_warmups
    return args.warmups


def check_phase_boundaries(result):
    """total = scan + compute, for both cold and warm. Cheap guard against a
    future change quietly moving work across a phase boundary."""
    for total, scan, compute in (
        ("total_cold_ms", "scan_cold_ms", "cold_ms"),
        ("total_warm_median_ms", "scan_warm_median_ms", "warm_median_ms"),
    ):
        expected = result[scan] + result[compute]
        if abs(result[total] - expected) > 1e-6 * max(1.0, abs(expected)):
            raise RuntimeError(
                f"{result['backend']} {result['workload']}: {total} is "
                f"{result[total]}, but {scan} + {compute} is {expected}"
            )


# How many trailing lines of the binary's output to keep in a failure record.
# Enough for a Velox exception's Reason/Expression block plus a little context.
FAILURE_OUTPUT_LINES = 40


def classify_failure(output, returncode):
    """Names the kind of failure from the binary's own output.

    Exit code alone cannot do this: 134 is what a VELOX_CHECK throw out of
    main() looks like, and that covers both a wrong answer and an unreadable
    file. Only the message separates them.
    """
    text = output.lower()
    if "checksum == expectedchecksum" in text or "expected_checksum" in text:
        return "checksum_mismatch"
    if any(
        marker in text
        for marker in (
            "out of memory",
            "bad_alloc",
            "cudaerrormemoryallocation",
            "rmm::out_of_memory",
        )
    ):
        return "oom"
    if any(
        marker in text
        for marker in (
            "pagereader",
            "fwrite failure",
            "no such file",
            "parquet fixture",
            "no registered file system",
            "corrupt",
        )
    ):
        return "read_error"
    if returncode in (134, -6) or "terminate called" in text or "sigabrt" in text:
        return "abort"
    return "unknown"


class BenchmarkFailure(RuntimeError):
    """A failed run, carrying enough of the binary's output that a reader of
    failures.json never has to re-run anything to learn what happened."""

    def __init__(self, message, returncode, output, failure_class=None):
        super().__init__(message)
        self.returncode = returncode
        self.output = output or ""
        self.failure_class = failure_class or classify_failure(
            self.output, returncode
        )

    def tail(self):
        lines = [line for line in self.output.splitlines() if line.strip()]
        return "\n".join(lines[-FAILURE_OUTPUT_LINES:])


def run_one(binary, backend, workload, batch_rows, scan_mode, args, gpu):
    command = [
        str(binary),
        f"--synthetic_workload={workload}",
        f"--synthetic_rows={args.rows}",
        f"--synthetic_batch_rows={batch_rows}",
        f"--synthetic_filter_percent={args.filter_percent}",
        f"--synthetic_group_cardinality={args.group_cardinality}",
        f"--synthetic_warmups={backend_warmups(backend, args)}",
        f"--synthetic_repetitions={args.repetitions}",
        f"--synthetic_input={args.input}",
        "--synthetic_format=json",
    ]
    chunk_read_bytes = 0
    if args.input == "parquet":
        command += [
            f"--synthetic_parquet_dir={args.parquet_dir}",
            f"--synthetic_parquet_row_group_rows="
            f"{row_group_rows(batch_rows, args)}",
            f"--synthetic_parquet_page_bytes={args.parquet_page_bytes}",
            f"--synthetic_parquet_compression={args.parquet_compression}",
            f"--synthetic_parquet_dictionary={str(args.parquet_dictionary).lower()}",
        ]
        if scan_mode == "gpu":
            # The GPU reader ignores every row-based batch knob; a byte budget
            # is the only equivalent, and the row figure it comes from is
            # nominal because the reader quantizes at page boundaries.
            chunk_read_bytes = batch_rows * BYTES_PER_ROW
            command += [
                "--synthetic_cudf_scan=gpu",
                f"--synthetic_cudf_chunk_read_bytes={chunk_read_bytes}",
                f"--synthetic_cudf_pass_read_bytes={args.cudf_pass_read_bytes}",
                f"--synthetic_cudf_buffered_input="
                f"{str(args.cudf_buffered_input).lower()}",
            ]
        else:
            command += [
                f"--synthetic_scan_batch_rows={batch_rows}",
                f"--synthetic_scan_repetitions={args.scan_repetitions}",
            ]
            if backend == "cudf":
                command.append("--synthetic_cudf_scan=cpu")
        if args.write_fixture:
            command.append("--synthetic_write_fixture=true")
    if args.skip_cpu_validation:
        command.append("--synthetic_cpu_validation=false")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Other agents share this GPU, and a concurrent workload silently corrupts
    # timings, so every measured run holds an exclusive lock for its whole
    # duration. Blocking here is intended.
    locked = [
        "flock",
        str(args.gpu_lock),
        "-c",
        f"{settle_script(gpu)}; {shlex.join(command)}",
    ]
    completed = subprocess.run(
        locked,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode:
        sys.stderr.write(completed.stdout)
        raise BenchmarkFailure(
            f"{backend} {workload} failed with exit code {completed.returncode}",
            returncode=completed.returncode,
            output=completed.stdout,
        )
    result = parse_result(completed.stdout)
    if not result["correct"]:
        raise BenchmarkFailure(
            f"{backend} {workload} failed correctness validation",
            returncode=0,
            output=completed.stdout,
            failure_class="checksum_mismatch",
        )
    if args.input == "parquet":
        # Still holds in GPU-scan mode, where the scan fields are zero because
        # the scan is fused into the measured plan.
        check_phase_boundaries(result)
    result["chunk_read_bytes"] = chunk_read_bytes
    return result


def write_results(results, output_prefix):
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    fields = [
        "backend",
        "gpu",
        "cudf_pin",
        "workload",
        "input_mode",
        "scan_mode",
        "rows",
        "batch_rows",
        "input_bytes",
        "filter_percent",
        "group_cardinality",
        "warmups",
        "repetitions",
        "checksum",
        "expected_checksum",
        "correct",
        "num_batches",
        "max_batch_rows",
        "min_batch_rows",
        "shared_fixture",
        "row_group_rows",
        "scan_batch_rows",
        "chunk_read_bytes",
        "scan_cold_ms",
        "scan_warm_median_ms",
        "cold_ms",
        "warm_median_ms",
        "warm_p95_ms",
        "total_cold_ms",
        "total_warm_median_ms",
        "rows_per_second",
        "input_gb_per_second",
    ]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    return json_path, csv_path


def print_comparison(results, fail_on_mismatch=True):
    by_workload = {}
    mismatches = []
    for result in results:
        # Keyed on the scan mode as well as the backend. Wave decodes Parquet on
        # the CPU in every configuration, so a gpu-decode total shares no scan
        # floor with it; letting one win the plain "cudf" slot would divide two
        # numbers that have nothing in common.
        key = (result["backend"], result.get("scan_mode", "cpu-prepass"))
        best = by_workload.setdefault(result["workload"], {}).get(key)
        if best is None or result["warm_median_ms"] < best["warm_median_ms"]:
            by_workload[result["workload"]][key] = result
    parquet = any(result.get("input_mode") == "parquet" for result in results)
    if parquet:
        print(
            f"{'workload':<16} {'wave scan':>10} {'wave gpu':>10} "
            f"{'wave total':>11} {'cuDF scan':>10} {'cuDF gpu':>10} "
            f"{'cuDF total':>11} {'gpu speedup':>12} {'checksums':>10}"
        )
    else:
        print(
            f"{'workload':<16} {'wave warm ms':>14} {'cuDF warm ms':>14} "
            f"{'Wave speedup':>18} {'checksums':>12}"
        )
    for workload in WORKLOADS:
        pair = by_workload.get(workload, {})
        wave = pair.get(("wave", "cpu-prepass"))
        cudf = pair.get(("cudf", "cpu-prepass"))
        cudfGpu = pair.get(("cudf", "gpu-decode"))
        if wave is not None and cudf is not None:
            speedup = cudf["warm_median_ms"] / wave["warm_median_ms"]
            checksums = (
                "match" if wave["checksum"] == cudf["checksum"] else "MISMATCH"
            )
            if parquet:
                print(
                    f"{workload:<16} {wave['scan_warm_median_ms']:>10.1f} "
                    f"{wave['warm_median_ms']:>10.1f} "
                    f"{wave['total_warm_median_ms']:>11.1f} "
                    f"{cudf['scan_warm_median_ms']:>10.1f} "
                    f"{cudf['warm_median_ms']:>10.1f} "
                    f"{cudf['total_warm_median_ms']:>11.1f} "
                    f"{speedup:>12.3f} {checksums:>10} "
                    f"(cuDF batch {cudf['batch_rows']})"
                )
            else:
                print(
                    f"{workload:<16} {wave['warm_median_ms']:>14.3f} "
                    f"{cudf['warm_median_ms']:>14.3f} {speedup:>18.3f} "
                    f"{checksums:>12}"
                )
            if checksums != "match":
                mismatches.append(workload)
        if cudfGpu is not None:
            # Its own line, with the scan columns dashed out, because a fused
            # GPU decode has no separable scan phase. The only ratio that means
            # anything here is against cuDF's own cpu-prepass total.
            versusCpu = (
                f"{cudf['total_warm_median_ms'] / cudfGpu['total_warm_median_ms']:.3f}"
                if cudf is not None
                else "-"
            )
            print(
                f"{workload + ' [cuDF gpu scan]':<16} {'-':>10} {'-':>10} "
                f"{'-':>11} {'-':>10} {'-':>10} "
                f"{cudfGpu['total_warm_median_ms']:>11.1f} "
                f"{versusCpu:>12} {'':>10} "
                f"(chunk {cudfGpu.get('chunk_read_bytes', 0)} B)"
            )
    if parquet:
        print(
            "\ncpu-prepass rows decode Parquet on the CPU before the measured "
            "pipeline. The scan is reported per backend, because the two "
            "backends read a fixture each unless --shared-row-group-rows pins "
            "one. gpu speedup compares the compute phase only.\n"
            "gpu-decode rows decode Parquet on the device inside one fused "
            "pipeline: no separable scan phase, and no scan floor in common "
            "with Wave, which decodes on the CPU in every configuration. Their "
            "ratio column is cuDF cpu-prepass total over cuDF gpu-decode "
            "total, never anything involving Wave."
        )
    else:
        print("\nWave scan timings use the wavemock reader, not Parquet or ORC I/O.")
    if mismatches and fail_on_mismatch:
        raise RuntimeError(
            f"backend checksum mismatch for {', '.join(mismatches)}"
        )
    return mismatches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=Path("_build/release"))
    # The device is pinned rather than chosen dynamically: GPU 0 is the
    # reserved device and every agent contends on one lock file.
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--gpu-lock", type=Path, default=Path("/tmp/velox_gpu0_benchmark.lock")
    )
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--cudf-pin", default="26.08")
    parser.add_argument("--filter-percent", type=int, default=10)
    parser.add_argument("--group-cardinality", type=int, default=1_000)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--wave-warmups", type=int)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=WORKLOADS)
    parser.add_argument(
        "--backends", nargs="+", choices=["wave", "cudf"], default=["wave", "cudf"]
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-cpu-validation", action="store_true")
    parser.add_argument("--input", choices=["generated", "parquet"], default="generated")
    # A list, so one invocation can produce both the matched CPU-scan
    # comparison and the cuDF-native one.
    parser.add_argument(
        "--cudf-scan", nargs="+", choices=["cpu", "gpu"], default=["cpu"]
    )
    parser.add_argument("--cudf-pass-read-bytes", type=int, default=0)
    parser.add_argument(
        "--cudf-buffered-input",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--parquet-dir", type=Path, default=Path("/tmp/velox_synthetic_parquet")
    )
    parser.add_argument("--parquet-page-bytes", type=int, default=1 << 20)
    parser.add_argument("--parquet-compression", default="none")
    parser.add_argument("--parquet-dictionary", action="store_true")
    parser.add_argument("--write-fixture", action="store_true")
    parser.add_argument("--scan-repetitions", type=int, default=2)
    # Batch size is per backend, and each backend is swept over its list. 0
    # means "the whole input in one batch".
    parser.add_argument("--wave-batch-rows", nargs="+", type=int, default=[100_000])
    parser.add_argument(
        "--cudf-batch-rows", nargs="+", type=int, default=[10_000_000, 100_000_000]
    )
    # Ten BIGINT columns at this many rows is under the 2 GB that the local file
    # sink can write in one call. Larger batches are stitched after the scan.
    parser.add_argument("--max-row-group-rows", type=int, default=10_000_000)
    # Makes every backend read one byte-identical fixture, so the scan phase is
    # comparable across backends instead of being per-backend tuned.
    parser.add_argument("--shared-row-group-rows", type=int)
    parser.add_argument(
        "--output-prefix", type=Path, default=Path("synthetic_gpu_results")
    )
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    binaries = {
        "wave": build_dir
        / "velox/experimental/wave/exec/tests/velox_wave_synthetic_benchmark",
        "cudf": build_dir
        / "velox/experimental/cudf/benchmarks/velox_cudf_synthetic_benchmark",
    }
    for binary in binaries.values():
        if not binary.is_file():
            parser.error(f"benchmark binary not found: {binary}")

    results = []
    failures = []
    gpu = args.gpu
    for workload in args.workloads:
        for backend in args.backends:
            binary = binaries[backend]
            for scan_mode in backend_scan_modes(backend, args):
                for batch_rows in backend_batch_rows(backend, args, args.rows):
                    print(
                        f"Running {backend} {workload} at {batch_rows} rows/batch, "
                        f"{scan_mode} scan, on GPU {gpu} "
                        f"(waiting for {args.gpu_lock})",
                        file=sys.stderr,
                    )
                    try:
                        result = run_one(
                            binary,
                            backend,
                            workload,
                            batch_rows,
                            scan_mode,
                            args,
                            gpu,
                        )
                    except RuntimeError as error:
                        if not args.continue_on_error:
                            raise
                        failure_class = getattr(error, "failure_class", "unknown")
                        print(
                            f"FAILED ({failure_class}): {error}", file=sys.stderr
                        )
                        failures.append(
                            {
                                "backend": backend,
                                "gpu": gpu,
                                "cudf_pin": args.cudf_pin,
                                "workload": workload,
                                "scan_mode": scan_mode,
                                "rows": args.rows,
                                "batch_rows": batch_rows,
                                "error": str(error),
                                "failure_class": failure_class,
                                "exit_code": getattr(error, "returncode", None),
                                "output_tail": (
                                    error.tail()
                                    if isinstance(error, BenchmarkFailure)
                                    else ""
                                ),
                            }
                        )
                        continue
                    result["gpu"] = gpu
                    result["cudf_pin"] = args.cudf_pin
                    result["batch_rows"] = batch_rows
                    result["shared_fixture"] = bool(args.shared_row_group_rows)
                    results.append(result)
                    # Long matrices are worth checkpointing; a later crash
                    # should not throw away everything measured so far.
                    write_results(results, args.output_prefix)

    mismatches = print_comparison(results, not args.continue_on_error)
    for workload in mismatches:
        failures.append(
            {
                "backend": "comparison",
                "gpu": args.gpu,
                "cudf_pin": args.cudf_pin,
                "workload": workload,
                "rows": args.rows,
                "error": "Wave and cuDF checksums differ",
                "failure_class": "checksum_mismatch",
            }
        )
    json_path, csv_path = write_results(results, args.output_prefix)
    print(f"\nJSON: {json_path}\nCSV:  {csv_path}")
    if failures:
        failures_path = args.output_prefix.with_suffix(".failures.json")
        failures_path.write_text(json.dumps(failures, indent=2) + "\n")
        print(f"Failures: {failures_path}")


if __name__ == "__main__":
    main()
