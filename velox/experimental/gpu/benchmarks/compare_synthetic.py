#!/usr/bin/env python3

"""Runs matched Wave and cuDF synthetic benchmarks on one GPU."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time


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


def gpu_statuses():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    devices = []
    for line in output.splitlines():
        index, utilization, memory = (int(value.strip()) for value in line.split(","))
        devices.append((utilization, memory, index))
    return devices


def select_gpu():
    devices = gpu_statuses()
    if not devices:
        raise RuntimeError("nvidia-smi reported no GPUs")
    return min(devices)[2]


def wait_for_gpu_free(index, timeout_seconds=60):
    deadline = time.monotonic() + timeout_seconds
    while True:
        for utilization, memory, gpu_index in gpu_statuses():
            if gpu_index != index:
                continue
            if utilization == 0 and memory == 0:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"GPU {index} did not become free within {timeout_seconds}s: "
                    f"utilization={utilization}%, memory.used={memory} MiB"
                )
            print(
                f"Waiting for GPU {index}: utilization={utilization}%, "
                f"memory.used={memory} MiB",
                file=sys.stderr,
            )
            time.sleep(1)
            break
        else:
            raise RuntimeError(f"GPU {index} was not reported by nvidia-smi")


def parse_result(output):
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "backend" in value:
            return value
    raise RuntimeError(f"Benchmark produced no JSON result:\n{output}")


def run_one(binary, backend, workload, args, gpu):
    command = [
        str(binary),
        f"--synthetic_workload={workload}",
        f"--synthetic_rows={args.rows}",
        f"--synthetic_batch_rows={args.batch_rows}",
        f"--synthetic_filter_percent={args.filter_percent}",
        f"--synthetic_group_cardinality={args.group_cardinality}",
        f"--synthetic_warmups={args.warmups}",
        f"--synthetic_repetitions={args.repetitions}",
        "--synthetic_format=json",
    ]
    if args.skip_cpu_validation:
        command.append("--synthetic_cpu_validation=false")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    completed = subprocess.run(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode:
        sys.stderr.write(completed.stdout)
        raise RuntimeError(
            f"{backend} {workload} failed with exit code {completed.returncode}"
        )
    result = parse_result(completed.stdout)
    if not result["correct"]:
        raise RuntimeError(f"{backend} {workload} failed correctness validation")
    return result


def write_results(results, output_prefix):
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    fields = [
        "backend",
        "gpu",
        "workload",
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
        "cold_ms",
        "warm_median_ms",
        "warm_p95_ms",
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
        by_workload.setdefault(result["workload"], {})[result["backend"]] = result
    print(
        f"{'workload':<16} {'wave warm ms':>14} {'cuDF warm ms':>14} "
        f"{'Wave speedup':>18} {'checksums':>12}"
    )
    for workload in WORKLOADS:
        pair = by_workload.get(workload, {})
        if "wave" not in pair or "cudf" not in pair:
            continue
        wave = pair["wave"]
        cudf = pair["cudf"]
        speedup = cudf["warm_median_ms"] / wave["warm_median_ms"]
        checksums = "match" if wave["checksum"] == cudf["checksum"] else "MISMATCH"
        print(
            f"{workload:<16} {wave['warm_median_ms']:>14.3f} "
            f"{cudf['warm_median_ms']:>14.3f} {speedup:>18.3f} "
            f"{checksums:>12}"
        )
        if checksums != "match":
                mismatches.append(workload)
    if mismatches and fail_on_mismatch:
        raise RuntimeError(
            f"backend checksum mismatch for {', '.join(mismatches)}"
        )
    return mismatches
    print("\nWave scan timings use the wavemock reader, not Parquet or ORC I/O.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=Path("_build/release"))
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--batch-rows", type=int, default=100_000)
    parser.add_argument("--filter-percent", type=int, default=10)
    parser.add_argument("--group-cardinality", type=int, default=1_000)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=WORKLOADS)
    parser.add_argument(
        "--backends", nargs="+", choices=["wave", "cudf"], default=["wave", "cudf"]
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-cpu-validation", action="store_true")
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
    for workload in args.workloads:
        for backend in args.backends:
            binary = binaries[backend]
            gpu = args.gpu if args.gpu is not None else select_gpu()
            if args.gpu is not None:
                wait_for_gpu_free(gpu)
            print(f"Running {backend} {workload} on GPU {gpu}", file=sys.stderr)
            try:
                result = run_one(binary, backend, workload, args, gpu)
            except RuntimeError as error:
                if not args.continue_on_error:
                    raise
                print(f"FAILED: {error}", file=sys.stderr)
                failures.append(
                    {
                        "backend": backend,
                        "gpu": gpu,
                        "workload": workload,
                        "rows": args.rows,
                        "batch_rows": args.batch_rows,
                        "error": str(error),
                    }
                )
                continue
            result["gpu"] = gpu
            result["batch_rows"] = args.batch_rows
            results.append(result)

    mismatches = print_comparison(results, not args.continue_on_error)
    for workload in mismatches:
        failures.append(
            {
                "backend": "comparison",
                "gpu": args.gpu,
                "workload": workload,
                "rows": args.rows,
                "batch_rows": args.batch_rows,
                "error": "Wave and cuDF checksums differ",
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
