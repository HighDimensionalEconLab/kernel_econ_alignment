import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_K0_VALUES = [
    0.5,
    1.0,
    1.5,
    1.75,
    1.846,
    1.9,
    1.95,
    2.0,
    2.1,
    3.0,
    4.0,
]

CHILD_CODE = """
import json

from neoclassical_growth_concave_convex_matern import neoclassical_growth_concave_convex_matern

sol = neoclassical_growth_concave_convex_matern(
    k_0=float(__import__("sys").argv[1]),
    train_points=int(__import__("sys").argv[2]),
    test_points=int(__import__("sys").argv[3]),
)
print(json.dumps({
    "valid_solution": bool(sol["valid_solution"]),
    "rejection_reason": sol["rejection_reason"],
    "candidate_branch": sol["candidate_branch"],
    "solver_status": sol["solver_status"],
    "solve_time": float(sol["solve_time"]),
    "max_helper_residual": float(sol["max_helper_residual"]),
    "max_train_residual": float(sol["max_train_residual"]),
    "max_validation_residual": float(sol["max_validation_residual"]),
    "p_lower_violation": float(sol["p_lower_violation"]),
    "p_upper_violation": float(sol["p_upper_violation"]),
    "active_branch_switch_train": bool(sol["active_branch_switch_train"]),
    "min_branch_gap_train": float(sol["min_branch_gap_train"]),
    "candidate_diagnostics": sol["candidate_diagnostics"],
}))
"""


def default_threshold_grid():
    return [0.5 + i * (4.0 - 0.5) / 39 for i in range(40)]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe concave-convex direct JAX/UNO solves under hard timeouts."
    )
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--train-points", type=int, default=41)
    parser.add_argument("--test-points", type=int, default=41)
    parser.add_argument("--threshold-grid", action="store_true")
    parser.add_argument("--output-dir", default="tmp")
    parser.add_argument("k0", nargs="*", type=float)
    return parser.parse_args()


def probe_one(k_0, timeout, train_points, test_points):
    start = time.perf_counter()
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                CHILD_CODE,
                str(float(k_0)),
                str(int(train_points)),
                str(int(test_points)),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "k_0": float(k_0),
            "run_status": "timeout",
            "wall_time": time.perf_counter() - start,
            "valid_solution": False,
            "rejection_reason": "subprocess_timeout",
        }

    row = {
        "k_0": float(k_0),
        "run_status": "ok" if result.returncode == 0 else "error",
        "wall_time": time.perf_counter() - start,
    }
    if result.returncode != 0:
        row["valid_solution"] = False
        row["rejection_reason"] = "subprocess_error"
        row["stderr_tail"] = result.stderr.splitlines()[-8:]
        return row

    payload = None
    for line in reversed(result.stdout.splitlines()):
        if line.startswith("{"):
            payload = json.loads(line)
            break
    if payload is None:
        row["run_status"] = "no_json"
        row["valid_solution"] = False
        row["rejection_reason"] = "missing_payload"
        row["stdout_tail"] = result.stdout.splitlines()[-8:]
        return row

    row.update(payload)
    return row


def main():
    args = parse_args()
    if args.k0:
        k0_values = args.k0
    elif args.threshold_grid:
        k0_values = default_threshold_grid()
    else:
        k0_values = DEFAULT_K0_VALUES

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        probe_one(k_0, args.timeout, args.train_points, args.test_points)
        for k_0 in k0_values
    ]

    jsonl_path = output_dir / "concave_convex_probe.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    csv_path = output_dir / "concave_convex_probe.csv"
    fieldnames = [
        "k_0",
        "run_status",
        "valid_solution",
        "rejection_reason",
        "candidate_branch",
        "solver_status",
        "solve_time",
        "wall_time",
        "max_helper_residual",
        "max_train_residual",
        "max_validation_residual",
        "p_lower_violation",
        "p_upper_violation",
        "active_branch_switch_train",
        "min_branch_gap_train",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    accepted = sum(1 for row in rows if row.get("valid_solution"))
    rejected = len(rows) - accepted
    print(
        f"wrote {jsonl_path} and {csv_path}; accepted={accepted} rejected={rejected}"
    )


if __name__ == "__main__":
    main()
