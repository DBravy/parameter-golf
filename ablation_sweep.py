"""
Launch a sweep over lambda values for the hybrid loss ablation.

Usage:
    python ablation_sweep.py
    python ablation_sweep.py --lambdas "0.0,0.1,0.3,0.5" --seeds "1337"
    python ablation_sweep.py --nproc 4
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

LAMBDAS_DEFAULT = [
    -1.0,  # baseline (standard CE)
    0.0,   # pure mistake-only
    0.05,
    0.1,
    0.2,
    0.3,
    0.5,
    0.7,
    1.0,
]

SEEDS_DEFAULT = [1337, 42, 7]


def parse_final_metrics(log_lines: str) -> dict:
    """Extract the last val_loss and val_bpb from training output."""
    metrics = {}
    for line in log_lines.splitlines():
        m = re.search(r"val_loss:([\d.]+)\s+val_bpb:([\d.]+)", line)
        if m:
            metrics["val_loss"] = float(m.group(1))
            metrics["val_bpb"] = float(m.group(2))
    return metrics


def run_sweep(lambdas, seeds, nproc):
    results = []
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    for seed in seeds:
        for lam in lambdas:
            run_id = f"hybrid_lam{lam:g}_seed{seed}"
            log_path = log_dir / f"{run_id}.txt"

            if log_path.exists():
                print(f"Skipping {run_id} (log exists at {log_path})")
                log_text = log_path.read_text()
                metrics = parse_final_metrics(log_text)
                if metrics:
                    results.append({"lambda": lam, "seed": seed, **metrics})
                continue

            print(f"\n{'=' * 60}")
            print(f"  lambda={lam:g}  seed={seed}")
            print(f"{'=' * 60}\n")

            env = os.environ.copy()
            env["SEED"] = str(seed)
            env["RUN_ID"] = run_id

            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={nproc}",
                "ablation_runner.py",
                "--lam",
                str(lam),
            ]

            with open(log_path, "w") as log_file:
                proc = subprocess.run(
                    cmd,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )

            if proc.returncode != 0:
                print(f"  FAILED (exit code {proc.returncode}) — see {log_path}")
                continue

            log_text = log_path.read_text()
            metrics = parse_final_metrics(log_text)
            if metrics:
                results.append({"lambda": lam, "seed": seed, **metrics})
                print(f"  val_bpb={metrics['val_bpb']:.4f}  val_loss={metrics['val_loss']:.4f}")
            else:
                print(f"  WARNING: no metrics found in {log_path}")

    return results


def print_summary(results):
    if not results:
        print("\nNo results to summarize.")
        return

    by_lam = defaultdict(list)
    for r in results:
        by_lam[r["lambda"]].append(r)

    print(f"\n{'=' * 62}")
    print("  ABLATION SUMMARY")
    print(f"{'=' * 62}")
    print(f"  {'Lambda':>8}  {'val_bpb (mean)':>15}  {'val_bpb (std)':>13}  {'n':>3}")
    print(f"  {'-' * 45}")

    for lam in sorted(by_lam.keys()):
        runs = by_lam[lam]
        bpbs = [r["val_bpb"] for r in runs if "val_bpb" in r]
        if not bpbs:
            continue
        mean = sum(bpbs) / len(bpbs)
        if len(bpbs) > 1:
            var = sum((b - mean) ** 2 for b in bpbs) / (len(bpbs) - 1)
            std = var**0.5
        else:
            std = 0.0
        label = "baseline" if lam < 0 else f"{lam:g}"
        print(f"  {label:>8}  {mean:>15.4f}  {std:>13.4f}  {len(bpbs):>3}")

    # Save full results
    out_path = "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid loss lambda sweep")
    parser.add_argument(
        "--lambdas",
        type=str,
        default=None,
        help="Comma-separated lambda values (default: -1,0,0.05,0.1,0.2,0.3,0.5,0.7,1.0)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds (default: 1337,42,7)",
    )
    parser.add_argument("--nproc", type=int, default=8, help="GPUs per run")
    args = parser.parse_args()

    lambdas = (
        [float(x) for x in args.lambdas.split(",")]
        if args.lambdas
        else LAMBDAS_DEFAULT
    )
    seeds = (
        [int(x) for x in args.seeds.split(",")]
        if args.seeds
        else SEEDS_DEFAULT
    )

    results = run_sweep(lambdas, seeds, args.nproc)
    print_summary(results)
