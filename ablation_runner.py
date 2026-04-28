"""
Single-run entry point for hybrid loss ablation.

Usage:
    torchrun --nproc_per_node=8 ablation_runner.py --lam 0.3
    torchrun --nproc_per_node=8 ablation_runner.py --lam -1   # baseline (standard CE)

All train_gpt.py env vars (SEED, ITERATIONS, etc.) are respected.
"""

import argparse
import os

import train_gpt
from losses import hybrid_ce_loss_fn, mistake_only_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lam",
        type=float,
        required=True,
        help="Lambda for hybrid loss. 0.0=mistake-only, 0.5~standard CE, -1=baseline.",
    )
    args = parser.parse_args()

    # Encode lambda in RUN_ID so logs are distinguishable
    if "RUN_ID" not in os.environ:
        seed = os.environ.get("SEED", "1337")
        os.environ["RUN_ID"] = f"hybrid_lam{args.lam:g}_seed{seed}"

    if args.lam < 0:
        train_gpt._TRAIN_LOSS_FN = None  # standard CE (default)
    elif args.lam == 0.0:
        train_gpt._TRAIN_LOSS_FN = mistake_only_loss
    else:
        train_gpt._TRAIN_LOSS_FN = hybrid_ce_loss_fn(args.lam)

    train_gpt.main()


if __name__ == "__main__":
    main()
