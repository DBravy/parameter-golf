"""
Single-run entry point for ablation experiments.

Usage:
    # Hybrid loss ablation
    torchrun --nproc_per_node=8 ablation_runner.py --lam 0.3
    torchrun --nproc_per_node=8 ablation_runner.py --lam -1   # baseline (standard CE)

    # Noise pretraining ablation
    torchrun --nproc_per_node=8 ablation_runner.py --noise_steps 500
    torchrun --nproc_per_node=8 ablation_runner.py --noise_steps 0   # baseline (no noise pretrain)

All train_gpt.py env vars (SEED, ITERATIONS, etc.) are respected.
"""

import argparse
import os

import torch

import train_gpt
from losses import hybrid_ce_loss_fn, mistake_only_loss


def make_noise_pretrain_hook(noise_steps, noise_mode, noise_lr, noise_batch_size):
    """Return a hook fn(model, device, args) that does noise pretraining."""

    def hook(model, device, args):
        if noise_steps <= 0:
            return

        vocab_size = args.vocab_size
        seq_len = args.train_seq_len

        # Use standard CE during noise pretrain regardless of _TRAIN_LOSS_FN
        saved_loss_fn = model.loss_fn
        model.loss_fn = None

        opt = torch.optim.AdamW(model.parameters(), lr=noise_lr, weight_decay=0.01)

        model.train()
        for step in range(1, noise_steps + 1):
            if noise_mode == "token":
                x = torch.randint(0, vocab_size, (noise_batch_size, seq_len), device=device)
                y = torch.randint(0, vocab_size, (noise_batch_size, seq_len), device=device)
            else:
                raise ValueError(f"unknown noise_mode: {noise_mode}")

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step <= 5 or step % 100 == 0 or step == noise_steps:
                print(f"  noise_pretrain step:{step}/{noise_steps} loss:{loss.item():.4f}")

        model.loss_fn = saved_loss_fn

    return hook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lam",
        type=float,
        default=None,
        help="Lambda for hybrid loss. 0.0=mistake-only, 0.5~standard CE, -1=baseline.",
    )
    parser.add_argument("--noise_steps", type=int, default=0,
                        help="Number of noise pretraining steps (0=disabled).")
    parser.add_argument("--noise_mode", type=str, default="token",
                        choices=["token"],
                        help="Noise pretraining mode.")
    parser.add_argument("--noise_lr", type=float, default=3e-4,
                        help="Learning rate for noise pretraining.")
    parser.add_argument("--noise_batch_size", type=int, default=64,
                        help="Batch size for noise pretraining.")
    args = parser.parse_args()

    # Build RUN_ID
    seed = os.environ.get("SEED", "1337")
    parts = []
    if args.lam is not None:
        parts.append(f"lam{args.lam:g}")
    if args.noise_steps > 0:
        parts.append(f"noise{args.noise_steps}")
    if not parts:
        parts.append("baseline")
    if "RUN_ID" not in os.environ:
        os.environ["RUN_ID"] = f"{'_'.join(parts)}_seed{seed}"

    # Set loss function
    if args.lam is not None:
        if args.lam < 0:
            train_gpt._TRAIN_LOSS_FN = None
        elif args.lam == 0.0:
            train_gpt._TRAIN_LOSS_FN = mistake_only_loss
        else:
            train_gpt._TRAIN_LOSS_FN = hybrid_ce_loss_fn(args.lam)

    # Set noise pretraining hook
    if args.noise_steps > 0:
        train_gpt._PRE_TRAIN_HOOK = make_noise_pretrain_hook(
            args.noise_steps, args.noise_mode, args.noise_lr, args.noise_batch_size
        )

    train_gpt.main()


if __name__ == "__main__":
    main()
