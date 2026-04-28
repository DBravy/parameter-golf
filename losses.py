"""
Pluggable loss functions for train_gpt.py ablation experiments.

All functions have signature: (logits: Tensor, targets: Tensor) -> Tensor
where logits is (N, V) float32 and targets is (N,) int64.

Uses clamp(min=1.0) instead of .item() branching to stay
compatible with torch.compile(fullgraph=True).
"""

import torch
import torch.nn.functional as F


def standard_ce_loss(logits, targets):
    """Baseline: standard mean cross-entropy over all tokens."""
    return F.cross_entropy(logits, targets, reduction="mean")


def mistake_only_loss(logits, targets):
    """CE averaged only over positions where argmax(logits) != target."""
    per_token = F.cross_entropy(logits, targets, reduction="none")
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        mask = (preds != targets).float()
    n_mistakes = mask.sum().clamp(min=1.0)
    return (per_token * mask).sum() / n_mistakes


def hybrid_ce_loss_fn(lam: float):
    """
    Returns closure: loss = mean_CE_over_mistakes + lam * mean_CE_over_correct.

    lam=0.0 -> pure mistake-only
    lam=0.5 -> ~standard CE (at ~67% miss rate)
    lam=1.0 -> equal weighting of mistake and correct means
    """

    def _loss(logits, targets):
        per_token = F.cross_entropy(logits, targets, reduction="none")
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            mis_mask = (preds != targets).float()
            cor_mask = 1.0 - mis_mask
        n_mis = mis_mask.sum().clamp(min=1.0)
        n_cor = cor_mask.sum().clamp(min=1.0)
        mis_term = (per_token * mis_mask).sum() / n_mis
        cor_term = (per_token * cor_mask).sum() / n_cor
        return mis_term + lam * cor_term

    return _loss
