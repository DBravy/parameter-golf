"""
Per-layer hippocampal memory pathway.

Drop-in companion to train_gpt.py's CausalSelfAttention. Every transformer
block runs this in parallel with local attention, and its output is added
to the residual stream behind a learnable per-channel scale.

The pathway has two information sources, concatenated into one memory bank
that every query attends to:

  1. A persistent FIFO cache of compressed slots that survives across
     training batches. Stop-gradient on the read; stop-gradient on the
     write. This is the cross-episode "hippocampal" buffer.

  2. Within-batch segment summaries. The current sequence is split into
     `num_segments` chunks; the compressor produces one summary per chunk;
     a query in chunk k may attend to summaries from chunks 0..k-1. This
     is what gives the compressor a within-batch training signal: loss on
     tokens in chunk k flows back through the read of summaries[0..k-1]
     and into the compressor's parameters.

The same compressor is used for both the within-batch summaries (gradient
flows through them) and the cross-batch cache writes (under .detach()), so
storage skill learned in (2) transfers to (1) by shared parameters.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MemoryAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        mem_dim: int,
        num_segments: int,
        cache_size: int,
    ):
        super().__init__()
        self.dim = dim
        self.mem_dim = mem_dim
        self.num_segments = num_segments
        self.cache_size = cache_size

        # Compressor: softmax-pooling over the seg_len axis.
        # gate = softmax(W_g x); summary = sum_t gate[t] * (W_v x)[t].
        self.compress_v = nn.Linear(dim, mem_dim, bias=False)
        self.compress_g = nn.Linear(dim, mem_dim, bias=False)

        # Single-head memory attention. Q from x, K/V from the memory bank.
        self.q_proj = nn.Linear(dim, mem_dim, bias=False)
        self.k_proj = nn.Linear(mem_dim, mem_dim, bias=False)
        self.v_proj = nn.Linear(mem_dim, mem_dim, bias=False)
        self.o_proj = nn.Linear(mem_dim, dim, bias=False)
        # Picked up by GPT._init_weights() in train_gpt.py: zero this so the
        # whole pathway starts as identity and gets phased in by training.
        self.o_proj._zero_init = True

        # Persistent cross-batch cache. Buffer (not parameter): saved with
        # state_dict, mutated in place at end of forward, no optimizer step.
        # DDP runs with broadcast_buffers=False in train_gpt.py, so each rank
        # keeps its own cache reflecting the contiguous text it has seen.
        self.register_buffer("cache", torch.zeros(cache_size, mem_dim))

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        n_seg = self.num_segments
        seg_len = T // n_seg
        assert T == n_seg * seg_len, (
            f"T={T} must be divisible by num_segments={n_seg}"
        )

        # 1. Compress each segment to one summary vector.
        x_seg = x.view(B, n_seg, seg_len, D)
        v = self.compress_v(x_seg)                                 # (B, n_seg, seg_len, mem_dim)
        g = self.compress_g(x_seg)
        gates = F.softmax(g.float(), dim=2).to(v.dtype)
        summaries = (gates * v).sum(dim=2)                         # (B, n_seg, mem_dim)

        # 2. Memory bank: persistent cache (stop-grad) || within-batch summaries.
        cache_b = self.cache.detach().to(summaries.dtype)
        cache_b = cache_b.unsqueeze(0).expand(B, -1, -1)           # (B, M, mem_dim)
        mem = torch.cat([cache_b, summaries], dim=1)               # (B, M+n_seg, mem_dim)

        k = self.k_proj(mem)
        v_mem = self.v_proj(mem)
        q = self.q_proj(x)                                         # (B, T, mem_dim)

        # 3. Visibility mask: persistent slots always visible; within-batch
        #    summary s visible to token t iff s < (t // seg_len).
        device = x.device
        token_seg = torch.arange(T, device=device) // seg_len      # (T,)
        sum_idx = torch.arange(n_seg, device=device)               # (n_seg,)
        sum_mask = sum_idx.unsqueeze(0) < token_seg.unsqueeze(1)   # (T, n_seg)
        persist_mask = torch.ones(
            T, self.cache_size, device=device, dtype=torch.bool,
        )
        mask = torch.cat([persist_mask, sum_mask], dim=1)          # (T, M+n_seg)

        # 4. Single-head dot-product attention with the mask.
        scores = torch.einsum("btc,bsc->bts", q, k) / math.sqrt(self.mem_dim)
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("bts,bsc->btc", attn, v_mem)            # (B, T, mem_dim)
        out = self.o_proj(out)                                     # (B, T, dim)

        # 5. Cross-batch cache write. Average over batch dim, FIFO push.
        # Pure .detach() (no torch.no_grad context) keeps this compile-friendly
        # under torch.compile(fullgraph=True). The .detach() on self.cache[n_new:]
        # is redundant for a buffer (no autograd) but makes the data flow explicit.
        new_slots = summaries.detach().mean(dim=0).to(self.cache.dtype)
        n_new = new_slots.size(0)
        self.cache.copy_(
            torch.cat([self.cache[n_new:].detach(), new_slots], dim=0)
        )

        return out
