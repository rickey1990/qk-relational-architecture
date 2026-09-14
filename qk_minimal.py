"""Minimal runnable PyTorch implementation of the tested QK architecture.

Architecture used in the September 2026 competitor audit:
- 2 layers, d_model=48, byte vocabulary=256
- no attention heads and no learned value projection in QK blocks
- ordinary field A0: separate Q0/K0 + fixed distance prior + causal self-allowed mask
- hop field Ah (final block only): separate Qh/Kh + strict causal no-self mask
- explicit extra hops: A0 @ Ah^k @ H, implemented by recurrent Ah @ state
- FF widths: 240 in block 1, 192 in block 2
- total parameters: 80,352

This file is intentionally compact and self-contained. It can train as a byte-level
next-token language model on any local file and can also run a smoke test without data.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def distance_prior_log(n: int, device, dtype):
    """Return delta=i-j and log D(|i-j|), with D(r)=.02/(1+r)+.98/(1+r)^2."""
    pos = torch.arange(n, device=device)
    delta = pos[:, None] - pos[None, :]
    r = delta.abs().to(dtype)
    log_prior = torch.log(0.02 / (1.0 + r) + 0.98 / (1.0 + r).square())
    return delta, log_prior


def qk_fields(h, q0, k0, qh=None, kh=None, need_hop=False):
    """Build ordinary A0 and, optionally, strict hop Ah.

    Shapes:
        h, q0, k0, qh, kh: [batch, sequence, d_model]
        A0, Ah:            [batch, sequence, sequence]
    """
    n = h.shape[1]
    d = q0.shape[-1]
    delta, log_prior = distance_prior_log(n, h.device, h.dtype)

    # Ordinary causal reading: j <= i, so the diagonal/self is legal.
    ordinary_mask = delta >= 0
    scores0 = (q0 @ k0.transpose(-2, -1)) / math.sqrt(d)
    scores0 = scores0 + log_prior
    A0 = scores0.masked_fill(~ordinary_mask, -torch.inf).softmax(dim=-1)

    Ah = None
    if need_hop:
        if qh is None or kh is None:
            raise ValueError("qh and kh are required when need_hop=True")

        # Explicit hop: j < i. A hop must move to an earlier relation.
        strict_mask = delta > 0
        strict_mask = strict_mask.clone()

        # Row 0 has no earlier key. A single fallback self entry keeps softmax finite.
        # No later row is allowed to self-loop.
        strict_mask[0, 0] = True

        scoresh = (qh @ kh.transpose(-2, -1)) / math.sqrt(d)
        Ah = scoresh.masked_fill(~strict_mask, -torch.inf).softmax(dim=-1)

    return A0, Ah


class QKBlock(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, has_hop: bool):
        super().__init__()
        self.has_hop = has_hop
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.q0 = nn.Linear(d_model, d_model, bias=False)
        self.k0 = nn.Linear(d_model, d_model, bias=False)

        if has_hop:
            self.qh = nn.Linear(d_model, d_model, bias=False)
            self.kh = nn.Linear(d_model, d_model, bias=False)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim, bias=False),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model, bias=False),
        )

    def forward(self, x, extra_hops: int = 0):
        h = self.ln1(x)
        q0, k0 = self.q0(h), self.k0(h)

        if extra_hops > 0:
            if not self.has_hop:
                raise ValueError("extra_hops may only be applied to the hop-enabled final block")
            qh, kh = self.qh(h), self.kh(h)
            A0, Ah = qk_fields(h, q0, k0, qh, kh, need_hop=True)

            # A0 Ah^k H: apply Ah repeatedly to H, then make the ordinary A0 read.
            z = h
            for _ in range(extra_hops):
                z = Ah @ z
            out = A0 @ z
        else:
            A0, _ = qk_fields(h, q0, k0, need_hop=False)
            out = A0 @ h

        x = x + out
        return x + self.ff(self.ln2(x))


class QKLanguageModel(nn.Module):
    def __init__(self, vocab_size=256, d_model=48, max_context=2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            QKBlock(d_model, ff_dim=240, has_hop=False),
            QKBlock(d_model, ff_dim=192, has_hop=True),
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Fixed sinusoidal positions, matching the tested small language model.
        pos = torch.arange(max_context)[:, None].float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_context, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("position", pe, persistent=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens, extra_hops: int = 0):
        if tokens.shape[1] > self.position.shape[0]:
            raise ValueError("sequence is longer than max_context")

        h = self.embed(tokens) * math.sqrt(self.d_model)
        h = h + self.position[: tokens.shape[1]]

        h = self.blocks[0](h, extra_hops=0)
        h = self.blocks[1](h, extra_hops=extra_hops)
        return self.head(self.norm(h))


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def make_byte_batch(data, batch_size, context, rng):
    starts = rng.integers(0, len(data) - context - 1, size=batch_size)
    windows = torch.stack([
        data[int(s): int(s) + context + 1] for s in starts
    ])
    return windows[:, :-1], windows[:, 1:]


def train_byte_file(path, steps=1000, context=128, batch_size=8, seed=11, device="cpu"):
    """Small next-byte training loop. Normal language training uses extra_hops=0."""
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed + 10000)

    raw = Path(path).read_bytes()
    if len(raw) < context + 2:
        raise ValueError("training file is too small for the selected context")

    data = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).astype(np.int64))
    model = QKLanguageModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    model.train()
    for step in range(1, steps + 1):
        x, y = make_byte_batch(data, batch_size, context, np_rng)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x, extra_hops=0)
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()

        if step == 1 or step % 100 == 0 or step == steps:
            print(f"step={step:5d} loss={loss.item():.4f}")

    return model


def smoke_test():
    torch.manual_seed(123)
    model = QKLanguageModel()
    assert parameter_count(model) == 80352, parameter_count(model)

    x = torch.randint(0, 256, (2, 32))
    y0 = model(x, extra_hops=0)
    y3 = model(x, extra_hops=3)
    assert y0.shape == (2, 32, 256)
    assert y3.shape == (2, 32, 256)
    assert torch.isfinite(y0).all() and torch.isfinite(y3).all()

    # Mask sanity check.
    h = model.blocks[1].ln1(
        model.embed(x) * math.sqrt(model.d_model) + model.position[: x.shape[1]]
    )
    b = model.blocks[1]
    A0, Ah = qk_fields(h, b.q0(h), b.k0(h), b.qh(h), b.kh(h), need_hop=True)
    assert torch.allclose(A0.sum(-1), torch.ones_like(A0.sum(-1)), atol=1e-6)
    assert torch.allclose(Ah.sum(-1), torch.ones_like(Ah.sum(-1)), atol=1e-6)
    diag = torch.diagonal(Ah, dim1=-2, dim2=-1)
    assert torch.all(diag[:, 1:] == 0), "strict hop rows must not self-loop"

    # qh/kh are dormant at k=0 and active when hops are requested.
    model.zero_grad(set_to_none=True)
    model(x, extra_hops=0).sum().backward()
    assert b.qh.weight.grad is None and b.kh.weight.grad is None

    model.zero_grad(set_to_none=True)
    model(x, extra_hops=1).sum().backward()
    assert b.qh.weight.grad is not None and torch.isfinite(b.qh.weight.grad).all()
    assert b.kh.weight.grad is not None and torch.isfinite(b.kh.weight.grad).all()

    print("smoke test: PASS")
    print("parameters:", parameter_count(model))
    print("normal logits:", tuple(y0.shape))
    print("+3-hop logits:", tuple(y3.shape))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run architecture checks")
    parser.add_argument("--train-byte-file", type=str, default=None,
                        help="train a next-byte model on a local file")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.smoke or args.train_byte_file is None:
        smoke_test()

    if args.train_byte_file:
        model = train_byte_file(
            args.train_byte_file,
            steps=args.steps,
            context=args.context,
            batch_size=args.batch,
            seed=args.seed,
            device=args.device,
        )
        torch.save(model.state_dict(), "qk_language_model.pt")
        print("saved qk_language_model.pt")


if __name__ == "__main__":
    main()
