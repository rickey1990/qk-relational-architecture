"""Minimal runnable PyTorch starter for the fixed-window QK Relational Architecture.

Companion architecture (September 2026):
- 2 layers, default d_model=96, byte vocabulary=256
- ordinary QK field uses a fixed causal window W = c*d (default c=1, W=96)
- ordinary reading is headless and has no learned value projection
- fixed distance prior D(r)=0.02/(1+r)+0.98/(1+r)^2
- final block owns separate Qh/Kh projections for optional strict no-self hops
- requested extra hops compute A0^(W) @ Ah^k @ H
- default FF widths are 5d in block 1 and 4d in block 2
- default tested-size parameter count: 271,296

Ordinary k=0 operation never constructs an N x N A0 matrix.  The local reader
uses small contiguous query/key blocks, so ordinary relationship work is O(N*W*d).
When extra_hops > 0, the current Ah implementation is still dense/full-history and
therefore quadratic in sequence length.  This file prioritizes clarity and fidelity
rather than reproducing every optimization used in the timing benchmark.

Documentation/explanatory text: CC BY-NC 4.0.
Source code: PolyForm Noncommercial License 1.0.0.
Contact: rickeyernest1990@gmail.com
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


DECAY_A = 0.02
DECAY_B = 0.98


def local_qk_read(values, q, k, window, block_size=64):
    """Exact fixed-window causal QK read without an N x N ordinary score matrix.

    Args:
        values: state to transmit, [B, N, d]. This is raw H at k=0, or Ah^k H.
        q, k:    ordinary Q0/K0 projections from the same normalized H, [B, N, d].
        window:  maximum number of causal positions visible to each query.
        block_size: small query block used only for implementation efficiency.

    Returns:
        [B, N, d] local relational read.
    """
    B, N, d = values.shape
    W = min(int(window), N)
    pieces = []

    for start in range(0, N, block_size):
        end = min(N, start + block_size)
        key_start = max(0, start - W + 1)
        key_end = end

        qb = q[:, start:end, :]
        kb = k[:, key_start:key_end, :]
        vb = values[:, key_start:key_end, :]

        scores = (qb @ kb.transpose(-2, -1)) / math.sqrt(d)

        qi = torch.arange(start, end, device=values.device)[:, None]
        kj = torch.arange(key_start, key_end, device=values.device)[None, :]
        delta = qi - kj
        valid = (delta >= 0) & (delta < W)

        r = delta.clamp_min(0).to(values.dtype)
        log_prior = torch.log(
            DECAY_A / (1.0 + r) + DECAY_B / (1.0 + r).square()
        )

        scores = scores + log_prior.unsqueeze(0)
        scores = scores.masked_fill(~valid.unsqueeze(0), -torch.inf)
        weights = scores.softmax(dim=-1)
        pieces.append(weights @ vb)

    return torch.cat(pieces, dim=1)


def strict_hop_matrix(h, qh, kh):
    """Build the current full-history strict hop field Ah.

    Rows i>0 may attend only to j<i. Row zero receives one fallback self entry so
    softmax has a legal element. No later row may self-loop.
    """
    B, N, d = h.shape
    scores = (qh @ kh.transpose(-2, -1)) / math.sqrt(d)

    pos = torch.arange(N, device=h.device)
    strict = pos[:, None] > pos[None, :]
    strict = strict.clone()
    strict[0, 0] = True

    return scores.masked_fill(~strict.unsqueeze(0), -torch.inf).softmax(dim=-1)


def dense_local_reference(values, q, k, window):
    """Dense N x N reference used only by the smoke test, not normal operation."""
    B, N, d = values.shape
    W = min(int(window), N)
    pos = torch.arange(N, device=values.device)
    delta = pos[:, None] - pos[None, :]
    valid = (delta >= 0) & (delta < W)
    r = delta.clamp_min(0).to(values.dtype)

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
    scores = scores + torch.log(
        DECAY_A / (1.0 + r) + DECAY_B / (1.0 + r).square()
    )
    weights = scores.masked_fill(~valid.unsqueeze(0), -torch.inf).softmax(dim=-1)
    return weights @ values, weights


class FixedWindowQKBlock(nn.Module):
    def __init__(self, d_model, ff_dim, window, has_hop=False, block_size=64):
        super().__init__()
        self.window = int(window)
        self.has_hop = bool(has_hop)
        self.block_size = int(block_size)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.q0 = nn.Linear(d_model, d_model, bias=False)
        self.k0 = nn.Linear(d_model, d_model, bias=False)

        if self.has_hop:
            self.qh = nn.Linear(d_model, d_model, bias=False)
            self.kh = nn.Linear(d_model, d_model, bias=False)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim, bias=False),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model, bias=False),
        )

    def forward(self, x, extra_hops=0):
        h = self.ln1(x)
        q0, k0 = self.q0(h), self.k0(h)

        if extra_hops > 0:
            if not self.has_hop:
                raise ValueError("extra_hops may only be used in the hop-enabled final block")

            qh, kh = self.qh(h), self.kh(h)
            Ah = strict_hop_matrix(h, qh, kh)

            # Exact order: A0^(W) Ah^k H.
            z = h
            for _ in range(int(extra_hops)):
                z = Ah @ z

            out = local_qk_read(z, q0, k0, self.window, self.block_size)
        else:
            # Ordinary mode: qh, kh and Ah are completely dormant.
            out = local_qk_read(h, q0, k0, self.window, self.block_size)

        x = x + out
        return x + self.ff(self.ln2(x))


class FixedWindowQKLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size=256,
        d_model=96,
        window_multiplier=1.0,
        max_context=8192,
        block_size=64,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.window_multiplier = float(window_multiplier)
        self.window = max(1, int(round(self.window_multiplier * self.d_model)))

        self.embed = nn.Embedding(self.vocab_size, self.d_model)
        self.blocks = nn.ModuleList([
            FixedWindowQKBlock(
                self.d_model,
                ff_dim=5 * self.d_model,
                window=self.window,
                has_hop=False,
                block_size=block_size,
            ),
            FixedWindowQKBlock(
                self.d_model,
                ff_dim=4 * self.d_model,
                window=self.window,
                has_hop=True,
                block_size=block_size,
            ),
        ])
        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.vocab_size, bias=False)

        pos = torch.arange(max_context)[:, None].float()
        div = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(max_context, self.d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("position", pe, persistent=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens, extra_hops=0):
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


def train_byte_file(
    path,
    steps=1000,
    context=512,
    batch_size=4,
    seed=11,
    device="cpu",
    save_path="qkra_fixed_window.pt",
):
    """Small next-byte training loop. Normal language training uses k=0."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 10000)

    raw = Path(path).read_bytes()
    if len(raw) < context + 2:
        raise ValueError("training file is too small for the selected context")

    data = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).astype(np.int64))
    model = FixedWindowQKLanguageModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    model.train()
    for step in range(1, int(steps) + 1):
        x, y = make_byte_batch(data, batch_size, context, rng)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x, extra_hops=0)
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()

        if step == 1 or step % 100 == 0 or step == steps:
            print(f"step={step:5d} loss={loss.item():.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"saved {save_path}")
    return model


def smoke_test():
    torch.manual_seed(123)
    model = FixedWindowQKLanguageModel()
    assert model.window == 96
    assert parameter_count(model) == 271296, parameter_count(model)

    # 1) Local operator matches the equivalent dense windowed mathematics.
    h = torch.randn(2, 137, 96)
    q = torch.randn_like(h)
    k = torch.randn_like(h)
    local = local_qk_read(h, q, k, window=96)
    dense, _ = dense_local_reference(h, q, k, window=96)
    local_err = float((local - dense).abs().max())
    assert local_err < 3e-6, local_err

    # 2) Normal and hopped model passes are finite.
    x = torch.randint(0, 256, (2, 128))
    y0 = model(x, extra_hops=0)
    y3 = model(x, extra_hops=3)
    assert y0.shape == (2, 128, 256)
    assert y3.shape == (2, 128, 256)
    assert torch.isfinite(y0).all() and torch.isfinite(y3).all()

    # 3) Hop field is row-normalized and strictly no-self after row zero.
    final = model.blocks[-1]
    h2 = final.ln1(
        model.blocks[0](
            model.embed(x) * math.sqrt(model.d_model) + model.position[: x.shape[1]],
            extra_hops=0,
        )
    )
    Ah = strict_hop_matrix(h2, final.qh(h2), final.kh(h2))
    assert torch.allclose(Ah.sum(-1), torch.ones_like(Ah.sum(-1)), atol=1e-6)
    diag = torch.diagonal(Ah, dim1=-2, dim2=-1)
    assert torch.all(diag[:, 1:] == 0), "strict hop rows must not self-loop"

    # 4) Hop parameters are dormant at k=0 and active when hopping is requested.
    model.zero_grad(set_to_none=True)
    model(x, extra_hops=0).sum().backward()
    assert final.qh.weight.grad is None and final.kh.weight.grad is None

    model.zero_grad(set_to_none=True)
    model(x, extra_hops=1).sum().backward()
    assert final.qh.weight.grad is not None and torch.isfinite(final.qh.weight.grad).all()
    assert final.kh.weight.grad is not None and torch.isfinite(final.kh.weight.grad).all()

    print("smoke test: PASS")
    print("parameters:", parameter_count(model))
    print("d_model:", model.d_model)
    print("window W=c*d:", model.window)
    print("local-vs-dense max abs:", f"{local_err:.3e}")
    print("normal logits:", tuple(y0.shape))
    print("+3-hop logits:", tuple(y3.shape))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run architecture checks")
    parser.add_argument("--train-byte-file", type=str, default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save", type=str, default="qkra_fixed_window.pt")
    args = parser.parse_args()

    if args.smoke or args.train_byte_file is None:
        smoke_test()

    if args.train_byte_file:
        train_byte_file(
            args.train_byte_file,
            steps=args.steps,
            context=args.context,
            batch_size=args.batch,
            seed=args.seed,
            device=args.device,
            save_path=args.save,
        )


if __name__ == "__main__":
    main()
