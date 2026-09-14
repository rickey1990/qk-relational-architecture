# Experimental Full-Pair QK Relational Architecture

A small experimental attention architecture built around a single full-pair QK relationship field for ordinary language processing, plus a separately learned strict-causal QK field for optional multi-hop relational propagation.

The current tested form is intentionally small and experimental. It is **not** presented as a replacement for the Transformer, and the results are not evidence of general multi-step reasoning. The repository is intended to make the architecture, current evidence, and minimal implementation easy to inspect and reproduce.

## Current architecture

For ordinary causal reading:

```text
A0 = softmax(Q0 K0^T / sqrt(d) + log(distance_prior) + causal_mask)
```

with

```text
D(r) = 0.02 / (1 + r) + 0.98 / (1 + r)^2
```

For explicit relational hopping, the final block uses separate hop projections and a strict causal mask that forbids self-loops:

```text
Ah = softmax(Qh Kh^T / sqrt(d) + strict_causal_mask)
```

The requested extra-hop computation is:

```text
Y = A0 Ah^k H
```

The implementation applies `Ah` repeatedly to the state and then performs the ordinary `A0` read. At `k = 0`, the hop-specific `Qh` and `Kh` projections are dormant.

## Included files

```text
Experimental_Full_Pair_QK_Relational_Architecture.pdf
    Main paper, equations, experimental results, limitations, and implementation notes.

QK_Relational_Architecture_Python_Quickstart.pdf
    Standalone Python-oriented implementation guide.

qk_minimal.py
    Minimal runnable PyTorch implementation of the tested 2-layer, d=48 model.

LICENSE-DOCUMENTATION.txt
    Documentation/figures/tables licensing notice: CC BY-NC 4.0.

LICENSE-CODE.txt
    Source-code and code-listing licensing notice: PolyForm Noncommercial 1.0.0.
```

## Quick start

Python 3.10+ is recommended.

```bash
pip install torch numpy
python qk_minimal.py --smoke
```

Expected smoke-test output includes:

```text
smoke test: PASS
parameters: 80352
normal logits: (2, 32, 256)
+3-hop logits: (2, 32, 256)
```

To train the minimal next-byte model on any local byte/text file:

```bash
python qk_minimal.py \
  --train-byte-file path/to/data.txt \
  --steps 1000 \
  --context 128 \
  --batch 8 \
  --seed 11
```

This saves `qk_language_model.pt` in the current directory.

## Model used in the latest small competitor audit

The language comparison used:

- 2 layers
- model width `d = 48`
- byte vocabulary of 256
- context length 128 during training
- QK model: headless, no learned value projection
- Transformer baseline: 4 attention heads, 12 dimensions per head
- exactly 80,352 stored parameters in each model
- three fixed seeds
- 4,194,304 training-target presentations per seed per model

Mean held-out results across the three seeds were:

| Model | Loss (nats/byte, lower is better) | Next-byte accuracy |
|---|---:|---:|
| QK | 1.7516 | 52.55% |
| 4-head Transformer | 1.7595 | 51.34% |

These results show that the tested QK model was competitive with the matched small Transformer in this experiment. Three seeds and one small real-text setting are **not sufficient to establish superiority**.

## Hop-mechanism result

In controlled randomized relation-chain tests, the separate strict-causal hop field learned a very sharp next-relation transition using final-answer training only. Across three tested seeds, the correct next relation was top-1 approximately 99.982% to 99.994% of the time in the transition diagnostic.

A model trained on relation depths up to 8 also showed useful extrapolation to substantially deeper requested hop counts in the controlled task. This is evidence for compositional reuse of the learned hop operator in that synthetic setting. It should **not** be interpreted as 20–30 steps of unrestricted natural-language reasoning.

The current endpoint/readout construction still has a boundary failure at the absolute deepest tested chain, even when the transition field itself continues to trace the chain correctly.

## Efficiency observations

The current implementation remains a **full-pair quadratic architecture in sequence length** because it forms QK token-pair scores.

At equal parameter count, the ordinary QK path had a modestly lower estimated major matrix-operation count in the tested context-128 setup and completed the small CPU training runs faster than the Transformer implementation. However:

- ordinary Transformer inference was faster in the measured PyTorch setup;
- the current dense QK implementation did **not** demonstrate a RAM advantage;
- both architectures remain quadratic in sequence length;
- optimized GPU/fused-kernel comparisons have not yet been performed.

The optional hop count mainly acts as a variable-compute control. Extra hops reuse the learned hop field rather than adding model parameters.

## Reproducibility and scope

The included `qk_minimal.py` reproduces the current architecture definition and passes internal architecture/masking/gradient smoke checks. The PDFs document the larger experimental protocol and limitations.

Important limitations include:

- small model scale;
- limited number of seeds;
- limited real-text benchmark coverage;
- no optimized GPU comparison;
- no demonstrated memory advantage;
- controlled synthetic hop tasks are not equivalent to general reasoning;
- results have not been independently peer reviewed.

## Licensing

This repository uses a split noncommercial license structure.

**Documentation, explanatory text, figures, and tables** are licensed under **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**. See `LICENSE-DOCUMENTATION.txt` and the canonical license terms linked there.

**Source code and code listings implementing the architecture** are licensed under the **PolyForm Noncommercial License 1.0.0**. See `LICENSE-CODE.txt` and the canonical license terms linked there.

Where a PDF contains both explanatory material and source-code listings, the corresponding license applies to each part as described above.

Neither license grants unrestricted commercial-use rights. See the main paper for the project contact and licensing note.
