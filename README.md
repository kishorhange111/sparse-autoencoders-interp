# sparse-autoencoders-interp

[![CI](https://github.com/kishorhange111/sparse-autoencoders-interp/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/sparse-autoencoders-interp/actions/workflows/ci.yml)

Sparse autoencoders (SAEs) for mechanistic interpretability, implemented from scratch and
evaluated two ways:

1. **Against ground truth**, on a superposition toy model where the true features are known.
   Do the SAEs find them?
2. **On GPT-2 small's residual stream**, by splicing the SAE's reconstruction into the forward
   pass and measuring how much of the language-model loss survives.

## Implementation ([`model.py`](src/sae/model.py))

| variant | sparsity mechanism | from |
|---|---|---|
| `relu` | ReLU + L1 weighted by decoder norms, `λ Σ fᵢ ‖W_dec,ᵢ‖`, so the penalty can't be dodged by rescaling | Anthropic, *Towards Monosemanticity* (2023) and the 2024 update |
| `topk` | keep the k largest pre-activations: L0 is fixed and there's no L1 shrinkage | Gao et al., *Scaling and evaluating sparse autoencoders* (2024) |
| `batchtopk` | top k·B over the batch, so L0 adapts per input and averages to k | Bussmann et al. (2024) |

Also implemented:
- the **AuxK** loss, which reconstructs the residual from the top dead latents to revive them;
- unit-norm decoder columns, with the **parallel gradient component removed** so the
  constraint and Adam don't fight;
- `b_dec` initialised at the **geometric median** (Weiszfeld);
- tied encoder/decoder initialisation.

## Evaluation

- **Toy model** ([`toy.py`](src/sae/toy.py)): 256 unit feature directions in 64 dimensions
  (4× superposition), each active with probability p. The metrics are MMCS (mean over true
  features of the best cosine with any decoder column) and the fraction recovered at
  cosine > 0.9, alongside L0, FVU and dead latents. It sweeps λ for the L1 SAE and k for
  the TopK variants.
- **GPT-2** ([`lm.py`](src/sae/lm.py)): residual stream at layer 6 on OpenWebText-10k,
  with 16× expansion (12,288 latents) and the same token budget for every variant. It
  reports FVU, L0, dead and dense latents, and **CE loss recovered**
  `(CE_zero − CE_sae) / (CE_zero − CE_clean)` with the reconstruction spliced in through a
  forward pre-hook. It also writes top-activating-context dashboards for a random sample of
  live latents.

```
pytest -q                                   # 11 tests (incl. the splice on a tiny random GPT-2)
python -m sae.toy --out results             # toy sweep
python -m sae.lm --layer 6 --out results    # GPT-2 (GPU), resumable per variant
```

The tests check, among other things:
- TopK's L0 is exact, and BatchTopK's averages to k;
- the L1 term scales with decoder norm;
- the parallel gradient really is removed;
- AuxK fires only when latents are dead;
- the geometric median ignores outliers;
- a perfect dictionary scores MMCS = 1;
- an exact-reconstruction SAE recovers 100% of GPT-2's loss.

**Results are pending.** Both experiments run on Colab, and the tables will be filled in from
`results/`.
