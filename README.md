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
- **GPT-2** ([`lm.py`](src/sae/lm.py)): residual stream at layer 6 on the first 10k documents of the Pile (`NeelNanda/pile-10k`),
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

## Results

### Toy model (Tesla T4, [`results/toy_recovery.md`](results/toy_recovery.md))

256 true features in d=64 (4× superposition), each active w.p. 0.02 (true L0 ≈ 5.1); SAEs with 512 latents, 3000 steps × 1024.

| SAE | knob | L0 | FVU | MMCS | recovered (cos > 0.9) | dead latents |
|---|---|---|---|---|---|---|
| relu | λ=0.001 | 149.6 | 0.003 | 0.387 | 0% | 3% |
| relu | λ=0.003 | 149.4 | 0.008 | 0.410 | 0% | 1% |
| relu | λ=0.01 | 139.5 | 0.006 | 0.410 | 0% | 0% |
| relu | λ=0.03 | 137.1 | 0.020 | 0.486 | 0% | 0% |
| relu | λ=0.1 | 76.5 | 0.090 | 0.782 | 10% | 0% |
| topk | k=2 | 2.0 | 0.332 | 0.995 | 100% | 50% |
| topk | k=4 | 4.0 | 0.150 | 0.999 | 100% | 46% |
| topk | k=6 | 6.0 | 0.120 | 0.998 | 100% | 34% |
| topk | k=8 | 8.0 | 0.114 | 0.996 | 100% | 28% |
| topk | k=16 | 16.0 | 0.098 | 0.974 | 100% | 28% |
| batchtopk | k=2 | 2.0 | 0.314 | 0.947 | 83% | 52% |
| batchtopk | k=4 | 4.0 | 0.243 | 0.903 | 61% | 50% |
| batchtopk | k=6 | 6.0 | 0.196 | 0.904 | 68% | 54% |
| batchtopk | k=8 | 8.0 | 0.170 | 0.902 | 74% | 53% |
| batchtopk | k=16 | 16.0 | 0.136 | 0.753 | 30% | 45% |

What it shows:
- **TopK finds the ground-truth dictionary; L1 does not.** At every k from 2 to 16, TopK
  recovers 100% of the 256 true features (MMCS ≥ 0.97). No L1 setting comes close: with
  λ ≤ 0.03, L0 stays around 140–150 against a true L0 of about 5, the decoder learns dense
  mixtures (MMCS ≈ 0.4) and nothing is recovered. Its FVU is the *lowest* in the table, so
  reconstruction error alone would have ranked the worst dictionary first.
- **Raising λ trades reconstruction for sparsity, slowly.** λ = 0.1 cuts L0 to 76 and
  lifts MMCS to 0.78, but recovers only 10% of features while FVU is already worse than
  TopK k = 8.
- **BatchTopK is worse than TopK here.** It recovers 61–83% for k ≤ 8 and 30% at k = 16.
  Sharing the k·B budget across the batch lets frequent-looking latents absorb several
  features. On this data, where every input has about the same number of active features,
  the per-sample constraint is the right prior.
- **Dead latents are expected.** 512 latents for 256 true features means a perfect TopK
  dictionary leaves about half unused, which is roughly what k = 2–4 shows.

### GPT-2 layer 6

Pending: the Colab run of `python -m sae.lm` has not finished yet.
