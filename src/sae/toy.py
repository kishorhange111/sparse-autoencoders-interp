"""Superposition toy model with known ground truth (after Sharkey et al. 2022, Elhage et al.
2022): ``n_true`` unit feature directions in ``d < n_true`` dimensions, each active
independently with probability ``p`` and magnitude U(0, 1). Because the true dictionary is
known, we can measure what reconstruction metrics can't: whether the SAE found the features.

- **MMCS**: mean over true features of the max cosine similarity with any decoder column.
- **recovered**: fraction of true features with a decoder column at cosine > 0.9.

    python -m sae.toy --out results
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from .model import SAE, fvu


def make_features(n_true: int, d: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(n_true, d, generator=g)
    return f / f.norm(dim=1, keepdim=True)


def sample(feats: torch.Tensor, n: int, p: float, g: torch.Generator) -> torch.Tensor:
    """Sampled on CPU with a seeded generator (identical data on any device), then moved."""
    active = torch.rand(n, feats.shape[0], generator=g) < p
    coef = torch.rand(n, feats.shape[0], generator=g) * active
    return coef.to(feats.device) @ feats


@torch.no_grad()
def recovery(sae: SAE, feats: torch.Tensor) -> dict:
    dec = sae.W_dec / sae.W_dec.norm(dim=0, keepdim=True)
    cos = feats @ dec  # (n_true, n_latents)
    best = cos.max(1).values
    return {"mmcs": float(best.mean()), "recovered": float((best > 0.9).float().mean())}


def train_toy(
    kind: str,
    knob: float,
    feats: torch.Tensor,
    p: float,
    n_latents: int,
    steps: int = 3000,
    bs: int = 1024,
    lr: float = 1e-3,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 1)
    d = feats.shape[1]
    kw = {"l1": knob} if kind == "relu" else {"k": int(knob), "k_aux": 2 * int(knob),
                                              "dead_after": 200}  # fmt: skip
    sae = SAE(d, n_latents, kind, **kw).to(feats.device)
    sae.init_b_dec(sample(feats, 4096, p, g))
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    for _ in range(steps):
        x = sample(feats, bs, p, g)
        _, _, losses = sae(x)
        opt.zero_grad()
        sum(losses.values()).backward()
        if kind != "relu":
            sae.remove_parallel_grad()
        opt.step()
        if kind != "relu":
            sae.normalize_decoder()
    sae.eval()
    x = sample(feats, 8192, p, torch.Generator().manual_seed(999))
    x_hat, f, _ = sae(x)
    return {
        "kind": kind,
        "knob": knob,
        "l0": float((f > 0).float().sum(1).mean()),
        "fvu": fvu(x, x_hat),
        "dead_frac": float(((f > 0).sum(0) == 0).float().mean()),
        **recovery(sae, feats),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-true", type=int, default=256)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--p", type=float, default=0.02)  # expected L0 of the data = n_true * p
    ap.add_argument("--latents", type=int, default=512)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    feats = make_features(a.n_true, a.d).to(a.device)
    sweep = [("relu", l1) for l1 in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)]
    sweep += [(kind, k) for kind in ("topk", "batchtopk") for k in (2, 4, 6, 8, 16)]
    rows = []
    for kind, knob in sweep:
        t = time.time()
        r = train_toy(kind, knob, feats, a.p, a.latents, a.steps)
        r["seconds"] = time.time() - t
        rows.append(r)
        print(json.dumps(r), flush=True)
    true_l0 = a.n_true * a.p
    lines = [
        f"{a.n_true} true features in d={a.d} ({a.n_true / a.d:.0f}× superposition), each "
        f"active w.p. {a.p} (true L0 ≈ {true_l0:.1f}); SAEs with {a.latents} latents, "
        f"{a.steps} steps × 1024.\n",
        "| SAE | knob | L0 | FVU | MMCS | recovered (cos > 0.9) | dead latents |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        knob = f"λ={r['knob']:g}" if r["kind"] == "relu" else f"k={int(r['knob'])}"
        lines.append(
            f"| {r['kind']} | {knob} | {r['l0']:.1f} | {r['fvu']:.3f} | {r['mmcs']:.3f} | "
            f"{100 * r['recovered']:.0f}% | {100 * r['dead_frac']:.0f}% |"
        )
    with open(os.path.join(a.out, "toy_recovery.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(a.out, "toy_recovery.json"), "w") as f:
        json.dump({"args": vars(a), "rows": rows}, f, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
