"""Sparse autoencoders for dictionary learning on activations.

``x ≈ x̂ = W_dec f(x) + b_dec`` with ``f(x) = act(W_enc (x - b_dec) + b_enc)`` sparse.

Variants (``kind=``):

- ``"relu"``: ReLU latents with an L1 penalty weighted by decoder column norms,
  ``λ Σ_i f_i ‖W_dec[:, i]‖`` (Anthropic, "Towards Monosemanticity" and its 2024 update).
  Weighting by the norm stops the model from cheating the L1 term by shrinking f and growing
  the decoder.
- ``"topk"``: keep the k largest pre-activations and zero the rest (Gao et al. 2024). L0 is
  set directly, no λ to tune, and there is no shrinkage bias from an L1 term.
- ``"batchtopk"``: top ``k·B`` over the whole batch instead of per sample (Bussmann et al.
  2024), so L0 varies per input but averages to k.

TopK variants support the **AuxK** loss: the residual is reconstructed from the top
``k_aux`` *dead* latents, which revives them (Gao et al.). Decoder columns are kept at unit
norm for TopK variants (``normalize_decoder``), and the gradient component parallel to each
column is removed so the constraint and the optimiser don't fight.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SAE(nn.Module):
    def __init__(self, d: int, n_latents: int, kind: str = "topk", k: int = 32,
                 l1: float = 1e-3, k_aux: int = 0, dead_after: int = 1000) -> None:  # fmt: skip
        super().__init__()
        if kind not in ("relu", "topk", "batchtopk"):
            raise ValueError(kind)
        self.kind, self.k, self.l1, self.k_aux, self.dead_after = kind, k, l1, k_aux, dead_after
        w = torch.randn(d, n_latents)
        w = w / w.norm(dim=0, keepdim=True)
        self.W_dec = nn.Parameter(w.clone())  # (d, n): columns are dictionary directions
        self.W_enc = nn.Parameter(w.t().clone())  # tied init (Gao et al., Conerly et al.)
        self.b_enc = nn.Parameter(torch.zeros(n_latents))
        self.b_dec = nn.Parameter(torch.zeros(d))
        self.register_buffer("steps_since_fired", torch.zeros(n_latents, dtype=torch.long))

    @torch.no_grad()
    def init_b_dec(self, x: torch.Tensor, iters: int = 50) -> None:
        """Geometric median of a data sample (Weiszfeld), more robust than the mean."""
        y = x.mean(0)
        for _ in range(iters):
            w = 1 / (x - y).norm(dim=1).clamp(min=1e-8)
            y = (w[:, None] * x).sum(0) / w.sum()
        self.b_dec.copy_(y)

    def pre(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.W_enc.t() + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pre(x)
        if self.kind == "relu":
            return F.relu(z)
        if self.kind == "topk":
            v, i = z.topk(self.k, dim=-1)
            return torch.zeros_like(z).scatter_(-1, i, F.relu(v))
        flat = z.flatten()  # batchtopk
        v, i = flat.topk(self.k * z.shape[0])
        return torch.zeros_like(flat).scatter_(0, i, F.relu(v)).view_as(z)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec.t() + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        z = self.pre(x)
        f = self.encode(x)
        x_hat = self.decode(f)
        mse = (x_hat - x).pow(2).sum(-1).mean()
        losses = {"mse": mse}
        if self.kind == "relu":
            losses["l1"] = self.l1 * (f * self.W_dec.norm(dim=0)).sum(-1).mean()
        if self.training:
            fired = (f > 0).any(0)
            self.steps_since_fired += 1
            self.steps_since_fired[fired] = 0
        if self.k_aux and self.kind != "relu":
            dead = self.steps_since_fired > self.dead_after
            if dead.any():
                zd = z.masked_fill(~dead, float("-inf"))
                k = min(self.k_aux, int(dead.sum()))
                v, i = zd.topk(k, dim=-1)
                fa = torch.zeros_like(z).scatter_(-1, i, F.relu(v))
                resid = (x - x_hat).detach()
                aux = (fa @ self.W_dec.t() - resid).pow(2).sum(-1).mean()
                losses["aux"] = aux / resid.pow(2).sum(-1).mean().clamp(min=1e-8) * 1 / 32
        return x_hat, f, losses

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec.div_(self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8))

    @torch.no_grad()
    def remove_parallel_grad(self) -> None:
        if self.W_dec.grad is None:
            return
        w = self.W_dec / self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.grad -= (self.W_dec.grad * w).sum(0, keepdim=True) * w


def fvu(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Fraction of variance unexplained."""
    x, x_hat = x.detach(), x_hat.detach()
    return float((x - x_hat).pow(2).sum() / (x - x.mean(0)).pow(2).sum())
