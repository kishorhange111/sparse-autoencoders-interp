"""SAEs on GPT-2 small's residual stream (GPU; run on Colab).

    python -m sae.lm --layer 6 --tokens 30_000_000 --out results

Pipeline: stream 10k Pile documents (NeelNanda/pile-10k) through GPT-2, take the residual
stream after block ``layer`` (non-BOS positions only), shuffle in a large activation
buffer, and train each SAE variant on the same token budget. Evaluation on held-out text:

- **FVU** and **L0**, the reconstruction-sparsity trade;
- **CE loss recovered**: splice the SAE reconstruction into the forward pass in place of the
  real activation. ``(CE_zero − CE_sae) / (CE_zero − CE_clean)``, where CE_zero
  zero-ablates the layer's residual. This is the metric that says whether the dictionary
  keeps what the model uses, not just what has variance;
- **dead / dense latents**: never firing, or firing on more than 10% of tokens;
- **dashboards**: top-activating contexts for a random sample of latents, as markdown.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from .model import SAE, fvu


def load_text(n_docs: int | None = None):
    from datasets import load_dataset

    # A parquet dataset: script-based ones (e.g. stas/openwebtext-10k) no longer load in
    # recent versions of ``datasets``.
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    docs = [d["text"] for d in ds]
    return docs[: n_docs or len(docs)]


class Acts:
    """Generates (B*T, d) residual activations from GPT-2 for a stream of token windows."""

    def __init__(self, model, tok, docs, layer, ctx=128, bs=32, device="cuda"):
        self.model, self.layer, self.ctx, self.bs, self.device = model, layer, ctx, bs, device
        ids = tok("\n\n".join(docs), return_tensors="pt", verbose=False).input_ids[0]
        n = len(ids) // (ctx - 1)
        windows = ids[: n * (ctx - 1)].view(n, ctx - 1)
        bos = torch.full((n, 1), tok.bos_token_id)
        self.windows = torch.cat([bos, windows], 1)  # BOS at position 0 (excluded from SAE data)
        self.i = 0

    def __len__(self):
        return len(self.windows)

    @torch.no_grad()
    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.i + self.bs > len(self.windows):
            self.i = 0
        w = self.windows[self.i : self.i + self.bs].to(self.device)
        self.i += self.bs
        out = self.model(w, output_hidden_states=True)
        h = out.hidden_states[self.layer]  # after block layer-1 == input to block `layer`
        return h[:, 1:].reshape(-1, h.shape[-1]).float(), w


class Buffer:
    def __init__(self, acts: Acts, n_batches: int = 64):
        self.acts, self.n = acts, n_batches
        self.data = torch.empty(0)
        self.refill()

    def refill(self):
        self.data = torch.cat([self.acts.next()[0] for _ in range(self.n)])
        self.data = self.data[torch.randperm(len(self.data), device=self.data.device)]
        self.ptr = 0

    def get(self, bs):
        if self.ptr + bs > len(self.data) // 2:  # refill at half-consumed for better mixing
            keep = self.data[self.ptr :]
            fresh = torch.cat([self.acts.next()[0] for _ in range(self.n // 2)])
            self.data = torch.cat([keep, fresh])
            self.data = self.data[torch.randperm(len(self.data), device=self.data.device)]
            self.ptr = 0
        x = self.data[self.ptr : self.ptr + bs]
        self.ptr += bs
        return x


def train_sae(sae: SAE, buf: Buffer, tokens: int, bs: int = 4096, lr: float = 2e-4,
              scale: float = 1.0, log=print) -> SAE:  # fmt: skip
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    steps = tokens // bs
    sae.init_b_dec(buf.get(bs * 4)[:16384] * scale)
    t0 = time.time()
    for step in range(1, steps + 1):
        for g in opt.param_groups:  # linear decay over the last 20%
            g["lr"] = lr * min(1.0, (steps - step) / (0.2 * steps))
        x = buf.get(bs) * scale
        _, f, losses = sae(x)
        loss = sum(losses.values())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if sae.kind != "relu":
            sae.remove_parallel_grad()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
        if sae.kind != "relu":
            sae.normalize_decoder()
        if step % 500 == 0 or step == steps:
            l0 = (f > 0).float().sum(1).mean()
            dead = (sae.steps_since_fired > 2000).float().mean()
            log(
                f"  step {step}/{steps} loss {loss.item():.4f} L0 {l0:.1f} dead {dead:.3f} "
                f"{time.time() - t0:.0f}s"
            )
    return sae


@torch.no_grad()
def evaluate(sae: SAE, model, acts_eval: Acts, scale: float, n_batches: int = 20) -> dict:
    sae.eval()
    layer = acts_eval.layer
    block = model.transformer.h[layer]
    mode = {"v": "clean"}

    def hook(_mod, args, kwargs):
        if mode["v"] == "clean":
            return None
        h = args[0] if args else kwargs["hidden_states"]
        if mode["v"] == "zero":
            new = torch.zeros_like(h)
        else:
            flat = h[:, 1:].reshape(-1, h.shape[-1]).float() * scale
            rec = (sae(flat)[0] / scale).to(h.dtype).view(h.shape[0], -1, h.shape[-1])
            new = torch.cat([h[:, :1], rec], 1)  # keep the BOS position untouched
        if args:
            return (new, *args[1:]), kwargs
        return args, {**kwargs, "hidden_states": new}

    handle = block.register_forward_pre_hook(hook, with_kwargs=True)
    ce = {"clean": [], "sae": [], "zero": []}
    fvus, l0s, fire_counts, n_tok = [], [], None, 0
    acts_eval.i = 0
    try:
        for _ in range(n_batches):
            x, w = acts_eval.next()
            xs = x * scale
            x_hat, f, _ = sae(xs)
            fvus.append(fvu(xs, x_hat))
            l0s.append(float((f > 0).float().sum(1).mean()))
            c = (f > 0).sum(0)
            fire_counts = c if fire_counts is None else fire_counts + c
            n_tok += len(x)
            for m in ce:
                mode["v"] = m
                logits = model(w).logits[:, 1:-1]
                ce[m].append(float(F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                                   w[:, 2:].reshape(-1))))  # fmt: skip
            mode["v"] = "clean"
    finally:
        handle.remove()
    m = {k: sum(v) / len(v) for k, v in ce.items()}
    freq = fire_counts.float() / n_tok
    sae.train()
    return {
        "fvu": sum(fvus) / len(fvus),
        "l0": sum(l0s) / len(l0s),
        "ce_clean": m["clean"],
        "ce_sae": m["sae"],
        "ce_zero": m["zero"],
        "ce_loss_recovered": (m["zero"] - m["sae"]) / (m["zero"] - m["clean"]),
        "dead_frac": float((freq == 0).float().mean()),
        "dense_frac": float((freq > 0.1).float().mean()),
    }


@torch.no_grad()
def dashboards(sae: SAE, acts_eval: Acts, tok, scale: float, n_latents: int = 12,
               n_batches: int = 30, top: int = 6, seed: int = 0) -> str:  # fmt: skip
    g = torch.Generator().manual_seed(seed)
    acts_eval.i = 0
    best: dict[int, list] = {}
    cand = None
    for _ in range(n_batches):
        x, w = acts_eval.next()
        f = sae.encode(x * scale)
        if cand is None:
            alive = ((f > 0).sum(0) > 0).nonzero().flatten().cpu()
            cand = alive[torch.randperm(len(alive), generator=g)[:n_latents]].tolist()
        T = w.shape[1] - 1
        for j in cand:
            v, i = f[:, j].topk(top)
            for val, idx in zip(v.tolist(), i.tolist(), strict=True):
                if val > 0:
                    b, t = divmod(idx, T)
                    best.setdefault(j, []).append((val, w[b].cpu(), t + 1))
    out = []
    for j in cand:
        rows = sorted(best.get(j, []), key=lambda r: -r[0])[:top]
        out.append(f"#### latent {j}\n")
        for val, ids, t in rows:
            left = tok.decode(ids[max(1, t - 12) : t]).replace("\n", "⏎")
            cur = tok.decode(ids[t : t + 1]).replace("\n", "⏎")
            right = tok.decode(ids[t + 1 : t + 4]).replace("\n", "⏎")
            out.append(f"- `{val:.2f}` …{left}**[{cur}]**{right}…")
        out.append("")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--tokens", type=float, default=30e6)
    ap.add_argument("--expansion", type=int, default=16)
    ap.add_argument("--variants", nargs="+", default=["topk:32", "batchtopk:32", "relu:5"])
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer, GPT2LMHeadModel

    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "lm_runs.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(x)["variant"] for x in open(path) if x.strip()}
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float16).to(dev).eval()
    docs = load_text()
    split = int(0.95 * len(docs))
    acts = Acts(model, tok, docs[:split], a.layer, device=dev)
    acts_eval = Acts(model, tok, docs[split:], a.layer, device=dev)
    buf = Buffer(acts)
    # Scale activations so E[||x||] = sqrt(d) (makes lr and l1 transferable across layers).
    sample = buf.get(8192)
    scale = math.sqrt(sample.shape[1]) / float(sample.norm(dim=1).mean())
    d = sample.shape[1]
    for spec in a.variants:
        if spec in done:
            continue
        kind, knob = spec.split(":")
        n = d * a.expansion
        if kind == "relu":
            sae = SAE(d, n, "relu", l1=float(knob))
        else:
            sae = SAE(d, n, kind, k=int(knob), k_aux=512, dead_after=2000)
        sae = sae.to(dev)
        print(f"== {spec}: {n} latents, {a.tokens:.0e} tokens", flush=True)
        t = time.time()
        train_sae(sae, buf, int(a.tokens), scale=scale)
        ev = evaluate(sae, model, acts_eval, scale)
        rec = {"variant": spec, "layer": a.layer, "latents": n, "tokens": a.tokens,
               "minutes": (time.time() - t) / 60, **ev}  # fmt: skip
        print(json.dumps(rec), flush=True)
        with open(os.path.join(a.out, f"dashboards_{spec.replace(':', '_')}.md"), "w",
                  encoding="utf-8") as f:  # fmt: skip
            f.write(f"# Top-activating contexts, {spec}, GPT-2 layer {a.layer}\n\n")
            f.write(dashboards(sae, acts_eval, tok, scale))
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
