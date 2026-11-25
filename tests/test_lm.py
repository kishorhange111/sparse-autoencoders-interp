"""The GPT-2 splice evaluation, on a tiny randomly initialised GPT-2 (no download)."""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from sae.lm import Acts, evaluate  # noqa: E402
from sae.model import SAE  # noqa: E402


class _Tok:
    bos_token_id = 0

    def __call__(self, text, return_tensors=None, verbose=False):
        ids = torch.tensor([[1 + (ord(c) % 60) for c in text]])
        return type("Enc", (), {"input_ids": ids})


def _setup():
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(n_layer=2, n_embd=16, n_head=2, vocab_size=64, n_positions=32)
    model = transformers.GPT2LMHeadModel(cfg).eval()
    acts = Acts(model, _Tok(), ["hello world " * 40], layer=1, ctx=16, bs=4, device="cpu")
    return model, acts


def test_exact_reconstruction_recovers_all_loss():
    model, acts = _setup()
    d = 16
    sae = SAE(d, 2 * d, "relu", l1=0.0)  # relu([x, -x]) decoded by [I, -I] is the identity
    eye = torch.eye(d)
    with torch.no_grad():
        sae.W_enc.copy_(torch.cat([eye, -eye]))
        sae.W_dec.copy_(torch.cat([eye, -eye], 1))
        sae.b_enc.zero_()
        sae.b_dec.zero_()
    ev = evaluate(sae, model, acts, scale=1.0, n_batches=2)
    assert ev["fvu"] < 1e-10
    assert ev["ce_sae"] == pytest.approx(ev["ce_clean"], abs=1e-5)
    assert ev["ce_loss_recovered"] == pytest.approx(1.0, abs=1e-4)


def test_untrained_sae_recovers_less():
    model, acts = _setup()
    ev = evaluate(SAE(16, 64, "topk", k=2), model, acts, scale=1.0, n_batches=2)
    assert ev["ce_loss_recovered"] < 0.99
    assert ev["ce_zero"] != ev["ce_clean"]
