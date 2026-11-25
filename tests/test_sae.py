import pytest
import torch

from sae.model import SAE, fvu
from sae.toy import make_features, recovery, sample, train_toy


def test_topk_has_exact_l0():
    sae = SAE(8, 32, "topk", k=3)
    with torch.no_grad():
        sae.b_enc.fill_(5.0)  # every pre-activation positive, so exactly k survive
    f = sae.encode(torch.randn(10, 8))
    assert ((f > 0).sum(1) == 3).all()


def test_batchtopk_l0_averages_to_k():
    sae = SAE(8, 32, "batchtopk", k=4)
    with torch.no_grad():
        sae.b_enc.fill_(5.0)
    f = sae.encode(torch.randn(16, 8))
    assert int((f > 0).sum()) == 4 * 16


def test_relu_l1_is_weighted_by_decoder_norm():
    torch.manual_seed(0)
    sae = SAE(8, 16, "relu", l1=1.0)
    x = torch.randn(4, 8)
    base = sae(x)[2]["l1"]
    with torch.no_grad():
        sae.W_dec.mul_(2.0)
    assert sae(x)[2]["l1"].item() == pytest.approx(2 * base.item(), rel=1e-5)


def test_parallel_grad_removed_and_decoder_normalised():
    torch.manual_seed(0)
    sae = SAE(8, 16, "topk", k=2)
    _, _, losses = sae(torch.randn(32, 8))
    losses["mse"].backward()
    sae.remove_parallel_grad()
    w = sae.W_dec / sae.W_dec.norm(dim=0, keepdim=True)
    assert (sae.W_dec.grad * w).sum(0).abs().max() < 1e-5
    with torch.no_grad():
        sae.W_dec.mul_(3.0)
    sae.normalize_decoder()
    torch.testing.assert_close(sae.W_dec.norm(dim=0), torch.ones(16))


def test_auxk_loss_appears_only_with_dead_latents():
    torch.manual_seed(0)
    sae = SAE(8, 16, "topk", k=2, k_aux=4, dead_after=3)
    with torch.no_grad():  # latents 0 and 1 always win; the rest can never fire
        sae.b_enc[:2] = 100.0
        sae.b_enc[2:] = -100.0
    x = torch.randn(32, 8)
    assert "aux" not in sae(x)[2]
    for _ in range(3):
        losses = sae(x)[2]
    assert int((sae.steps_since_fired > 3).sum()) == 14
    assert "aux" in losses and losses["aux"] > 0


def test_geometric_median_ignores_outliers():
    x = torch.zeros(101, 2)  # four symmetric clusters: unique median at the origin
    x[:25, 0], x[25:50, 0], x[50:75, 1], x[75:100, 1] = -1.0, 1.0, -1.0, 1.0
    x[100] = torch.tensor([1e6, 1e6])
    sae = SAE(2, 4)
    sae.init_b_dec(x)
    assert sae.b_dec.abs().max() < 0.1  # the mean would be ~1e4


def test_recovery_of_perfect_dictionary_is_one():
    feats = make_features(16, 8)
    sae = SAE(8, 16)
    with torch.no_grad():
        sae.W_dec.copy_(feats.t() * 3.0)  # scale doesn't matter
    r = recovery(sae, feats)
    assert r["mmcs"] == pytest.approx(1.0) and r["recovered"] == 1.0


def test_fvu_bounds():
    x = torch.randn(100, 4)
    assert fvu(x, x) == 0.0
    assert fvu(x, x.mean(0).expand_as(x)) == pytest.approx(1.0)


def test_toy_training_beats_random_dictionary():
    feats = make_features(32, 16)
    g = torch.Generator().manual_seed(0)
    assert sample(feats, 4, 0.1, g).shape == (4, 16)
    rnd = recovery(SAE(16, 64), feats)["mmcs"]
    r = train_toy("topk", 3, feats, 0.1, 64, steps=300, bs=256, lr=3e-3)
    assert r["mmcs"] > rnd + 0.2
    assert r["l0"] == pytest.approx(3.0)
