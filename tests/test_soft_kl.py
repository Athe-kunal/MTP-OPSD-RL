"""CPU unit tests for mtp_opsd_rl.soft_kl. Run from repo root: uv run pytest tests -q"""
import torch
import torch.nn.functional as F

from mtp_opsd_rl import soft_kl
from mtp_opsd_rl.soft_kl import ce_sum, soft_kl_loss

N, d, V, TAU = 37, 16, 101, 1.1


def repo_forward_kl(h_s, h_t, W, labels, tau):
    """Same math as OPSDTrainer.generalized_jsd_loss with beta=0, token_clip=None."""
    s = F.log_softmax((h_s @ W.T) / tau, dim=-1)
    t = F.log_softmax((h_t @ W.T) / tau, dim=-1)
    jsd = F.kl_div(s, t, reduction="none", log_target=True)
    mask = labels != -100
    return jsd[mask].sum() / mask.sum()


def make(seed=0):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(V, d, generator=g, dtype=torch.float64, requires_grad=True)
    h_s = torch.randn(N, d, generator=g, dtype=torch.float64, requires_grad=True)
    h_t = torch.randn(N, d, generator=g, dtype=torch.float64)
    labels = torch.randint(0, V, (N,), generator=g)
    labels[::5] = -100  # some ignored positions (prompt / padding)
    return W, h_s, h_t, labels


def grad_hs(loss, h_s):
    return torch.autograd.grad(loss, h_s, retain_graph=True)[0]


def test_ce_sum_matches_plain_ce():
    W, h_s, _, labels = make()
    ref = F.cross_entropy(h_s @ W.T, labels, ignore_index=-100, reduction="sum")
    assert torch.allclose(ce_sum(W, h_s, labels, chunk=8), ref)


def test_gradient_equals_forward_kl():
    W, h_s, h_t, labels = make()
    g_ref = grad_hs(repo_forward_kl(h_s, h_t, W, labels, TAU), h_s)
    g_soft = grad_hs(soft_kl_loss(W, h_s / TAU, h_t / TAU, labels), h_s)
    assert torch.allclose(g_soft, g_ref, atol=1e-10)


def test_chunking_does_not_change_gradient():
    W, h_s, h_t, labels = make(1)
    g_big = grad_hs(soft_kl_loss(W, h_s, h_t, labels), h_s)
    orig = soft_kl.ce_sum
    soft_kl.ce_sum = lambda W_, h_, y_: orig(W_, h_, y_, chunk=7)
    try:
        g_small = grad_hs(soft_kl_loss(W, h_s, h_t, labels), h_s)
    finally:
        soft_kl.ce_sum = orig
    assert torch.allclose(g_big, g_small, atol=1e-12)


def test_label_choice_is_irrelevant():
    W, h_s, h_t, labels = make(2)
    other = torch.randint(0, V, (N,))
    other[labels == -100] = -100  # same mask, different token ids
    g1 = grad_hs(soft_kl_loss(W, h_s, h_t, labels), h_s)
    g2 = grad_hs(soft_kl_loss(W, h_s, h_t, other), h_s)
    assert torch.allclose(g1, g2, atol=1e-10)


def test_ignored_positions_get_zero_grad_and_head_untouched():
    W, h_s, h_t, labels = make(3)
    soft_kl_loss(W, h_s, h_t, labels).backward()
    assert torch.all(h_s.grad[labels == -100] == 0)
    assert W.grad is None  # W is detached inside the loss
