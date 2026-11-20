"""Soft-KL: forward KL(teacher || student) gradients from two hard-label cross-entropies.

Implements the KL gradient-recovery trick (Teoh 2026, "Multi-Token Prediction Done Right")
for OPSD. No [N, V] logit tensor is kept for backward.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _acc_dtype(t):
    # at least float32 (bf16 -> fp32 for a stable log-softmax), never downcast fp64
    return torch.promote_types(t.dtype, torch.float32)


def _ce_chunk(h, W, labels):
    # logits for one chunk only
    logits = (h @ W.T).to(_acc_dtype(h))
    return F.cross_entropy(logits, labels, ignore_index=-100, reduction="sum")


def ce_sum(W, h, labels, chunk=1024):
    """Sum of CE(labels, softmax(h @ W.T)) over positions with labels != -100.

    h: [N, d], labels: [N], W: [V, d] (no grad).
    Chunked + checkpointed: each chunk's [chunk, V] logits are freed after the
    forward and recomputed during backward, so peak memory is one chunk.
    """
    total = h.new_zeros((), dtype=_acc_dtype(h))
    for s in range(0, h.shape[0], chunk):
        hc, yc = h[s:s + chunk], labels[s:s + chunk]
        if (yc != -100).any():
            total = total + checkpoint(_ce_chunk, hc, W, yc, use_reentrant=False)
    return total


@torch.no_grad()
def per_token_forward_kl(W, h_s, h_t, chunk=1024):
    """KL(q_i || p_i) VALUE per position, [N]. No grad, chunked: peak memory is one chunk of logits."""
    out = []
    for s in range(0, h_s.shape[0], chunk):
        lp = F.log_softmax((h_s[s:s + chunk] @ W.T).to(_acc_dtype(h_s)), -1)
        lq = F.log_softmax((h_t[s:s + chunk] @ W.T).to(_acc_dtype(h_t)), -1)
        out.append((lq.exp() * (lq - lp)).sum(-1))
    return torch.cat(out)


def soft_kl_loss(W, h_s, h_t, labels, token_clip=None, compute_kl=False, stats=None, fp32=False, tau=1.0):
    """Loss whose gradient w.r.t. h_s equals that of
        sum_i KL(q_i || p_i) / num_valid                          (token_clip=None)
        sum_i clamp(KL(q_i || p_i), max=token_clip) / num_valid   (per-TOKEN clip)

    W:      [V, d] LM head (detached here; must be frozen)
    h_s:    [N, d] student post-norm hidden, requires grad
    h_t:    [N, d] teacher post-norm hidden
    tau:    softmax temperature; applied to the hidden states AFTER the optional fp32 upcast
            (W(h/tau) = (Wh)/tau; dividing a bf16 h by tau first would round every logit)
    labels: [N]    any valid token ids at completion positions, -100 elsewhere
    compute_kl: also compute the true KL value (one extra no-grad chunked pass) for logging
    stats:  optional dict, filled with "kl_mean" (and "clip_frac") when the KL is computed
    fp32:   run the CE matmuls in fp32. Each CE gradient is dominated by the large -W[x] term,
            and the KL gradient is their small difference; in bf16 that subtraction loses
            precision (catastrophic cancellation). Costs one fp32 copy of W (V*d*4 bytes).

    The returned VALUE is CE_student + CE_teacher (not the KL); only its gradient is meaningful.
    """
    W, h_t = W.detach(), h_t.detach()
    if fp32:
        W, h_s, h_t = W.float(), h_s.float(), h_t.float()
    if tau != 1.0:
        h_s, h_t = h_s / tau, h_t / tau
    valid = labels != -100
    n_valid = valid.sum().clamp_min(1)  # clipped tokens stay in the denominator, as with clamp()

    if token_clip is not None or compute_kl:
        kl = per_token_forward_kl(W, h_s.detach(), h_t)
        if stats is not None:
            stats["kl_mean"] = (kl[valid].sum() / n_valid).item()
        if token_clip is not None:
            clipped = valid & (kl > token_clip)          # clamp(max=c) has zero grad here
            labels = labels.masked_fill(clipped, -100)   # drop these tokens from BOTH CEs
            if stats is not None:
                stats["clip_frac"] = (clipped.sum() / n_valid).item()

    # Straight-through routing: forward value is exactly h_t,
    # but backward sends  -(grad w.r.t. h_t_aug)  into h_s.
    h_t_aug = h_t + (h_s.detach() - h_s)

    # grad wrt h_s:  W^T(p - e_x)  +  (-1) * W^T(q - e_x)  =  W^T(p - q)
    return (ce_sum(W, h_s, labels) + ce_sum(W, h_t_aug, labels)) / n_valid
