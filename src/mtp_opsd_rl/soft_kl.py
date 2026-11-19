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


def ce_sum(W, h, labels, chunk=4096):
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


def soft_kl_loss(W, h_s, h_t, labels):
    """Loss whose gradient w.r.t. h_s equals that of  sum_i KL(q_i || p_i) / num_valid.

    W:      [V, d] LM head (detached here; must be frozen)
    h_s:    [N, d] student post-norm hidden, already divided by tau, requires grad
    h_t:    [N, d] teacher post-norm hidden, already divided by tau
    labels: [N]    any valid token ids at completion positions, -100 elsewhere

    The returned VALUE is CE_student + CE_teacher (not the KL); only its gradient is meaningful.
    """
    W, h_t = W.detach(), h_t.detach()
    n_valid = (labels != -100).sum().clamp_min(1)

    # Straight-through routing: forward value is exactly h_t,
    # but backward sends  -(grad w.r.t. h_t_aug)  into h_s.
    h_t_aug = h_t + (h_s.detach() - h_s)

    # grad wrt h_s:  W^T(p - e_x)  +  (-1) * W^T(q - e_x)  =  W^T(p - q)
    return (ce_sum(W, h_s, labels) + ce_sum(W, h_t_aug, labels)) / n_valid
