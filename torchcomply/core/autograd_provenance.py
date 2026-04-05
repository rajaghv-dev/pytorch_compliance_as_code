"""
autograd_provenance.py — Data provenance tracking via custom Autograd Function.

ProvenanceLinear is a drop-in replacement for the first linear layer in a model.
It records which data-subject IDs were in each forward batch, then logs those IDs
against the gradient norm in the backward pass.

This is the third of three PyTorch compliance mechanisms:
  1. Forward hooks   — audit.py          (module-level: *what* ran)
  2. __torch_function__ — dispatcher_hooks.py (operator-level: *which ops* ran)
  3. Custom Autograd — this file          (gradient-level: *whose data* influenced weights)

Machine Unlearning Workflow
---------------------------
When a data subject invokes their GDPR Article 17 right to erasure:
  1. Query ``ProvenanceLinear.get_provenance_log()`` for entries containing their ID.
  2. Identify which training steps (and therefore which weight updates) they influenced.
  3. Retrain from the last clean checkpoint, omitting their data — or apply influence
     function perturbations to reverse their contribution (see Koh & Liang, 2017).

This is significantly cheaper than full retraining when the subject's influence is
localised to a small number of gradient steps.

Regulatory references:
  GDPR Art. 17 — Right to Erasure ("Right to be Forgotten")
    https://gdpr-info.eu/art-17-gdpr/
  EU AI Act Art. 12 — Record-Keeping
    https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689  (Article 12)

Related work:
  Koh & Liang (2017) — Understanding Black-box Predictions via Influence Functions
    https://arxiv.org/abs/1703.04730
  Cao & Yang (2015) — Towards Making Systems Forget with Machine Unlearning
    https://arxiv.org/abs/1906.00707

PyTorch custom Autograd Functions:
  https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd
"""

from __future__ import annotations

import time
from typing import ClassVar, List, Optional

import torch
from torch import Tensor


class ProvenanceLinear(torch.autograd.Function):
    """
    Custom Autograd Function that tracks data-subject provenance through
    the backward pass.

    Usage::

        output = ProvenanceLinear.apply(x, weight, bias, subject_ids)

    Where ``subject_ids`` is a 1-D tensor with one ID per sample in the batch.
    The gradient step in ``backward`` logs which subjects influenced the update.
    """

    _provenance_log: ClassVar[List[dict]] = []

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        subject_ids: Tensor,
    ) -> Tensor:
        # ctx.save_for_backward stores tensors in the autograd graph.
        # Using save_for_backward (not storing on ctx directly) is required
        # for correct behaviour with second-order gradients and with
        # gradient checkpointing. See PyTorch docs for details.
        ctx.save_for_backward(input, weight, bias)
        # subject_ids is not a differentiable tensor, so we store it directly.
        ctx.subject_ids = subject_ids
        output = input @ weight.t()
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        input, weight, bias = ctx.saved_tensors

        # Record which subjects influenced this gradient update.
        # grad_output.norm() gives the L2 norm of the upstream gradient signal —
        # a proxy for how much this batch's data affected the weight update.
        # A larger norm means a larger influence on model parameters.
        sids = ctx.subject_ids
        record = {
            "subject_ids": sids.tolist() if hasattr(sids, "tolist") else list(sids),
            "grad_norm": float(grad_output.norm().item()),
            "timestamp": time.time_ns(),
        }
        ProvenanceLinear._provenance_log.append(record)

        grad_input = grad_output @ weight
        grad_weight = input.t() @ grad_output
        grad_bias = grad_output.sum(0) if bias is not None else None
        return grad_input, grad_weight.t(), grad_bias, None  # None for subject_ids

    @classmethod
    def get_provenance_log(cls) -> List[dict]:
        return list(cls._provenance_log)

    @classmethod
    def clear_log(cls) -> None:
        cls._provenance_log.clear()
