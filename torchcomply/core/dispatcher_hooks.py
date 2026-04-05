"""
dispatcher_hooks.py — Dispatcher-level compliance via __torch_function__.

ComplianceTensor is a torch.Tensor subclass that intercepts EVERY tensor
operation at the dispatcher level — matmul, add, conv2d, relu — without
touching a single line of model code.

This is the second of three PyTorch compliance mechanisms:
  1. Forward hooks   — audit.py          (module-level granularity)
  2. __torch_function__ — this file       (operator-level granularity)
  3. Custom Autograd — autograd_provenance.py (gradient-level granularity)

__torch_function__ is PyTorch's official extension point for tensor subclasses.
It is called for every ATen operator invoked on a subclassed tensor, giving
complete visibility into computation without modifying model code.

Maps to EU AI Act Article 12 at the operator level.

Regulatory references:
  EU AI Act Art. 12 — Record-keeping and traceability
    https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689  (Article 12)

PyTorch extension points:
  __torch_function__ protocol:
    https://pytorch.org/docs/stable/notes/extending.html#extending-torch-python-api
  Tensor subclassing guide:
    https://pytorch.org/docs/stable/notes/extending.html#subclassing-torch-tensor

Reentrancy:
  Accessing .shape or .dtype on a ComplianceTensor inside __torch_function__
  triggers another dispatch call, causing infinite recursion. We guard against
  this with a thread-local ``active`` flag (threading.local is per-thread, so
  it is safe for DataLoader workers). See also: Python threading.local docs:
    https://docs.python.org/3/library/threading.html#thread-local-data
"""

from __future__ import annotations

import threading
import time
from typing import ClassVar, List


import torch

# Thread-local reentrancy guard: prevents recursive __torch_function__ calls
# that would occur when accessing .shape / .size() on ComplianceTensor instances
# inside the logging block (those attribute accesses can trigger dispatch again).
_RECORDING = threading.local()


class ComplianceTensor(torch.Tensor):
    """
    Tensor subclass that logs every operation at the Dispatcher level.

    Usage::

        x = ComplianceTensor(torch.randn(8, 10))
        out = model(x)                       # all ops inside model are logged
        log = ComplianceTensor.get_log()     # retrieve the operator log
    """

    _compliance_log: ClassVar[List[dict]] = []

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # Reentrancy guard: nested dispatch calls (e.g. from .shape access) just
        # execute the op without logging to avoid infinite recursion.
        if getattr(_RECORDING, "active", False):
            return super().__torch_function__(func, types, args, kwargs)

        _RECORDING.active = True
        try:
            result = super().__torch_function__(func, types, args, kwargs)
            input_shapes = [tuple(a.shape) for a in args if isinstance(a, torch.Tensor)]
            output_shape = tuple(result.shape) if isinstance(result, torch.Tensor) else None
            cls._compliance_log.append(
                {
                    "operator": func.__name__,
                    "timestamp": time.time_ns(),
                    "input_shapes": input_shapes,
                    "output_shape": output_shape,
                }
            )
            return result
        finally:
            _RECORDING.active = False

    @classmethod
    def get_log(cls) -> List[dict]:
        """Return a copy of the operator log."""
        return list(cls._compliance_log)

    @classmethod
    def clear_log(cls) -> None:
        """Reset the log (call before each measurement)."""
        cls._compliance_log.clear()
