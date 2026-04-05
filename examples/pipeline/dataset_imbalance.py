"""
Dataset imbalance detector — Art. 10 (Data Governance).

WHAT THIS DEMONSTRATES
----------------------
Wraps a PyTorch DataLoader to monitor class distribution during the first N
batches and emit a warning when the max/min class ratio exceeds 3:1.

This is an Art. 10 §2(f) compliance control: high-risk AI systems must use
training data that is free of biases that could lead to prohibited
discrimination or dangerous inaccuracies.

HOW IT WORKS
------------
1. A custom collate wrapper intercepts each batch from the DataLoader.
2. It accumulates class counts across the first MONITOR_BATCHES batches.
3. After the monitoring window, it computes the max/min class ratio.
4. If the ratio exceeds IMBALANCE_THRESHOLD, it emits a compliance warning
   that includes the specific class counts.

USAGE
-----
    from examples.dataset_imbalance import ImbalanceMonitor

    loader = DataLoader(dataset, batch_size=32)
    monitor = ImbalanceMonitor(loader, threshold=3.0, monitor_batches=50)

    for batch in monitor:
        images, labels = batch
        # training loop ...

    if monitor.is_imbalanced:
        print(f"Class counts: {monitor.class_counts}")

REGULATORY MAPPING
------------------
  Art. 10 §2(f): Training, validation and testing data sets shall take
                 into account demographic and other characteristics or
                 factors in order to mitigate possible biases.
  Art. 10 §3:    Training, validation and testing data sets shall be
                 relevant, sufficiently representative, and as free of
                 errors as possible.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger("pct.example.dataset_imbalance")

# ----------------------------------------------------------------------- #
# Configuration
# ----------------------------------------------------------------------- #

# Alert if (most frequent class count) / (least frequent class count) > this.
IMBALANCE_THRESHOLD = 3.0

# How many batches to monitor before producing the imbalance report.
MONITOR_BATCHES = 50

# Index of the label in each batch tuple (0 = first element is image, 1 = label).
LABEL_INDEX = 1


# ----------------------------------------------------------------------- #
# ImbalanceMonitor
# ----------------------------------------------------------------------- #

class ImbalanceMonitor:
    """
    Wraps a DataLoader to monitor class distribution in the first N batches.

    Parameters
    ----------
    loader : DataLoader
        The DataLoader to wrap.
    threshold : float
        Max/min class count ratio that triggers an Art.10 warning.
    monitor_batches : int
        Number of batches to examine before producing the report.
    label_index : int
        Which element of each batch tuple contains the class labels.
    """

    def __init__(
        self,
        loader: DataLoader,
        threshold: float = IMBALANCE_THRESHOLD,
        monitor_batches: int = MONITOR_BATCHES,
        label_index: int = LABEL_INDEX,
    ) -> None:
        self.loader          = loader
        self.threshold       = threshold
        self.monitor_batches = monitor_batches
        self.label_index     = label_index

        self._class_counts: Counter = Counter()
        self._batches_seen:  int    = 0
        self._report_done:   bool   = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def class_counts(self) -> dict[int, int]:
        """Class counts accumulated so far."""
        return dict(self._class_counts)

    @property
    def imbalance_ratio(self) -> float:
        """
        Current max / min class count ratio.
        Returns 1.0 if zero or one class has been seen.
        """
        return _compute_ratio(self._class_counts)

    @property
    def is_imbalanced(self) -> bool:
        """True if the ratio exceeds the threshold."""
        return self.imbalance_ratio > self.threshold

    def __iter__(self) -> Iterator[Any]:
        """
        Iterate over the wrapped DataLoader, monitoring class distribution.
        """
        for batch in self.loader:
            self._observe_batch(batch)
            yield batch

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _observe_batch(self, batch: Any) -> None:
        """
        Extract labels from the batch and update class counts.

        Calls _check_imbalance() once the monitoring window is complete.
        """
        # Extract labels; handle (data, labels) tuples and plain tensors.
        labels = _extract_labels(batch, self.label_index)
        if labels is None:
            return

        for lbl in labels.view(-1).tolist():
            self._class_counts[int(lbl)] += 1

        self._batches_seen += 1

        # Produce the report once after the monitoring window.
        if self._batches_seen == self.monitor_batches and not self._report_done:
            self._check_imbalance()
            self._report_done = True

    def _check_imbalance(self) -> None:
        """
        Compute the max/min class ratio and emit a compliance warning if
        the threshold is exceeded.
        """
        if len(self._class_counts) < 2:
            return   # Not enough classes to compute a ratio.

        ratio = _compute_ratio(self._class_counts)

        if ratio > self.threshold:
            # Find the most and least common classes for the warning message.
            most_common  = self._class_counts.most_common(1)[0]
            least_common = self._class_counts.most_common()[-1]

            logger.warning(
                "COMPLIANCE WARNING [Art.10 §2(f)]: Dataset imbalance detected "
                "after %d batches. Max/min class ratio = %.1f (threshold: %.1f). "
                "Most frequent class: %s (%d samples). "
                "Least frequent class: %s (%d samples). "
                "Class distribution: %s. "
                "Consider oversampling, undersampling, or class-weighted loss.",
                self.monitor_batches,
                ratio,
                self.threshold,
                most_common[0],  most_common[1],
                least_common[0], least_common[1],
                dict(self._class_counts.most_common()),
            )
        else:
            logger.info(
                "ImbalanceMonitor: class ratio %.2f is within threshold %.2f "
                "after %d batches — Art.10 §2(f) check passed",
                ratio,
                self.threshold,
                self.monitor_batches,
            )


# ----------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------- #

def _extract_labels(batch: Any, label_index: int) -> torch.Tensor | None:
    """
    Extract the label tensor from a batch.

    Handles common patterns:
    - (inputs, labels) tuple
    - (inputs, labels, metadata) tuple
    - plain tensor (batch is all labels)
    """
    if isinstance(batch, (list, tuple)) and len(batch) > label_index:
        item = batch[label_index]
        if isinstance(item, torch.Tensor):
            return item
    elif isinstance(batch, torch.Tensor):
        return batch
    return None


def _compute_ratio(counts: Counter) -> float:
    """Return max / min class count. Returns 1.0 if fewer than 2 classes."""
    if len(counts) < 2:
        return 1.0
    values = list(counts.values())
    min_count = min(values)
    if min_count == 0:
        return float("inf")
    return max(values) / min_count


# ----------------------------------------------------------------------- #
# Demo
# ----------------------------------------------------------------------- #

def _demo() -> None:
    """Demonstrate ImbalanceMonitor with a synthetic imbalanced dataset."""
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    from torch.utils.data import TensorDataset

    print("\n── DatasetImbalance Demo ──────────────────────────────────────")
    print("Art.10 §2(f) compliance check: max/min class ratio monitor")
    print()

    # Create imbalanced dataset: 900 class-0 samples, 100 class-1 samples.
    n_samples = 1000
    labels = torch.cat([
        torch.zeros(900, dtype=torch.long),
        torch.ones(100,  dtype=torch.long),
    ])
    features = torch.randn(n_samples, 8)
    dataset  = TensorDataset(features, labels)
    loader   = DataLoader(dataset, batch_size=32, shuffle=True)

    monitor = ImbalanceMonitor(loader, threshold=3.0, monitor_batches=10)

    print("  Running through DataLoader with ImbalanceMonitor …")
    for i, (x, y) in enumerate(monitor):
        if i >= 15:
            break   # Process 15 batches for demo purposes.

    print(f"\n  Class counts: {monitor.class_counts}")
    print(f"  Imbalance ratio: {monitor.imbalance_ratio:.1f}x")
    print(f"  Is imbalanced: {monitor.is_imbalanced}")

    if monitor.is_imbalanced:
        print("\n  ✗ Art.10 §2(f) flag raised: dataset imbalance exceeds 3:1")
        print("    Recommended actions:")
        print("    - Use WeightedRandomSampler for oversampling minority class")
        print("    - Apply class-weighted cross-entropy loss")
        print("    - Collect more data for under-represented classes")
    else:
        print("\n  ✓ Art.10 §2(f) check passed: class distribution is balanced")


if __name__ == "__main__":
    _demo()
