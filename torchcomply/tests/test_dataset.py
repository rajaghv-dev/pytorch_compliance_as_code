"""Tests for CompliantDataset and ConsentRegistry — GDPR Articles 6 and 7."""

import pytest
import torch
from torch.utils.data import Dataset

from torchcomply.core.dataset import (
    CompliantDataset,
    ConsentRegistry,
    ConsentViolation,
)


class _Synthetic(Dataset):
    """Simple dataset: (feature, label, subject_id)."""

    def __init__(self, n: int = 20):
        self.data = [(torch.randn(5), i % 3, f"sub_{i:03d}") for i in range(n)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _registry(n: int = 20, denied_ids=("sub_005",)):
    records = {}
    for i in range(n):
        sid = f"sub_{i:03d}"
        records[sid] = {
            "consent": sid not in denied_ids,
            "purposes": ["training"],
        }
    return ConsentRegistry(records)


def test_consent_granted():
    reg = _registry()
    ds = CompliantDataset(_Synthetic(10), reg, "training")
    item = ds[0]  # sub_000 has consent
    assert item is not None


def test_consent_denied():
    reg = _registry(denied_ids=("sub_003",))
    ds = CompliantDataset(_Synthetic(10), reg, "training")
    with pytest.raises(ConsentViolation) as exc_info:
        _ = ds[3]
    assert exc_info.value.subject_id == "sub_003"


def test_access_log_summary():
    reg = _registry(denied_ids=("sub_001", "sub_002"))
    ds = CompliantDataset(_Synthetic(10), reg, "training")
    for i in (0, 1, 2, 3):
        try:
            _ = ds[i]
        except ConsentViolation:
            pass
    summary = reg.access_log_summary()
    assert summary["granted"] == 2
    assert summary["denied"] == 2


def test_dataset_profile_imbalance():
    """Force 10:1 class imbalance and verify warning is set."""

    class Imbalanced(Dataset):
        def __len__(self):
            return 110

        def __getitem__(self, idx):
            label = 0 if idx < 100 else 1
            return torch.randn(5), label, f"s{idx}"

    reg = ConsentRegistry({f"s{i}": {"consent": True, "purposes": ["t"]} for i in range(110)})
    ds = CompliantDataset(Imbalanced(), reg, "t")
    assert ds.profile.max_class_ratio > 5.0
    assert len(ds.profile.warnings) > 0


def test_len():
    reg = _registry()
    ds = CompliantDataset(_Synthetic(15), reg, "training")
    assert len(ds) == 15
