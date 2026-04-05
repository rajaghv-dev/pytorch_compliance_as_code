"""
Example 5: Consent-Aware Data Loading — GDPR Articles 6, 7
Demonstrates ConsentRegistry, CompliantDataset, and class-imbalance profiling.
"""

import pathlib
import time

import torch
from torch.utils.data import DataLoader, Dataset

from torchcomply.core.dataset import (
    CompliantDataset,
    ConsentRegistry,
    ConsentViolation,
)


class _SyntheticDataset(Dataset):
    """500-sample dataset with 5 classes (intentional 10:1 imbalance)."""

    # class 0: 200, class 1: 150, class 2: 80, class 3: 50, class 4: 20
    _DIST = [200, 150, 80, 50, 20]

    def __init__(self):
        items = []
        for label, count in enumerate(self._DIST):
            for _ in range(count):
                items.append(items.__len__())  # will be replaced below
        # rebuild with proper data
        self._data = []
        idx = 0
        for label, count in enumerate(self._DIST):
            for _ in range(count):
                subject_id = f"subject_{idx+1:03d}"
                self._data.append((torch.randn(10), label, subject_id))
                idx += 1

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]


def _make_registry(n: int = 500, n_denied: int = 15, seed: int = 7) -> ConsentRegistry:
    """485 subjects consent, 15 opt out (randomly selected — realistic consent withdrawal pattern)."""
    import random
    rng = random.Random(seed)
    all_ids = [f"subject_{i+1:03d}" for i in range(n)]
    denied_ids = set(rng.sample(all_ids, n_denied))
    records = {}
    for i in range(n):
        sid = f"subject_{i+1:03d}"
        records[sid] = {
            "consent": sid not in denied_ids,
            # Purpose-scoped: some subjects consent to classification but not analytics
            "purposes": ["classification"] if sid not in denied_ids else [],
        }
    return ConsentRegistry(records)


def main():
    t0 = time.time()
    print("=" * 70)
    print("EXAMPLE 5: Consent-Aware Data Loading — GDPR Articles 6, 7")
    print("=" * 70)

    base_ds = _SyntheticDataset()
    registry = _make_registry(n=500, n_denied=15)
    ds = CompliantDataset(base_ds, registry, purpose="classification")

    # ------------------------------------------------------------------
    # Print dataset profile
    # ------------------------------------------------------------------
    profile = ds.profile
    print(f"\nDataset profile:")
    print(f"  Samples  : {profile.num_samples}")
    print(f"  Classes  : {dict(sorted(profile.class_distribution.items()))}")
    print(f"  Max class ratio: {profile.max_class_ratio:.1f}:1")
    for w in profile.warnings:
        print(f"  ⚠️  WARNING: {w}")

    # ------------------------------------------------------------------
    # Load through DataLoader — count granted / denied
    # ------------------------------------------------------------------
    print("\nLoading all samples (per-sample, catching ConsentViolation)…")
    granted = 0
    denied = 0
    denied_log = []

    for i in range(len(ds)):
        try:
            _ = ds[i]
            granted += 1
        except ConsentViolation as cv:
            denied += 1
            denied_log.append(cv)

    print(f"\n  Loaded: {granted} samples | Denied: {denied} samples")

    # ------------------------------------------------------------------
    # Denial log sample
    # ------------------------------------------------------------------
    print(f"\nDenial log (first 5):")
    for cv in denied_log[:5]:
        print(f"  {cv.subject_id} | purpose={cv.purpose} | DENIED (consent_withdrawn)")

    summary = registry.access_log_summary()
    print(f"\nAccess log: {summary['granted']} GRANTED, {summary['denied']} DENIED")

    # ------------------------------------------------------------------
    # Purpose scoping demo — same subjects, different purpose
    # ------------------------------------------------------------------
    print("\n" + "─" * 50)
    print("Purpose scoping demo (GDPR Art.7 — consent is purpose-specific):")
    print("  Same registry, but requesting purpose='analytics' instead of 'classification'")
    # Most subjects only consented to 'classification'; rebuild with partial analytics consent
    analytics_records = {}
    import random
    rng2 = random.Random(42)
    for sid, rec in registry._records.items():
        analytics_granted = rec["consent"] and rng2.random() < 0.6  # only 60% extend to analytics
        analytics_records[sid] = {
            "consent": rec["consent"],
            "purposes": rec["purposes"] + (["analytics"] if analytics_granted else []),
        }
    analytics_registry = ConsentRegistry(analytics_records)
    analytics_ds = CompliantDataset(base_ds, analytics_registry, purpose="analytics")
    an_granted, an_denied = 0, 0
    for i in range(len(analytics_ds)):
        try:
            _ = analytics_ds[i]
            an_granted += 1
        except ConsentViolation:
            an_denied += 1
    print(f"  purpose='classification': {granted} granted, {denied} denied")
    print(f"  purpose='analytics':      {an_granted} granted, {an_denied} denied")
    print(f"  → {an_denied - denied} extra denials when purpose scope changes")

    # ------------------------------------------------------------------
    # Visualisation 1 — Consent scatter
    # ------------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    denied_set = {cv.subject_id for cv in denied_log}

    # Group subjects into 10 buckets of 50 each for a readable bar chart
    N_GROUPS = 10
    subjects_per_group = 500 // N_GROUPS
    group_granted = [0] * N_GROUPS
    group_denied = [0] * N_GROUPS
    for i, (_, _, sid) in enumerate(base_ds):
        g = min(i // subjects_per_group, N_GROUPS - 1)
        if sid in denied_set:
            group_denied[g] += 1
        else:
            group_granted[g] += 1

    x = np.arange(N_GROUPS)
    width = 0.4
    group_labels = [f"S{i*subjects_per_group+1}–{(i+1)*subjects_per_group}" for i in range(N_GROUPS)]

    fig, ax = plt.subplots(figsize=(12, 4))
    bars_g = ax.bar(x - width / 2, group_granted, width, color="seagreen", alpha=0.85,
                    label=f"Granted ({sum(group_granted)})", edgecolor="white")
    bars_d = ax.bar(x + width / 2, group_denied, width, color="crimson", alpha=0.85,
                    label=f"Denied ({sum(group_denied)})", edgecolor="white")
    for bar, v in zip(bars_d, group_denied):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(v), ha="center", fontsize=8, color="crimson", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=30, ha="right", fontsize=8.5)
    ax.set_xlabel("Subject group (by sample index)")
    ax.set_ylabel("Count")
    ax.set_title(
        "Consent Access Log — GDPR Article 6\n"
        f"{sum(group_granted)} granted, {sum(group_denied)} denied | Denials distributed randomly across groups"
    )
    ax.legend(fontsize=9)
    plt.tight_layout()

    out_dir = pathlib.Path(__file__).parent / "sample_output"
    out_dir.mkdir(exist_ok=True)
    out1 = out_dir / "consent_scatter.png"
    plt.savefig(out1, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nVisualization 1 saved → {out1}")

    # ------------------------------------------------------------------
    # Visualisation 2 — Class distribution bar chart
    # ------------------------------------------------------------------
    dist = dict(sorted(profile.class_distribution.items()))
    classes = list(dist.keys())
    counts = list(dist.values())
    balanced_level = sum(counts) / len(counts)

    colors = ["steelblue" if c >= balanced_level * 0.8 else "darkorange" for c in counts]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(classes, counts, color=colors, edgecolor="white")
    ax.axhline(
        balanced_level,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Balanced level ({balanced_level:.0f})",
    )
    ax.annotate(
        f"{profile.max_class_ratio:.0f}:1 imbalance detected",
        xy=(classes[-1], counts[-1]),
        xytext=(classes[len(classes) // 2], max(counts) * 0.8),
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red",
        fontsize=10,
    )
    ax.set_xticks(classes)
    ax.set_xlabel("Class")
    ax.set_ylabel("Sample count")
    ax.set_title("Dataset Class Distribution — EU AI Act Art. 10 (Data Governance)")
    ax.legend()
    plt.tight_layout()

    out2 = out_dir / "class_distribution.png"
    plt.savefig(out2, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Visualization 2 saved → {out2}")
    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
