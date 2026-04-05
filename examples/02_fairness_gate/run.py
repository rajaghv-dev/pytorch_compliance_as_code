"""
Example 2: Fairness Gate — EU AI Act Article 10
Shows a FairnessGate blocking biased training on synthetic data.
"""

import pathlib
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchcomply.core.fairness import ComplianceViolation, FairnessGate, compute_demographic_parity


def compute_equalized_odds(predictions, labels, protected_attr):
    """
    Equalized odds: equal TPR *and* FPR across protected groups.
    Returns max disparity across TPR and FPR separately.
    A stricter metric than demographic parity — required when
    group base rates differ (which they do in this dataset).
    """
    groups = protected_attr.unique()
    tpr_list, fpr_list = [], []
    for g in groups:
        mask = protected_attr == g
        y_true = labels[mask].float()
        y_pred = predictions[mask].float()
        tp = ((y_pred == 1) & (y_true == 1)).float().sum()
        fn = ((y_pred == 0) & (y_true == 1)).float().sum()
        fp = ((y_pred == 1) & (y_true == 0)).float().sum()
        tn = ((y_pred == 0) & (y_true == 0)).float().sum()
        tpr = float(tp / (tp + fn + 1e-9))
        fpr = float(fp / (fp + tn + 1e-9))
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    tpr_gap = max(tpr_list) - min(tpr_list)
    fpr_gap = max(fpr_list) - min(fpr_list)
    return tpr_gap, fpr_gap, tpr_list, fpr_list


def make_biased_dataset(seed: int = 42):
    """
    Group 0: 300 samples, 50/50 label split.
    Group 1: 300 samples, 80/20 label split (intentionally biased).
    """
    torch.manual_seed(seed)
    # Group 0 — balanced
    x0 = torch.randn(300, 10)
    y0 = (torch.rand(300) < 0.5).long()
    x0[y0 == 1] += 0.5
    g0 = torch.zeros(300, dtype=torch.long)

    # Group 1 — skewed 80/20 (high bias to demonstrate the gate blocking consistently)
    x1 = torch.randn(300, 10)
    y1 = (torch.rand(300) < 0.80).long()
    x1[y1 == 1] += 0.5
    g1 = torch.ones(300, dtype=torch.long)

    X = torch.cat([x0, x1])
    Y = torch.cat([y0, y1])
    G = torch.cat([g0, g1])
    return TensorDataset(X, Y, G)


def main():
    t0 = time.time()
    print("=" * 70)
    print("EXAMPLE 2: Fairness Gate — EU AI Act Article 10 (Data Governance)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    dataset = make_biased_dataset()
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=256, shuffle=False)

    model = nn.Sequential(
        nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()
    # Threshold of 0.10 (10%) follows common operational practice in hiring/lending
    # fairness literature (e.g., US EEOC 4/5ths rule, EU non-discrimination guidelines).
    # EU AI Act Art.10 does not specify a numeric threshold — this is organisation-configurable.
    gate = FairnessGate(threshold=0.10)

    losses = []
    parities = []
    eq_odds_tpr = []
    eq_odds_fpr = []
    statuses = []
    blocked_epoch = None

    print(f"\n{'Epoch':<6} | {'Loss':>8} | {'Parity':>8} | {'EqOdds TPR':>11} | {'EqOdds FPR':>11} | Status")
    print("-" * 65)

    for epoch in range(8):
        model.train()
        total_loss = 0.0
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        # Compute parity manually for display (gate does it internally too)
        model.eval()
        all_p, all_g = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                out = model(x)
                all_p.append(out.argmax(dim=-1).cpu())
                all_g.append(batch[2])
        preds = torch.cat(all_p)
        groups = torch.cat(all_g)
        # also collect ground-truth labels for equalized odds
        all_y = []
        with torch.no_grad():
            for batch in val_loader:
                all_y.append(batch[1])
        labels = torch.cat(all_y)

        parity = compute_demographic_parity(preds, groups)
        tpr_gap, fpr_gap, _, _ = compute_equalized_odds(preds, labels, groups)

        losses.append(avg_loss)
        parities.append(parity)
        eq_odds_tpr.append(tpr_gap)
        eq_odds_fpr.append(fpr_gap)

        try:
            gate.on_epoch_end(model, val_loader, epoch, protected_attr_idx=2)
            status = "✅ PASSED"
            statuses.append("passed")
        except ComplianceViolation as cv:
            status = "❌ BLOCKED"
            statuses.append("blocked")
            if blocked_epoch is None:
                blocked_epoch = epoch
                print(f"  {epoch+1:<4}  | {avg_loss:>8.3f} | {parity:>8.3f} | {tpr_gap:>11.3f} | {fpr_gap:>11.3f} | {status}")
                print(f"     ComplianceViolation: parity {cv.disparity:.3f} > threshold {cv.threshold}")
                print(f"     Equalized odds gap — TPR: {tpr_gap:.3f}, FPR: {fpr_gap:.3f}")
                print(f"     (Training continues for visualization — in production the pipeline would halt here)")
                print(f"  {' ':4}  {'  (continuing remaining epochs to show trajectory)':}")
            else:
                print(f"  {epoch+1:<4}  | {avg_loss:>8.3f} | {parity:>8.3f} | {tpr_gap:>11.3f} | {fpr_gap:>11.3f} | {status}")
            continue

        print(f"  {epoch+1:<4}  | {avg_loss:>8.3f} | {parity:>8.3f} | {tpr_gap:>11.3f} | {fpr_gap:>11.3f} | {status}")

    # ------------------------------------------------------------------
    # Fairness log table
    # ------------------------------------------------------------------
    print("\nFairness log:")
    print(f"  {'Epoch':<6} | {'Disparity':>10} | {'Threshold':>10} | Status")
    print("  " + "-" * 44)
    for entry in gate.get_log():
        print(
            f"  {entry['epoch']+1:<6} | {entry['disparity']:>10.4f} | "
            f"{entry['threshold']:>10.4f} | {entry['status'].upper()}"
        )

    # ------------------------------------------------------------------
    # Visualisation — fairness trajectory
    # ------------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    epochs_range = list(range(1, len(losses) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: demographic parity (the gate metric)
    ax1 = axes[0]
    ax1b = ax1.twinx()
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss", color="steelblue")
    ax1.plot(epochs_range, losses, color="steelblue", linewidth=2, marker="o", label="Loss")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1b.set_ylabel("Demographic Parity Disparity", color="darkorange")
    ax1b.plot(epochs_range, parities, color="darkorange", linewidth=2, marker="s", label="DP Parity")
    ax1b.axhline(y=gate.threshold, color="red", linestyle="--", linewidth=1.5,
                 label=f"Gate threshold ({gate.threshold})")
    ax1b.tick_params(axis="y", labelcolor="darkorange")
    for i, (e, s) in enumerate(zip(epochs_range, statuses)):
        if s == "passed":
            ax1b.annotate("✅", xy=(e, parities[i]), fontsize=12, ha="center", va="bottom")
        else:
            ax1b.axvspan(e - 0.5, e + 0.5, color="red", alpha=0.15)
            ax1b.annotate("❌", xy=(e, parities[i]), fontsize=12, ha="center", va="bottom", color="red")
    ax1.set_title("Demographic Parity Gate (gate metric)\nBlocks training if parity > 0.10")
    ax1b.legend(loc="upper left", fontsize=8)

    # Right plot: equalized odds (stricter metric, shown for context)
    ax2 = axes[1]
    ax2.plot(epochs_range, eq_odds_tpr, color="purple", linewidth=2, marker="^", label="TPR gap")
    ax2.plot(epochs_range, eq_odds_fpr, color="teal", linewidth=2, marker="v", label="FPR gap")
    ax2.axhline(y=gate.threshold, color="red", linestyle="--", linewidth=1.5,
                label=f"Parity threshold ({gate.threshold})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Equalized Odds Gap")
    ax2.set_title("Equalized Odds (stricter metric)\nTPR gap + FPR gap across groups")
    ax2.legend(fontsize=9)
    ax2.text(0.02, 0.97,
             "Equalized odds requires equal TPR *and* FPR\nacross groups — stricter than demographic parity.\n"
             "EU AI Act does not mandate a specific metric;\norganisations should choose based on deployment context.",
             transform=ax2.transAxes, fontsize=7.5, va="top", color="grey",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    fig.suptitle(
        "Training with Compliance Fairness Gate — EU AI Act Article 10\n"
        f"Threshold: {gate.threshold} | {'Blocked at epoch ' + str(blocked_epoch+1) if blocked_epoch is not None else 'All epochs passed'}",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()

    out_dir = pathlib.Path(__file__).parent / "sample_output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "fairness_trajectory.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nVisualization saved → {out_path}")
    print(f"Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
