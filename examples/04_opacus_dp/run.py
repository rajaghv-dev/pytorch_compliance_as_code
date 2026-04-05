"""
Example 4: Differential Privacy Training — GDPR Article 25 (Privacy by Design)
Trains a small CNN on CIFAR-10 with Opacus DP-SGD and compliance monitoring.
"""

import pathlib
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from torchcomply.integrations.opacus_bridge import CompliancePrivacyEngine, EpsilonBudgetExceeded


class _SmallCNN(nn.Module):
    """Small CNN without BatchNorm — compatible with Opacus out of the box."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            correct += (out.argmax(1) == y.to(device)).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def main():
    t0 = time.time()
    print("=" * 70)
    print("EXAMPLE 4: Differential Privacy — GDPR Article 25 (Privacy by Design)")
    print("Model: Small CNN | Dataset: CIFAR-10 (5000 samples) | Opacus DP-SGD")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device.upper()}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    tfm = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_full = torchvision.datasets.CIFAR10(
        root="/tmp/cifar10", train=True, download=True, transform=tfm
    )
    test_full = torchvision.datasets.CIFAR10(
        root="/tmp/cifar10", train=False, download=True, transform=tfm
    )

    train_ds = Subset(train_full, range(5000))
    test_ds = Subset(test_full, range(1000))

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    # ------------------------------------------------------------------
    # Model + Opacus
    # ------------------------------------------------------------------
    model = _SmallCNN()
    model = ModuleValidator.fix(model)  # replaces BatchNorm if any
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=3e-3)
    criterion = nn.CrossEntropyLoss()

    TARGET_EPSILON = 50.0
    TARGET_DELTA = 1e-5
    EPOCHS = 5

    print(f"\n⚠️  Note: ε={TARGET_EPSILON} does NOT constitute strong privacy.")
    print(f"   For GDPR Art.25 compliance, ε ≤ 8 is recommended (ε ≤ 1 is strong).")
    print(f"   This demo uses ε={TARGET_EPSILON} for convergence speed on a small dataset.")
    print(f"   Lower ε → more noise → lower accuracy → stronger privacy guarantee.\n")

    # ------------------------------------------------------------------
    # Non-DP baseline (same model, same data, no privacy noise)
    # ------------------------------------------------------------------
    print("Training non-DP baseline for honest accuracy comparison…")
    baseline_model = _SmallCNN()
    baseline_model = ModuleValidator.fix(baseline_model).to(device)
    baseline_opt = optim.Adam(baseline_model.parameters(), lr=3e-3)
    baseline_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    for _ in range(EPOCHS):
        baseline_model.train()
        for x, y in baseline_loader:
            baseline_opt.zero_grad()
            criterion(baseline_model(x.to(device)), y.to(device)).backward()
            baseline_opt.step()
    baseline_acc = _accuracy(baseline_model, test_loader, device)
    print(f"Non-DP baseline accuracy: {baseline_acc:.1%}  (trained identically, no noise)\n")

    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=EPOCHS,
        target_epsilon=TARGET_EPSILON,
        target_delta=TARGET_DELTA,
        max_grad_norm=1.0,
    )

    # ε enforcement: max_epsilon=8.0 mirrors the fairness gate — same mechanism,
    # same ComplianceViolation-class exception (EpsilonBudgetExceeded).
    # At TARGET_EPSILON=50 the budget far exceeds 8 after epoch 1; we catch the
    # exception, report it, then continue training with enforcement disabled so
    # the full training run completes for the rest of the demo.
    cpe = CompliancePrivacyEngine(
        privacy_engine,
        regulations=["gdpr_art_25", "gdpr_art_32"],
        max_epsilon=8.0,   # enforced hard cap — recommended for GDPR Art.25
    )
    print(f"\n⚙️  CompliancePrivacyEngine configured with max_epsilon={cpe.max_epsilon}")
    print(f"   check_epsilon() will raise EpsilonBudgetExceeded if ε > {cpe.max_epsilon}")
    print(f"   (same hard-stop semantics as FairnessGate.on_epoch_end)\n")

    print(f"Training for {EPOCHS} epochs with ε={TARGET_EPSILON}, δ={TARGET_DELTA}\n")
    print(f"{'Epoch':<6} | {'Loss':>8} | {'Acc':>7} | {'ε':>8} | δ")
    print("-" * 44)

    epoch_accs = []
    epoch_epsilons = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            cpe.step()
            total_loss += loss.item()

        acc = _accuracy(model, test_loader, device)
        try:
            eps = cpe.check_epsilon(delta=TARGET_DELTA)  # enforces max_epsilon
            budget_status = "within budget"
        except EpsilonBudgetExceeded as exc:
            eps = exc.epsilon
            budget_status = f"⚠️  EXCEEDS max={cpe.max_epsilon}"
            # In production: stop training here. For the demo we continue.
            if epoch == 1:
                print(f"\n  [EpsilonBudgetExceeded] ε={eps:.1f} > max_epsilon={cpe.max_epsilon}")
                print(f"  → In production: raise and stop training (same as FairnessGate)")
                print(f"  → Demo continues with enforcement disabled for remaining epochs\n")
                cpe.max_epsilon = None  # disable for remaining epochs

        epoch_accs.append(acc)
        epoch_epsilons.append(eps)

        print(
            f"  {epoch:<4}  | {total_loss/len(train_loader):>8.3f} | {acc:>7.1%} | {eps:>8.1f} | 1e-5  {budget_status}"
        )

    final_eps = epoch_epsilons[-1]
    final_acc = epoch_accs[-1]

    print(f"\n{'─'*54}")
    print(f"  Final epsilon         : {final_eps:.1f}  (⚠️  vacuous at this level)")
    print(f"  Final accuracy (DP)   : {final_acc:.1%}")
    print(f"  Baseline accuracy     : {baseline_acc:.1%}  (same model, no noise)")
    print(
        f"  Privacy-accuracy cost : {abs(baseline_acc - final_acc)*100:.1f}% accuracy gap for ε={final_eps:.1f}"
    )
    print(f"  max_grad_norm=1.0 means per-sample gradients clipped to L2-norm ≤ 1.0")
    print(f"  → individual contributions bounded → membership inference harder")

    # Per-class accuracy comparison: DP vs non-DP
    print(f"\nPer-class accuracy (DP vs non-DP baseline):")
    print(f"  {'Class':<6} | {'DP acc':>8} | {'Baseline':>9} | {'Drop':>6}")
    print(f"  {'─'*38}")
    classes_list = test_full.classes[:10]
    for cls_idx in range(10):
        # filter test samples for this class
        cls_mask = [test_full.targets[i] == cls_idx for i in range(len(test_ds))]
        cls_indices = [i for i, m in enumerate(cls_mask) if m]
        if not cls_indices:
            continue
        cls_ds = torch.utils.data.Subset(test_ds, cls_indices)
        cls_loader = DataLoader(cls_ds, batch_size=256, shuffle=False, num_workers=0)
        dp_cls_acc = _accuracy(model, cls_loader, device)
        base_cls_acc = _accuracy(baseline_model, cls_loader, device)
        drop = base_cls_acc - dp_cls_acc
        print(f"  {cls_idx:<6} | {dp_cls_acc:>8.1%} | {base_cls_acc:>9.1%} | {drop:>+6.1%}")

    # ------------------------------------------------------------------
    # Visualisation 1 — DP Budget Gauge
    # ------------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ratio = final_eps / TARGET_EPSILON
    bar_color = "green" if ratio < 0.70 else ("darkorange" if ratio < 0.90 else "crimson")

    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.barh(0, final_eps, color=bar_color, height=0.5, label=f"ε consumed = {final_eps:.1f}")
    ax.barh(
        0,
        TARGET_EPSILON - final_eps,
        left=final_eps,
        color="lightgrey",
        height=0.5,
        label=f"Remaining",
    )
    ax.set_xlim(0, TARGET_EPSILON * 1.05)
    ax.set_yticks([])
    ax.set_xlabel("Privacy budget (ε)")
    ax.axvline(TARGET_EPSILON, color="red", linestyle="--", linewidth=1)
    ax.text(
        final_eps / 2,
        0,
        f"ε = {final_eps:.1f} / {TARGET_EPSILON:.1f}",
        va="center",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="white",
    )
    ax.set_title(
        "Differential Privacy Budget — GDPR Article 25 (Privacy by Design)\n"
        f"Opacus DP-SGD | δ = {TARGET_DELTA}"
    )
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    out_dir = pathlib.Path(__file__).parent / "sample_output"
    out_dir.mkdir(exist_ok=True)
    out1 = out_dir / "dp_budget_gauge.png"
    plt.savefig(out1, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nVisualization 1 saved → {out1}")

    # ------------------------------------------------------------------
    # Visualisation 2 — Accuracy vs Epsilon tradeoff
    # ------------------------------------------------------------------
    epochs_range = list(range(1, EPOCHS + 1))
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(
        epochs_range,
        [a * 100 for a in epoch_accs],
        color="steelblue",
        marker="o",
        linewidth=2,
        label="Accuracy (%)",
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test Accuracy (%)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(
        epochs_range,
        epoch_epsilons,
        color="darkorange",
        marker="s",
        linewidth=2,
        label="ε consumed",
    )
    ax2.set_ylabel("Privacy budget ε consumed", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    ax1.axhline(baseline_acc * 100, color="steelblue", linestyle=":", linewidth=1.5,
                label=f"Non-DP baseline ({baseline_acc:.1%})", alpha=0.7)
    ax1.legend(loc="lower left", fontsize=8)
    ax1.set_title(
        "Privacy-Accuracy Tradeoff During DP-SGD Training\nGDPR Article 25 — Opacus DP-SGD\n"
        f"⚠️  ε={TARGET_EPSILON} is vacuous privacy — shown for convergence demo only"
    )
    fig.tight_layout()

    out2 = out_dir / "dp_accuracy_tradeoff.png"
    plt.savefig(out2, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Visualization 2 saved → {out2}")
    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
