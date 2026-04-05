"""
Example 9: Deployment Monitoring — EU AI Act Articles 9, 14, 15
Post-deployment inference monitoring with bias drift detection.
"""

import pathlib
import time
from collections import Counter

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import torchvision.datasets as dsets


def _entropy(probs: torch.Tensor) -> float:
    """Shannon entropy of a probability distribution."""
    p = probs.clamp(min=1e-9)
    return float(-(p * p.log()).sum().item())


def _alert_handler(batch_id: int, metric_name: str, value: float, threshold: float) -> None:
    """
    Stub alert handler — represents the integration point for real alerting.
    In production: call PagerDuty, Slack webhook, MLflow alert, SIEM log, etc.
    """
    print(f"  [ALERT] batch={batch_id} metric={metric_name} value={value:.3f} > threshold={threshold}")
    print(f"  [ALERT] → In production: stop serving / page oncall / switch fallback model")


def main():
    t0 = time.time()
    print("=" * 70)
    print("EXAMPLE 9: Deployment Monitoring — EU AI Act Articles 9, 14, 15")
    print("Bias drift detection over 1000 simulated inference requests")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device.upper()}")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval()

    # Load CIFAR-10 test set for class-based concept drift simulation
    # Pre-drift: batches from vehicle/transport classes (0=airplane,1=auto,8=ship,9=truck)
    # Post-drift: batches from animal classes (2=bird,3=cat,4=deer,5=dog,6=frog,7=horse)
    # This is a real concept drift — the input distribution fundamentally changes domain.
    print("\nLoading CIFAR-10 for class-based concept drift simulation…")
    tfm = T.Compose([
        T.Resize(224), T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    try:
        cifar_test = dsets.CIFAR10(root="/tmp/cifar10", train=False, download=True, transform=tfm)
        # CIFAR classes: airplane=0, auto=1, bird=2, cat=3, deer=4, dog=5, frog=6, horse=7, ship=8, truck=9
        vehicle_classes = {0, 1, 8, 9}   # transport domain (pre-drift)
        animal_classes  = {2, 3, 4, 5, 6, 7}  # living things domain (post-drift)
        vehicle_imgs = [cifar_test[i][0] for i in range(len(cifar_test)) if cifar_test.targets[i] in vehicle_classes][:700]
        animal_imgs  = [cifar_test[i][0] for i in range(len(cifar_test)) if cifar_test.targets[i] in animal_classes][:300]
        all_imgs = vehicle_imgs + animal_imgs
        use_cifar = True
        print(f"  Pre-drift pool:  {len(vehicle_imgs)} vehicle images (classes 0,1,8,9)")
        print(f"  Post-drift pool: {len(animal_imgs)} animal images (classes 2-7)")
    except Exception as e:
        print(f"  CIFAR-10 unavailable ({e}) — falling back to synthetic inputs")
        use_cifar = False

    BATCH_SIZE = 50
    TOTAL = 1000
    N_BATCHES = TOTAL // BATCH_SIZE
    DRIFT_AT = 700  # requests 700+ use animal classes (concept drift)
    HUMAN_CONF_THRESHOLD = 0.5

    # Entropy-based drift detection: ResNet-18 (ImageNet) is familiar with CIFAR vehicle
    # classes (airplanes, cars, ships → ImageNet vehicle categories → confident, low entropy).
    # When the domain shifts to CIFAR animals (broad categories vs. ImageNet's 120 dog breeds),
    # the model becomes uncertain across many classes → entropy rises.
    # We detect drift when per-batch average entropy exceeds the baseline range.
    DRIFT_THRESHOLD_ENTROPY = 4.7  # baseline (vehicles): 3.6–4.5; post-drift (animals): 4.9–5.2
    DRIFT_THRESHOLD_TOP3 = 0.40  # kept for visualization reference

    print(f"\nSimulating {TOTAL} requests in {N_BATCHES} batches of {BATCH_SIZE}…")
    print(f"Drift at request {DRIFT_AT}: domain shifts from vehicles → animals\n")
    print(f"{'Batch':<7} | {'Entropy':>8} | {'Top-3 conc':>11} | {'HumanReview':>12} | Status")
    print("─" * 62)

    batch_entropies = []
    batch_top3_rates = []
    batch_human_counts = []
    drift_detected_batch = None
    all_statuses = []
    img_ptr_veh, img_ptr_ani = 0, 0

    for b in range(N_BATCHES):
        start_req = b * BATCH_SIZE
        in_drift = start_req >= DRIFT_AT

        if use_cifar:
            if not in_drift:
                batch_imgs = vehicle_imgs[img_ptr_veh:img_ptr_veh + BATCH_SIZE]
                img_ptr_veh = (img_ptr_veh + BATCH_SIZE) % len(vehicle_imgs)
            else:
                batch_imgs = animal_imgs[img_ptr_ani:img_ptr_ani + BATCH_SIZE]
                img_ptr_ani = (img_ptr_ani + BATCH_SIZE) % len(animal_imgs)
            x = torch.stack(batch_imgs).to(device)
        else:
            x = torch.randn(BATCH_SIZE, 3, 224, 224, device=device)
            if in_drift:
                x[:, 0] += 3.0

        with torch.no_grad():
            logits = model(x)
        probs = torch.softmax(logits, dim=-1)

        avg_probs = probs.mean(dim=0)
        entropy = _entropy(avg_probs)
        top3_rate = float(avg_probs.topk(3).values.sum().item())
        # Human routing: low model confidence → flag for review
        human_count = int((probs.max(dim=-1).values < HUMAN_CONF_THRESHOLD).sum().item())

        batch_entropies.append(entropy)
        batch_top3_rates.append(top3_rate)
        batch_human_counts.append(human_count)

        # Drift = entropy rises above baseline (model uncertain on new domain)
        is_drift = entropy > DRIFT_THRESHOLD_ENTROPY
        if is_drift and drift_detected_batch is None:
            drift_detected_batch = b + 1

        status = "⚠️  DRIFT" if is_drift else "✅ OK"
        all_statuses.append(status)
        print(f"  {b+1:<5} | {entropy:>8.3f} | {top3_rate:>11.3f} | {human_count:>12} | {status}")

        if is_drift and drift_detected_batch == b + 1:
            print(
                f"     ⚠️  ENTROPY DRIFT at batch {b+1} (req {start_req+1}+) — entropy {entropy:.3f} > {DRIFT_THRESHOLD_ENTROPY}"
            )
            print(f"        Domain shift: vehicles → animals | model becomes uncertain")
            print(f"        Action: routing next {BATCH_SIZE} requests to human review")
            _alert_handler(b + 1, "entropy", entropy, DRIFT_THRESHOLD_ENTROPY)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    drift_batches = sum(1 for s in all_statuses if "DRIFT" in s)
    total_human = sum(batch_human_counts)
    avg_entropy_before = sum(batch_entropies[:14]) / 14
    avg_entropy_after = sum(batch_entropies[14:]) / max(len(batch_entropies) - 14, 1)
    # Entropy RISES after drift — model becomes more uncertain on unfamiliar domain
    entropy_change = avg_entropy_after - avg_entropy_before
    entropy_change_pct = entropy_change / max(avg_entropy_before, 1e-9)
    entropy_direction = "↑ increase" if entropy_change > 0 else "↓ decrease"

    print(f"\n{'─'*62}")
    print(f"Deployment monitoring summary:")
    print(f"  Total requests:              {TOTAL}")
    print(f"  Batches monitored:           {N_BATCHES}")
    if drift_detected_batch:
        print(f"  Drift detected:              batch {drift_detected_batch} (request {DRIFT_AT}+)")
    else:
        print(f"  No drift detected above threshold={DRIFT_THRESHOLD_ENTROPY}")
    print(f"  Human reviews routed:        {total_human} (confidence < {HUMAN_CONF_THRESHOLD})")
    print(f"  Entropy change after drift:  {entropy_change_pct:+.1%} ({entropy_direction})")
    print(f"  Note: entropy measures prediction spread — not accuracy.")
    print(f"  Accuracy requires ground-truth labels; this is a confidence distribution metric.")

    # ------------------------------------------------------------------
    # Visualisation 1 — bias drift over time
    # ------------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir = pathlib.Path(__file__).parent / "sample_output"
    out_dir.mkdir(exist_ok=True)

    batches = list(range(1, N_BATCHES + 1))
    colors = ["crimson" if "DRIFT" in s else "steelblue" for s in all_statuses]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(batches, batch_entropies, color=colors, alpha=0.85)
    ax.axhline(
        DRIFT_THRESHOLD_ENTROPY,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Entropy threshold ({DRIFT_THRESHOLD_ENTROPY})",
    )
    if drift_detected_batch:
        ax.axvspan(drift_detected_batch - 0.5, N_BATCHES + 0.5, color="red", alpha=0.07)
        ax.annotate(
            "DRIFT DETECTED",
            xy=(drift_detected_batch, DRIFT_THRESHOLD_ENTROPY + 0.05),
            fontsize=10,
            color="red",
            fontweight="bold",
        )
    ax.set_xlabel("Batch number")
    ax.set_ylabel("Shannon entropy (avg per batch)")
    drift_label = f"batch {drift_detected_batch} (req {DRIFT_AT}+)" if drift_detected_batch else "not detected"
    ax.set_title(
        "Entropy-Based Drift Detection Over Deployment — EU AI Act Articles 9, 14, 15\n"
        f"Concept drift: vehicles→animals domain at request {DRIFT_AT} | Detected: {drift_label}\n"
        f"Entropy threshold: {DRIFT_THRESHOLD_ENTROPY} | Pre-drift: ~{avg_entropy_before:.1f} | Post-drift: ~{avg_entropy_after:.1f}"
    )
    ax.legend()
    plt.tight_layout()
    out1 = out_dir / "bias_drift.png"
    plt.savefig(out1, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nVisualization 1 saved → {out1}")

    # Visualisation 2 — human intervention rate
    drift_split = (drift_detected_batch or N_BATCHES) - 1
    avg_pre = sum(batch_human_counts[:drift_split]) / max(drift_split, 1)
    avg_post = sum(batch_human_counts[drift_split:]) / max(len(batch_human_counts) - drift_split, 1)
    pct_increase = (avg_post - avg_pre) / max(avg_pre, 1e-9) * 100

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(batches, batch_human_counts, color="darkorange", linewidth=2, marker="o", markersize=4)
    ax.axvline(
        drift_detected_batch or N_BATCHES,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Drift start",
    )
    ax.fill_between(batches, batch_human_counts, alpha=0.15, color="darkorange")
    ax.text(
        0.02, 0.93,
        f"Pre-drift avg: {avg_pre:.0f}/batch → Post-drift: {avg_post:.0f}/batch ({pct_increase:+.0f}%)",
        transform=ax.transAxes, fontsize=9, va="top", color="darkred",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.set_xlabel("Batch number")
    ax.set_ylabel("Human review requests per batch")
    ax.set_title("Human Intervention Rate — EU AI Act Article 14 (Human Oversight)")
    ax.legend()
    plt.tight_layout()
    out2 = out_dir / "human_interventions.png"
    plt.savefig(out2, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Visualization 2 saved → {out2}")
    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
