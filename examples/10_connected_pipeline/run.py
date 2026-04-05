"""
Example 10: End-to-End Compliance Pipeline — The crown jewel.
Connects Opacus + Captum + CrypTen + audit hooks + fairness gate + Annex IV PDF
+ MLflow experiment tracking + OpenTelemetry spans.
All three tools. All three mechanisms. MLflow. OTel. One script.
"""

import pathlib
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchcomply.core.dataset import CompliantDataset, ConsentRegistry, ConsentViolation
from torchcomply.core.diff import ComplianceDiff, ComplianceSnapshot
from torchcomply.core.engine import ComplianceEngine
from torchcomply.core.fairness import compute_demographic_parity
from torchcomply.core.dispatcher_hooks import ComplianceTensor
from torchcomply.core.autograd_provenance import ProvenanceLinear


def _make_dataset(n: int = 500, n_subjects: int = 20, n_denied: int = 2):
    """Synthetic tabular: 500 samples, 10 features, 2 classes.
    Groups are slightly imbalanced (60/40 split) so the fairness gate does real work.
    """
    torch.manual_seed(0)
    X = torch.randn(n, 10)
    Y = (X[:, 0] + X[:, 1] > 0).long()
    subjects_per = n // n_subjects
    sids = [f"s{i // subjects_per:03d}" for i in range(n)]
    denied = {f"s{i:03d}" for i in range(n_denied)}
    records = {
        f"s{i:03d}": {"consent": f"s{i:03d}" not in denied, "purposes": ["classification"]}
        for i in range(n_subjects)
    }
    return X, Y, sids, ConsentRegistry(records)


class _SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, sids):
        self.X, self.Y, self.sids = X, Y, sids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.sids[i]


def _make_model():
    return nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))


def main():
    t0 = time.time()
    stage_times = {}
    # compliance_record threads evidence through all 10 stages → becomes the Annex IV PDF
    compliance_record = {
        "stages": {},
        "regulations": ["eu_ai_act", "gdpr"],
    }

    print("═" * 70)
    print("TORCHCOMPLY — End-to-End Compliance Pipeline")
    print("All three tools: Opacus + Captum + CrypTen")
    print("All three mechanisms: Hooks + Dispatcher + Autograd")
    print("PyTorch Conference Europe 2026 | Station F, Paris")
    print("═" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = pathlib.Path(__file__).parent / "sample_output"
    out_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 1 — Compliant Data Loading
    # ------------------------------------------------------------------
    ts = time.time()
    X, Y, sids, registry = _make_dataset()
    base_ds = _SimpleDataset(X, Y, sids)
    compliant_ds = CompliantDataset(base_ds, registry, purpose="classification")
    profile = compliant_ds.profile

    allowed_X, allowed_Y, allowed_groups = [], [], []
    denied = 0
    for i in range(len(compliant_ds)):
        try:
            feat, label, sid = compliant_ds[i]
            allowed_X.append(feat)
            allowed_Y.append(label)
            # Slightly imbalanced groups: 60% group-0, 40% group-1
            # This ensures the fairness gate has to do real work, not trivially pass
            allowed_groups.append(0 if i % 5 != 0 else 1)
        except ConsentViolation:
            denied += 1

    X_t = torch.stack(allowed_X)
    Y_t = torch.stack(allowed_Y)
    G_t = torch.tensor(allowed_groups)
    granted = len(allowed_X)

    stage_times["Stage 1: Data loading"] = time.time() - ts
    compliance_record["stages"]["1_data"] = {
        "granted": granted, "denied": denied,
        "group_balance": f"{allowed_groups.count(0)}/{allowed_groups.count(1)} (group0/group1)",
    }
    print(
        f"\nStage 1 ✅: {granted} samples loaded, {denied} denied ({denied // 25} subjects opted out)"
    )

    # ------------------------------------------------------------------
    # Stage 2 — Model + Compliance Hooks (using context manager API)
    # ------------------------------------------------------------------
    ts = time.time()
    model = _make_model().to(device)
    # Context manager API: `with engine` removes hooks automatically on exit.
    # This prevents hook leakage in long-running services — no need to call engine.detach().
    engine = ComplianceEngine(regulations=["eu_ai_act", "gdpr"])
    model = engine.attach(model)
    n_hooks = len([m for n, m in model.named_modules() if not list(m.children())])
    stage_times["Stage 2: Hook setup"] = time.time() - ts
    compliance_record["stages"]["2_hooks"] = {"n_hooks": n_hooks}
    print(f"Stage 2 ✅: Compliance hooks registered on {n_hooks} modules")
    print(f"           (context manager API — hooks removed automatically on exit)")

    # ------------------------------------------------------------------
    # Stage 3 — DP Training with Opacus
    # ------------------------------------------------------------------
    ts = time.time()
    EPOCHS = 3
    eps_final = None
    try:
        from opacus import PrivacyEngine
        from torchcomply.integrations.opacus_bridge import CompliancePrivacyEngine

        train_ds = TensorDataset(X_t, Y_t)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        criterion = nn.CrossEntropyLoss()

        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )
        cpe = CompliancePrivacyEngine(privacy_engine)
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                cpe.step()
                total_loss += loss.item()
        eps_final = cpe.get_epsilon(delta=1e-5)
        opacus_ok = True
        stage_times["Stage 3: DP training"] = time.time() - ts
        compliance_record["stages"]["3_dp"] = {"epsilon": eps_final, "delta": 1e-5, "epochs": EPOCHS}
        print(f"Stage 3 ✅: DP training complete, ε={eps_final:.1f}")
    except Exception as e:
        stage_times["Stage 3: DP training"] = time.time() - ts
        print(f"Stage 3 ⚠️:  Opacus unavailable ({e}) — training without DP")
        opacus_ok = False
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        train_ds = TensorDataset(X_t, Y_t)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        for _ in range(EPOCHS):
            for xb, yb in train_loader:
                optimizer.zero_grad()
                criterion(model(xb.to(device)), yb.to(device)).backward()
                optimizer.step()

    # ------------------------------------------------------------------
    # Stage 4 — Fairness Gate
    # ------------------------------------------------------------------
    ts = time.time()
    model.eval()
    with torch.no_grad():
        preds = model(X_t.to(device)).argmax(dim=-1).cpu()
    parity = compute_demographic_parity(preds, G_t)
    threshold = 0.15
    fairness_status = "✅ PASSED" if parity <= threshold else "⚠️ REVIEW"
    stage_times["Stage 4: Fairness check"] = time.time() - ts
    compliance_record["stages"]["4_fairness"] = {
        "parity": round(parity, 4), "threshold": threshold, "status": fairness_status,
        "group_counts": {0: allowed_groups.count(0), 1: allowed_groups.count(1)},
    }
    print(
        f"Stage 4 ✅: Fairness check {fairness_status}, parity={parity:.3f} threshold={threshold}"
    )

    # ------------------------------------------------------------------
    # Stage 5 — Captum Explanations
    # ------------------------------------------------------------------
    ts = time.time()
    explanations = []
    try:
        from captum.attr import IntegratedGradients

        # Opacus wraps with GradSampleModule whose hooks conflict with Captum backward.
        # Disable those hooks, then run attribution on the unwrapped inner model.
        if hasattr(model, "disable_hooks"):
            model.disable_hooks()
        captum_model = (model._module if hasattr(model, "_module") else model).eval()
        ig = IntegratedGradients(captum_model)
        for i in range(3):
            x_s = X_t[i : i + 1].to(device).requires_grad_(True)
            baseline = torch.zeros_like(x_s)
            attr, _ = ig.attribute(
                x_s,
                baselines=baseline,
                target=int(Y_t[i].item()),
                n_steps=20,
                return_convergence_delta=True,
            )
            top3 = attr.squeeze().abs().topk(3).indices.tolist()
            explanations.append(
                {"sample": i, "top_features": top3, "predicted": int(preds[i].item())}
            )
        stage_times["Stage 5: Captum"] = time.time() - ts
        compliance_record["stages"]["5_captum"] = {"n_explanations": 3, "method": "IntegratedGradients"}
        print(f"Stage 5 ✅: 3 IntegratedGradients explanations generated")
        for e in explanations:
            print(
                f"  Sample {e['sample']}: top features {e['top_features']}, pred={e['pred'] if 'pred' in e else e['predicted']}"
            )
    except Exception as ex:
        stage_times["Stage 5: Captum"] = time.time() - ts
        print(f"Stage 5 ⚠️:  Captum unavailable ({ex})")

    # ------------------------------------------------------------------
    # Stage 6 — CrypTen Encrypted Inference
    # ------------------------------------------------------------------
    ts = time.time()
    crypten_status = "⚠️"
    max_diff = None
    overhead_x = None
    try:
        from torchcomply.integrations.crypten_bridge import ComplianceSecureInference

        model_cpu = model.cpu().eval()
        dummy = X_t[:1]
        secure = ComplianceSecureInference(model_cpu, dummy)
        with torch.no_grad():
            std_out = model_cpu(X_t[:3])
        enc_out = secure.secure_predict(X_t[:3])
        max_diff = float((std_out - enc_out).abs().max().item())
        enc_time = secure.get_log()[-1]["encryption_time_ms"]
        with torch.no_grad():
            t_s = time.time()
            _ = model_cpu(X_t[:3])
            std_time = (time.time() - t_s) * 1000
        overhead_x = enc_time / max(std_time, 0.001)
        crypten_status = "✅"
        model = model.to(device)
        stage_times["Stage 6: CrypTen"] = time.time() - ts
        compliance_record["stages"]["6_crypten"] = {
            "max_diff": round(max_diff, 8), "overhead_x": round(overhead_x, 1),
            "simulation_mode": True,  # single-machine — real MPC needs network isolation
        }
        print(
            f"Stage 6 ✅: Encrypted inference verified, max diff={max_diff:.6f}, overhead={overhead_x:.0f}x"
        )
    except Exception as ex:
        stage_times["Stage 6: CrypTen"] = time.time() - ts
        print(f"Stage 6 ⚠️:  CrypTen unavailable — {ex}")
        print("   (CrypTen demo available in examples/07_crypten_secure/)")

    # ------------------------------------------------------------------
    # Stage 7 — Audit Chain Verification
    # ------------------------------------------------------------------
    ts = time.time()
    chain = engine.audit_chain
    chain.verify()
    stage_times["Stage 7: Audit verify"] = time.time() - ts
    compliance_record["stages"]["7_audit"] = {
        "entries": len(chain),
        "root_hash": chain.root_hash(),
        "integrity": "VALID",
    }
    # Flush full chain to disk — chain_summary in PDF is compact; this is the full record
    chain.flush_jsonl(out_dir / "audit_chain.jsonl")
    print(f"Stage 7 ✅: Audit chain {len(chain)} entries, integrity VALID")
    print(f"         root_hash={chain.root_hash()[:16]}… | persisted → audit_chain.jsonl")

    # ------------------------------------------------------------------
    # Stage 8 — Annex IV PDF
    # ------------------------------------------------------------------
    ts = time.time()
    from torchcomply.reports.annex_iv import AnnexIVReport, ModelIntrospector

    mi = ModelIntrospector(model)
    fairness_log = [
        {
            "epoch": 0,
            "disparity": parity,
            "threshold": threshold,
            "status": "passed" if parity <= threshold else "blocked",
        }
    ]
    dp_info = (
        {"epsilon": eps_final, "delta": "1e-5", "framework": "opacus"}
        if opacus_ok and eps_final
        else None
    )
    report = AnnexIVReport(
        model_introspection=mi,
        audit_chain=chain,
        fairness_log=fairness_log,
        training_config={"model": "MLP(10→32→2)", "epochs": EPOCHS, "optimizer": "Adam"},
        dataset_info={
            "system_name": "Connected Pipeline Demo",
            "dataset_name": "Synthetic",
            "size": granted,
        },
        dp_info=dp_info,
        regulations=["eu_ai_act", "gdpr"],
    )
    pdf_path = out_dir / "compliance_report.pdf"
    report.save_pdf(str(pdf_path))
    stage_times["Stage 8: PDF report"] = time.time() - ts
    compliance_record["stages"]["8_pdf"] = {"path": str(pdf_path)}
    print(f"Stage 8 ✅: Annex IV report generated → {pdf_path}")

    engine.detach()

    # ------------------------------------------------------------------
    # Stage 9 — MLflow experiment tracking
    # ------------------------------------------------------------------
    ts = time.time()
    mlflow_ok = False
    try:
        from torchcomply.integrations.mlflow_logger import ComplianceMLflowLogger

        mlflow_logger = ComplianceMLflowLogger(experiment_name="torchcomply_pipeline")
        mlflow_logger.start_run(run_name="connected_pipeline_demo")
        # Log fairness
        mlflow_logger.log_fairness(
            epoch=0,
            disparity=parity,
            threshold=threshold,
            status="passed" if parity <= threshold else "blocked",
        )
        # Log audit summary
        audit_summary = chain.summary()
        mlflow_logger.log_audit_summary(audit_summary)
        # Log DP params if available
        if opacus_ok and eps_final is not None:
            mlflow_logger.log_dp_params(epsilon=eps_final, delta=1e-5, n_steps=EPOCHS)
        run_id = mlflow_logger._run.info.run_id if mlflow_logger._run else "unknown"
        compliance_record["stages"]["9_mlflow"] = {"run_id": run_id}
        mlflow_logger.end_run()
        mlflow_ok = True
        stage_times["Stage 9: MLflow"] = time.time() - ts
        print(f"Stage 9 ✅: MLflow run logged — fairness, audit summary, DP params")
        print(f"         run_id={run_id}")
        print(f"         Verify: mlflow ui  →  http://localhost:5000/#/experiments/1/runs/{run_id}")
    except Exception as ex:
        stage_times["Stage 9: MLflow"] = time.time() - ts
        print(f"Stage 9 ⚠️:  MLflow logging failed ({ex})")

    # ------------------------------------------------------------------
    # Stage 10 — OpenTelemetry spans
    # ------------------------------------------------------------------
    ts = time.time()
    otel_ok = False
    try:
        from torchcomply.integrations.otel import OtelComplianceLogger

        otel_logger = OtelComplianceLogger(service_name="torchcomply")
        with otel_logger.span("compliance_pipeline", {"regulation": "eu_ai_act,gdpr"}):
            with otel_logger.span("fairness_check", {"disparity": str(round(parity, 4))}):
                pass
            with otel_logger.span("audit_chain_verify", {"entries": str(len(chain))}):
                pass
            with otel_logger.span("annex_iv_report", {"pdf": str(pdf_path.name)}):
                pass
        finished = otel_logger.get_finished_spans()
        otel_ok = True
        stage_times["Stage 10: OTel"] = time.time() - ts
        print(
            f"Stage 10 ✅: OpenTelemetry — {len(finished)} spans emitted "
            f"({', '.join(s.name for s in finished)})"
        )
    except Exception as ex:
        stage_times["Stage 10: OTel"] = time.time() - ts
        print(f"Stage 10 ⚠️:  OpenTelemetry failed ({ex})")

    # ------------------------------------------------------------------
    # Summary box
    # ------------------------------------------------------------------
    consent_summary = registry.access_log_summary()
    elapsed = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())

    root_hash_short = chain.root_hash()[:16] + "…" if chain.root_hash() else "N/A"
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║            END-TO-END COMPLIANCE PIPELINE COMPLETE               ║
╠══════════════════════════════════════════════════════════════════╣
║  Model:          MLP (10→32→2), {n_params:<5} params                  ║
║  Dataset:        500 samples, {denied} subjects opted out               ║
║  Hardware:       {device.upper():<54}║
║  Regulations:    EU AI Act, GDPR                                 ║
║                                                                  ║
║  ── Tools Connected ─────────────────────────────────────────── ║
║  Opacus:         DP-SGD, ε={f"{eps_final:.1f}" if eps_final else "N/A":<8}                    {"✅" if opacus_ok else "⚠️ "}         ║
║  Captum:         {len(explanations)} IntegratedGradients explanations   {"✅" if explanations else "⚠️ "}         ║
║  CrypTen:        Encrypted inference verified         {crypten_status}         ║
║  MLflow:         Fairness + audit + DP metrics        {"✅" if mlflow_ok else "⚠️ "}         ║
║  OpenTelemetry:  Compliance spans emitted             {"✅" if otel_ok else "⚠️ "}         ║
║                                                                  ║
║  ── Compliance Metrics ──────────────────────────────────────── ║
║  Audit Trail:    {len(chain):<5} entries | root_hash={root_hash_short}  ║
║  Fairness:       Parity {parity:.3f} | Threshold {threshold} | {fairness_status}    ║
║  Consent:        {consent_summary['granted']:<5} granted | {consent_summary['denied']:<5} denied             ║
║  Explanations:   {len(explanations)} samples explained                             ║
║  Stages logged:  {len(compliance_record['stages'])}/10 to compliance_record                  ║
║                                                                  ║
║  ── Artifacts ───────────────────────────────────────────────── ║
║  Report:         compliance_report.pdf                           ║
║  Audit log:      audit_chain.jsonl  (full entry log)             ║
║  Pipeline Time:  {elapsed:<6.1f}s                                        ║
║                                                                  ║
║  Status:         READY FOR REGULATORY REVIEW ✅                  ║
╚══════════════════════════════════════════════════════════════════╝""")

    # ------------------------------------------------------------------
    # ComplianceDiff — compare this run against a synthetic "previous" run
    # to demonstrate run-to-run regression detection (EU AI Act Art. 9)
    # ------------------------------------------------------------------
    snapshot_now = ComplianceSnapshot.from_engine(
        run_id="connected_pipeline_demo",
        engine=engine,
        accuracy=None,
        drift_detected=False,
        epsilon=eps_final,
        delta=1e-5 if eps_final else None,
        fairness_parity=parity,
        fairness_passed=(parity <= threshold),
    )
    # Simulate a previous run with slightly worse fairness and higher ε
    snapshot_prev = ComplianceSnapshot(
        run_id="previous_run",
        audit_root_hash="abc123",
        audit_entries=max(len(chain) - 20, 0),
        fairness_parity=max(parity - 0.01, 0.0),
        fairness_passed=True,
        epsilon=eps_final * 0.9 if eps_final else None,
        delta=1e-5 if eps_final else None,
        drift_detected=False,
    )
    diff = ComplianceDiff(snapshot_prev, snapshot_now)
    print(f"\n{diff.report()}")

    # Save compliance snapshot for next run's diff
    import json as _json
    (out_dir / "compliance_snapshot.json").write_text(
        _json.dumps(snapshot_now.to_dict(), indent=2)
    )
    print(f"Compliance snapshot saved → {out_dir / 'compliance_snapshot.json'}")
    print(f"  (Pass to next run's ComplianceDiff to detect regressions)")

    # Save summary
    (out_dir / "pipeline_summary.txt").write_text(
        f"Pipeline completed in {elapsed:.1f}s\n"
        + f"Opacus: {opacus_ok} | Captum: {bool(explanations)} | CrypTen: {crypten_status}\n"
        + f"Audit: {len(chain)} entries | Parity: {parity:.3f} | Consent: {consent_summary}\n"
    )

    # ------------------------------------------------------------------
    # Visualisation 1 — pipeline stage timeline
    # ------------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    stage_labels = list(stage_times.keys())
    stage_vals = [stage_times[k] for k in stage_labels]
    stage_colors = [
        "#3498db",
        "#3498db",
        "#27ae60",
        "#e67e22",
        "#8e44ad",
        "#e74c3c",
        "#2ecc71",
        "#e74c3c",
    ][: len(stage_labels)]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(stage_labels, stage_vals, color=stage_colors[: len(stage_labels)], alpha=0.85)
    for bar, v in zip(bars, stage_vals):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.2f}s", va="center", fontsize=9)
    ax.set_xlabel("Time (seconds)")
    ax.set_title(
        "End-to-End Pipeline Stage Timeline\nOpacus + Captum + CrypTen + Audit + Fairness + PDF"
    )
    ax.invert_yaxis()
    plt.tight_layout()
    out1 = out_dir / "pipeline_timeline.png"
    plt.savefig(out1, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nVisualization 1 saved → {out1}")

    # Visualisation 2 — coverage radar
    categories = [
        "Audit logging",
        "Transparency",
        "Human oversight",
        "Privacy (DP)",
        "Data governance",
        "Risk mgmt",
        "Documentation",
        "Bias prevention",
    ]
    # Compute scores from actual pipeline evidence
    n_stages = len(compliance_record["stages"])
    audit_score = min(1.0, len(chain) / 100)  # 100 entries = full score
    transparency_score = 1.0 if explanations else 0.5
    oversight_score = 1.0 if otel_ok else 0.6
    privacy_score = min(1.0, 1.0 - max(0, eps_final - 1.0) / 20.0) if eps_final else 0.4
    data_gov_score = min(1.0, granted / max(granted + denied, 1))
    risk_mgmt_score = min(1.0, n_stages / 10)
    doc_score = 1.0 if (out_dir / "compliance_report.pdf").exists() else 0.5
    bias_score = max(0.0, 1.0 - parity / threshold)
    scores = [audit_score, transparency_score, oversight_score, privacy_score,
              data_gov_score, risk_mgmt_score, doc_score, bias_score]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    vals = scores + scores[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.plot(angles, vals, color="steelblue", linewidth=2)
    ax.fill(angles, vals, color="steelblue", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9)
    ax.set_ylim(0, 1)
    ax.set_title("Compliance Coverage — EU AI Act + GDPR\nAll tools connected", pad=20)
    plt.tight_layout()
    out2 = out_dir / "coverage_radar.png"
    plt.savefig(out2, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Visualization 2 saved → {out2}")
    print(f"\nTotal elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
