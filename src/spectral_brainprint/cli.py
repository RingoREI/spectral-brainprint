"""Small executable entry points; research stages remain explicit."""
import argparse
import json
from pathlib import Path
import numpy as np
from spectral_brainprint.features.power import features_from_psd
from spectral_brainprint.features.spatial_blind import covariance_blocks
from spectral_brainprint.spectra.welch import welch_psd
from spectral_brainprint.models.lda import make_lda
from spectral_brainprint.evaluation.reproduce import reproduce, environment, validate_partition


def synthetic_demo():
    rng = np.random.default_rng(20261112)
    y = np.repeat(np.arange(1, 4), 12)
    eeg = rng.normal(size=(len(y), 1200, 18))
    time = np.arange(1200) / 300
    # Artificial class-specific oscillations; no real people or biometric evidence.
    for i, label in enumerate(y):
        eeg[i] += (0.5 + label / 2) * np.sin(2 * np.pi * (5 + 4 * label) * time)[:, None]
    f, p = welch_psd(eeg)
    config = {"bands": [{"low_hz": a, "high_hz": b} for a, b in [(4, 8), (8, 13), (13, 30)]],
              "relative_denominator_hz": [1, 40], "numerical_power_floor": 1e-300}
    power, invalid = features_from_psd(p, f, config)
    spatial, _, _, _ = covariance_blocks(eeg)
    x = np.concatenate([power, spatial["logcov_upper"]], axis=1)
    train = np.flatnonzero(np.arange(len(y)) % 12 < 8)
    validation = np.flatnonzero(np.arange(len(y)) % 12 >= 8)
    validate_partition(train, validation, y)
    model = make_lda(3).fit(x[train], y[train])
    pred = model.predict(x[validation])
    return {"scope": "Synthetic engineering demonstration only; no biometric performance claim",
            "seed": 20261112, "feature_shape": list(x.shape),
            "n_train": len(train), "n_validation": len(validation),
            "invalid_power_channels": int(invalid.sum()),
            "synthetic_accuracy": float(np.mean(pred == y[validation])),
            "environment": environment()}


def main():
    parser = argparse.ArgumentParser(description="Spectral Brainprint research utilities")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run a synthetic EEG -> features -> LDA example")
    demo.add_argument("--output", type=Path, required=True)
    rep = commands.add_parser("reproduce", help="Reproduce one historical Train-only case")
    rep.add_argument("--package-root", type=Path, required=True)
    rep.add_argument("--feature-set", default="power108")
    rep.add_argument("--protocol", default="within_task_blocked_provisional")
    rep.add_argument("--task", choices=["EC", "EO", "Imagery", "SSVEP"], default="EC")
    rep.add_argument("--fold", type=int, default=0)
    rep.add_argument("--output", type=Path, required=True)
    mlp = commands.add_parser("predict-mlp20", help="Run the frozen team MLP20 ensemble")
    mlp.add_argument("--features", type=Path, required=True)
    mlp.add_argument("--checkpoint", type=Path, nargs=3, required=True)
    mlp.add_argument("--calibration", type=Path, required=True)
    mlp.add_argument("--template", type=Path)
    mlp.add_argument("--device", default="cpu")
    mlp.add_argument("--trust-legacy-checkpoint", action="store_true",
                     help="Allow pickle when loading trusted original team checkpoints")
    mlp.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new run path")
    if args.command == "predict-mlp20":
        from spectral_brainprint.models.team_inference import predict_mlp20

        result = predict_mlp20(args.features, args.checkpoint, args.calibration, args.output,
                               template=args.template, device=args.device,
                               trust_legacy_checkpoint=args.trust_legacy_checkpoint)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    result = synthetic_demo() if args.command == "demo" else reproduce(
        args.package_root, args.feature_set, args.protocol, args.task, args.fold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != "predictions"}, ensure_ascii=False, indent=2))
    if args.command == "reproduce" and not result["matches_history_at_1e_12"]:
        raise SystemExit(2)
