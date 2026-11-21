"""Train paired fixed-slot MLPs for the prospective P/A feature ablation."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler


VARIANTS = {"M00": (0, 0), "M10": (1, 0), "M01": (0, 1), "M11": (1, 1)}
LABELS = np.arange(1, 21)
ROLES = ("train", "inner_dev", "outer_eval")
SEEDS = (20261102, 20261103, 20261104)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def load_features(path):
    with np.load(path, allow_pickle=False) as data:
        required = {"X", "y", "record_ids", "tasks", "paths", "metadata_json"}
        if not required.issubset(data.files):
            raise ValueError(f"Prepared features missing keys: {sorted(required - set(data.files))}")
        arrays = {name: np.asarray(data[name]) for name in required - {"metadata_json"}}
        raw_metadata = np.asarray(data["metadata_json"])
        if raw_metadata.ndim != 0:
            raise ValueError("metadata_json must be a scalar JSON string")
        metadata = json.loads(str(raw_metadata.item()))
    x, y, ids = arrays["X"], arrays["y"], arrays["record_ids"]
    if x.ndim != 2 or x.shape[1] != 315 or len(x) == 0 or x.dtype.kind not in "fiu":
        raise ValueError("Prepared X must be a nonempty numeric [N,315] matrix")
    if np.isinf(x).any():
        raise ValueError("Infinite prepared features are not permitted")
    if y.shape != (len(x),) or y.dtype.kind not in "iu" or not np.isin(y, LABELS).all():
        raise ValueError("Prepared y must contain registered identity labels 1 through 20")
    if ids.shape != y.shape or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
        raise ValueError("record_ids must be unique integers aligned to X")
    for field in ("tasks", "paths"):
        if arrays[field].shape != y.shape or arrays[field].dtype.kind not in "US":
            raise ValueError(f"{field} must be an aligned string vector")
        if not all(str(item) for item in arrays[field]):
            raise ValueError(f"{field} contains empty values")
    if len(set(arrays["paths"].tolist())) != len(y):
        raise ValueError("Prepared paths must be unique")
    if metadata.get("a36_accepted") is not True:
        raise ValueError("Prepared metadata must explicitly declare a36_accepted=true")
    if not np.isfinite(x[:, 279:315]).any():
        raise ValueError("No finite A36 observations; cannot run an aperiodic ablation")
    arrays["X"] = np.asarray(x, dtype=np.float64)
    return arrays, metadata


def load_partitions(path, arrays, task, folds):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"task", "fold", "role", "record_id"}.issubset(reader.fieldnames or ()):
            raise ValueError("Split CSV requires task,fold,role,record_id")
        rows = list(reader)
    lookup = {int(key): index for index, key in enumerate(arrays["record_ids"])}
    partitions = {}
    seen_outer = set()
    for fold in folds:
        selected = [r for r in rows if r["task"] == task and int(r["fold"]) == fold]
        if not selected or {r["role"] for r in selected} != set(ROLES):
            raise ValueError(f"{task} fold {fold}: require all three split roles")
        ids = [int(row["record_id"]) for row in selected]
        if len(ids) != len(set(ids)) or not set(ids).issubset(lookup):
            raise ValueError(f"{task} fold {fold}: duplicate or unrecognized split record IDs")
        if any(str(arrays["tasks"][lookup[key]]) != task for key in ids):
            raise ValueError(f"{task} fold {fold}: task metadata mismatch")
        parts = {role: np.array([lookup[int(r["record_id"])] for r in selected if r["role"] == role], dtype=int)
                 for role in ROLES}
        for role, indices in parts.items():
            if not len(indices) or set(arrays["y"][indices]) != set(LABELS):
                raise ValueError(f"{task} fold {fold}: {role} must cover all 20 enrolled identities")
        outer_ids = set(arrays["record_ids"][parts["outer_eval"]].tolist())
        if outer_ids & seen_outer:
            raise ValueError("Outer evaluation record IDs overlap across selected folds")
        seen_outer.update(outer_ids)
        group_field = next((name for name in ("source_group", "group") if name in selected[0]), None)
        if group_field:
            groups = {role: {r[group_field] for r in selected if r["role"] == role} for role in ROLES}
            if any(not group for values in groups.values() for group in values):
                raise ValueError("Provided source groups must be nonempty")
            if any(groups[a] & groups[b] for a, b in (("train", "inner_dev"), ("train", "outer_eval"), ("inner_dev", "outer_eval"))):
                raise ValueError("Source group crosses partition roles")
        partitions[fold] = parts
    return partitions


def fit_preprocessor(x, train_rows):
    imputer = SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=False)
    training = imputer.fit_transform(x[train_rows])
    scaler = StandardScaler().fit(training)
    return imputer, scaler


def transformed(x, preprocessing, p, a):
    imputer, scaler = preprocessing
    result = np.asarray(scaler.transform(imputer.transform(x)), dtype=np.float32)
    if not p:
        result[:, 171:279] = 0
    if not a:
        result[:, 279:315] = 0
    if result.shape[1] != 315 or not np.isfinite(result).all():
        raise ValueError("Preprocessing did not produce finite fixed-slot inputs")
    return result


def require_torch(device):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError('Install the project [deep] extra before training the MLP ablation') from error
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
    return torch


def evaluate(model, x, y, torch, device, batch_size=256):
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            tensor = torch.as_tensor(x[start:start + batch_size], device=device)
            logits.append(model(tensor)["logits"].detach().cpu())
    logits = torch.cat(logits)
    probabilities = torch.softmax(logits, dim=1).numpy().astype(np.float64)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    nll = float(log_loss(y, probabilities, labels=LABELS))
    predicted = probabilities.argmax(axis=1) + 1
    return {"accuracy": float(accuracy_score(y, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "macro_f1": float(f1_score(y, predicted, labels=LABELS, average="macro", zero_division=0)),
            "nll": nll}, predicted, probabilities


def fit_selected_model(x_train, y_train, x_dev, y_dev, seed, fold, config, device, torch):
    """Select a checkpoint using only training and inner-development arrays."""
    from spectral_brainprint.models.mlp import SimpleMLP

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = SimpleMLP(20, input_dim=315, hidden_dims=(256, 128), dropout=.2).to(device)
    initial_hash = hashlib.sha256(b"".join(value.detach().cpu().numpy().tobytes()
                                           for value in model.state_dict().values())).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    generator = np.random.default_rng(np.random.SeedSequence([seed, fold]))
    inputs = torch.as_tensor(x_train, device=device)
    targets = torch.as_tensor(y_train - 1, dtype=torch.long, device=device)
    best_nll, patience_anchor = float("inf"), float("inf")
    best_epoch, best_state, bad_epochs = 0, None, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        order = generator.permutation(len(x_train))
        loss_sum = 0.0
        for start in range(0, len(order), config["batch_size"]):
            batch = torch.as_tensor(order[start:start + config["batch_size"]], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(inputs[batch])["logits"], targets[batch])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(batch)
        dev_metrics, _, _ = evaluate(model, x_dev, y_dev, torch, device)
        if not np.isfinite(dev_metrics["nll"]):
            raise ValueError("Nonfinite inner-development loss")
        if dev_metrics["nll"] < best_nll:
            best_nll, best_epoch = dev_metrics["nll"], epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if dev_metrics["nll"] < patience_anchor - config["early_stopping"]["min_delta"]:
            patience_anchor, bad_epochs = dev_metrics["nll"], 0
        else:
            bad_epochs += 1
        history.append({"epoch": epoch, "train_nll": loss_sum / len(x_train),
                        "development_nll": dev_metrics["nll"],
                        "development_balanced_accuracy": dev_metrics["balanced_accuracy"]})
        if epoch >= config["early_stopping"]["min_epochs"] and bad_epochs >= config["early_stopping"]["patience"]:
            break
    if best_state is None:
        raise ValueError("No finite development checkpoint was selected")
    model.load_state_dict(best_state)
    model.to(device).eval()
    return model, {"selected_epoch": best_epoch, "epochs_completed": len(history),
                   "best_development_nll": best_nll, "training_curve": history,
                   "training_seconds": time.perf_counter() - started,
                   "initial_state_sha256": initial_hash,
                   "parameters": sum(parameter.numel() for parameter in model.parameters())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--task", choices=("EC", "EO", "Imagery", "SSVEP"), default="EC")
    parser.add_argument("--fold", type=int, choices=(0, 1, 2))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--pilot", action="store_true", help="Permit a single-fold/single-seed engineering run")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-epochs", type=int, default=80)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("--output-dir exists; choose a fresh run directory")
    if len(args.seeds) != len(set(args.seeds)) or any(not 0 <= seed < 2 ** 32 for seed in args.seeds):
        parser.error("Seeds must be distinct integers in [0,2**32)")
    if not args.pilot and (len(args.seeds) < 3 or args.fold is not None):
        parser.error("Formal development runs require all three folds and at least three seeds; use --pilot for a subset")
    if args.max_epochs <= 0:
        parser.error("--max-epochs must be positive")
    if not args.pilot and args.max_epochs < 10:
        parser.error("Formal runs require a maximum budget of at least the 10 minimum epochs")
    arrays, metadata = load_features(args.features)
    folds = (args.fold,) if args.fold is not None else (0, 1, 2)
    partitions = load_partitions(args.splits, arrays, args.task, folds)
    torch = require_torch(args.device)
    config = {"model": "SimpleMLP", "input_dim": 315, "hidden_dims": [256, 128], "dropout": .2,
              "preprocessing": "train-only median imputation then StandardScaler; gates after scaling",
              "class_mapping": {str(label): int(label - 1) for label in LABELS},
              "learning_rate": .0005, "weight_decay": .01, "batch_size": 32,
              "max_epochs": args.max_epochs, "gradient_clip": 5., "optimizer": "AdamW",
              "scheduler": None, "augmentation": None,
              "early_stopping": {"monitor": "inner_development_nll", "min_epochs": 10,
                                 "patience": 12, "min_delta": .0001},
              "seeds": args.seeds, "folds": list(folds), "device": args.device,
              "torch_threads": 1, "deterministic_algorithms": True,
              "evaluation_nll": "sklearn log_loss from float64 normalized softmax probabilities",
              "selection": "minimum inner-development NLL; min_delta controls patience only; no outer evaluation during fitting"}
    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   **{name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn")}}
    hashes = {"features": sha256(args.features), "splits": sha256(args.splits)}
    scope = "pilot" if args.pilot else "development"
    manifest = {"status": "running", "scope": scope, "task": args.task,
                "protocol": metadata.get("protocol", "within_task_blocked_provisional"),
                "config": config, "environment": environment, "input_sha256": hashes,
                "source_files": {"features": args.features.name, "splits": args.splits.name},
                "trainer_sha256": sha256(__file__), "features_metadata": metadata}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "run_manifest.json", manifest)
    try:
        for fold, parts in partitions.items():
            preprocessing = fit_preprocessor(arrays["X"], parts["train"])
            target = args.output_dir / "preprocessing"
            target.mkdir(exist_ok=True)
            imputer, scaler = preprocessing
            np.savez_compressed(target / f"fold{fold}.npz", median=imputer.statistics_, mean=scaler.mean_,
                                scale=scaler.scale_, train_record_ids=arrays["record_ids"][parts["train"]])
            for seed in args.seeds:
                initial_hash = None
                for variant, (p, a) in VARIANTS.items():
                    x_train = transformed(arrays["X"][parts["train"]], preprocessing, p, a)
                    x_dev = transformed(arrays["X"][parts["inner_dev"]], preprocessing, p, a)
                    model, trained = fit_selected_model(x_train, arrays["y"][parts["train"]],
                                                        x_dev, arrays["y"][parts["inner_dev"]],
                                                        seed, fold, config, args.device, torch)
                    if initial_hash is None:
                        initial_hash = trained["initial_state_sha256"]
                    elif initial_hash != trained["initial_state_sha256"]:
                        raise ValueError("Paired factor arms did not start from the same weights")
                    evaluation_rows = parts["outer_eval"]
                    x_eval = transformed(arrays["X"][evaluation_rows], preprocessing, p, a)
                    evaluate_started = time.perf_counter()
                    metrics, predicted, probability = evaluate(model, x_eval, arrays["y"][evaluation_rows], torch, args.device)
                    inference_seconds = time.perf_counter() - evaluate_started
                    stem = f"{args.task}-fold{fold}-seed{seed}-{variant}"
                    model_path = args.output_dir / "checkpoints" / f"{stem}.pt"
                    model_path.parent.mkdir(exist_ok=True)
                    torch.save({"model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                                "config": config, "variant": variant, "selected_epoch": trained["selected_epoch"]}, model_path)
                    result = {"task": args.task, "protocol": manifest["protocol"], "scope": scope,
                              "seed": seed, "fold": fold, "variant": variant, "p": p, "a": a,
                              "n_train": len(parts["train"]), "n_development": len(parts["inner_dev"]),
                              "n_validation": len(evaluation_rows), "metrics": metrics, **trained,
                              "inference_seconds": inference_seconds, "config": config,
                              "environment": environment, "input_sha256": hashes,
                              "predictions": [{"record_id": int(arrays["record_ids"][row]),
                                               "truth": int(arrays["y"][row]), "prediction": int(predicted[index]),
                                               "probabilities": probability[index].astype(float).tolist()}
                                              for index, row in enumerate(evaluation_rows)]}
                    write_json(args.output_dir / "results" / f"result-{stem}.json", result)
                    print(f"{stem}: BA={metrics['balanced_accuracy']:.4%}; epoch={trained['selected_epoch']}", flush=True)
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failure"] = f"{type(error).__name__}: {error}"
        write_json(args.output_dir / "run_manifest.json", manifest)
        raise
    write_json(args.output_dir / "run_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
