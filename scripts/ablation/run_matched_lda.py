"""Fit deterministic LDA controls on exactly the MLP inner-training records."""

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import time
import warnings

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from threadpoolctl import threadpool_limits


TASKS = ("EC", "EO", "Imagery", "SSVEP")
VARIANTS = {"M00": (0, 0), "M10": (1, 0), "M01": (0, 1), "M11": (1, 1)}
LABELS = np.arange(1, 21)


def load_workflow():
    spec = importlib.util.spec_from_file_location("matched_lda_workflow", Path(__file__).with_name("run.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def active_columns(p, a):
    return np.concatenate([np.arange(171), np.arange(171, 279) if p else np.empty(0, int),
                           np.arange(279, 315) if a else np.empty(0, int)])


def fit_arm(x_train, y_train):
    """No development/evaluation observations are accepted by this fit function."""
    if x_train.ndim != 2 or len(x_train) != len(y_train) or not np.isfinite(x_train).all():
        raise ValueError("Invalid LDA fitting array")
    if set(np.asarray(y_train).tolist()) != set(LABELS):
        raise ValueError("LDA fitting rows must contain all 20 registered identities")
    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=np.full(20, .05))
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started = time.perf_counter()
        model.fit(np.asarray(x_train, dtype=np.float64), y_train)
        seconds = time.perf_counter() - started
    if not np.array_equal(model.classes_, LABELS):
        raise ValueError("Unexpected LDA class ordering")
    return model, seconds, [f"{type(w.message).__name__}: {w.message}" for w in caught]


def evaluate(model, x, y):
    with threadpool_limits(limits=1):
        probabilities = np.asarray(model.predict_proba(np.asarray(x, dtype=np.float64)), dtype=np.float64)
    if (probabilities.shape != (len(y), 20) or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-12)):
        raise ValueError("Invalid LDA class probabilities")
    predicted = LABELS[probabilities.argmax(axis=1)]
    metrics = {"accuracy": float(accuracy_score(y, predicted)),
               "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
               "macro_f1": float(f1_score(y, predicted, labels=LABELS, average="macro", zero_division=0)),
               "nll": float(log_loss(y, probabilities, labels=LABELS))}
    return metrics, predicted, probabilities


def check_mlp_source(folder, arrays, partitions, hashes, workflow):
    manifest_path = folder / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest["scope"] != "development":
        raise ValueError("MLP source must be a complete development run")
    if manifest["input_sha256"] != hashes or manifest["config"]["folds"] != [0, 1, 2]:
        raise ValueError("MLP features/splits do not match this control")
    task = manifest["task"]
    seeds = manifest["config"]["seeds"]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("MLP source requires distinct paired seeds")
    files = sorted((folder / "results").glob("result*.json"))
    if len(files) != 12 * len(seeds):
        raise ValueError("Incomplete MLP result source")
    seen = set()
    for path in files:
        result = json.loads(path.read_text(encoding="utf-8"))
        fold, seed, variant = result["fold"], result["seed"], result["variant"]
        key = fold, seed, variant
        if (key in seen or fold not in partitions or seed not in seeds or variant not in VARIANTS
                or result["task"] != task or result["input_sha256"] != hashes):
            raise ValueError("Mismatched or duplicate MLP source result")
        seen.add(key)
        parts = partitions[fold]
        if (result["n_train"] != len(parts["train"]) or result["n_development"] != len(parts["inner_dev"])
                or result["n_validation"] != len(parts["outer_eval"])):
            raise ValueError("MLP partition role counts differ")
        expected = list(zip(arrays["record_ids"][parts["outer_eval"]].tolist(), arrays["y"][parts["outer_eval"]].tolist()))
        if [(r["record_id"], r["truth"]) for r in result["predictions"]] != expected:
            raise ValueError("MLP evaluation IDs or truth do not match prepared data")
    for fold, parts in partitions.items():
        imputer, scaler = workflow.fit_preprocessor(arrays["X"], parts["train"])
        with np.load(folder / f"preprocessing/fold{fold}.npz", allow_pickle=False) as saved:
            for field, value in (("median", imputer.statistics_), ("mean", scaler.mean_), ("scale", scaler.scale_),
                                 ("train_record_ids", arrays["record_ids"][parts["train"]])):
                if not np.array_equal(saved[field], value, equal_nan=True):
                    raise ValueError(f"MLP inner-training preprocessing differs: fold {fold} {field}")
    return workflow.sha256(manifest_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("Choose a fresh output directory")
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("Tasks must be distinct")
    workflow = load_workflow()
    prepared = {}
    source_hashes = {}
    for task in args.tasks:
        folder = args.artifact_root / ("ec/prepared" if task == "EC" else "shared/prepared_other_tasks")
        features, splits = folder / "features.npz", folder / "splits.csv"
        arrays, metadata = workflow.load_features(features)
        if not all(str(path).replace("\\", "/").startswith("Train/") for path in arrays["paths"]):
            raise ValueError("The matched control accepts labeled Train source records only")
        partitions = workflow.load_partitions(splits, arrays, task, (0, 1, 2))
        outer = np.concatenate([arrays["record_ids"][p["outer_eval"]] for p in partitions.values()])
        if set(outer.tolist()) != set(arrays["record_ids"][arrays["tasks"] == task].tolist()):
            raise ValueError("Outer folds do not cover the task cohort exactly once")
        hashes = {"features": workflow.sha256(features), "splits": workflow.sha256(splits)}
        mlp_hash = check_mlp_source(args.artifact_root / task.lower() / "full/runs", arrays, partitions, hashes, workflow)
        prepared[task] = arrays, metadata, partitions, hashes
        source_hashes[task] = dict(input_sha256=hashes, mlp_manifest_sha256=mlp_hash)
    config = dict(model="LDA", solver="lsqr", shrinkage="auto", priors="uniform20", input_slots=315,
                  preprocessing="Identical inner-train median/no-indicator and StandardScaler as MLP; cast float32 as MLP, then promote exact values to float64 for LDA",
                  active_features={v: active_columns(*factors).tolist() for v, factors in VARIANTS.items()},
                  inactive_columns="Remove disabled columns before automatic covariance shrinkage; no dummy zero columns",
                  selection="Fixed configuration; inner-development rows excluded from fit and unused for tuning",
                  repeats="Deterministic one fit per task/fold/condition; no synthetic seed repetitions",
                  comparison="Per-seed MLP scores minus the same deterministic LDA score on matched evaluation records",
                  scope="Post-MLP development control, not independent confirmatory evaluation")
    environment = {"python": platform.python_version(), **{p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn", "threadpoolctl")}}
    manifest = dict(status="running", scope="development", tasks=args.tasks, config=config, environment=environment,
                    source_hashes=source_hashes, trainer_sha256=workflow.sha256(__file__),
                    methods_sha256=workflow.sha256(Path(__file__).resolve().parents[2] / "docs/09_ablation_methods.md"),
                    shared_preprocessing_implementation_sha256=workflow.sha256(Path(__file__).with_name("run.py")))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    workflow.write_json(args.output_dir / "run_manifest.json", manifest)
    try:
        for task, (arrays, metadata, partitions, hashes) in prepared.items():
            for fold, parts in partitions.items():
                preprocessing = workflow.fit_preprocessor(arrays["X"], parts["train"])
                imputer, scaler = preprocessing
                target = args.output_dir / "preprocessing"
                target.mkdir(exist_ok=True)
                np.savez_compressed(target / f"{task}-fold{fold}.npz", median=imputer.statistics_, mean=scaler.mean_, scale=scaler.scale_,
                                    train_record_ids=arrays["record_ids"][parts["train"]],
                                    development_record_ids=arrays["record_ids"][parts["inner_dev"]], evaluation_record_ids=arrays["record_ids"][parts["outer_eval"]])
                for variant, (p, a) in VARIANTS.items():
                    columns = active_columns(p, a)
                    train_x = workflow.transformed(arrays["X"][parts["train"]], preprocessing, p, a)[:, columns].astype(np.float64)
                    model, seconds, fitting_warnings = fit_arm(train_x, arrays["y"][parts["train"]])
                    dev_x = workflow.transformed(arrays["X"][parts["inner_dev"]], preprocessing, p, a)[:, columns].astype(np.float64)
                    dev_metrics, _, _ = evaluate(model, dev_x, arrays["y"][parts["inner_dev"]])
                    indices = parts["outer_eval"]
                    eval_x = workflow.transformed(arrays["X"][indices], preprocessing, p, a)[:, columns].astype(np.float64)
                    started = time.perf_counter()
                    metrics, predicted, probabilities = evaluate(model, eval_x, arrays["y"][indices])
                    elapsed = time.perf_counter() - started
                    stem = f"{task}-fold{fold}-{variant}"
                    weights = args.output_dir / "weights"
                    weights.mkdir(exist_ok=True)
                    np.savez_compressed(weights / f"{stem}.npz", coefficients=model.coef_, intercept=model.intercept_,
                                        classes=model.classes_, covariance=model.covariance_, means=model.means_, priors=model.priors_,
                                        active_columns=columns)
                    result = dict(model="LDA", task=task, protocol=metadata["protocol"], scope="development", fold=fold, variant=variant, p=p, a=a,
                                  n_train=len(parts["train"]), n_development=len(parts["inner_dev"]), n_validation=len(indices),
                                  metrics=metrics, inner_development_metrics=dev_metrics, config=config, environment=environment, input_sha256=hashes,
                                  training_seconds=seconds, inference_seconds=elapsed, active_dimension=len(columns),
                                  discriminant_coefficients=int(model.coef_.size + model.intercept_.size), fitting_warnings=fitting_warnings,
                                  predictions=[dict(record_id=int(arrays["record_ids"][row]), truth=int(arrays["y"][row]),
                                                    prediction=int(predicted[i]), probabilities=probabilities[i].tolist()) for i, row in enumerate(indices)])
                    workflow.write_json(args.output_dir / "results" / f"result-{stem}.json", result)
                    print(f"{stem}: BA={metrics['balanced_accuracy']:.4%}; train={len(parts['train'])}; features={len(columns)}", flush=True)
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failure"] = f"{type(error).__name__}: {error}"
        workflow.write_json(args.output_dir / "run_manifest.json", manifest)
        raise
    workflow.write_json(args.output_dir / "run_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
