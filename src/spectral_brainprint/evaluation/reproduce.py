"""Reproduce one frozen Train-only case without rewriting the source handoff."""
from pathlib import Path
import hashlib
import importlib.metadata
import platform
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from threadpoolctl import threadpool_limits
from spectral_brainprint.io.legacy_handoff import load_periodic, read_csv
from spectral_brainprint.models.lda import make_lda


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def environment():
    return {"python": platform.python_version(), **{
        name: importlib.metadata.version(name)
        for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl")}}


def validate_partition(train, validation, y):
    train, validation = np.asarray(train), np.asarray(validation)
    for part in (train, validation):
        if part.ndim != 1 or len(part) == 0 or not np.issubdtype(part.dtype, np.integer):
            raise ValueError("Partitions must be nonempty integer row arrays")
        if len(np.unique(part)) != len(part) or part.min() < 0 or part.max() >= len(y):
            raise ValueError("Duplicate or invalid partition rows")
    if set(train) & set(validation):
        raise ValueError("Train/validation record leakage")
    if not set(y[validation]).issubset(set(y[train])):
        raise ValueError("Closed-set validation has unregistered identities")


def reproduce(package_root, feature_set, protocol, task, fold):
    root = Path(package_root)
    x, y, meta = load_periodic("Train", feature_set, package_root=root)
    assignment = root / "payload/ssvep_covariance_v4/reports/case_assignments.csv"
    lookup = {int(key): i for i, key in enumerate(meta["record_ids"])}
    rows = [r for r in read_csv(assignment)
            if (r["protocol"], r["task"], int(r["fold"])) == (protocol, task, fold)]
    if not rows or {r["role"] for r in rows} != {"train", "validation"}:
        raise ValueError("Case missing or roles invalid")
    tr, va = [np.array([lookup[int(r["record_id"])] for r in rows if r["role"] == role])
              for role in ("train", "validation")]
    validate_partition(tr, va, y)
    if set(y[tr]) != set(range(1, 21)):
        raise ValueError("Historical reproduction requires identities 1 through 20")
    model = make_lda()
    with threadpool_limits(limits=1):
        model.fit(x[tr], y[tr])
        pred = model.predict(x[va])
    metrics = {"accuracy": float(accuracy_score(y[va], pred)),
               "balanced_accuracy": float(balanced_accuracy_score(y[va], pred)),
               "macro_f1": float(f1_score(y[va], pred, labels=np.arange(1, 21),
                                           average="macro", zero_division=0))}
    reference = [r for r in read_csv(root / "payload/ssvep_covariance_v4/reports/fold_metrics.csv")
                 if (r["feature_set"], r["protocol"], r["task"], int(r["fold"]))
                 == (feature_set, protocol, task, fold)]
    if len(reference) != 1:
        raise ValueError("Expected exactly one historical metric row")
    errors = {key: abs(value - float(reference[0][key])) for key, value in metrics.items()}
    inputs = [assignment, root / "payload/periodic_power_v1/features/train_power_v1.npz",
              root / "payload/ssvep_covariance_v4/features/train_blind_covariance_v4.npz",
              root / "payload/periodic_power_v1/features/feature_dictionary.csv",
              root / "payload/ssvep_covariance_v4/features/feature_dictionary.csv"]
    return {"feature_set": feature_set, "protocol": protocol, "task": task, "fold": fold,
            "n_train": len(tr), "n_validation": len(va), "metrics": metrics,
            "absolute_metric_errors": errors, "matches_history_at_1e_12": max(errors.values()) < 1e-12,
            "environment": environment(),
            "input_sha256": {p.relative_to(root).as_posix(): sha256(p) for p in inputs},
            "predictions": [{"record_id": int(meta["record_ids"][i]), "truth": int(y[i]),
                             "prediction": int(p)} for i, p in zip(va, pred)],
            "scope": "One same-session provisional case; no Test or rejection calibration"}
