"""Portable, read-only feature loading and strict sample alignment for fusion."""
from pathlib import Path
import csv
import numpy as np

MODEL_BLOCKS={
    "power108":["power"],"cov_oas171":["cov_oas_upper"],"cov_trace171":["cov_trace_upper"],
    "corr153":["corr_pearson_off"],"logcov171":["logcov_upper"],"logcov_trace171":["logcov_trace_upper"],
    "power_logcov279":["power","logcov_upper"],"power_corr261":["power","corr_pearson_off"],
    "power_logcov_trace279":["power","logcov_trace_upper"],"blind_log144":["blind_logpower"],
    "blind_shape144":["blind_shape"],"blind_local120":["blind_local_contrast"],"blind_harmonic60":["blind_harmonic_pattern"],
    "blind_log_local264":["blind_logpower","blind_local_contrast"],"power_blind252":["power","blind_logpower"],
    "power_logcov_blind423":["power","logcov_upper","blind_logpower"]}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as handle:return list(csv.DictReader(handle))


def _keys(values,name):
    values=np.asarray(values)
    if values.ndim!=1:raise ValueError(f"{name} must be one-dimensional")
    result=[str(v) for v in values]
    if any(not v for v in result) or len(set(result))!=len(result):raise ValueError(f"{name} contains empty or duplicate keys")
    return result


def load_periodic(split="Train",feature_set="power_logcov279",task=None,package_root=None):
    """Return X, y_or_None, metadata; works after moving this entire package."""
    if split not in ["Train","Test"]:raise ValueError("split must be Train or Test")
    if task is not None and task not in ["EC","EO","Imagery","SSVEP"]:raise ValueError("Unknown task")
    if package_root is None:
        raise ValueError("An explicit package_root is required; no private path is inferred")
    root=Path(package_root)
    power_path=root/f"payload/periodic_power_v1/features/{split.lower()}_power_v1.npz"
    v4_path=root/f"payload/ssvep_covariance_v4/features/{split.lower()}_blind_covariance_v4.npz"
    with np.load(power_path,allow_pickle=False) as power,np.load(v4_path,allow_pickle=False) as v4:
        for key in ["record_ids","paths","samples","tasks","sessions"]:
            if not np.array_equal(power[key],v4[key]):raise ValueError(f"Power/V4 metadata mismatch: {key}")
        _keys(v4["record_ids"],"record_ids");_keys(v4["paths"],"paths")
        selected=np.ones(len(v4["record_ids"]),bool) if task is None else v4["tasks"]==task
        blocks=MODEL_BLOCKS[feature_set]
        x=np.concatenate([(power["X"] if b=="power" else v4[b])[selected] for b in blocks],axis=1)
        y=v4["identity_labels"][selected] if split=="Train" else None
        if split=="Train" and not np.array_equal(power["identity_labels"][selected],y):raise ValueError("Identity label mismatch")
        if split=="Test" and ("identity_labels" in power.files or "identity_labels" in v4.files):raise ValueError("Unexpected Test identity labels")
        meta={key:v4[key][selected] for key in ["record_ids","paths","samples","tasks","sessions"]}
        definitions=read_csv(root/"payload/ssvep_covariance_v4/features/feature_dictionary.csv")
        names=[]
        for b in blocks:names.extend(list(power["feature_names"]) if b=="power" else [r["name"] for r in definitions if r["block"]==b])
        if x.shape[1]!=len(names) or not np.isfinite(x).all():raise ValueError("Invalid periodic features")
        meta["feature_names"]=np.asarray(names);meta["feature_set"]=feature_set
    return x,y,meta


def align_branch(base_meta,branch_x,*,branch_record_ids=None,branch_paths=None,
                 base_y=None,branch_y=None,branch_feature_names=None,prefix="branch"):
    """Reorder an exactly matching branch into base order. No dropping/filling/fitting.

    Use canonical full paths if branch IDs use another numbering scheme. When both
    paths and IDs are supplied, both must agree. NaN may represent fit failure;
    impute only in each model's training fold. Infinity is rejected.
    """
    x=np.asarray(branch_x,dtype=np.float64)
    if x.ndim!=2 or np.isinf(x).any():raise ValueError("branch_x must be a numeric 2-D matrix without infinity")
    if branch_record_ids is None and branch_paths is None:raise ValueError("Provide canonical record_ids or full paths")
    if branch_paths is not None:
        expected=_keys(base_meta["paths"],"base paths");actual=_keys(branch_paths,"branch paths")
    else:
        expected=_keys(base_meta["record_ids"],"base record_ids");actual=_keys(branch_record_ids,"branch record_ids")
    if len(actual)!=len(x):raise ValueError("Key count and feature rows differ")
    missing=set(expected)-set(actual);extra=set(actual)-set(expected)
    if missing or extra:raise ValueError(f"Record set differs: missing={len(missing)}, extra={len(extra)}; select the same split/task scope first")
    index={key:i for i,key in enumerate(actual)};order=np.array([index[key] for key in expected])
    if branch_record_ids is not None:
        ids=np.asarray(branch_record_ids)
        _keys(ids,"branch record_ids")
        if len(ids)!=len(x) or not np.array_equal(ids[order],base_meta["record_ids"]):raise ValueError("Record IDs disagree with canonical order/path mapping")
    if branch_paths is not None and not np.array_equal(np.asarray(branch_paths)[order],base_meta["paths"]):raise ValueError("Path mismatch")
    if branch_y is not None:
        if base_y is None:raise ValueError("Branch labels supplied without a labeled base; never invent Test labels")
        labels=np.asarray(branch_y)
        if labels.ndim!=1 or len(labels)!=len(x) or not np.array_equal(labels[order],base_y):raise ValueError("Identity labels disagree")
    if branch_feature_names is None:names=[f"{prefix}__{i}" for i in range(x.shape[1])]
    else:
        raw=np.asarray(branch_feature_names)
        if raw.ndim!=1 or len(raw)!=x.shape[1]:raise ValueError("Feature name count mismatch")
        names=[f"{prefix}__{name}" for name in raw]
    if len(set(names))!=len(names):raise ValueError("Duplicate feature names within branch")
    return x[order],np.asarray(names)
