"""Read historical team features in the frozen 705-column MLP order.

The adapter and normalization follow Brainprint_ML's
brainprint_mlp20_candidate_20261105/training.py and deployment.py.
"""

from pathlib import Path

import numpy as np


TEAM_VECTOR_DIM = 705
_BRANCH_SHAPES = {
    "periodic": (279,),
    "aperiodic": (18, 2),
    "artifact_global": (296,),
    "blink_events": (12, 47),
}
_TEXT_METADATA = ("id", "sample", "session", "group", "signal_hash")
_INTEGER_METADATA = ("record_id", "task", "role", "label")


def vector_input(raw, masks):
    """Return raw float32 inputs before the training-fitted normalization.

    The power and spatial block has 279 columns. The nonperiodic block has
    36 columns in channel order, with offset then exponent for each channel.
    ICA/global features have 296 columns. Blink events contribute another
    47 columns each for the population mean and standard deviation.
    """
    missing = _BRANCH_SHAPES.keys() - (raw.keys() & masks.keys())
    if missing:
        raise ValueError(f"Missing feature block or mask: {', '.join(sorted(missing))}")
    arrays = {name: np.asarray(raw[name]) for name in _BRANCH_SHAPES}
    validity = {}
    n_rows = arrays["periodic"].shape[0] if arrays["periodic"].ndim else 0
    for name, shape in _BRANCH_SHAPES.items():
        values = arrays[name]
        mask = np.asarray(masks[name])
        if n_rows == 0 or values.shape != (n_rows, *shape) or values.dtype.kind not in "fiu":
            raise ValueError(f"{name} must be a nonempty numeric array of shape (N, {shape})")
        if mask.shape != values.shape[:-1]:
            raise ValueError(f"Invalid mask shape for {name}: expected {values.shape[:-1]}")
        validity[name] = mask.astype(bool, copy=False)

    events = arrays["blink_events"]
    valid = validity["blink_events"][..., None] & np.isfinite(events)
    count = np.maximum(valid.sum(axis=1), 1)
    mean = np.where(valid, events, 0).sum(axis=1) / count
    variance = np.where(valid, (events - mean[:, None, :]) ** 2, 0).sum(axis=1) / count

    periodic = np.where(validity["periodic"][:, None], arrays["periodic"], np.nan)
    aperiodic = np.where(validity["aperiodic"][..., None], arrays["aperiodic"], np.nan)
    # The legacy vectorizer uses the global block as stored, without its mask.
    return np.concatenate(
        (
            periodic,
            aperiodic.reshape(n_rows, 36),
            arrays["artifact_global"],
            mean,
            np.sqrt(variance),
        ),
        axis=1,
    ).astype(np.float32)


def vector_normalizer(x, rows):
    """Fit the historical median and IQR on the specified training rows."""
    x = np.asarray(x)
    rows = np.asarray(rows)
    if x.ndim != 2 or not x.shape[1] or x.dtype.kind not in "fiu":
        raise ValueError("Expected a numeric feature matrix with at least one column")
    if (rows.ndim != 1 or rows.dtype.kind not in "iu" or not len(rows)
            or np.any(rows < 0) or np.any(rows >= len(x)) or len(np.unique(rows)) != len(rows)):
        raise ValueError("Training rows must be distinct, in-bounds integer indices")
    train = x[rows]
    train = np.where(np.isfinite(train), train, np.nan)
    center = np.nan_to_num(np.nanmedian(train, axis=0))
    q25, q75 = np.nanquantile(train, [0.25, 0.75], axis=0)
    scale = np.nan_to_num(q75 - q25, nan=1.0)
    return {
        "center": center.astype(np.float32),
        "scale": np.where(scale > 1e-8, scale, 1.0).astype(np.float32),
    }


def normalized_input(x, normalizer):
    """Fill nonfinite values with the fitted median, standardize, and clip."""
    x = np.asarray(x)
    center = np.asarray(normalizer["center"])
    scale = np.asarray(normalizer["scale"])
    if x.ndim != 2 or x.dtype.kind not in "fiu":
        raise ValueError("Expected a numeric feature matrix")
    if center.shape != (x.shape[1],) or scale.shape != center.shape:
        raise ValueError("Normalizer must have one center and scale per column")
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Normalizer center and scale must be finite, with positive scales")
    standardized = (np.where(np.isfinite(x), x, center) - center) / scale
    return np.clip(standardized, -12, 12).astype(np.float32)


def _load_h5(path):
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Reading fused HDF5 requires h5py. Install the 'eeg' extra.") from exc

    with h5py.File(path, "r") as handle:
        missing = [name for name in _BRANCH_SHAPES if f"features/{name}" not in handle]
        if missing:
            raise ValueError(f"Missing HDF5 feature blocks: {', '.join(missing)}")
        raw = {}
        masks = {}
        for name in _BRANCH_SHAPES:
            dataset = handle[f"features/{name}"]
            if dataset.dtype.kind not in "fiu" or dataset.shape[1:] != _BRANCH_SHAPES[name]:
                raise ValueError(f"Invalid HDF5 feature type or shape: {name}")
            values = dataset[:].astype(np.float32)
            mask = handle.get(f"masks/{name}")
            if mask is not None and mask.shape != values.shape[:-1]:
                raise ValueError(f"Invalid HDF5 mask shape: {name}")
            valid = np.isfinite(values).any(axis=-1)
            if mask is not None:
                valid &= mask[:].astype(bool)
            raw[name] = values
            masks[name] = valid

        metadata = {}
        for key in (*_TEXT_METADATA, *_INTEGER_METADATA):
            if key in handle:
                dataset = handle[key]
                is_text = h5py.check_string_dtype(dataset.dtype) is not None
                metadata[key] = dataset.asstr()[:] if is_text else dataset[:]
    return vector_input(raw, masks), metadata


def _load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        if "x" not in data.files:
            raise ValueError("Packed NPZ must contain the original MLP vector key 'x'")
        vectors = data["x"]
        metadata = {
            key: data[key] for key in (*_TEXT_METADATA, *_INTEGER_METADATA) if key in data.files
        }
    return vectors, metadata


def load_team_vectors(path):
    """Read fused HDF5 or packed NPZ and retain available row identifiers.

    Return (x, metadata). The matrix has shape [N, 705] and dtype float32.
    The metadata dictionary contains original per-row column names. The NPZ
    may additionally carry a scalar source_role, which is not per-row metadata.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        vectors, metadata = _load_h5(path)
    elif suffix == ".npz":
        vectors, metadata = _load_npz(path)
    else:
        raise ValueError("Provide a fused .h5/.hdf5 or a packed .npz file")

    if (vectors.ndim != 2 or vectors.shape[1] != TEAM_VECTOR_DIM or
            not len(vectors) or vectors.dtype.kind not in "fiu"):
        raise ValueError("Expected a nonempty numeric [N, 705] vector matrix")
    vectors = vectors.astype(np.float32, copy=False)
    if np.isinf(vectors).any():
        raise ValueError("Packed features contain infinity")

    for key, column in metadata.items():
        if column.shape != (len(vectors),):
            raise ValueError(f"Metadata shape differs from features: {key}")
        if key in _TEXT_METADATA:
            if column.dtype.kind == "S":
                column = np.char.decode(column, "utf-8")
            elif column.dtype.kind == "O" and all(isinstance(value, str) for value in column):
                column = column.astype(str)
            if column.dtype.kind != "U":
                raise ValueError(f"Metadata must contain text: {key}")
            metadata[key] = column
        elif column.dtype.kind not in "iu":
            raise ValueError(f"Metadata must contain integers: {key}")
    ids = [key for key in ("id", "record_id") if key in metadata]
    if not ids and "sample" in metadata:
        ids = ["sample"]
    if not ids:
        raise ValueError("Team inputs require 'id', 'record_id', or unique 'sample' metadata")
    for key in ids:
        column = metadata[key]
        if (column.dtype.kind == "U" and np.any(column == "")) or len(np.unique(column)) != len(column):
            raise ValueError(f"Empty or duplicate values in record key: {key}")
    return vectors, metadata
