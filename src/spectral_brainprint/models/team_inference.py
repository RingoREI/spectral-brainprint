"""Use a frozen three-seed MLP20 deployment without refitting on inference inputs."""

import csv
import json
from pathlib import Path

import numpy as np

from spectral_brainprint.io.team_vectors import load_team_vectors, normalized_input


def _read_member(path, trust_legacy_checkpoint):
    import torch

    from spectral_brainprint.models.mlp import SimpleMLP

    checkpoint = torch.load(path, map_location="cpu", weights_only=not trust_legacy_checkpoint)
    if checkpoint.get("kind") != "mlp":
        raise ValueError(f"Expected an MLP deployment checkpoint: {path.name}")
    config = checkpoint["config"]
    mlp = config["mlp"]
    if mlp["input_dim"] != 705 or list(mlp["hidden_dims"]) != [256, 128]:
        raise ValueError("Checkpoint does not have the historical MLP20 architecture")
    if not np.array_equal(np.asarray(checkpoint["enrolled_labels"]), np.arange(1, 21)):
        raise ValueError("MLP20 checkpoint class order must be identities 1 through 20")
    model = SimpleMLP(20, mlp["input_dim"], mlp["hidden_dims"], mlp["dropout"])
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval(), checkpoint


def predict_mlp20(features, checkpoints, calibration, output, *, device="cpu",
                  template=None, trust_legacy_checkpoint=False):
    """Write a Sample,Label CSV using frozen calibration and exact class order."""
    try:
        import torch
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        raise RuntimeError("Install the project's [deep] extra to run MLP20 inference") from error

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if len(checkpoints) != 3:
        raise ValueError("MLP20 deployment requires three checkpoints")
    paths = [Path(path).resolve(strict=True) for path in checkpoints]
    if len(set(paths)) != 3:
        raise ValueError("The three checkpoints must be distinct files")
    frozen = json.loads(Path(calibration).read_text(encoding="utf-8-sig"))
    threshold = float(frozen["threshold"])
    if not np.isfinite(threshold) or frozen.get("equal_seed_weights") is not True:
        raise ValueError("Expected the frozen equal-weight ensemble calibration")

    members = {}
    baseline_config = None
    for path in paths:
        model, checkpoint = _read_member(path, trust_legacy_checkpoint)
        config = checkpoint["config"]
        if baseline_config is None:
            baseline_config = config
        elif config != baseline_config:
            raise ValueError("Ensemble checkpoints use different configuration versions")
        seed = checkpoint["signature"]["seed"]
        if seed in members:
            raise ValueError("Duplicate training seed in ensemble")
        members[seed] = model, checkpoint["normalizer"]
    seeds = frozen["seeds"]
    if seeds != baseline_config["seeds"] or set(seeds) != set(members):
        raise ValueError("Calibration and model seeds differ")
    if frozen["policy"] != baseline_config["calibration_policy"]:
        raise ValueError("Calibration and model policies differ")

    vectors, meta = load_team_vectors(features)
    samples = meta.get("sample")
    if samples is None or len(set(samples)) != len(vectors) or any(not key for key in samples):
        raise ValueError("Export requires one unique, nonempty sample name per record")
    order = samples.tolist()
    if template is not None:
        with Path(template).open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["Sample", "Label"]:
                raise ValueError("Expected Sample,Label template columns")
            order = [row["Sample"] for row in reader]
        if len(order) != len(samples) or set(order) != set(samples):
            raise ValueError("Template does not match the input sample set")

    batch_size = int(baseline_config["eval_batch_size"])
    if batch_size < 1:
        raise ValueError("Checkpoint batch size must be positive")
    scores = []
    with torch.inference_mode():
        for seed in seeds:
            model, normalizer = members[seed]
            model.to(device)
            x = normalized_input(vectors, normalizer)
            batches = [model(torch.from_numpy(x[start:start + batch_size]).to(device))["logits"]
                       .float().cpu().numpy() for start in range(0, len(x), batch_size)]
            scores.append(np.concatenate(batches))
    mean = np.mean(scores, axis=0)
    if not np.isfinite(mean).all():
        raise ValueError("Ensemble produced nonfinite logits")
    labels = np.where(mean.max(1) >= threshold, mean.argmax(1) + 1, 0)
    lookup = dict(zip(samples, labels.tolist()))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Sample", "Label"])
        writer.writerows((sample, lookup[sample]) for sample in order)
    return {"rows": len(order), "rejected": int((labels == 0).sum()), "threshold": threshold}
