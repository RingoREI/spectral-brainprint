"""Plot a preselected fold/seed's training and inner-development loss curves."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    records = {}
    for path in args.input_dir.rglob("result*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value["fold"] == args.fold and value["seed"] == args.seed:
            if value["variant"] in records:
                raise ValueError("Duplicate task/variant for selected fold and seed")
            records[value["variant"]] = value
    if set(records) != {"M00", "M10", "M01", "M11"}:
        parser.error("Selected fold/seed must have exactly four factor conditions")
    tasks = {value["task"] for value in records.values()}
    if len(tasks) != 1:
        raise ValueError("Curves mix tasks")
    for suffix in ("png", "pdf"):
        if args.output_prefix.with_suffix("." + suffix).exists():
            parser.error("Figure exists; choose a fresh output prefix")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    style = {"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
             "ps.fonttype": 42, "figure.facecolor": "white", "savefig.facecolor": "white"}
    with plt.rc_context(style):
        figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), sharey=True)
        figure.subplots_adjust(left=.08, right=.97, top=.83, bottom=.15, hspace=.38, wspace=.20)
        for variant, axis, label in zip(("M00", "M01", "M10", "M11"), axes.flat,
                                         ("S", "S + A", "S + P", "S + P + A")):
            value = records[variant]
            curve = value["training_curve"]
            epochs = [row["epoch"] for row in curve]
            axis.plot(epochs, [row["train_nll"] for row in curve], color="#336A9E", label="Train minibatches")
            axis.plot(epochs, [row["development_nll"] for row in curve], color="#CE772B", label="Inner development")
            selected = value["selected_epoch"]
            axis.axvline(selected, color="#737C87", linestyle="--", linewidth=1,
                         label="Selected checkpoint")
            axis.set_title(f"{variant}: {label} | selected epoch {selected}", loc="left", fontsize=10)
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Negative log likelihood")
            axis.set_ylim(bottom=0)
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color="#E5E8EB", linewidth=.6)
            axis.set_axisbelow(True)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(.52, .92), ncol=3, frameon=False)
        task = next(iter(tasks))
        figure.suptitle(f"{task} | Checkpoint selection inside the training partition", y=.99, fontsize=14)
        figure.text(.08, .065, f"Fold {args.fold}, seed {args.seed}; same-session provisional filename blocks.", fontsize=9)
        figure.text(.08, .025, "Training loss includes dropout; development uses eval mode. Outer evaluation is not used for selection.", fontsize=8.5)
        for suffix in ("png", "pdf"):
            figure.savefig(args.output_prefix.with_suffix("." + suffix), dpi=300, bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    main()
