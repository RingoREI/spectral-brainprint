# Fixed-slot P/A ablation

This directory contains the implementation of [Methods 09](../../docs/09_ablation_methods.md). It keeps the historical LDA scripts separate. Real features, checkpoints, predictions and generated figures stay under the Git-ignored `artifacts/ablation/` directory.

| Script | Purpose |
|---|---|
| `prepare.py` | Restore A36 from read-only `data_basic` HDF5; align with frozen S/P; freeze nested provisional filename-block splits |
| `run.py` | Train M00/M10/M01/M11 from matched initializations, with 315 input slots and training-only transforms |
| `summarize.py` | Validate predictions, pool folds per seed, compute all eight BA effects and secondary measures, export tables and matrix/effect figures |
| `plot_curves.py` | Plot a preselected fold/seed's training and inner-development NLL, without consulting outer-evaluation trajectories |
| `overview.py` | Combine the four complete task reports into a task matrix and primary-contrast figure, without pooling task records |

## Setup

Run commands from the repository root in its virtual environment. For a CPU installation:

```powershell
.\.venv\Scripts\python.exe -m pip install "torch>=2.7.1,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[ablation]"
```

The extraction extra pins FOOOF 1.1.0 to recover the team's frozen method, despite the package's deprecation in favor of specparam. Changing the estimator/version is a new extraction protocol. The actual execution environment is recorded separately in `environments/ablation-20261119.txt` and each run manifest.

## Prepare real data and splits

Set `$periodic` to the handoff containing `payload/` and `handoff_manifest.json`. Set `$raw` to the preprocessing directory containing `train_basic.h5` and `unique_index.csv`. These source directories remain read-only. Use a fresh destination:

```powershell
$periodic = 'PATH_TO_PERIODIC_HANDOFF'
$raw = 'PATH_TO_PREPROCESSED_HDF5'
.\.venv\Scripts\python.exe scripts/ablation/prepare.py --package-root $periodic --raw-root $raw --tasks EC --output-dir artifacts/ablation/reproductions/my-run/ec/prepared --workers 4
```

Preparation restores only the frozen `median1s` A36 method: 300 Hz, one-second periodic Hann windows, 50% overlap, median Welch, FOOOF fixed fit over 2–40 Hz, widths 1–8 Hz, six maximum peaks and peak threshold 2. A36 is ordered **offset18 then exponent18** and appended after `[S171, P108]`. Validity and fit-quality metrics are saved separately from identity inputs. No quality-based row deletion is performed, and failed fits are represented as NaN for training-only imputation. Bad alignment or an entirely missing A column stops preparation.

`--resume` is only for an interrupted matching extraction: it validates chunk fingerprints. Do not rerun preparation on an accepted dataset after training; use fresh destinations to preserve the prepared-input hashes. `--tasks EO Imagery SSVEP` can create a shared preparation for those tasks; training still runs separately by task.

The historical three outer validation blocks are recovered exactly from frozen assignments. For outer fold f, use the remaining block `(f+1)%3` as inner development and `(f+2)%3` as inner training, restricting both to the old outer-training rows. Exclude training records within **two filename-order ranks** of an inner-development record of the same identity/task. Actual trial numbers are not necessarily consecutive. These are historical filename blocks, not verified acquisition Runs.

For EC, the resulting train / inner-development / outer-evaluation counts are **312/318/362**, **322/312/358**, and **278/362/352**. These training sets are smaller than those in the published LDA report; absolute model scores must not be interpreted as a matched LDA-versus-MLP comparison.

## Four-cell training

The fixed architecture is SimpleMLP `315 → 256 → 128 → 20`, LayerNorm/GELU/dropout 0.2. Inputs have the same 315 slots for every arm; P/A gates are applied after median imputation and StandardScaler fitted only on inner-training records. Each condition starts from identical weights within a seed and uses paired data-order RNG.

Settings are frozen in `run_manifest.json` before fitting: AdamW, learning rate 0.0005, weight decay 0.01, batch 32, gradient clip 5, maximum 80 epochs, no scheduler or augmentation. Checkpoints minimize actual inner-development NLL. Patience is 12, minimum epochs 10, and min_delta 0.0001 controls patience only. Outer evaluation happens after checkpoint selection. The CPU implementation uses one Torch thread and deterministic algorithms. Default seeds are 20261102, 20261103 and 20261104. This is a fixed-budget development experiment, not a completed hyperparameter search.

An engineering pilot and the complete experiment have separate destinations:

```powershell
$prepared = 'artifacts/ablation/reproductions/my-run/ec/prepared'
.\.venv\Scripts\python.exe scripts/ablation/run.py --features "$prepared/features.npz" --splits "$prepared/splits.csv" --task EC --pilot --fold 0 --seeds 20261102 --output-dir artifacts/ablation/reproductions/my-run/ec/pilot/runs --device cpu
.\.venv\Scripts\python.exe scripts/ablation/run.py --features "$prepared/features.npz" --splits "$prepared/splits.csv" --task EC --output-dir artifacts/ablation/reproductions/my-run/ec/full/runs --device cpu
```

Full runs require three folds and at least three seeds: **36 fits per task**, with the pilot excluded from the effect summary. Do not change settings in response to outer-evaluation scores and silently combine versions. The maximum budget may be reached; selected epochs and training curves must accompany the results.

## Summaries and figures

```powershell
.\.venv\Scripts\python.exe scripts/ablation/summarize.py --input-dir artifacts/ablation/reproductions/my-run/ec/full/runs --output-dir artifacts/ablation/reproductions/my-run/ec/full/report
.\.venv\Scripts\python.exe scripts/ablation/plot_curves.py --input-dir artifacts/ablation/reproductions/my-run/ec/full/runs --output-prefix artifacts/ablation/reproductions/my-run/ec/full/report/training_curves --fold 0 --seed 20261102
```

For a pilot summary, add `--pilot`. Existing run/report destinations are rejected; choose new paths. `--skip-plots` exports tables only.

Reports contain per-seed pooled cell scores, all eight pp effects, RBER with the undefined-denominator rule, per-identity metrics, paired corrections/regressions, execution summaries and training curves. Each report exports the 2×2 matrix and paired effects as 300 dpi PNG and vector PDF; the curve script adds a third figure pair. Variability is descriptive seed variation; no confidence intervals, p-values or significance stars are generated.

After all four task reports exist under `<artifact-root>/<task>/full/report/`, create the cross-task overview in a fresh directory:

```powershell
.\.venv\Scripts\python.exe scripts/ablation/overview.py --artifact-root artifacts/ablation --output-dir artifacts/ablation/overview
```

For another reproduction root, replace both paths. The overview requires complete reports for EC, EO, Imagery and SSVEP; it does not include pilots. `metrics_by_task.csv` stores BA on the 0–1 scale; `effects_by_task.csv` stores BA differences in pp. The PNG/PDF overview plots seed means and individual paired primary effects.

The restored A extractor and historical P extractor use different Welch conventions. This experiment tests the incremental utility of these frozen feature pipelines, not the unique physiological contribution of two perfectly isolated signal components. True source-recording independence, cross-session generalization, open-set rejection, raw-EEG fusion and GNN performance remain untested.

## Local layout

```text
artifacts/ablation/
  ec/
    prepared/             real S/P/A features, extraction chunks, quality, splits
    validation/           one-record old/new A extraction regression
    pilot/{runs,report}/  engineering pilot, excluded from full results
    full/{runs,report}/   three-fold, three-seed analysis
  shared/
    prepared_other_tasks/ common EO/Imagery/SSVEP input, with task-specific splits
  {eo,imagery,ssvep}/
    full/{runs,report}/   separate task training and analysis
  overview/               four-task tables, figure and source hashes
  ABLATION_REPORT.md      English local report of the executed experiment
  README.md               task-level result index
```

The shared directory is data preparation only; it does not mix task-specific fitted models or pool their scores. Existing `artifacts/lda/` results remain unchanged.

## Completed diagnostic review and matched-role LDA control

The original MLP outputs are immutable inputs to these follow-up scripts. Use fresh report destinations. Diagnostics do not change training settings, remove difficult records or replace the primary contrast.

| Script | Purpose |
|---|---|
| `diagnose_convergence.py` | Inspect all train/inner-dev curves, actual minima and budget boundaries, without using outer outcomes |
| `analyze_errors.py` | Inspect paired identity recall, corrections/regressions, confusion pairs and unique-record stability |
| `run_matched_lda.py` | Fit the 48 deterministic LDA controls using exactly the MLP inner-training rows |
| `summarize_matched_lda.py` | Revalidate 48 LDA/144 MLP results and export matched model scores and feature-effect differences |

```powershell
.\.venv\Scripts\python.exe scripts/ablation/diagnose_convergence.py --input-root artifacts/ablation --output-dir artifacts/ablation/review-rerun/convergence
.\.venv\Scripts\python.exe scripts/ablation/analyze_errors.py --artifact-root artifacts/ablation --output-dir artifacts/ablation/review-rerun/errors
.\.venv\Scripts\python.exe scripts/ablation/run_matched_lda.py --artifact-root artifacts/ablation --output-dir artifacts/ablation/review-rerun/lda/runs
.\.venv\Scripts\python.exe scripts/ablation/summarize_matched_lda.py --lda-dir artifacts/ablation/review-rerun/lda/runs --artifact-root artifacts/ablation --output-dir artifacts/ablation/review-rerun/lda/report
```

LDA uses lsqr/automatic shrinkage/uniform priors, with no inner-dev search or refit on development rows. It receives the active 171/279/207/315 columns after the same MLP preprocessing and float32 quantization, promoted to float64 arithmetic. Disabled zero columns are removed because automatic shrinkage depends on feature dimension. Deterministic LDA cells have one score and no fabricated seed SD; each MLP seed compares against that same score.

Current accepted follow-up results are in `artifacts/ablation/diagnostics/{convergence,errors}/`, `artifacts/ablation/matched_lda/{runs,report}/`, and the local English `FOLLOWUP_REPORT.md`. Each diagnosis exports two PNG/PDF figure pairs, CSVs and provenance. The first fixed-budget feature-level stage can close; real source independence, cross-session generalization, raw-EEG CNN fusion and graph-model claims remain future work.

## Source audit and common-PSD sensitivity study

Use the environment in `environments/ablation-20261119.txt` and the completed mixed-PSD inputs under `artifacts/ablation/`. [Methods and results](../../docs/10_source_audit_and_matched_psd.md) describe the comparison. Run from the repository root. Set `$periodic` and `$raw` as in the original preparation instructions and `$competition` to the competition folder. Choose fresh destinations.

```powershell
$python = '.\.venv\Scripts\python.exe'
$mixed = 'artifacts/ablation'
$audit = "$mixed/reproductions/source-review"
$study = "$mixed/reproductions/common-psd"
& $python scripts/ablation/source_audit.py --raw-root $raw --competition-root $competition --package-root $periodic --ablation-root $mixed --output-dir "$audit/provenance"
& $python scripts/ablation/audit_blocks.py --ablation-root $mixed --output-dir "$audit/blocks"

& $python scripts/ablation/prepare_matched_psd.py --repo-root . --raw-root $raw --group ec --output-dir "$study/ec/prepared"
& $python scripts/ablation/prepare_matched_psd.py --repo-root . --raw-root $raw --group other --output-dir "$study/shared/prepared_other_tasks"
& $python scripts/ablation/verify_matched_psd.py --repo-root . --raw-root $raw --ec-prepared "$study/ec/prepared" --other-prepared "$study/shared/prepared_other_tasks" --output-dir "$study/full_verification" --workers 7

foreach ($task in 'EC','EO','Imagery','SSVEP') {
    $prepared = if ($task -eq 'EC') { "$study/ec/prepared" } else { "$study/shared/prepared_other_tasks" }
    $full = "$study/$($task.ToLowerInvariant())/full"
    & $python scripts/ablation/run.py --features "$prepared/features.npz" --splits "$prepared/splits.csv" --task $task --output-dir "$full/runs" --device cpu
    & $python scripts/ablation/summarize.py --input-dir "$full/runs" --output-dir "$full/report"
    & $python scripts/ablation/plot_curves.py --input-dir "$full/runs" --output-prefix "$full/report/training_curves" --fold 0 --seed 20261102
}
& $python scripts/ablation/overview.py --artifact-root $study --output-dir "$study/overview"
& $python scripts/ablation/run_matched_lda.py --artifact-root $study --output-dir "$study/matched_lda/runs"
& $python scripts/ablation/summarize_matched_lda.py --lda-dir "$study/matched_lda/runs" --artifact-root $study --output-dir "$study/matched_lda/report"
& $python scripts/ablation/compare_matched_psd.py --mixed-root $mixed --common-root $study --output-dir "$study/comparison"
```

The local accepted outputs use `artifacts/ablation/source_audit/` and `artifacts/ablation/matched_psd/`. The comparison exports the paired cell scores, all eight paired effects, the primary effect by task, and the M00/M01 reproducibility checks. Seeds describe training variation. The source audit records the evidence available for each delivered trial and each upstream field marked unknown.
