# LDA Experiment Report

Technical report · Linear discriminant analysis (LDA) with band power and spatial covariance · 17–18 November 2026

## 1. Scope and data

This baseline compares three feature sets for closed-set EEG identity recognition. The primary comparison adds band power to a spatial log-covariance classifier. The inputs are the frozen Train feature matrices and fold assignments in `periodic_handoff_20261102`.

The cohort contains 11,304 four-second clips from 20 registered identities in one labeled session. Each clip contains 1,200 samples from 18 channels at 300 Hz, in the `data_basic` view with the supplied reference and device-native amplitude units. Each task has its own training and validation records. The counts below include the historical exclusions at block boundaries.

| Task | Validation clips across folds | Fold 0 train / validation | Fold 1 train / validation | Fold 2 train / validation |
|---|---:|---:|---:|---:|
| Eyes closed (EC) | 1,072 | 670 / 362 | 634 / 358 | 680 / 352 |
| Eyes open (EO) | 1,070 | 670 / 360 | 633 / 357 | 677 / 353 |
| Imagery | 5,504 | 3,624 / 1,840 | 3,589 / 1,835 | 3,635 / 1,829 |
| SSVEP | 3,658 | 2,392 / 1,226 | 2,357 / 1,221 | 2,407 / 1,211 |

## 2. Method

| Feature set | Columns | Definition |
|---|---:|---|
| P (`power108`) | 108 | Log₁₀ band power and power relative to 1–40 Hz in each channel and each of three bands (4–8, 8–13 and 13–30 Hz). The source estimator uses mean Welch spectra with 2 s Hann windows and 50% overlap. |
| S (`logcov171`) | 171 | Matrix logarithm of the within-clip covariance after oracle approximating shrinkage (OAS). The upper triangle is vectorized with off-diagonal entries multiplied by √2. |
| S+P (`power_logcov279`) | 279 | Concatenation of the saved power and log-covariance columns. |

We fitted one LDA per task, feature set and fold under `within_task_blocked_provisional`, giving 36 fits. Each validation block was defined by filename order within task and identity. All training and validation folds contained all 20 identities. The loader checked record IDs, full paths and identity labels across the two feature matrices. Within each fold, the three runs used identical ordered validation IDs and labels.

Median imputation with missingness indicators enabled and standardization were fitted only on the training rows. LDA used the least-squares solver (`lsqr`), automatic covariance shrinkage and equal class priors. Each validation clip contributed one prediction per feature set to its task's pooled metrics. Balanced accuracy (BA) is the mean of the 20 identity recalls. The primary effect in percentage points (pp) is 100 × [BA(S+P) − BA(S)], using BA on the 0–1 scale. [Power configuration](configs/features/power_v1.json) · [Covariance configuration](configs/features/spatial_blind_v4.json) · [Classifier implementation](src/spectral_brainprint/models/lda.py)

## 3. Results

**Pooled validation metrics.** P, S and S+P columns show BA in percent. The last two columns give overall accuracy and macro-F1 for the combined model. The CSV includes all three metrics and correct/error counts for every feature set. [Full pooled table](docs/results/lda-within-task-metrics.csv)

| Task | P (%) | S (%) | S+P (%) | Gain over S (pp) | S+P accuracy (%) | S+P macro-F1 (%) |
|---|---:|---:|---:|---:|---:|---:|
| EC | 81.84 | 92.25 | 95.23 | +2.97 | 95.62 | 95.55 |
| EO | 76.17 | 87.97 | 91.33 | +3.36 | 92.15 | 91.70 |
| Imagery | 92.62 | 95.85 | 97.21 | +1.36 | 97.20 | 97.21 |
| SSVEP | 91.96 | 95.97 | 98.27 | +2.31 | 98.39 | 98.35 |

![LDA balanced accuracy by task and feature set, with pooled and individual fold gains from adding power.](docs/results/lda-within-task-overview.png)

**Figure 1.** Diamonds show the S+P gain calculated from pooled validation predictions. Circles show the three individual fold gains. [Vector PDF](docs/results/lda-within-task-overview.pdf)

**Paired changes from S to S+P.** “Corrected” counts clips that changed from a wrong prediction to a correct one. “Regressed” counts the reverse. The last column counts identities by the direction of change in pooled recall.

| Task | BA gain by fold, 0 / 1 / 2 (pp) | Corrected | Regressed | Errors, S → S+P | Identities, higher / equal / lower recall |
|---|---:|---:|---:|---:|---:|
| EC | +4.58 / +1.64 / +2.68 | 40 | 7 | 80 → 47 | 16 / 3 / 1 |
| EO | +2.90 / +2.08 / +5.12 | 47 | 10 | 121 → 84 | 12 / 4 / 4 |
| Imagery | +1.46 / +0.87 / +1.75 | 106 | 30 | 230 → 154 | 15 / 2 / 3 |
| SSVEP | +3.39 / +1.17 / +2.36 | 93 | 11 | 141 → 59 | 13 / 6 / 1 |

![EC recall and validation support for the 20 enrolled identities under each feature set.](docs/results/lda-ec-within-task/lda_identity_recall.png)

**Figure 2.** Per-identity recall and support for EC. One identity fell from 38/40 to 35/40 correct after adding power. Each task directory includes fold scores, paired errors, per-identity metrics, confusion counts and provenance. [EC](docs/results/lda-ec-within-task/README.md) · [EO](docs/results/lda-eo-within-task/README.md) · [Imagery](docs/results/lda-imagery-within-task/README.md) · [SSVEP](docs/results/lda-ssvep-within-task/README.md)

## 4. Interpretation and limits

S+P had the highest pooled BA on all four tasks. Gains over S ranged from 1.36 to 3.36 pp and were positive in all 12 validation folds. Between one and four identities per task nevertheless lost recall. All 36 fits reproduced the historical accuracy, BA and macro-F1 with absolute differences below `1e-12`.

These results describe within-task, same-session recognition on filename blocks. Fold training sets overlap. The continuous parent recordings, acquisition Runs and source window offsets remain unknown. Cross-recording and cross-session performance require additional acquisition information. Band powers include the aperiodic background. The separate P/A ablation estimates the effect of explicit aperiodic parameters and uses smaller inner-training subsets for its matched LDA–MLP comparison. [Source audit](docs/10_source_audit_and_matched_psd.md) · [Ablation Experiment Report](ABLATION_EXPERIMENT_REPORT.md)

## 5. Reproduction

Run the following PowerShell commands from the repository root. The recorded environment used Python 3.13.17, NumPy 2.5.5, SciPy 1.18.1, scikit-learn 1.9.1, threadpoolctl 3.8.0 and Matplotlib 3.11.2. Create a virtual environment once and install the project with its plotting dependencies.

```powershell
python -m venv .venv
$python = '.\.venv\Scripts\python.exe'
& $python -m pip install -e ".[plots]"
```

**Redraw Figure 1 from repository data.** The plotting script reads the pooled CSV and each task's `fold_metrics.csv` from `docs/results/`. It writes PNG and PDF files with the chosen output prefix.

```powershell
& $python scripts/lda/plot_lda_overview.py --results-dir docs/results --output-prefix artifacts/lda/reproductions/overview/lda-overview
```

**Refit the 36 LDA cases from the separate handoff.** Set `$handoff` to the directory containing `handoff_manifest.json` and `payload/`. The runner reads the Train feature matrices and both feature dictionaries from `periodic_power_v1` and `ssvep_covariance_v4`, plus the saved `case_assignments.csv` and reference `fold_metrics.csv` from the latter. Raw EEG is unnecessary at this stage. Using another dataset requires compatible feature columns and ordering, record IDs, labels, splits and reference metrics. Choose fresh run and report directories.

```powershell
$handoff = 'PATH_TO_PERIODIC_HANDOFF'
$destination = 'artifacts/lda/reproductions/four-tasks'
foreach ($task in 'EC','EO','Imagery','SSVEP') {
    $folder = Join-Path $destination $task.ToLowerInvariant()
    & $python scripts/lda/run_lda_ablation.py --package-root $handoff --task $task --output-dir "$folder/runs"
    if ($LASTEXITCODE -ne 0) { throw "Refit or historical check failed for $task" }
    & $python scripts/lda/summarize_lda_results.py --input-dir "$folder/runs" --output-dir "$folder/report"
    if ($LASTEXITCODE -ne 0) { throw "Summary failed for $task" }
}
```

Each task produces nine prediction JSON files and a nine-row `summary.csv` in `runs/`. All nine rows should have `matches_history_at_1e_12=True`. The per-task `report/` contains aggregate CSVs, PNG/PDF figures and an input provenance manifest.
