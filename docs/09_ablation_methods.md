# Methods: 2 × 2 Feature Ablation and Effect Estimation

**Status:** implementation and execution record, 19 November 2026. The predeclared 2 × 2 matrix, effect estimands and uncertainty rules below are retained. A36 preparation and all four task experiments are complete: EC, EO, Imagery and SSVEP each have 36 fixed-budget MLP fits (144 total, excluding the pilot). Completion is verified by the run manifests and prediction-derived reports. Results are local under `artifacts/ablation/`, with an English `ABLATION_REPORT.md` and four-task overview. The [LDA Experiment Report](../LDA_EXPERIMENT_REPORT.md) remains a separate historical comparison.

## 1. Objective and scope

We evaluate whether explicit band-power features (P108) and aperiodic spectral parameters (A36) improve closed-set EEG identity recognition beyond a fixed log-covariance representation (S171). The principal question is whether A36 provides incremental predictive value when S171 and P108 are already available.

The experimental factors are the availability of feature blocks to the classifier, not interventions on physiological signal generators. P108 contains contributions from both periodic activity and the aperiodic background. S171 is also a mixed signal statistic. Consequently, this study estimates conditional feature-representation benefits and does not isolate the causal effects of pure periodic or aperiodic neural activity. S is present in every condition, so its contribution cannot be estimated by this four-cell design. Spectral interpretation follows [Donoghue et al. (2020)](https://www.nature.com/articles/s41593-020-00744-x).

## 2. Factorial design

Let p and a be binary indicators for providing P108 and A36, respectively. The first subscript always denotes P, the second A. All four conditions use the same model family, cohort, partitions and evaluation rules.

|  | A absent: a = 0 | A present: a = 1 |
|---|---|---|
| **P absent: p = 0** | **M00:** S171; effective dimension 171; score y00 | **M01:** S171 + A36; effective dimension 207; score y01 |
| **P present: p = 1** | **M10:** S171 + P108; effective dimension 279; score y10 | **M11:** S171 + P108 + A36; effective dimension 315; score y11 |

This is a two-factor, two-level full factorial design. The four combinations permit estimation of conditional increments, average main effects and an interaction contrast; the design alone supplies neither independent replication nor a valid significance test. See the [NIST factorial-design handbook](https://www.itl.nist.gov/div898/handbook/pri/section3/pri3331.htm).

The current protocol concerns the S-based MLP study. A later raw-EEG CNN study may use the analogous R / R+P / R+A / R+P+A matrix, but constitutes a separate model comparison. Scores from different classifiers or studies must not be combined into one four-cell matrix.

## 3. Matched data and training controls

The implemented scope is the 20 registered identities in the historical same-session development data. Tasks are trained and evaluated separately, without record-weighted cross-task pooling. EC was executed first; all four tasks have now completed their four conditions, three folds and three seeds.

A36 was recovered from the read-only `data_basic` waveform collection. Preparation verifies one-to-one record IDs, original paths, labels, task/session metadata, channel order and frozen feature dictionaries. EC contains 1,072 records and the other tasks 10,232, for 11,304 records in total. No channel fit failed, and no identity-score-based quality exclusions were applied. Fit quality, including negative exponents, remains diagnostic and is not appended to identity inputs. Missing-value handling is fitted only on inner-training rows. The [data contract](05_contracts.md) retains the metadata and missingness requirements; extraction settings and numerical-regression scope are recorded in Section 11.

Outer evaluation retains the historical `within_task_blocked_provisional` validation assignments. For each identity/task, the historical validation-fold membership recovers three contiguous filename blocks, checked against numeric filename order. In outer fold f, inner development uses block (f+1) mod 3 and inner training block (f+2) mod 3, both restricted to the original outer-training records. Inner-training records whose within-identity/task filename rank is within two positions of an inner-development record are removed. Historical outer boundary exclusions are retained. This yields:

| Task | Fold 0 train / dev / evaluation | Fold 1 train / dev / evaluation | Fold 2 train / dev / evaluation |
|---|---:|---:|---:|
| EC | 312 / 318 / 362 | 322 / 312 / 358 | 278 / 362 / 352 |
| EO | 313 / 317 / 360 | 320 / 313 / 357 | 277 / 360 / 353 |
| Imagery | 1,789 / 1,795 / 1,840 | 1,800 / 1,789 / 1,835 | 1,755 / 1,840 / 1,829 |
| SSVEP | 1,171 / 1,181 / 1,226 | 1,186 / 1,171 / 1,221 | 1,141 / 1,226 / 1,211 |

All three roles contain the 20 registered identities. Groups are named by identity, task and recovered filename block; their role sets are disjoint. These are **provisional filename groups**, not verified acquisition Runs, continuous-recording groups or independent windows. No augmentation is applied. Original/derived-record grouping remains a requirement for any later protocol that introduces derived windows.

All arms are retrained from scratch using the same 315 slots ordered [S171, P108, A36]. A36 occupies 18 offsets followed by 18 exponents. The same median imputer and StandardScaler are fitted on inner-training rows only. Disabled P/A slots are set to zero after preprocessing. The four effective feature dimensions in Section 2 do not represent four input-layer sizes.

The model is the repository's SimpleMLP with 315 input slots, hidden widths 256 and 128, Linear–LayerNorm–GELU–Dropout blocks, dropout 0.2 and 20 output classes: **117,140 nominal parameters** in every arm. Original identity labels 1–20 map to target indices 0–19. Equal nominal parameter counts do not imply identical effective capacity or optimization difficulty. The old 705-dimensional deployment model is not restored or reused by this experiment.

Seeds are fixed to 20261102, 20261103 and 20261104. Within each seed/fold, every arm starts from the same parameter state, verified by SHA256, and receives the same shuffled row order. NumPy's SeedSequence([seed, fold]) drives minibatch permutations; Torch is reset to the paired seed before model creation. Training uses AdamW, learning rate 0.0005, weight decay 0.01, batch size 32, cross-entropy loss and gradient-norm clipping at 5.0. There is no learning-rate scheduler, augmentation or deployment ensemble. The maximum budget is 80 epochs.

Checkpoint selection uses the minimum **inner-development NLL**, evaluated from normalized float64 softmax probabilities with sklearn log_loss. Patience is 12 epochs, min_delta is 0.0001 and the minimum stopping epoch is 10; min_delta controls patience, while the saved checkpoint tracks the actual minimum development NLL. Outer evaluation is accessed only after the selected checkpoint is restored. Fitting uses one Torch CPU thread and deterministic algorithms; the run manifest freezes configuration and environment before optimization.

This inner-development design uses fewer fitting records than the historical LDA reproduction. Accordingly, its raw scores must not be directly interpreted as an architecture ranking against published LDA results. Any new classifier comparison requires matched training/development roles. Budget exhaustion and best checkpoints at epoch 80 are reported from the saved execution summaries and curves; no budget change or post-hoc search is introduced by this methods update.

## 4. Outcome definition and aggregation

For each task t, condition (p,a) and seed s, concatenate the predictions from the three outer folds, retaining each evaluation record exactly once. Verify identical ordered (record_id, truth) pairs across the four conditions. Let C = 20, n_c be the number of pooled evaluation records of identity c, and TP_c the number predicted correctly. Define

$$
y_{pa}^{(t,s)}=\operatorname{BA}_{pa}^{(t,s)}
=\frac{1}{C}\sum_{c=1}^{C}\frac{TP_{c,pa}^{(t,s)}}{n_c}.
$$

All registered identities must have evaluation support. Scores are stored on the **0–1 scale**, without chance adjustment. This is macro-average recall, consistent with [scikit-learn's balanced-accuracy definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html). Overall BA is recomputed from pooled predictions, not obtained by averaging fold BA. A single-fold smoke test must be labeled as a pilot and is not a complete three-fold result.

The task and seed indices are suppressed below. Every contrast is first computed within the same task and seed. A difference multiplied by 100 has units of **percentage points (pp)**. Rounding occurs only for display.

## 5. Effect-size definitions

### 5.1 Baseline-relative gains

The baseline-relative quantities are

$$
D_{00}=0,\quad D_{10}=100(y_{10}-y_{00}),\quad
D_{01}=100(y_{01}-y_{00}),\quad D_{11}=100(y_{11}-y_{00}).
$$

Absolute BA differences are the primary effect-size scale: they are unstandardized, directly interpretable performance contrasts. D10 and D01 are gains from adding one block to S. D11 is the **joint gain** of adding both blocks; it must not be attributed exclusively to P or A. Baseline-relative differences do not, on their own, characterize the full factorial effects.

### 5.2 Conditional increments and the primary contrast

| Contrast ID | Definition, in pp | Interpretation | Role |
|---|---|---|---|
| `A_given_P` | 100(y11 − y10) | A increment when S and P are already available | **Primary contrast for each task** |
| `P_without_A` | 100(y10 − y00) | P increment when A is absent | Secondary |
| `A_without_P` | 100(y01 − y00) | A increment when P is absent | Secondary |
| `P_given_A` | 100(y11 − y01) | P increment when A is already available | Secondary |
| `joint_vs_S` | 100(y11 − y00) | Combined-feature gain over S | Secondary |

These are conditional predictive increments, not unique allocations of physiological information. Positive and negative estimates are retained. The primary contrast must not be changed after observing which condition or task improves most.

### 5.3 Average main effects and interaction

Average main effects assign equal weight to the other factor's two levels:

$$
M_P=\frac{\Delta_{P\mid A=0}+\Delta_{P\mid A=1}}{2},\qquad
M_A=\frac{\Delta_{A\mid P=0}+\Delta_{A\mid P=1}}{2},
$$

where each delta is the corresponding pp contrast from Section 5.2. The interaction **contrast** is

$$
I_{PA}=100(y_{11}-y_{10}-y_{01}+y_{00})
=\Delta_{A\mid P=1}-\Delta_{A\mid P=0}
=\Delta_{P\mid A=1}-\Delta_{P\mid A=0}.
$$

I_PA > 0 denotes super-additivity on the BA scale; I_PA < 0 denotes sub-additivity. Neither sign alone establishes physiological synergy, redundancy or statistical significance. Feature overlap, estimation noise, optimization and the 100% accuracy ceiling can affect the contrast. The four simple increments remain visible even when average main effects are reported.

For clarity about coding, the saturated identity y_pa = beta0 + betaP·p + betaA·a + betaPA·p·a has betaP = y10 − y00, betaA = y01 − y00 and betaPA = y11 − y10 − y01 + y00 on the 0–1 scale. With 0/1 coding, betaP and betaA are **not** the average main effects. This four-cell identity is an algebraic description; four cell means do not provide residual degrees of freedom for an ANOVA.

## 6. Supplementary measures

| Measure | Reporting rule | Purpose and limitation |
|---|---|---|
| Accuracy and macro-F1 differences | Candidate minus its stated reference, multiplied by 100 | Complement BA; do not replace the predeclared endpoint |
| Per-identity recall differences | Same identity and evaluation records, in pp, with support | Show heterogeneous gains and regressions |
| Corrected/regressed predictions | Counts for the same paired records, overall and by identity | Make changes auditable; raw counts weight larger identity groups more heavily |
| Relative balanced-error reduction (RBER) | Defined below; secondary only | Express improvement relative to remaining balanced error, with a baseline-sensitive denominator |
| Negative log likelihood, if probabilities are retained | Candidate minus reference; negative is better | Optional probability-quality analysis; logit/target mapping and any calibration must be fixed before evaluation |
| Parameters and extraction/training/inference time | Report alongside performance on stated hardware | Describe complexity; these are not substitutes for a predictive effect estimate |

For a candidate score y_new and its appropriate reference y_ref, define balanced error BER = 1 − BA and

$$
\operatorname{RBER}_{\mathrm{new:ref},\%}
=100\frac{(1-y_{\mathrm{ref}})-(1-y_{\mathrm{new}})}{1-y_{\mathrm{ref}}}
=100\frac{y_{\mathrm{new}}-y_{\mathrm{ref}}}{1-y_{\mathrm{ref}}}.
$$

The numerical result is displayed as a percentage. For the primary A increment, y_ref = y10 and y_new = y11. RBER is computed per seed and then summarized; it is not the ratio of already averaged scores. Negative values are allowed. At y_ref = 1 it is undefined: store null with a reason, without an epsilon replacement. Report the number of defined seeds K_valid/K. If any seed has undefined RBER, mark the all-seed RBER summary unavailable and retain the defined seed-level values rather than silently dropping the undefined seed. Close to 1, report the denominator and raw BA values because the ratio is sensitive. This is a reduction in **class-balanced error**, not necessarily the percentage reduction in the number of misclassified records.

The relative accuracy increase 100(y_new − y_ref)/y_ref may be included as explicitly labeled context, but is not the primary effect size. It answers a different denominator-based question. Cohen's d or Hedges' g are not pre-specified here: a standardization variance must correspond to a justified replication unit, and the SD of three dependent folds is not an appropriate default denominator.

If b_c counts reference-wrong/candidate-correct records of identity c and r_c counts reference-correct/candidate-wrong records, then

$$
\Delta\operatorname{BA}_{\mathrm{pp}}=\frac{100}{C}\sum_c\frac{b_c-r_c}{n_c},
\qquad \Delta\operatorname{Accuracy}_{\mathrm{pp}}=100\frac{\sum_c(b_c-r_c)}{\sum_c n_c}.
$$

Thus, corrected-minus-regressed counts recover the accuracy change, while BA requires identity weighting. A McNemar p-value is a hypothesis-test result, not an effect size; the ordinary record-level test targets unweighted paired correctness and does not automatically handle source clustering or CV training dependence. It is not a default test for the BA endpoint. See [Dietterich (1998)](https://web.engr.oregonstate.edu/~tgd/publications/dietterich-approximate-statistical-tests-nc1998.pdf).

## 7. Training variability and uncertainty

For each task, calculate every contrast separately for each seed **after pooling its folds**. Report the arithmetic mean, sample SD (denominator K−1), minimum and maximum of the K paired contrast estimates. Also retain the four cell scores for each seed. Do not subtract a best-seed score from an average, stack the predictions of K seeds as K times as many records, or silently substitute an ensemble score.

These summaries describe training randomness conditional on the selected data and partitions. Fold scores are descriptive partitions, not independent replications. Three folds × three seeds must not be analyzed as nine independent observations. Ordinary t-based confidence intervals or two-factor ANOVA on those nine scores are not part of this protocol. Overlapping CV training sets invalidate naive independence assumptions; see [Bengio and Grandvalet (2004)](https://www.jmlr.org/papers/v5/grandvalet04a.html) and [Bouthillier et al. (2021)](https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html).

**Default for the present provisional development data:** report effect estimates and descriptive seed/fold/identity variation without inferential confidence intervals, p-values or significance stars. Unknown source-window dependence is not repaired by bootstrapping individual records. Bootstrapping the 20 identities changes the uncertainty target and does not automatically address shared training or within-session dependence.

**Conditional extension for future independent held-out evaluation:** if acquisition/source groups are verified, can reasonably be treated as independent, and are nested within each registered identity, a paired identity-stratified group bootstrap may be used. Each identity must have an adequately justified number of independent groups; at least two is necessary but not sufficient. Sample groups with replacement within each identity, retaining all windows/derived records with multiplicity. Use exactly the same resampling indices for all four arms and all seeds. Recompute BA and the contrasts, then average each contrast over the fixed seeds. With 10,000 resamples and a recorded bootstrap RNG seed, report the 2.5th and 97.5th percentiles, the group definition and group counts. This interval is conditional on the trained models and selected seeds; it does not include retraining or hyperparameter-selection uncertainty. If the grouping assumptions or independent holdout are unavailable, any identity has only one group, group counts are inadequate, or the resampling distribution is degenerate, report the interval as unavailable with a reason. A change to the resampling hierarchy or a formal testing/multiplicity procedure requires a protocol amendment before evaluating outcomes.

Each task has its own estimates. Tasks share enrolled identities and are not independent replications. No overall superiority claim is based on selecting the largest task gain. A positive point estimate indicates direction; it is not a statistical-significance claim. A minimum worthwhile gain, if needed for model acceptance, must be specified from the application and cost requirements before results are examined; this document does not invent a universal pp or standardized-effect cutoff.

## 8. Reporting and acceptance of analysis inputs

Each analysis must retain the task, protocol/cohort/split hashes, feature definitions, input view, model/preprocessing configuration, seed and factor settings. Per-record predictions remain local. Cohort, label and row-order mismatches must fail analysis rather than be fixed by silently dropping rows.

The minimum report contains: (1) four cell BA scores per seed; (2) the five simple/joint contrasts, M_P, M_A and I_PA, with mean/SD/range and units; (3) accuracy, macro-F1 and identity-level changes; (4) paired correction/regression counts and valid RBER summaries; (5) uncertainty scope and the reason for any unavailable interval; and (6) parameter counts, cost and fit-quality diagnostics. Plot labels must distinguish descriptive seed SD from a confidence interval. Negative results and identity regressions are retained. The historical LDA report is a reference document, not an additional neural training replicate.

The primary effect-size summary is **A_given_P in BA percentage points**, reported separately by task. All secondary contrasts remain available to explain dependence on the feature background and the chosen performance scale.

## 9. Illustrative arithmetic, not experimental results

For synthetic scores y00 = 0.90, y10 = 0.93, y01 = 0.92 and y11 = 0.94:

- P_without_A = +3 pp; A_without_P = +2 pp.
- A_given_P = +1 pp; P_given_A = +2 pp; joint_vs_S = +4 pp.
- M_P = +2.5 pp; M_A = +1.5 pp; I_PA = −1 pp.
- RBER for A_given_P = 100 × (0.94 − 0.93) / (1 − 0.93) = 14.29%.

Here +4 pp is the joint gain, not an independent attribution to either feature block. The negative interaction describes sub-additivity on BA; it does not establish physiological redundancy. These values exist only to explain the calculations.

## 10. Execution status and related documents

A36 acceptance, the explicit inner/outer role manifests, the paired seed list and the fixed MLP training configuration have now been implemented. The EC pilot checked the execution path. Complete EC, EO, Imagery and SSVEP development runs each contain four conditions × three folds × three seeds, verified by completed manifests, aligned predictions and recomputed aggregate metrics. Pilot fits remain separate and are not additional independent evidence.

The [ablation workflow](../scripts/ablation/README.md) documents preparation, execution, summarization and curve generation. The [team integration document](08_team_model_integration.md) distinguishes this new feature-level model from the still-unrecovered 705-dimensional team deployment weights. The [general experimental protocol](06_experiments.md) covers later open-set and perturbation studies. Statistical estimands, primary comparisons and the descriptive uncertainty policy in Sections 4–8 have not been changed after fitting.

Real acquisition/source groups and independently labeled sessions remain unresolved. Raw-EEG CNN fusion and GNN studies are not executed by this workflow. If future periodic features are background-corrected peaks, or PSD estimators/views change, the feature definitions and matched controls must be versioned rather than silently relabeled P108.

## 11. Implementation freeze, recovery and provenance

The preparation implementation is `scripts/ablation/prepare.py`; fitting uses `run.py`, effect estimation uses `summarize.py`, and saved learning curves are rendered with `plot_curves.py`, all under `scripts/ablation/`. `overview.py` combines the four accepted task reports without pooling task records, and exports matched secondary accuracy/macro-F1/NLL differences. Each output directory preserves its configuration before execution. The code, feature dictionaries, raw index/waveform sources and split inputs are hashed. Individual predictions, preprocessing parameters and model weights remain in the ignored `artifacts/ablation/` collection.

### A36 recovery settings and estimator distinction

The frozen A36 recovery uses 18 channels at 300 Hz, 1,200 samples per record, the original reference and device-native amplitude units. Welch uses a periodic Hann window, nperseg = nfft = 300, noverlap = 150, constant detrending, density scaling and **median** aggregation. FOOOF 1.1.0 fits 2–40 Hz in fixed mode, with peak_width_limits = [1,8], max_n_peaks = 6, min_peak_height = 0 and peak_threshold = 2. The saved A36 order is offset18 followed by exponent18; this differs from the old HDF5 per-channel [offset,exponent] layout and requires explicit adaptation.

The completed preparation covers **11,304 records and 203,472 channel fits**: EC contributes 19,296 fits, and the other three tasks 184,176. Zero fits failed, and quality rows are retained rather than used as identity features or performance-based exclusion criteria. Acceptance does not establish neural specificity or longitudinal stability.

A current-environment numerical regression against the recovered legacy extractor covered one EC record, with maximum absolute parameter difference 0 across its 36 A36 values. A separate audit refitted the first/middle/last EC raw HDF5 records and one record per other task (108 channel spectra in total), again obtaining maximum absolute difference 0. These checks do **not** prove equality to a historical full A36 matrix, which was not supplied.

P108 retains the historical Welch convention: **mean** aggregation with nperseg = nfft = 600 and noverlap = 300. Its relative-power denominator is 1–40 Hz. Thus, P and A currently also differ in spectral-estimation settings. The ablation evaluates these explicitly defined feature pipelines; it is not a clean physiological periodic-versus-aperiodic decomposition. A matched-PSD comparison would require a separately versioned protocol and controls.

### Frozen files and hashes

The authoritative preparation records are `artifacts/ablation/ec/prepared/acceptance.json`, `extraction_config.json`, `quality.csv` and `splits.csv`, and the corresponding files under `artifacts/ablation/shared/prepared_other_tasks/`. The shared preparation contains EO, Imagery and SSVEP. Execution records for all four tasks are `artifacts/ablation/<task>/full/runs/run_manifest.json`; each manifest includes the feature/split hashes, trainer hash, exact settings, seed list and completion status. Effect-table provenance is under each task's `full/report/provenance.json`.

| Frozen item | SHA256 |
|---|---|
| EC prepared features | `013bc5213b87e52378bdc78d7baeec7d49a10de10868ec728dd95d1b75af0c9c` |
| EC prepared splits | `428eac7ae17be5c5197d12bbdf90ca68ca2078fe7dbe370495e49a901b87a35a` |
| EO/shared other-task prepared features | `6fd12278c5d4dcbdd487567aab127f9d43d8137f6d2925e8d8d8a358779c7f62` |
| EO/shared other-task prepared splits | `a5d9a011e2cc5e5d718b46b1c20b26ad09e3b67c679567f38a6ae21e97ef1bcc` |
| Trainer used by all four tasks | `c4dcc50cbbe7a83eadffeef4b3a18c8849e5f5dfb67657c36f495ddc85513894` |
| A36 preparation implementation | `8464a8fd5dda52b65791835f8ff66cbb0e68c7c00df7dd959763c91fd47e2cfd` |

The run environment records Python 3.13.17, PyTorch 2.15.0+cpu, NumPy 2.5.5 and scikit-learn 1.9.1. Extraction records SciPy 1.18.1, h5py 3.16.0 and FOOOF 1.1.0. These are current-execution records, not a claim that the original deployment environment was reconstructed. Source changes, new budgets or altered selection rules require a new run/configuration rather than rewriting completed artifacts.


## 12. Follow-up diagnostics and matched-role LDA control

This follow-up is specified on 19 November 2026 after inspecting the original MLP development results. It is a development-stage control, not an independent preregistered confirmation. The primary feature contrast and statistical definitions in Sections 4–9 remain unchanged. Original MLP weights, outputs, seeds and the 80-epoch budget remain frozen; the diagnostics do not trigger automatic retraining or outer-score-driven tuning.

### Convergence and error inspection

Convergence diagnostics use only the saved training and inner-development trajectories, selection epochs and frozen maximum budget. They report all runs, the proportion reaching the budget, selected checkpoints at the budget, final versus best development NLL, and descriptive last-ten-epoch changes/slopes. No new threshold declares convergence or statistical significance. Training NLL is measured over changing minibatches with dropout enabled; development NLL uses evaluation mode. Their difference is not a controlled same-checkpoint estimate of overfitting.

Error inspection retains M11 versus M10 as the primary comparison. Within each task and seed, predictions are pooled over the three disjoint evaluation folds, and the same record/label pairs are checked across all four arms. Per-identity recall differences, paired corrections/regressions, persistent errors and confusion pairs are descriptive. Cross-seed means and sample SDs do not multiply the number of independently acquired records. Negative changes remain visible.

### Deterministic LDA control, fixed before fitting

Each task/fold/condition uses exactly the MLP's **inner-training** records. Inner-development and outer-evaluation rows remain excluded from imputation, scaling and LDA fitting. The same median imputer (no missingness indicators) and StandardScaler are reconstructed from inner-training rows and checked against the saved MLP parameters. Transformed features are quantized to float32 exactly as for the MLP, then promoted without additional precision loss to float64 for the LDA calculations. All labeled Train records and the frozen A36 extraction are retained; unlabeled Test is unused.

The fixed classifier uses `LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=uniform20)`, with no development-set hyperparameter search. Inner-development scores may be recorded after fitting, but do not select settings. The primary controlled comparison concerns matched input information and fitting records, not equal model parameter counts or identical optimization.

LDA reads the active feature columns only: 171, 279, 207 and 315 for M00, M10, M01 and M11. It does not receive dummy disabled zero columns. Automatic Ledoit–Wolf covariance shrinkage depends on feature dimension, so appending zero columns can change the covariance estimate on the remaining dimensions. Fixed 315-slot gating remains the original MLP capacity control; it is not imposed on LDA's covariance estimator. This choice is fixed for all arms before fitting, without selecting between representations using evaluation scores.

There are **4 tasks × 3 folds × 4 conditions = 48 deterministic LDA fits**, not three seed repetitions of identical LDA models. Each LDA cell score pools its three nonduplicated folds before computing BA, accuracy, macro-F1 and NLL. All eight BA effects use the definitions above, with a single deterministic estimate and no invented seed SD. The same LDA prediction vector serves as the reference for each of the three matched MLP seeds. Report MLP-minus-LDA differences within task/condition/seed and the difference between their A_given_P estimates; summarize MLP seed variation descriptively. Do not mix LDA and MLP cells into one factorial matrix.

### Completion and interpretation

A reproducible first feature-level stage can be considered documented when all original MLP outputs, convergence/error diagnostics, 48 matched LDA controls, pairing checks, tables, figures and limitations are retained. Budget exhaustion does not establish convergence, and this stage does not complete the broader research program. Comparisons remain conditional on these provisional same-session filename blocks and fixed settings. True acquisition groups, independent labeled sessions, raw-EEG fusion, graph models and open-set evaluation require separate protocols. Any budget extension or feature-estimator change must be recorded as a new development experiment rather than silently replacing the current results.

### Follow-up execution record

The Section 12 control has now completed: 48 deterministic LDA fits, all 144 MLP convergence/error diagnostics, matched predictions and all comparison tables/figures. The 37-test suite passed. The LDA audit found no fitting warnings and exactly reproduced its stored probabilities from coefficients (maximum difference 0). Full local results are under `artifacts/ablation/diagnostics/`, `matched_lda/` and `FOLLOWUP_REPORT.md`. The methods and source files actually used for the LDA run are retained under `matched_lda/runs/frozen/` with hashes in its manifest. This execution record changes no effect definition, MLP checkpoint or training budget.
