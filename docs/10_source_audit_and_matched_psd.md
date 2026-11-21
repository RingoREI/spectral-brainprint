# Source audit and common-PSD sensitivity study

Frozen on 2026-11-21 before common-PSD model fitting. The earlier mixed-PSD development results informed the question.

## Source and partition audit

Trace every labeled Train record through its delivered MAT file, preprocessing index, HDF5 row, prepared features and model roles. Distinguish delivered trial files from continuous acquisitions and filename blocks from Runs. Mark unavailable parent and timing metadata as unknown. Verify that each evaluation identity occurs in training. Inspect the four original fold-1 development-loss anomalies and the three identities whose recall decreased at every seed. Retain all records and assignments.

## Fixed sensitivity design

Does A36 add balanced accuracy when P and A use the same spectral estimate? Choose the existing A estimator for both blocks. This holds S171 and A36 fixed while changing the spectrum underlying P108.

| Quantity | Specification |
|---|---|
| Input | Same ordered 11,304 labeled Train records, data_basic, 18 original channels, 1,200 samples at 300 Hz |
| Common spectrum | Periodic Hann Welch, length 300, overlap 150, FFT length 300, constant detrend per segment, one-sided density, median across seven segments, 1 Hz bins |
| P108 | Trapezoidal integration of linear PSD with inclusive boundaries at 4–8, 8–13 and 13–30 Hz. 54 log10 powers then 54 ratios to the 1–40 Hz integral. Preserve floor, channel order and dictionary |
| A36 | Fixed FOOOF 1.1.0 fit at 2–40 Hz, widths 1–8 Hz, up to six peaks, minimum height 0, threshold 2. Offset18 then exponent18. Accept cached A after raw-source, estimator, alignment and same-spectrum numerical checks |
| S171 and roles | Original values and exact inner-train, inner-development and outer-evaluation assignments including the rank buffer. Check records, truths, paths and split hashes |
| Four arms | M00 = S, M10 = S+P, M01 = S+A, M11 = S+P+A. Refit all arms |
| Primary model | Methods 09 fixed 315-slot MLP, training-only transformations, paired seeds 20261102/03/04, folds 0/1/2, same ordering, dropout, optimizer, 80 epochs maximum, minimum inner-development NLL checkpoint. Four task-specific matrices, 144 fits |
| Secondary control | Same inner-training rows, active-column LDA with automatic shrinkage and uniform priors as Methods 09 §12. One deterministic fit per fold and cell, 48 fits |

Pool the three disjoint outer folds per task and seed, then compute balanced accuracy as mean recall across the 20 enrolled identities. The primary effect in percentage points is 100 × [BA(M11) − BA(M10)]. Report individual seeds, mean, sample SD, and the paired change in this contrast from the original mixed-PSD study. Compare M10 and M11 scores separately to locate changes in the contrast. Retain the complete matrix and all eight Methods 09 contrasts. Report the LDA contrast separately. Verify the M00/M01 negative controls across versions at record level, including probabilities. Investigate discrepancies before interpreting changes as P effects.

Interpret persistence separately per task by the direction and size of the increment and disclose seed disagreements. Retain negative effects. Keep assignments, training rules, and the analysis fixed. All comparisons reuse development records. The original mean/2-second and new median/1-second estimators differ in several coupled settings. Conclusions concern pipeline sensitivity within this cohort. Independent-session performance, statistical significance and physiological specificity require additional evidence.

## Local outputs

The source map and block diagnostics live under `artifacts/ablation/source_audit/`. The common-PSD fits and paired comparisons live under `artifacts/ablation/matched_psd/`. Per-record outputs remain local.

## Results on the fixed development cohort

The source map aligns 11,304 labeled clips from 20 enrolled identities in session1. All 12 task/fold partitions were reconstructed. Every identity in outer evaluation appears in the corresponding inner training set. Historical MAT field and waveform fingerprints, HDF5 rows and feature labels could be linked for every clip. The original training MAT bytes and the manifest that would identify true recording boundaries were unavailable locally. All upstream Run, continuous-parent and source-window coordinates are marked unknown. A full waveform-hash check and a multichannel suffix–prefix screen allowing a per-channel DC offset found zero matches under the documented criterion (at least 75 samples at an integer shift).

In the original experiment, fold 1 trained on filename block 0 and developed on block 2. Across the four tasks, its median nearest train–development filename-rank gap within an identity was 23–100 ranks, compared with three in the other folds. All four tasks had their largest median within-identity training-to-development centroid displacement on fold 1 for each feature block. Fold 0 had the highest mean outer-evaluation loss for each task. This localizes a distribution difference while the timing and acquisition conditions remain unresolved.

The table shows the change in pooled balanced accuracy when A36 is added to S+P, in percentage points. MLP means and sample SDs summarize three paired seeds. LDA has one deterministic, matched-role estimate per task.

| Task | Original MLP | Common PSD MLP, mean ± seed SD | Positive seeds | Original LDA | Common PSD LDA |
|---|---:|---:|---:|---:|---:|
| EC | +0.950 | +0.849 ± 0.213 | 3/3 | +1.252 | +0.559 |
| EO | +0.387 | +0.080 ± 0.680 | 1/3 | +0.271 | +0.640 |
| Imagery | +0.213 | +0.086 ± 0.269 | 1/3 | +0.229 | +0.288 |
| SSVEP | +0.591 | +0.258 ± 0.305 | 2/3 | +0.666 | +0.537 |

A36 retained a positive mean increment in all four MLP tasks. Each mean decreased when P used the common estimator. EC retained a positive increment at every seed. EO and Imagery had two negative seed effects each, and SSVEP had one. All four deterministic LDA contrasts were positive. Among the originally selected identities, mean SSVEP identity 19 recall still fell by 5.20 pp in the common-PSD MLP comparison. The audit retained every record.

The full local report includes both sides of each contrast, all four cells, paired effects, sentinel checks, and identity follow-up. Reproduction commands and output paths are in the [script guide](../scripts/ablation/README.md). To test recording-independent performance, the required next input is a provider manifest keyed by full trial path or an independently labeled acquisition of the enrolled identities.
