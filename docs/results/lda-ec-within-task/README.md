# EC LDA result assets

The [LDA Experiment Report](../../../LDA_EXPERIMENT_REPORT.md) provides the shared protocol, complete four-task results, interpretation, limitations and reproduction commands. This directory retains EC's aggregate tables, figures and provenance.

## Tables and figures

- [Pooled metrics](metrics_summary.csv): accuracy, balanced accuracy, macro-F1 and error counts.
- [Fold metrics](fold_metrics.csv): training/validation counts and paired gains.
- [Per-identity metrics](per_identity.csv) and [paired errors](paired_errors.csv).
- Three `confusion_*.csv` files: raw confusion counts for each representation.
- [Provenance](provenance.json): input hashes, source result hashes and environment versions.
- [Performance figure (PDF)](lda_performance.pdf), [identity recall (PDF)](lda_identity_recall.pdf), [confusion matrix (PDF)](lda_confusion.pdf).

![Pooled metrics and observed validation folds](lda_performance.png)

![Identity recall and validation support](lda_identity_recall.png)

![Power plus log-covariance confusion matrix](lda_confusion.png)

Local individual predictions are under `artifacts/lda/runs/ec/`; the original generated report is under `artifacts/lda/reports/ec/`. Those local artifacts remain ignored by Git. Results cover the historical same-session, within-task filename-block protocol.
