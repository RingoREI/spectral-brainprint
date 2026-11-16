"""Training-only preprocessing and the frozen historical LDA configuration."""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_lda(n_classes=20):
    if n_classes < 2:
        raise ValueError("At least two registered classes are required")
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto",
                                   priors=np.full(n_classes, 1.0 / n_classes)),
    )
