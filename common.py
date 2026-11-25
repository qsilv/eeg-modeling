"""
common.py - shared paths and helpers used by feature_extraction.py and model_training.py
"""

import os

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

TRIAL_FOLDER = os.path.join('processed_data', 'trials')
FEATURE_FOLDER = os.path.join('processed_data', 'features')
METRICS_FOLDER = os.path.join('results', 'metrics')
FIGURES_FOLDER = os.path.join('results', 'figures')

# causal smoothing horizon for predicted probabilities (seconds of past windows)
SMOOTH_SECONDS = 5.0
RANDOM_STATE = 42


def smoothing_windows(step_s, seconds=SMOOTH_SECONDS):
    """number of consecutive window predictions that span `seconds` (at least 1)"""
    return max(1, int(seconds / step_s + 0.5))


def smooth_probabilities(proba, trial_ids, k):
    """
    causal moving average of P(focused) over the current and k-1 previous windows,
    computed separately inside each trial (rows must be in time order per trial).
    The first windows of a trial average over whatever history exists.
    """
    if k <= 1:
        return np.asarray(proba, dtype=float)
    s = pd.Series(np.asarray(proba, dtype=float))
    return (s.groupby(np.asarray(trial_ids), sort=False)
             .transform(lambda g: g.rolling(k, min_periods=1).mean())
             .to_numpy())


def zscore_by_subject(X, subjects, fit_mask):
    """
    z-score every feature separately for each subject, using only that subject's
    rows where fit_mask is True to estimate mean/std. Labels are never used, so
    for a held-out subject this is an unlabeled calibration step.
    """
    Xz = np.empty_like(X, dtype=np.float32)
    for subject in np.unique(subjects):
        rows = subjects == subject
        ref = X[rows & fit_mask]
        mu = ref.mean(axis=0)
        sd = ref.std(axis=0)
        sd[sd < 1e-8] = 1.0
        Xz[rows] = (X[rows] - mu) / sd
    return Xz


def balanced_lgbm(n_jobs=1, **params):
    defaults = dict(n_estimators=200, learning_rate=0.05, num_leaves=15,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                    class_weight='balanced', random_state=RANDOM_STATE,
                    n_jobs=n_jobs, verbose=-1)
    defaults.update(params)
    return LGBMClassifier(**defaults)
