"""
paper_comparison.py - 3-class results comparable to the MEMA dataset paper (optional)

The dataset paper (Liu et al., arXiv:2411.09879, Table II) classifies the three
attention states (attentive / neutral / relaxed) with all 32 channels, training
on each subject's first 9 trials (rounds 1-3) and testing on the last 3 (round 4),
and with leave-one-subject-out for cross-subject.

This script runs the same 3-class task with this project's pipeline:
4 s windows, 11 features per channel, per-subject z-scoring, and the models
model_training.py selected for the binary task (no new tuning):
  subject-dependent: per-subject shrinkage LDA, rounds 1-3 -> round 4
  cross-subject:     soft-voting LogReg + RF + LightGBM, leave-one-subject-out
for two electrode sets: the NeuroPawn 8 and all 29 EEG electrodes.
Accuracy and macro F1 are reported per window and after 5 s causal smoothing
of the class probabilities.

Needs: processed_data/trials (8 channels) and processed_data/trials_all
(python data_processing.py --all-channels).
Output: results/metrics/paper_comparison.csv
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (METRICS_FOLDER, RANDOM_STATE, TRIAL_FOLDER, balanced_lgbm,
                    smoothing_windows, zscore_by_subject)
from feature_extraction import build_feature_table, load_trials

warnings.filterwarnings('ignore')

STATES = ['a', 'n', 'r']
EOG = ['HEOL', 'HEOR']
WINDOW_S = 4

# the MEMA paper's Table II, attention task (accuracy %, F1 %)
PAPER = pd.DataFrame([
    ('SVM', 78.71, 54.82, 62.28, 43.35), ('RF', 80.55, 59.56, 56.03, 45.21),
    ('DT', 82.38, 57.83, 60.12, 41.92), ('EEGNet', 85.12, 59.64, 64.84, 45.13),
    ('ACRNN', 82.53, 57.94, 63.27, 43.69), ('DGCNN', 82.14, 47.59, 63.91, 43.65),
], columns=['method', 'dep_acc', 'dep_f1', 'cross_acc', 'cross_f1'])


def smooth_multiclass(proba, trial_ids, k):
    df = pd.DataFrame(proba)
    return (df.groupby(np.asarray(trial_ids), sort=False)
              .transform(lambda g: g.rolling(k, min_periods=1).mean()).to_numpy())


def scores(y, proba, trial_ids, k):
    out = {}
    for kind, p in (('raw', proba), ('smoothed', smooth_multiclass(proba, trial_ids, k))):
        pred = p.argmax(axis=1)
        out[f'{kind}_acc'] = accuracy_score(y, pred)
        out[f'{kind}_f1'] = f1_score(y, pred, average='macro', labels=[0, 1, 2],
                                     zero_division=0)
    return out


def ensemble():
    return VotingClassifier([
        ('logreg', Pipeline([('scale', StandardScaler()),
                             ('clf', LogisticRegression(C=0.01, class_weight='balanced',
                                                        max_iter=2000))])),
        ('rf', RandomForestClassifier(n_estimators=200, min_samples_leaf=10, max_samples=0.5,
                                      class_weight='balanced_subsample',
                                      random_state=RANDOM_STATE, n_jobs=1)),
        ('lgbm', balanced_lgbm(n_estimators=200, learning_rate=0.05, num_leaves=15)),
    ], voting='soft')


def cross_fold(X, y, subjects, trial_ids, s, k):
    own = subjects == s
    model = ensemble().fit(X[~own], y[~own])
    return scores(y[own], model.predict_proba(X[own]), trial_ids[own], k)


def evaluate(name, X, meta, k):
    y = meta.state.map({s: i for i, s in enumerate(STATES)}).values
    subjects, rounds, trials = meta.subject.values, meta['round'].values, meta.trial_id.values
    order = sorted(np.unique(subjects), key=lambda s: int(s.replace('Subject', '')))

    Zd = zscore_by_subject(X, subjects, rounds <= 3)
    dep = []
    for s in order:
        own = subjects == s
        tr, te = own & (rounds <= 3), own & (rounds == 4)
        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=[1 / 3] * 3)
        lda.fit(Zd[tr], y[tr])
        dep.append(scores(y[te], lda.predict_proba(Zd[te]), trials[te], k))

    Zc = zscore_by_subject(X, subjects, np.ones(len(y), bool))
    cross = Parallel(n_jobs=-1)(delayed(cross_fold)(Zc, y, subjects, trials, s, k)
                                for s in order)
    rows = []
    for scenario, results in (('subject_dependent', dep), ('cross_subject', cross)):
        table = pd.DataFrame(results)
        rows.append({'electrodes': name, 'scenario': scenario,
                     **{f'{c}_mean': table[c].mean() for c in table.columns},
                     **{f'{c}_std': table[c].std() for c in table.columns}})
    return rows


def main():
    start = time.time()
    rows = []
    for name, folder, drop in (('NeuroPawn 8', TRIAL_FOLDER, []),
                               ('All 29 EEG', os.path.join('processed_data', 'trials_all'), EOG)):
        metadata, signals, fs, channels = load_trials(folder)
        X, meta, step_s = build_feature_table(metadata, signals, fs, WINDOW_S, 0.5)
        del signals
        keep = np.array([i * 11 + j for i, c in enumerate(channels) if c not in drop
                         for j in range(11)])
        rows += evaluate(name, X[:, keep], meta, smoothing_windows(step_s))
        print(f'{name} done ({time.time() - start:.0f} s)', flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(METRICS_FOLDER, 'paper_comparison.csv'), index=False)
    pd.set_option('display.width', 200)
    show = result[['electrodes', 'scenario', 'raw_acc_mean', 'raw_f1_mean',
                   'smoothed_acc_mean', 'smoothed_f1_mean']].copy()
    show.iloc[:, 2:] = (show.iloc[:, 2:] * 100).round(1)
    print('\nThis project, 3-class (%):\n' + show.to_string(index=False))
    print('\nMEMA paper Table II, attention (%):\n' + PAPER.to_string(index=False))


if __name__ == '__main__':
    main()
