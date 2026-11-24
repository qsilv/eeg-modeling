"""
model_training.py - step 3: model selection, training and evaluation

Two scenarios:
  1. subject-dependent: train on rounds 1-3, test on round 4 of the same subject
  2. cross-subject:     leave-one-subject-out (train on 19 subjects, test on all
                        12 trials of the held-out subject)

Everything that is tuned (training pool, feature-selection method, model and
hyperparameters) is chosen by grouped cross-validation on training data only:
  - subject-dependent: leave-one-round-out over rounds 1-3 (3 folds); the same
    configuration is used for every subject.
  - cross-subject:     nested; for every held-out subject, GroupKFold by
    subject over the 19 training subjects picks the configuration.
The selection score is the mean smoothed CV accuracy across subjects.

Features are z-scored per subject with that subject's own statistics (for the
dependent scenario: training rounds only; for the held-out subject in the cross
scenario: all of its windows, without labels - an unlabeled calibration step).

Predicted probabilities are smoothed causally (mean over the past ~5 s, within
a trial). Both raw-window and smoothed metrics are reported.
"""

import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.feature_selection._base import SelectorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (FEATURE_FOLDER, METRICS_FOLDER, RANDOM_STATE, SMOOTH_SECONDS,
                    balanced_lgbm, smooth_probabilities, smoothing_windows,
                    zscore_by_subject)

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

INNER_FOLDS_CROSS = 4


# --- feature selection (all fit on training folds only, inside the Pipeline) ---

class CorrelationSelector(SelectorMixin, BaseEstimator):
    """keep features whose |Pearson r| with the label is >= the mean |r| (LCC)"""

    def fit(self, X, y):
        Xc = X - X.mean(axis=0)
        yc = y - y.mean()
        denom = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum()) + 1e-12
        self.scores_ = np.abs(Xc.T @ yc) / denom
        self.mask_ = self.scores_ >= self.scores_.mean()
        return self

    def _get_support_mask(self):
        return self.mask_


def get_selectors():
    return {
        'none': 'passthrough',
        'anova': SelectKBest(f_classif, k=30),
        'fi': SelectFromModel(
            RandomForestClassifier(n_estimators=50, max_depth=10, max_samples=0.5,
                                   class_weight='balanced_subsample',
                                   random_state=RANDOM_STATE, n_jobs=1),
            threshold='mean'),
        'lcc': CorrelationSelector(),
        'pca': Pipeline([('scale', StandardScaler()),
                         ('pca', PCA(n_components=0.95, random_state=RANDOM_STATE))]),
    }


def get_models():
    """(name, hyperparameter label) -> estimator. class_weight='balanced' where supported."""
    def scaled(model):
        return Pipeline([('scale', StandardScaler()), ('clf', model)])

    return {
        ('LDA', 'shrinkage=auto'): LinearDiscriminantAnalysis(
            solver='lsqr', shrinkage='auto', priors=[0.5, 0.5]),
        ('LogReg', 'C=0.01'): scaled(LogisticRegression(
            C=0.01, class_weight='balanced', max_iter=2000)),
        ('LogReg', 'C=1'): scaled(LogisticRegression(
            C=1.0, class_weight='balanced', max_iter=2000)),
        ('KNN', 'k=15'): scaled(KNeighborsClassifier(n_neighbors=15)),
        ('KNN', 'k=51'): scaled(KNeighborsClassifier(n_neighbors=51)),
        ('RF', 'leaf=10'): RandomForestClassifier(
            n_estimators=200, min_samples_leaf=10, max_samples=0.5,
            class_weight='balanced_subsample', random_state=RANDOM_STATE, n_jobs=1),
        ('LightGBM', '200x15'): balanced_lgbm(n_estimators=200, learning_rate=0.05,
                                              num_leaves=15),
        ('LightGBM', '500x31'): balanced_lgbm(n_estimators=500, learning_rate=0.03,
                                              num_leaves=31, min_child_samples=50,
                                              reg_lambda=1.0),
    }


def config_name(selector, model, params):
    return f'{model}[{params}]+{selector}'


def build_pipeline(selector, model_key):
    return Pipeline([('select', clone_or_pass(get_selectors()[selector])),
                     ('model', clone(get_models()[model_key]))])


def clone_or_pass(step):
    return step if isinstance(step, str) else clone(step)


# --- metrics ---

def window_metrics(y, proba):
    pred = (proba >= 0.5).astype(int)
    return {
        'accuracy': accuracy_score(y, pred),
        'balanced_accuracy': balanced_accuracy_score(y, pred),
        'f1_macro': f1_score(y, pred, average='macro', zero_division=0),
    }


def per_subject_scores(subjects, y, proba, trial_ids, k_smooth):
    """smoothed accuracy per subject (used as the CV selection score)"""
    smoothed = smooth_probabilities(proba, trial_ids, k_smooth)
    correct = (smoothed >= 0.5) == y
    return pd.Series(correct).groupby(subjects).mean()


# --- cross-validation of all configurations on one train/validation split ---

def evaluate_split(X, y, subjects, trial_ids, train_idx, val_idx, k_smooth):
    """
    fits every (selector, model) configuration on train_idx and returns the
    smoothed accuracy per validation subject. Each selector is fit once per
    split and shared by all models - identical to fitting Pipeline(selector, model).
    """
    rows = []
    for sel_name, selector in get_selectors().items():
        if isinstance(selector, str):
            Xtr, Xva = X[train_idx], X[val_idx]
        else:
            selector = clone(selector).fit(X[train_idx], y[train_idx])
            Xtr, Xva = selector.transform(X[train_idx]), selector.transform(X[val_idx])
        for (model_name, params), model in get_models().items():
            model = clone(model).fit(Xtr, y[train_idx])
            proba = model.predict_proba(Xva)[:, 1]
            scores = per_subject_scores(subjects[val_idx], y[val_idx], proba,
                                        trial_ids[val_idx], k_smooth)
            for subject, score in scores.items():
                rows.append({'selector': sel_name, 'model': model_name, 'params': params,
                             'subject': subject, 'cv_smoothed_acc': score})
    return rows


def summarize_cv(rows):
    cv = pd.DataFrame(rows)
    keys = [c for c in ['pool', 'selector', 'model', 'params'] if c in cv.columns]
    # average over folds per subject first, then over subjects
    per_subject = cv.groupby(keys + ['subject']).cv_smoothed_acc.mean().reset_index()
    return (per_subject.groupby(keys).cv_smoothed_acc.agg(['mean', 'std']).reset_index()
            .rename(columns={'mean': 'cv_acc', 'std': 'cv_acc_std'})
            .sort_values('cv_acc', ascending=False, ignore_index=True))


# --- data ---

def load_features(window_s):
    data = np.load(os.path.join(FEATURE_FOLDER, f'features_{window_s}s.npz'))
    meta = pd.DataFrame({k: data[k] for k in
                         ['subject', 'trial_id', 'round', 'state', 'label', 't_end_s']})
    return data['X'], meta, float(data['step_s'])


def subject_order(subjects):
    return sorted(np.unique(subjects), key=lambda s: int(s.replace('Subject', '')))


# --- scenario 1: subject-dependent ---

def run_subject_dependent(window_s):
    X, meta, step_s = load_features(window_s)
    k_smooth = smoothing_windows(step_s)
    y, subjects = meta.label.values, meta.subject.values
    trial_ids, rounds = meta.trial_id.values, meta['round'].values
    print(f'\n=== Subject-dependent ({window_s} s windows, smoothing over {k_smooth} windows) ===')

    # model selection: leave-one-round-out over rounds 1-3. Two training pools:
    #   own    - one model per subject, trained on that subject's data only
    #   pooled - one model trained on the training rounds of all 20 subjects
    #            (each subject z-scored with its own stats); still never sees round 4
    def cv_fold(held_round):
        fit_mask = (rounds <= 3) & (rounds != held_round)
        val_mask = rounds == held_round
        Xz = zscore_by_subject(X, subjects, fit_mask)
        rows = []
        for r in evaluate_split(Xz, y, subjects, trial_ids,
                                np.where(fit_mask)[0], np.where(val_mask)[0], k_smooth):
            rows.append({'pool': 'pooled', **r})
        for subject in subject_order(subjects):
            own = subjects == subject
            for r in evaluate_split(Xz, y, subjects, trial_ids,
                                    np.where(fit_mask & own)[0], np.where(val_mask & own)[0],
                                    k_smooth):
                rows.append({'pool': 'own', **r})
        return rows

    t0 = time.time()
    fold_rows = Parallel(n_jobs=3)(delayed(cv_fold)(r) for r in (1, 2, 3))
    cv = summarize_cv([r for rows in fold_rows for r in rows])
    cv.to_csv(os.path.join(METRICS_FOLDER, 'cv_subject_dependent.csv'), index=False)
    best = cv.iloc[0]
    print(f'CV done in {time.time() - t0:.0f} s. Chosen: pool={best.pool}, '
          f'{config_name(best.selector, best.model, best.params)} '
          f'(CV smoothed acc {best.cv_acc:.3f})')

    # final fit on rounds 1-3, test on round 4
    train_mask, test_mask = rounds <= 3, rounds == 4
    Xz = zscore_by_subject(X, subjects, train_mask)
    proba = np.zeros(len(y))

    def fit_predict(fit_rows, pred_rows):
        pipe = build_pipeline(best.selector, (best.model, best.params))
        pipe.fit(Xz[fit_rows], y[fit_rows])
        return pipe.predict_proba(Xz[pred_rows])[:, 1]

    if best.pool == 'pooled':
        proba[test_mask] = fit_predict(train_mask, test_mask)
    else:
        for subject in subject_order(subjects):
            own = subjects == subject
            proba[test_mask & own] = fit_predict(train_mask & own, test_mask & own)

    # majority-class baseline from each subject's training labels
    baseline = {}
    for subject in subject_order(subjects):
        own = subjects == subject
        majority = int(y[train_mask & own].mean() >= 0.5)
        baseline[subject] = np.mean(y[test_mask & own] == majority)

    test = meta[test_mask].reset_index(drop=True).assign(proba=proba[test_mask])
    choice = {'pool': best.pool, 'selector': best.selector, 'model': best.model,
              'params': best.params, 'cv_acc': float(best.cv_acc)}
    return test, k_smooth, baseline, choice


# --- scenario 2: cross-subject (nested leave-one-subject-out) ---

def cross_outer_fold(X, y, subjects, trial_ids, test_subject, k_smooth):
    train_idx = np.where(subjects != test_subject)[0]
    test_idx = np.where(subjects == test_subject)[0]

    # inner model selection on the 19 training subjects only
    rows = []
    for inner_tr, inner_va in GroupKFold(n_splits=INNER_FOLDS_CROSS).split(
            train_idx, groups=subjects[train_idx]):
        rows += evaluate_split(X, y, subjects, trial_ids,
                               train_idx[inner_tr], train_idx[inner_va], k_smooth)
    cv = summarize_cv(rows)
    best = cv.iloc[0]

    pipe = build_pipeline(best.selector, (best.model, best.params))
    pipe.fit(X[train_idx], y[train_idx])
    proba = pipe.predict_proba(X[test_idx])[:, 1]

    majority = int(y[train_idx].mean() >= 0.5)
    baseline = np.mean(y[test_idx] == majority)
    cv['outer_subject'] = test_subject
    return test_idx, proba, baseline, cv


def run_cross_subject(window_s):
    X, meta, step_s = load_features(window_s)
    k_smooth = smoothing_windows(step_s)
    y, subjects, trial_ids = meta.label.values, meta.subject.values, meta.trial_id.values
    print(f'\n=== Cross-subject ({window_s} s windows, smoothing over {k_smooth} windows) ===')

    # every subject (training and held-out) is z-scored with its own statistics
    Xz = zscore_by_subject(X, subjects, np.ones(len(y), bool))

    t0 = time.time()
    outer = Parallel(n_jobs=-1, verbose=5)(
        delayed(cross_outer_fold)(Xz, y, subjects, trial_ids, s, k_smooth)
        for s in subject_order(subjects))
    print(f'Nested LOSO done in {time.time() - t0:.0f} s')

    proba = np.zeros(len(y))
    baseline, cv_tables, choices = {}, [], []
    for (test_idx, p, base, cv), subject in zip(outer, subject_order(subjects)):
        proba[test_idx] = p
        baseline[subject] = base
        cv_tables.append(cv)
        best = cv.iloc[0]
        choices.append({'subject': subject, 'selector': best.selector, 'model': best.model,
                        'params': best.params, 'inner_cv_acc': best.cv_acc})

    cv_all = pd.concat(cv_tables, ignore_index=True)
    cv_all.to_csv(os.path.join(METRICS_FOLDER, 'cv_cross_subject_inner.csv'), index=False)
    choices = pd.DataFrame(choices)
    choices.to_csv(os.path.join(METRICS_FOLDER, 'cross_subject_choices.csv'), index=False)
    counts = choices.groupby(['model', 'params', 'selector']).size().sort_values(ascending=False)
    print('Configurations chosen across the 20 outer folds:\n' + counts.to_string())

    test = meta.assign(proba=proba)
    return test, k_smooth, baseline, choices


# --- reporting ---

def score_predictions(test, k_smooth, baseline, scenario):
    test = test.copy()
    test['proba_smoothed'] = smooth_probabilities(test.proba.values, test.trial_id.values,
                                                  k_smooth)
    rows, matrices = [], {}
    for subject in subject_order(test.subject.values):
        part = test[test.subject == subject]
        raw = window_metrics(part.label.values, part.proba.values)
        smooth = window_metrics(part.label.values, part.proba_smoothed.values)
        rows.append({'scenario': scenario, 'subject': subject, 'n_windows': len(part),
                     'baseline_acc': baseline[subject],
                     **{f'raw_{k}': v for k, v in raw.items()},
                     **{f'smoothed_{k}': v for k, v in smooth.items()}})
        for kind, col in (('raw', 'proba'), ('smoothed', 'proba_smoothed')):
            matrices[f'{scenario}_{kind}_{subject}'] = confusion_matrix(
                part.label.values, (part[col].values >= 0.5).astype(int), labels=[0, 1])
    return pd.DataFrame(rows), matrices, test


def main():
    os.makedirs(METRICS_FOLDER, exist_ok=True)
    with open(os.path.join(METRICS_FOLDER, 'window_choice.json')) as f:
        window_choice = json.load(f)

    start = time.time()
    dep_test, dep_k, dep_base, dep_choice = run_subject_dependent(
        window_choice['subject_dependent'])
    cross_test, cross_k, cross_base, cross_choices = run_cross_subject(
        window_choice['cross_subject'])

    per_subject, matrices, predictions = [], {}, []
    for scenario, test, k, base in (('subject_dependent', dep_test, dep_k, dep_base),
                                    ('cross_subject', cross_test, cross_k, cross_base)):
        table, mats, preds = score_predictions(test, k, base, scenario)
        per_subject.append(table)
        matrices.update(mats)
        predictions.append(preds.assign(scenario=scenario))

    per_subject = pd.concat(per_subject, ignore_index=True)
    per_subject.to_csv(os.path.join(METRICS_FOLDER, 'per_subject_results.csv'), index=False)
    np.savez_compressed(os.path.join(METRICS_FOLDER, 'confusion_matrices.npz'), **matrices)
    (pd.concat(predictions, ignore_index=True)
       [['scenario', 'subject', 'trial_id', 'round', 'state', 'label', 't_end_s',
         'proba', 'proba_smoothed']]
       .to_csv(os.path.join(METRICS_FOLDER, 'test_predictions.csv.gz'), index=False,
               float_format='%.5f'))

    metric_cols = [c for c in per_subject.columns if c.startswith(('raw_', 'smoothed_'))]
    summary = per_subject.groupby('scenario')[['baseline_acc'] + metric_cols].agg(['mean', 'std'])
    summary.columns = [f'{a}_{b}' for a, b in summary.columns]
    summary = summary.reset_index()
    summary['window_s'] = summary.scenario.map(window_choice)
    summary['smooth_windows'] = summary.scenario.map(
        {'subject_dependent': dep_k, 'cross_subject': cross_k})
    summary.to_csv(os.path.join(METRICS_FOLDER, 'summary.csv'), index=False)

    with open(os.path.join(METRICS_FOLDER, 'chosen_configs.json'), 'w') as f:
        json.dump({'smooth_seconds': SMOOTH_SECONDS,
                   'subject_dependent': dep_choice,
                   'cross_subject': cross_choices.to_dict(orient='records')}, f, indent=2)

    for row in summary.itertuples():
        print(f'\n{row.scenario}: baseline {row.baseline_acc_mean:.3f} | '
              f'raw acc {row.raw_accuracy_mean:.3f} +/- {row.raw_accuracy_std:.3f}, '
              f'bal {row.raw_balanced_accuracy_mean:.3f}, F1 {row.raw_f1_macro_mean:.3f} | '
              f'smoothed acc {row.smoothed_accuracy_mean:.3f} +/- '
              f'{row.smoothed_accuracy_std:.3f}, bal {row.smoothed_balanced_accuracy_mean:.3f}, '
              f'F1 {row.smoothed_f1_macro_mean:.3f}')
    print(f'\nTotal time {time.time() - start:.0f} s')


if __name__ == '__main__':
    main()
