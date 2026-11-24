"""
feature_extraction.py - step 2: continuous trials -> windowed features

Cuts each cleaned trial into windows (length and overlap are parameters),
computes 11 features per channel (88 total) and saves one feature table per
window length together with per-window subject / trial_id / round / label.

Window length is then chosen with cross-validation on rounds 1-3 only
(round 4 is never touched here). The comparison uses a fixed reference
pipeline: per-subject z-scoring + LightGBM + 5 s causal probability smoothing.
  - subject-dependent CV: per subject, leave-one-round-out over rounds 1-3
  - cross-subject CV:     GroupKFold by subject over rounds 1-3
The chosen window per scenario is written to results/metrics/window_choice.json.
"""

import json
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import signal, stats
from sklearn.model_selection import GroupKFold

from common import (FEATURE_FOLDER, METRICS_FOLDER, SMOOTH_SECONDS, TRIAL_FOLDER,
                    balanced_lgbm, smooth_probabilities, smoothing_windows,
                    zscore_by_subject)

WINDOW_SECONDS = [1, 2, 4]
OVERLAP = 0.5

BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45),
}
EPS = 1e-12


# --- loading and windowing ---

def load_trials():
    """returns metadata DataFrame and {trial_id: (samples, channels) float32 array}"""
    metadata = pd.read_csv(os.path.join(TRIAL_FOLDER, 'metadata.csv'))
    signals = {}
    for subject in metadata.subject.unique():
        with np.load(os.path.join(TRIAL_FOLDER, f'{subject}.npz')) as data:
            for key in data.files:
                signals[f'{subject}_{key}'] = data[key]
    with open(os.path.join(TRIAL_FOLDER, 'info.txt')) as f:
        info = dict(line.strip().split('=', 1) for line in f if '=' in line)
    return metadata, signals, int(info['fs']), info['channels'].split(',')


def cut_windows(trial, window_samples, step_samples):
    """(samples, channels) -> (n_windows, window_samples, channels); drops the partial tail"""
    n = (trial.shape[0] - window_samples) // step_samples + 1
    if n <= 0:
        return np.empty((0, window_samples, trial.shape[1]), dtype=trial.dtype)
    starts = np.arange(n) * step_samples
    index = starts[:, None] + np.arange(window_samples)[None, :]
    return trial[index]


# --- features ---

def feature_names(channels):
    per_channel = [f'logpow_{b}' for b in BANDS] + [
        'log_alpha_beta', 'log_theta_beta', 'log_theta_alpha',
        'log_std', 'skew', 'kurtosis']
    return [f'{ch}_{name}' for ch in channels for name in per_channel]


def extract_features(windows, fs):
    """
    windows: (n, samples, channels) -> (n, channels * 11)

    Per channel: log band power in 5 bands (Welch PSD integrated over the band),
    log alpha/beta, theta/beta, theta/alpha ratios, log standard deviation,
    skewness and kurtosis. The window mean is not used: after the 0.5 Hz
    high-pass it is ~0 and carries no information.
    """
    n_samples = windows.shape[1]
    nperseg = min(fs, n_samples // 2)
    freqs, psd = signal.welch(windows, fs=fs, nperseg=nperseg, axis=1)  # (n, f, ch)
    df = freqs[1] - freqs[0]

    log_power = {}
    for band, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        log_power[band] = np.log(psd[:, mask, :].sum(axis=1) * df + EPS)  # (n, ch)

    per_channel = [log_power[b] for b in BANDS] + [
        log_power['alpha'] - log_power['beta'],
        log_power['theta'] - log_power['beta'],
        log_power['theta'] - log_power['alpha'],
        np.log(windows.std(axis=1) + EPS),
        stats.skew(windows, axis=1),
        stats.kurtosis(windows, axis=1),
    ]
    # (n, ch, 11) -> (n, ch * 11), channel-major to match feature_names()
    return np.stack(per_channel, axis=2).reshape(windows.shape[0], -1).astype(np.float32)


def build_feature_table(metadata, signals, fs, window_s, overlap):
    window_samples = int(round(window_s * fs))
    step_samples = int(round(window_samples * (1 - overlap)))
    blocks, rows = [], []
    for trial in metadata.itertuples(index=False):
        windows = cut_windows(signals[trial.trial_id], window_samples, step_samples)
        if len(windows) == 0:
            continue
        blocks.append(extract_features(windows, fs))
        n = len(windows)
        rows.append(pd.DataFrame({
            'subject': trial.subject,
            'trial_id': trial.trial_id,
            'round': trial.round,
            'state': trial.state,
            'label': trial.label,
            'window_index': np.arange(n),
            't_end_s': (np.arange(n) * step_samples + window_samples) / fs,
        }))
    return np.concatenate(blocks), pd.concat(rows, ignore_index=True), step_samples / fs


def save_feature_table(path, X, meta, names, window_s, step_s):
    np.savez_compressed(
        path, X=X, feature_names=np.array(names),
        subject=meta.subject.values.astype(str), trial_id=meta.trial_id.values.astype(str),
        round=meta['round'].values, state=meta.state.values.astype(str),
        label=meta.label.values, window_index=meta.window_index.values,
        t_end_s=meta.t_end_s.values, window_s=window_s, step_s=step_s)


# --- window-length comparison (rounds 1-3 only) ---

def score_fold(X, y, trial_ids, train_idx, test_idx, k_smooth):
    model = balanced_lgbm(n_jobs=1)
    model.fit(X[train_idx], y[train_idx])
    proba = model.predict_proba(X[test_idx])[:, 1]
    smoothed = smooth_probabilities(proba, trial_ids[test_idx], k_smooth)
    y_test = y[test_idx]
    return np.mean((proba >= 0.5) == y_test), np.mean((smoothed >= 0.5) == y_test)


def compare_window(X, meta, step_s):
    train_mask = meta['round'].values <= 3
    X, meta = X[train_mask], meta[train_mask].reset_index(drop=True)
    y = meta.label.values
    subjects = meta.subject.values
    trial_ids = meta.trial_id.values
    rounds = meta['round'].values
    k_smooth = smoothing_windows(step_s)

    # subject-dependent: per-subject model, leave one training round out.
    # z-scoring stats come from the training rounds of the fold only.
    dep_jobs = []
    for subject in np.unique(subjects):
        idx = np.where(subjects == subject)[0]
        for held_round in (1, 2, 3):
            train_idx = idx[rounds[idx] != held_round]
            test_idx = idx[rounds[idx] == held_round]
            dep_jobs.append((train_idx, test_idx))

    def dep_fold(train_idx, test_idx):
        # only this subject's rows are involved; stats from the training rounds
        idx = np.concatenate([train_idx, test_idx])
        fit_mask = np.r_[np.ones(len(train_idx), bool), np.zeros(len(test_idx), bool)]
        Xz = zscore_by_subject(X[idx], subjects[idx], fit_mask)
        n_tr = len(train_idx)
        return score_fold(Xz, y[idx], trial_ids[idx], np.arange(n_tr),
                          np.arange(n_tr, len(idx)), k_smooth)

    dep = Parallel(n_jobs=-1)(delayed(dep_fold)(tr, te) for tr, te in dep_jobs)

    # cross-subject: GroupKFold over subjects; every subject z-scored with its own stats
    Xz = zscore_by_subject(X, subjects, np.ones(len(y), bool))
    cross = Parallel(n_jobs=5)(
        delayed(score_fold)(Xz, y, trial_ids, tr, te, k_smooth)
        for tr, te in GroupKFold(n_splits=5).split(Xz, y, subjects))

    dep, cross = np.array(dep), np.array(cross)
    return {
        'dep_raw': dep[:, 0].mean(), 'dep_smoothed': dep[:, 1].mean(),
        'cross_raw': cross[:, 0].mean(), 'cross_smoothed': cross[:, 1].mean(),
    }


def main():
    os.makedirs(FEATURE_FOLDER, exist_ok=True)
    os.makedirs(METRICS_FOLDER, exist_ok=True)
    metadata, signals, fs, channels = load_trials()
    names = feature_names(channels)
    print(f'Loaded {len(metadata)} trials ({len(channels)} channels, {fs} Hz)')

    rows = []
    for window_s in WINDOW_SECONDS:
        t0 = time.time()
        X, meta, step_s = build_feature_table(metadata, signals, fs, window_s, OVERLAP)
        save_feature_table(os.path.join(FEATURE_FOLDER, f'features_{window_s}s.npz'),
                           X, meta, names, window_s, step_s)
        print(f'{window_s} s windows: {X.shape[0]} windows x {X.shape[1]} features '
              f'({time.time() - t0:.0f} s)')

        scores = compare_window(X, meta, step_s)
        rows.append({'window_s': window_s, 'step_s': step_s, 'n_windows': X.shape[0],
                     'smooth_windows': smoothing_windows(step_s), **scores})
        print('   CV (rounds 1-3): ' + ', '.join(f'{k}={v:.3f}' for k, v in scores.items()))

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(METRICS_FOLDER, 'window_selection.csv'), index=False)

    choice = {
        'subject_dependent': int(table.loc[table.dep_smoothed.idxmax(), 'window_s']),
        'cross_subject': int(table.loc[table.cross_smoothed.idxmax(), 'window_s']),
    }
    with open(os.path.join(METRICS_FOLDER, 'window_choice.json'), 'w') as f:
        json.dump(choice, f, indent=2)
    print(table.round(3).to_string(index=False))
    print(f'Chosen window length (by smoothed CV accuracy): {choice}')


if __name__ == '__main__':
    main()
