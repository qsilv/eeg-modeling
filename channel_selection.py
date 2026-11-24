"""
channel_selection.py - which 8 electrodes work best? (optional experiment)

Needs the all-channel trials:  python data_processing.py --all-channels

The MEMA files carry no channel labels. The 8 NeuroPawn positions are known
(F3, Fz, F4, C3, C4, Pz, O1, O2); the other columns are named by index (cNN) and
described by the known electrodes their signals correlate with most.

Protocol (4 s windows, 2 s step, same 11 features per channel, per-subject
z-scoring, 5 s smoothing):
  1. Search, rounds 1-3 only. Greedy forward selection: start empty, repeatedly
     add the channel that most improves the CV score, up to 16 channels.
     Score = mean of subject-dependent CV (per-subject shrinkage LDA,
     leave-one-round-out over rounds 1-3) and cross-subject CV (LDA, 5-fold
     GroupKFold by subject). LDA is used because it is fast and was the
     subject-dependent model chosen in model_training.py.
     For context, 300 random 8-channel sets are scored the same way.
  2. Clean test on round 4, which the search never saw:
       subject-dependent: per-subject LDA trained on rounds 1-3, tested on round 4
       cross-subject:     random forest trained on the other 19 subjects (all
                          rounds), tested on the held-out subject's round 4
     for the NeuroPawn 8, the greedy best 8, and all 31 channels.

--exclude c30,c31 drops columns from the candidate pool (c30 and c31 do not
behave like the other EEG channels: high amplitude, mostly slow activity, and
weak correlation with every other electrode). Outputs then get a suffix.

Outputs: results/metrics/channel_selection_greedy{suffix}.csv, _random, _test,
channel_map.csv, results/figures/channel_selection{suffix}.png
"""

import argparse
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

from common import (FIGURES_FOLDER, METRICS_FOLDER, RANDOM_STATE, smooth_probabilities,
                    smoothing_windows, zscore_by_subject)
from feature_extraction import build_feature_table, feature_names, load_trials

warnings.filterwarnings('ignore')

TRIAL_FOLDER_ALL = os.path.join('processed_data', 'trials_all')
NEUROPAWN = ['F3', 'Fz', 'F4', 'C3', 'C4', 'Pz', 'O1', 'O2']
WINDOW_S = 4
MAX_CHANNELS = 16
N_RANDOM = 300
FEATURES_PER_CHANNEL = 11


def lda():
    return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=[0.5, 0.5])


def smoothed_acc_by_subject(proba, y, subjects, trial_ids, k):
    sm = smooth_probabilities(proba, trial_ids, k)
    return pd.Series((sm >= 0.5) == y).groupby(subjects).mean()


class SearchData:
    """rounds 1-3 only, with the z-scored copies each CV fold needs precomputed.
    Arrays are passed to workers as plain arguments so joblib can memory-map them."""

    def __init__(self, X, meta, k_smooth):
        keep = meta['round'].values <= 3
        y = meta.label.values[keep]
        subjects = meta.subject.values[keep]
        rounds = meta['round'].values[keep]
        X = X[keep]
        # subject-dependent folds: stats from the training rounds of the fold
        dep = np.stack([zscore_by_subject(X, subjects, rounds != h) for h in (1, 2, 3)])
        # cross-subject: each subject normalized with its own (unlabeled) stats
        cross = zscore_by_subject(X, subjects, np.ones(len(y), bool))
        fold_id = np.zeros(len(y), int)
        for f, (_, te) in enumerate(GroupKFold(n_splits=5).split(cross, y, subjects)):
            fold_id[te] = f
        self.args = dict(dep=dep, cross=cross, y=y, rounds=rounds,
                         subjects=subjects.astype(str), trials=meta.trial_id.values[keep].astype(str),
                         fold_id=fold_id, k=k_smooth)


def columns_for(channels, all_channels):
    idx = [all_channels.index(c) for c in channels]
    return np.concatenate([np.arange(i * FEATURES_PER_CHANNEL, (i + 1) * FEATURES_PER_CHANNEL)
                           for i in idx])


def cv_score(cols, dep, cross, y, rounds, subjects, trials, fold_id, k):
    dep_scores = []
    for h in (1, 2, 3):
        Z = dep[h - 1][:, cols]
        for s in np.unique(subjects):
            idx = np.where(subjects == s)[0]
            tr, te = idx[rounds[idx] != h], idx[rounds[idx] == h]
            proba = lda().fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
            sm = smooth_probabilities(proba, trials[te], k)
            dep_scores.append(np.mean((sm >= 0.5) == y[te]))
    Z = cross[:, cols]
    cross_scores = []
    for f in np.unique(fold_id):
        tr, te = fold_id != f, fold_id == f
        proba = lda().fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
        cross_scores.append(smoothed_acc_by_subject(proba, y[te], subjects[te], trials[te], k))
    d, c = np.mean(dep_scores), pd.concat(cross_scores).mean()
    return {'dep_cv': d, 'cross_cv': c, 'score': (d + c) / 2}


def greedy_search(data, channels):
    chosen, rows = [], []
    for step in range(MAX_CHANNELS):
        candidates = [c for c in channels if c not in chosen]
        scores = Parallel(n_jobs=-1)(
            delayed(cv_score)(columns_for(chosen + [c], channels), **data.args)
            for c in candidates)
        best = int(np.argmax([s['score'] for s in scores]))
        chosen.append(candidates[best])
        rows.append({'n_channels': step + 1, 'added': candidates[best],
                     'channels': ','.join(chosen), **scores[best]})
        print(f'  {step + 1:2d}: +{candidates[best]:4s} score {scores[best]["score"]:.3f} '
              f'(dep {scores[best]["dep_cv"]:.3f}, cross {scores[best]["cross_cv"]:.3f})',
              flush=True)
    return pd.DataFrame(rows)


def test_round4(X, meta, cols, k_smooth):
    y, subjects = meta.label.values, meta.subject.values
    trials, rounds = meta.trial_id.values, meta['round'].values
    Xs = X[:, cols]
    order = sorted(np.unique(subjects), key=lambda s: int(s.replace('Subject', '')))

    Zd = zscore_by_subject(Xs, subjects, rounds <= 3)
    Zc = zscore_by_subject(Xs, subjects, np.ones(len(y), bool))
    dep, cross = [], []
    for s in order:
        own = subjects == s
        tr, te = own & (rounds <= 3), own & (rounds == 4)
        proba = lda().fit(Zd[tr], y[tr]).predict_proba(Zd[te])[:, 1]
        sm = smooth_probabilities(proba, trials[te], k_smooth)
        dep.append(np.mean((sm >= 0.5) == y[te]))

        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=10, max_samples=0.5,
                                    class_weight='balanced_subsample',
                                    random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(Zc[~own], y[~own])
        proba = rf.predict_proba(Zc[te])[:, 1]
        sm = smooth_probabilities(proba, trials[te], k_smooth)
        cross.append(np.mean((sm >= 0.5) == y[te]))
    return np.array(dep), np.array(cross)


def channel_map(signals, channels):
    """describe every column by the known electrodes it correlates with most (relaxed trials)"""
    corr = []
    for trial_id, x in signals.items():
        if trial_id.endswith('_r1'):
            corr.append(np.corrcoef((x - np.median(x, axis=0)).T))
    corr = np.mean(corr, axis=0)
    known = [i for i, c in enumerate(channels) if c in NEUROPAWN]
    rows = []
    for i, c in enumerate(channels):
        nearest = [j for j in np.argsort(-corr[i]) if j != i and j in known][:2]
        rows.append({'channel': c, 'known_position': c in NEUROPAWN,
                     'closest_known': ', '.join(f'{channels[j]} (r={corr[i, j]:.2f})'
                                                for j in nearest),
                     'max_corr_any': np.sort(corr[i])[-2]})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exclude', default='', help='comma-separated channels to leave out')
    args = parser.parse_args()
    excluded = [c for c in args.exclude.split(',') if c]
    suffix = '_excl_' + '_'.join(excluded) if excluded else ''

    start = time.time()
    metadata, signals, fs, all_channels = load_trials(TRIAL_FOLDER_ALL)
    X, meta, step_s = build_feature_table(metadata, signals, fs, WINDOW_S, 0.5)
    k = smoothing_windows(step_s)
    cmap = channel_map(signals, all_channels)
    cmap.to_csv(os.path.join(METRICS_FOLDER, 'channel_map.csv'), index=False)
    del signals
    # drop excluded channels' feature columns
    channels = [c for c in all_channels if c not in excluded]
    X = X[:, columns_for(channels, all_channels)]
    print(f'{len(channels)} channels, {X.shape[0]} windows x {X.shape[1]} features '
          f'({time.time() - start:.0f} s)')

    data = SearchData(X, meta, k)
    print('Greedy forward selection (rounds 1-3 CV):')
    greedy = greedy_search(data, channels)
    greedy.to_csv(os.path.join(METRICS_FOLDER, f'channel_selection_greedy{suffix}.csv'), index=False)

    rng = np.random.default_rng(RANDOM_STATE)
    random_sets = [list(rng.choice(channels, 8, replace=False)) for _ in range(N_RANDOM)]
    named_sets = {'NeuroPawn 8': NEUROPAWN,
                  'Greedy best 8': greedy.channels[7].split(','),
                  f'All {len(channels)}': channels}
    to_score = random_sets + list(named_sets.values())
    scores = Parallel(n_jobs=-1)(delayed(cv_score)(columns_for(s, channels), **data.args)
                                 for s in to_score)
    random_df = pd.DataFrame([{'channels': ','.join(s), **sc}
                              for s, sc in zip(random_sets, scores[:N_RANDOM])])
    random_df.to_csv(os.path.join(METRICS_FOLDER, f'channel_selection_random{suffix}.csv'), index=False)
    named_cv = dict(zip(named_sets, scores[N_RANDOM:]))

    test_rows = []
    for name, chans in named_sets.items():
        dep, cross = test_round4(X, meta, columns_for(chans, channels), k)
        cv = named_cv[name]
        pct = (random_df.score < cv['score']).mean()
        test_rows.append({'set': name, 'channels': ','.join(chans),
                          'cv_dep': cv['dep_cv'], 'cv_cross': cv['cross_cv'],
                          'cv_percentile_vs_random8': pct if len(chans) == 8 else np.nan,
                          'test_dep_mean': dep.mean(), 'test_dep_std': dep.std(),
                          'test_cross_mean': cross.mean(), 'test_cross_std': cross.std(),
                          **{f'dep_{i + 1}': v for i, v in enumerate(dep)},
                          **{f'cross_{i + 1}': v for i, v in enumerate(cross)}})
    test = pd.DataFrame(test_rows)
    test.to_csv(os.path.join(METRICS_FOLDER, f'channel_selection_test{suffix}.csv'), index=False)

    pd.set_option('display.width', 200)
    print('\nChannel map:\n' + cmap.to_string(index=False))
    print('\nRandom 8-channel sets, CV score: '
          f'median {random_df.score.median():.3f}, 95th pct {random_df.score.quantile(.95):.3f}')
    print('\nRound-4 test (smoothed accuracy):')
    print(test[['set', 'channels', 'cv_dep', 'cv_cross', 'cv_percentile_vs_random8',
                'test_dep_mean', 'test_cross_mean']].round(3).to_string(index=False))
    # paired comparison NeuroPawn vs greedy, per subject
    a = test.iloc[0]
    b = test.iloc[1]
    for sc in ('dep', 'cross'):
        diff = np.array([b[f'{sc}_{i}'] - a[f'{sc}_{i}'] for i in range(1, 21)])
        print(f'Greedy - NeuroPawn, {sc}: mean {diff.mean():+.3f}, better for {np.sum(diff > 0)}'
              f'/20 subjects, worse for {np.sum(diff < 0)}')
    draw(greedy, random_df, named_cv, suffix, f'All {len(channels)}')
    print(f'Total {time.time() - start:.0f} s')


def draw(greedy, random_df, named_cv, suffix, all_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(greedy.n_channels, greedy.dep_cv, 'o-', label='Subject-dependent CV')
    ax.plot(greedy.n_channels, greedy.cross_cv, 'o-', label='Cross-subject CV')
    for name, color in (('NeuroPawn 8', 'k'), (all_name, 'gray')):
        ax.axhline(named_cv[name]['score'], color=color, ls=':',
                   label=f'{name} (mean of both CV scores)')
    for r in greedy.itertuples():
        ax.annotate(r.added, (r.n_channels, r.score), fontsize=8, xytext=(0, -12),
                    textcoords='offset points', ha='center')
    ax.plot(greedy.n_channels, greedy.score, 'k-', alpha=0.3, label='Mean (selection score)')
    ax.set_xlabel('Number of channels (greedy forward selection)')
    ax.set_ylabel('CV accuracy, rounds 1-3 (smoothed)')
    ax.set_title('Greedy electrode selection')
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(random_df.score, bins=30, color='#9ecae1')
    for name, color in (('NeuroPawn 8', 'k'), ('Greedy best 8', '#d62728')):
        ax.axvline(named_cv[name]['score'], color=color, lw=2, label=name)
    ax.set_xlabel('CV score (mean of dependent and cross-subject)')
    ax.set_ylabel('Random 8-channel sets')
    ax.set_title(f'{len(random_df)} random 8-electrode layouts')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_FOLDER, f'channel_selection{suffix}.png'), dpi=300,
                bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
