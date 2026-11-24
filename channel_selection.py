"""
channel_selection.py - which 8 electrodes work best? (optional experiment)

Needs the all-channel trials:  python data_processing.py --all-channels

Candidates are the 29 EEG electrodes (the CPz reference is constant and the two
horizontal EOG channels HEOL/HEOR are not EEG, so they are not candidates).

Protocol (4 s windows, 2 s step, same 11 features per channel, per-subject
z-scoring, 5 s smoothing):
  1. Search, rounds 1-3 only. Greedy forward selection: start empty, repeatedly
     add the electrode that most improves the CV score, up to 16 electrodes.
     Score = mean of subject-dependent CV (per-subject shrinkage LDA,
     leave-one-round-out over rounds 1-3) and cross-subject CV (LDA, 5-fold
     GroupKFold by subject). LDA is used because it is fast and was the
     subject-dependent model chosen in model_training.py.
     For context, 300 random 8-electrode sets are scored the same way.
  2. Clean test on round 4, which the search never saw:
       subject-dependent: per-subject LDA trained on rounds 1-3, tested on round 4
       cross-subject:     random forest trained on the other 19 subjects (all
                          rounds), tested on the held-out subject's round 4
     for the NeuroPawn 8, the greedy best 8, all 29 EEG electrodes, the 8
     columns an earlier version of this project used by mistake, and the two
     EOG channels alone (how much can eye movements alone tell apart the states?).

Outputs: results/metrics/channel_selection_{greedy,random,test}.csv,
results/figures/channel_selection.png
"""

import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

from common import (FIGURES_FOLDER, METRICS_FOLDER, RANDOM_STATE, smooth_probabilities,
                    smoothing_windows, zscore_by_subject)
from feature_extraction import build_feature_table, load_trials

warnings.filterwarnings('ignore')

TRIAL_FOLDER_ALL = os.path.join('processed_data', 'trials_all')
NEUROPAWN = ['F3', 'Fz', 'F4', 'C3', 'C4', 'Pz', 'O1', 'O2']
# columns 2, 3, 4, 12, 14, 23, 26, 28 - what the first version of the pipeline read
PREVIOUS_MISLABELLED = ['Fz', 'F3', 'F4', 'Cz', 'C4', 'P3', 'T6', 'O1']
EOG = ['HEOL', 'HEOR']
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
                         subjects=subjects.astype(str),
                         trials=meta.trial_id.values[keep].astype(str),
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


def greedy_search(data, candidates, all_channels):
    chosen, rows = [], []
    for step in range(MAX_CHANNELS):
        options = [c for c in candidates if c not in chosen]
        scores = Parallel(n_jobs=-1)(
            delayed(cv_score)(columns_for(chosen + [c], all_channels), **data.args)
            for c in options)
        best = int(np.argmax([s['score'] for s in scores]))
        chosen.append(options[best])
        rows.append({'n_channels': step + 1, 'added': options[best],
                     'channels': ','.join(chosen), **scores[best]})
        print(f'  {step + 1:2d}: +{options[best]:4s} score {scores[best]["score"]:.3f} '
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


def main():
    start = time.time()
    metadata, signals, fs, all_channels = load_trials(TRIAL_FOLDER_ALL)
    X, meta, step_s = build_feature_table(metadata, signals, fs, WINDOW_S, 0.5)
    del signals
    k = smoothing_windows(step_s)
    eeg = [c for c in all_channels if c not in EOG]
    print(f'{len(all_channels)} channels ({len(eeg)} EEG), {X.shape[0]} windows '
          f'({time.time() - start:.0f} s)')

    data = SearchData(X, meta, k)
    print('Greedy forward selection over EEG electrodes (rounds 1-3 CV):')
    greedy = greedy_search(data, eeg, all_channels)
    greedy.to_csv(os.path.join(METRICS_FOLDER, 'channel_selection_greedy.csv'), index=False)

    rng = np.random.default_rng(RANDOM_STATE)
    random_sets = [list(rng.choice(eeg, 8, replace=False)) for _ in range(N_RANDOM)]
    named_sets = {'NeuroPawn 8': NEUROPAWN,
                  'Greedy best 8': greedy.channels[7].split(','),
                  f'All {len(eeg)} EEG': eeg,
                  'Previous (mislabelled) 8': PREVIOUS_MISLABELLED,
                  'EOG only (HEOL, HEOR)': EOG}
    to_score = random_sets + list(named_sets.values())
    scores = Parallel(n_jobs=-1)(delayed(cv_score)(columns_for(s, all_channels), **data.args)
                                 for s in to_score)
    random_df = pd.DataFrame([{'channels': ','.join(s), **sc}
                              for s, sc in zip(random_sets, scores[:N_RANDOM])])
    random_df.to_csv(os.path.join(METRICS_FOLDER, 'channel_selection_random.csv'), index=False)
    named_cv = dict(zip(named_sets, scores[N_RANDOM:]))

    test_rows, per_subject = [], {}
    for name, chans in named_sets.items():
        dep, cross = test_round4(X, meta, columns_for(chans, all_channels), k)
        per_subject[name] = (dep, cross)
        cv = named_cv[name]
        row = {'set': name, 'channels': ','.join(chans),
               'cv_dep': cv['dep_cv'], 'cv_cross': cv['cross_cv'],
               'cv_percentile_vs_random8': ((random_df.score < cv['score']).mean()
                                            if len(chans) == 8 else np.nan),
               'test_dep_mean': dep.mean(), 'test_dep_std': dep.std(),
               'test_cross_mean': cross.mean(), 'test_cross_std': cross.std()}
        if name != 'NeuroPawn 8':
            base_dep, base_cross = per_subject['NeuroPawn 8']
            for sc, a, b in (('dep', base_dep, dep), ('cross', base_cross, cross)):
                diff = b - a
                row[f'{sc}_vs_neuropawn'] = diff.mean()
                row[f'{sc}_n_better'] = int(np.sum(diff > 0))
                row[f'{sc}_wilcoxon_p'] = wilcoxon(diff).pvalue if np.any(diff) else np.nan
        row.update({f'dep_{i + 1}': v for i, v in enumerate(dep)})
        row.update({f'cross_{i + 1}': v for i, v in enumerate(cross)})
        test_rows.append(row)
    test = pd.DataFrame(test_rows)
    test.to_csv(os.path.join(METRICS_FOLDER, 'channel_selection_test.csv'), index=False)

    pd.set_option('display.width', 220)
    print('\nRandom 8-electrode sets, CV score: '
          f'median {random_df.score.median():.3f}, 95th pct {random_df.score.quantile(.95):.3f}')
    print('\nRound-4 test (smoothed accuracy):')
    cols = ['set', 'channels', 'cv_dep', 'cv_cross', 'cv_percentile_vs_random8',
            'test_dep_mean', 'test_cross_mean', 'dep_vs_neuropawn', 'dep_n_better',
            'dep_wilcoxon_p', 'cross_vs_neuropawn', 'cross_n_better', 'cross_wilcoxon_p']
    print(test[cols].round(3).to_string(index=False))
    draw(greedy, random_df, named_cv, f'All {len(eeg)} EEG')
    print(f'Total {time.time() - start:.0f} s')


def draw(greedy, random_df, named_cv, all_name):
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
    ax.set_xlabel('Number of electrodes (greedy forward selection)')
    ax.set_ylabel('CV accuracy, rounds 1-3 (smoothed)')
    ax.set_title('Greedy electrode selection')
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(random_df.score, bins=30, color='#9ecae1')
    for name, color in (('NeuroPawn 8', 'k'), ('Greedy best 8', '#d62728')):
        ax.axvline(named_cv[name]['score'], color=color, lw=2, label=name)
    ax.set_xlabel('CV score (mean of dependent and cross-subject)')
    ax.set_ylabel('Random 8-electrode sets')
    ax.set_title(f'{len(random_df)} random 8-electrode layouts')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_FOLDER, 'channel_selection.png'), dpi=300,
                bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
