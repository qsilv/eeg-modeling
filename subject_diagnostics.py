"""
subject_diagnostics.py - step 5 (optional): why does accuracy differ between people?

For every subject, computed from the 4 s feature windows and the test predictions:

  signal strength   how different focused and unfocused windows look for this
                    person: mean |Cohen's d| of the 10 most different features
  typicality        correlation between this person's focused-vs-unfocused
                    pattern (d over all 88 features) and the average pattern of
                    the other 19 subjects. Low = their "focus signature" is unusual
  pattern stability correlation between the pattern in rounds 1-3 and in round 4
  round-4 drift     how far round-4 features moved from rounds 1-3, per state,
                    in units of the rounds 1-3 standard deviation
  artifact rate     % of windows whose log signal std is > 3 robust SDs above
                    the subject's own median (bursts of large-amplitude noise)
  muscle share      median share of gamma (30-45 Hz) in total band power
  eye share         median share of delta (0.5-4 Hz) power at frontal electrodes
  per-round acc.    leave-one-round-out accuracy of the subject-dependent
                    pipeline (per-subject shrinkage LDA, 5 s smoothing) when each
                    of the 4 rounds is held out in turn

Outputs: results/metrics/subject_diagnostics.csv, diagnostics_correlations.csv,
results/figures/subject_diagnostics.png
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from common import (FEATURE_FOLDER, FIGURES_FOLDER, METRICS_FOLDER, smooth_probabilities,
                    smoothing_windows, zscore_by_subject)

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
FRONTAL = ['F3', 'Fz', 'F4']


def cohens_d(X, y):
    a, b = X[y == 1], X[y == 0]
    pooled = np.sqrt((a.var(axis=0) * (len(a) - 1) + b.var(axis=0) * (len(b) - 1))
                     / (len(a) + len(b) - 2))
    return (a.mean(axis=0) - b.mean(axis=0)) / np.where(pooled < 1e-8, 1, pooled)


def band_power(X, names, channels, band):
    cols = [names.index(f'{ch}_logpow_{band}') for ch in channels]
    return np.exp(X[:, cols])


def main():
    with open(os.path.join(METRICS_FOLDER, 'window_choice.json')) as f:
        window_s = json.load(f)['subject_dependent']
    data = np.load(os.path.join(FEATURE_FOLDER, f'features_{window_s}s.npz'))
    X, names = data['X'], list(data['feature_names'])
    subjects, y, rounds = data['subject'], data['label'], data['round']
    trial_ids = data['trial_id']
    k_smooth = smoothing_windows(float(data['step_s']))
    channels = list(dict.fromkeys(n.split('_')[0] for n in names))
    order = sorted(np.unique(subjects), key=lambda s: int(s.replace('Subject', '')))

    # focused-vs-unfocused pattern per subject (own z-scoring, all rounds)
    Xz = zscore_by_subject(X, subjects, np.ones(len(y), bool))
    d_all = {s: cohens_d(Xz[subjects == s], y[subjects == s]) for s in order}
    mean_d = np.mean([d_all[s] for s in order], axis=0)

    log_std_cols = [names.index(f'{ch}_log_std') for ch in channels]
    total_power = sum(band_power(X, names, channels, b) for b in BANDS)
    gamma_share = band_power(X, names, channels, 'gamma') / total_power
    frontal_total = sum(band_power(X, names, FRONTAL, b) for b in BANDS)
    delta_share = band_power(X, names, FRONTAL, 'delta') / frontal_total

    rows, round_rows = [], []
    for s in order:
        own = subjects == s
        d = d_all[s]
        others = np.mean([d_all[o] for o in order if o != s], axis=0)

        early, late = own & (rounds <= 3), own & (rounds == 4)
        Xe = zscore_by_subject(X[own], subjects[own], rounds[own] <= 3)
        r_own, y_own = rounds[own], y[own]
        d_early = cohens_d(Xe[r_own <= 3], y_own[r_own <= 3])
        d_late = cohens_d(Xe[r_own == 4], y_own[r_own == 4])
        drift = np.mean([np.abs(Xe[(r_own == 4) & (y_own == c)].mean(axis=0)
                                - Xe[(r_own <= 3) & (y_own == c)].mean(axis=0)).mean()
                         for c in (0, 1)])

        # artifact bursts: windows far above this subject's typical amplitude
        log_std = X[own][:, log_std_cols]
        med = np.median(log_std, axis=0)
        mad = stats.median_abs_deviation(log_std, axis=0, scale='normal')
        artifact = np.mean(((log_std - med) / np.where(mad < 1e-8, 1, mad) > 3).any(axis=1))

        # leave-one-round-out with the chosen subject-dependent pipeline
        for held in (1, 2, 3, 4):
            tr, te = r_own != held, r_own == held
            Xr = zscore_by_subject(X[own], subjects[own], tr)
            model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto',
                                               priors=[0.5, 0.5]).fit(Xr[tr], y_own[tr])
            proba = smooth_probabilities(model.predict_proba(Xr[te])[:, 1],
                                         trial_ids[own][te], k_smooth)
            round_rows.append({'subject': s, 'held_out_round': held,
                               'accuracy': np.mean((proba >= 0.5) == y_own[te])})

        top = np.sort(np.abs(d))[-10:]
        rows.append({
            'subject': s,
            'signal_strength': top.mean(),
            'typicality': np.corrcoef(d, others)[0, 1],
            'pattern_stability': np.corrcoef(d_early, d_late)[0, 1],
            'round4_drift_sd': drift,
            'artifact_rate': artifact,
            'muscle_share': np.median(gamma_share[own]),
            'eye_share': np.median(delta_share[own]),
            'top_feature': names[int(np.argmax(np.abs(d)))],
            'top_feature_d': d[int(np.argmax(np.abs(d)))],
        })

    diag = pd.DataFrame(rows)
    per_round = pd.DataFrame(round_rows)
    wide = per_round.pivot(index='subject', columns='held_out_round', values='accuracy')
    wide.columns = [f'loro_acc_round{c}' for c in wide.columns]
    diag = diag.merge(wide, left_on='subject', right_index=True)

    results = pd.read_csv(os.path.join(METRICS_FOLDER, 'per_subject_results.csv'))
    acc = results.pivot(index='subject', columns='scenario', values='smoothed_accuracy')
    diag = diag.merge(acc.rename(columns={'subject_dependent': 'dep_acc',
                                          'cross_subject': 'cross_acc'}),
                      left_on='subject', right_index=True)
    diag.to_csv(os.path.join(METRICS_FOLDER, 'subject_diagnostics.csv'), index=False)

    # Spearman correlation of each diagnostic with test accuracy (n = 20, indicative only)
    metrics = ['signal_strength', 'typicality', 'pattern_stability', 'round4_drift_sd',
               'artifact_rate', 'muscle_share', 'eye_share']
    corr_rows = []
    for m in metrics:
        for target in ('dep_acc', 'cross_acc'):
            rho, p = stats.spearmanr(diag[m], diag[target])
            corr_rows.append({'diagnostic': m, 'accuracy': target, 'spearman_rho': rho,
                              'p_value': p})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(os.path.join(METRICS_FOLDER, 'diagnostics_correlations.csv'), index=False)

    pd.set_option('display.width', 200)
    show = diag.copy()
    show['subject'] = show.subject.str.replace('Subject', 'S')
    print(show.drop(columns=['top_feature_d']).round(3).to_string(index=False))
    print('\nSpearman correlation with smoothed test accuracy (n=20):')
    print(corr.pivot(index='diagnostic', columns='accuracy', values='spearman_rho').round(2))
    print('\nMean pattern (top 8 features by |mean d| across subjects):')
    top = np.argsort(-np.abs(mean_d))[:8]
    for i in top:
        signs = np.mean([np.sign(d_all[s][i]) == np.sign(mean_d[i]) for s in order])
        print(f'  {names[i]:22s} mean d = {mean_d[i]:+.2f}   same sign in {signs:.0%} of subjects')

    draw(diag)


def draw(diag):
    os.makedirs(FIGURES_FOLDER, exist_ok=True)
    diag = diag.copy()
    diag['n'] = diag.subject.str.replace('Subject', '').astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.5))

    panels = [
        (axes[0], 'signal_strength', 'dep_acc',
         'Signal strength (mean |d|, top 10 features)', 'Subject-dependent accuracy'),
        (axes[1], 'typicality', 'cross_acc',
         'Typicality (corr. with other subjects\' pattern)', 'Cross-subject accuracy'),
    ]
    for ax, xcol, ycol, xlabel, ylabel in panels:
        ax.scatter(diag[xcol], diag[ycol], color='#3182bd', s=40)
        for r in diag.itertuples():
            ax.annotate(str(r.n), (getattr(r, xcol), getattr(r, ycol)), fontsize=9,
                        xytext=(4, 3), textcoords='offset points')
        rho = stats.spearmanr(diag[xcol], diag[ycol])[0]
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'Spearman rho = {rho:.2f}')

    heat = diag.set_index('n')[[f'loro_acc_round{r}' for r in (1, 2, 3, 4)]]
    heat.columns = ['R1', 'R2', 'R3', 'R4']
    sns.heatmap(heat.sort_index(), annot=True, fmt='.0%', cmap='RdYlGn', vmin=0.4, vmax=1.0,
                cbar=False, ax=axes[2], annot_kws={'fontsize': 7})
    axes[2].set_title('Accuracy with each round held out\n(per-subject LDA, smoothed)')
    axes[2].set_xlabel('Held-out round')
    axes[2].set_ylabel('Subject')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_FOLDER, 'subject_diagnostics.png'), dpi=300,
                bbox_inches='tight')
    plt.close(fig)
    print('saved subject_diagnostics.png')


if __name__ == '__main__':
    main()
