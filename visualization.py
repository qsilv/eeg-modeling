"""
visualization.py - step 4: figures for the report

  cv_model_comparison.png  CV accuracy (training data only) per model x feature method
  window_selection.png     window-length comparison on rounds 1-3
  per_subject.png          test accuracy per subject, raw vs smoothed
  smoothing_curve.png      test accuracy vs smoothing horizon (sensitivity only;
                           the reported numbers use the pre-set 5 s)
  confusion_matrix.png     aggregated confusion matrices of the final pipelines
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import FIGURES_FOLDER, METRICS_FOLDER, SMOOTH_SECONDS, smooth_probabilities

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context('paper', font_scale=1.3)

SCENARIOS = {'subject_dependent': 'Subject-dependent', 'cross_subject': 'Cross-subject'}
METHOD_ORDER = ['none', 'anova', 'fi', 'lcc', 'pca']
MODEL_ORDER = ['LDA', 'LogReg', 'KNN', 'RF', 'LightGBM', 'Ensemble']


def save(fig, name):
    fig.savefig(os.path.join(FIGURES_FOLDER, name), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {name}')


def best_per_model_method(cv, extra_keys=()):
    """best hyperparameter setting for each (model, selector) pair"""
    keys = list(extra_keys) + ['model', 'selector']
    return cv.sort_values('cv_acc', ascending=False).groupby(keys, as_index=False).first()


def draw_cv_comparison():
    dep = pd.read_csv(os.path.join(METRICS_FOLDER, 'cv_subject_dependent.csv'))
    cross = pd.read_csv(os.path.join(METRICS_FOLDER, 'cv_cross_subject_inner.csv'))
    # cross-subject: average inner-CV score over the 20 outer folds
    cross = (cross.groupby(['model', 'params', 'selector']).cv_acc.mean().reset_index())

    dep_best = best_per_model_method(dep[dep['pool'] == dep.iloc[0]['pool']])
    cross_best = best_per_model_method(cross)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, data, title in ((axes[0], dep_best, f'Subject-dependent ({dep.iloc[0]["pool"]} training pool)'),
                            (axes[1], cross_best, 'Cross-subject (mean of inner CV)')):
        sns.barplot(data=data, x='model', y='cv_acc', hue='selector', order=MODEL_ORDER,
                    hue_order=METHOD_ORDER, palette='viridis', ax=ax)
        ax.set_title(title)
        ax.set_xlabel('')
        ax.set_ylabel('CV accuracy (smoothed, rounds/subjects held out)')
        ax.set_ylim(0.45, max(0.82, data.cv_acc.max() + 0.08))
        ax.axhline(0.5, color='r', ls='--', alpha=0.5)
        ax.legend(title='Feature method', fontsize=9, title_fontsize=10, loc='upper center',
                  ncol=5)
    fig.suptitle('Model selection: cross-validated accuracy on training data only', y=1.02)
    save(fig, 'cv_model_comparison.png')


def draw_window_selection():
    table = pd.read_csv(os.path.join(METRICS_FOLDER, 'window_selection.csv'))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    styles = {'dep_raw': ('Subject-dep., raw', 'C0', '--'),
              'dep_smoothed': ('Subject-dep., smoothed', 'C0', '-'),
              'cross_raw': ('Cross-subject, raw', 'C1', '--'),
              'cross_smoothed': ('Cross-subject, smoothed', 'C1', '-')}
    for col, (label, color, ls) in styles.items():
        ax.plot(table.window_s, table[col], marker='o', color=color, ls=ls, label=label)
    ax.set_xticks(table.window_s)
    ax.set_xlabel('Window length (s), 50% overlap')
    ax.set_ylabel('CV accuracy (rounds 1-3)')
    ax.set_title('Window length comparison (LightGBM reference)')
    ax.legend()
    save(fig, 'window_selection.png')


def draw_per_subject():
    table = pd.read_csv(os.path.join(METRICS_FOLDER, 'per_subject_results.csv'))
    table['n'] = table.subject.str.replace('Subject', '').astype(int)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, (scenario, title) in zip(axes, SCENARIOS.items()):
        part = table[table.scenario == scenario].sort_values('n')
        x = np.arange(len(part))
        ax.bar(x - 0.2, part.raw_accuracy, 0.4, label='Raw window', color='#9ecae1')
        ax.bar(x + 0.2, part.smoothed_accuracy, 0.4, label=f'Smoothed ({SMOOTH_SECONDS:.0f} s)',
               color='#3182bd')
        ax.plot(x, part.baseline_acc, 'k_', markersize=14, mew=2, label='Majority baseline')
        ax.set_xticks(x, part.n)
        ax.set_ylim(0.3, 1.0)
        ax.set_ylabel('Test accuracy')
        ax.set_title(f'{title}: mean {part.smoothed_accuracy.mean():.1%} '
                     f'(raw {part.raw_accuracy.mean():.1%})')
        ax.legend(loc='lower right', fontsize=9, ncol=3)
    axes[1].set_xlabel('Subject')
    fig.tight_layout()
    save(fig, 'per_subject.png')


def draw_smoothing_curve():
    preds = pd.read_csv(os.path.join(METRICS_FOLDER, 'test_predictions.csv.gz'))
    summary = pd.read_csv(os.path.join(METRICS_FOLDER, 'summary.csv'))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (scenario, title) in enumerate(SCENARIOS.items()):
        part = preds[preds.scenario == scenario]
        row = summary[summary.scenario == scenario].iloc[0]
        step = row.window_s / 2
        horizons = np.arange(0, 31, 1)
        accs = []
        for h in horizons:
            k = max(1, int(h / step + 0.5))
            sm = smooth_probabilities(part.proba.values, part.trial_id.values, k)
            per_subj = pd.Series((sm >= 0.5) == part.label.values).groupby(part.subject.values).mean()
            accs.append(per_subj.mean())
        ax.plot(horizons, accs, color=f'C{i}', label=title)
    ax.axvline(SMOOTH_SECONDS, color='gray', ls=':', label=f'Reported setting ({SMOOTH_SECONDS:.0f} s)')
    ax.set_xlabel('Causal smoothing horizon (s)')
    ax.set_ylabel('Mean test accuracy')
    ax.set_title('Sensitivity to smoothing horizon (not used for selection)')
    ax.legend()
    save(fig, 'smoothing_curve.png')


def draw_confusion_matrices():
    data = np.load(os.path.join(METRICS_FOLDER, 'confusion_matrices.npz'))
    with open(os.path.join(METRICS_FOLDER, 'chosen_configs.json')) as f:
        chosen = json.load(f)
    dep = chosen['subject_dependent']
    cross_models = pd.DataFrame(chosen['cross_subject']).model.value_counts()
    titles = {
        'subject_dependent': f'Subject-dependent ({dep["model"]} + {dep["selector"]})',
        'cross_subject': 'Cross-subject (' + ', '.join(
            f'{m} x{n}' for m, n in cross_models.items()) + ')',
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for row, scenario in enumerate(SCENARIOS):
        for col, kind in enumerate(('raw', 'smoothed')):
            total = sum(data[k] for k in data.files if k.startswith(f'{scenario}_{kind}_'))
            pct = total / total.sum(axis=1, keepdims=True)
            ax = axes[row, col]
            sns.heatmap(pct, annot=True, fmt='.1%', cmap='Blues' if row == 0 else 'Oranges',
                        xticklabels=['Unfocused', 'Focused'],
                        yticklabels=['Unfocused', 'Focused'], vmin=0, vmax=1, ax=ax, cbar=False)
            ax.set_title(f'{titles[scenario]}\n{kind} windows', fontsize=11)
            ax.set_ylabel('True state')
            ax.set_xlabel('Predicted state')
    fig.tight_layout()
    save(fig, 'confusion_matrix.png')


# Earlier project phases, recorded from previous runs of this repository:
#   phase 1 - original pipeline (1 s windows, no normalization, best of 20 configs
#             picked by test accuracy), first layout (Fz, F3, F4, Cz, C4, P3, T6, O1)
#   phase 2 - this pipeline (smoothed accuracy), same first layout
# The final phase is read from results/metrics/summary.csv.
EARLIER_PHASES = [
    ('1. Original pipeline\n(first layout)', 0.669, 0.580),
    ('2. Improved method\n(first layout)', 0.772, 0.737),
]


def draw_phase_summary():
    summary = pd.read_csv(os.path.join(METRICS_FOLDER, 'summary.csv')).set_index('scenario')
    phases = EARLIER_PHASES + [(
        '3-4. Final: adjusted\nattention layout',
        summary.loc['subject_dependent', 'smoothed_accuracy_mean'],
        summary.loc['cross_subject', 'smoothed_accuracy_mean'])]
    baselines = (summary.loc['subject_dependent', 'baseline_acc_mean'],
                 summary.loc['cross_subject', 'baseline_acc_mean'])

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(phases))
    for i, (label, color, base) in enumerate((('Calibrated user (subject-dependent)', '#3182bd',
                                               baselines[0]),
                                              ('New user (cross-subject)', '#e6550d',
                                               baselines[1]))):
        values = [p[1 + i] for p in phases]
        bars = ax.bar(x + (i - 0.5) * 0.38, values, 0.38, color=color, label=label)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f'{v:.1%}', ha='center',
                    fontsize=10)
        ax.axhline(base, color=color, ls='--', alpha=0.6, lw=1)
    ax.text(-0.5, 0.46, 'dashed lines: majority-class baselines', fontsize=9, color='gray')
    ax.set_xticks(x, [p[0] for p in phases])
    ax.set_ylim(0.45, 0.85)
    ax.set_ylabel('Test accuracy (mean over 20 subjects)')
    ax.set_title('Accuracy through the phases of the project')
    ax.legend(loc='upper left', fontsize=9)
    save(fig, 'phase_summary.png')


def main():
    os.makedirs(FIGURES_FOLDER, exist_ok=True)
    draw_phase_summary()
    draw_window_selection()
    draw_cv_comparison()
    draw_per_subject()
    draw_smoothing_curve()
    draw_confusion_matrices()


if __name__ == '__main__':
    main()
