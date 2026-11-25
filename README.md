# EEG Attention State Classification

Classical machine learning for classifying focused vs. unfocused mental states
from EEG, restricted to what a budget 8-channel, 125 Hz headset (NeuroPawn: a helmet
with 8 movable dry electrodes) can record. The report is [`eeg_attention_report.pdf`](eeg_attention_report.pdf)
(source: `report.tex`).

## Dataset

**XJTU-EEG MEMA** (Multi-label EEG for Mental Attention states),
<https://github.com/XJTU-EEG/MEMA>, paper: <https://arxiv.org/abs/2411.09879>.

- 20 subjects x 12 trials (4 rounds x attentive / neutral / relaxed), ~45 min per subject
- The pipeline reads the `For_graph/` text files (32 columns at 500 Hz)
- The NeuroPawn's electrodes are movable, so the layout is a choice. The main pipeline
  keeps F3, Fz, F4, P3, P4, Pz, O1, O2 (chosen in `channel_selection.py`, see below).
  Columns are located by name with the dataset's channel-to-electrode table
  (`COLUMN_ELECTRODES` in `data_processing.py`); column 18 is the CPz reference
  (constant) and columns 31-32 are horizontal EOG (HEOL, HEOR).
- Labels: `a` (attentive) = focused (1); `n` (neutral) and `r` (relaxed) = unfocused (0)

Put the extracted data at `MEMA Dataset/MEMA/For_graph/SubjectN/SubjectN_<state><round>.txt`.
Neither the raw data nor the processed trials are committed: the dataset has no
stated license, so redistribution is not clearly permitted, and the processed
float32 trials are ~190 MB. Everything under `processed_data/` is regenerated
by the pipeline.

## Pipeline

Run from the project root, in order (`run_pipeline.bat` does the first five):

```bash
pip install -r requirements.txt
python data_processing.py      # raw .txt -> cleaned continuous trials
python feature_extraction.py   # trials -> windowed features, window-length CV
python model_training.py       # model selection (grouped CV), training, evaluation
python visualization.py        # report figures
python subject_diagnostics.py  # optional: why accuracy differs between subjects

# optional: all 32 channels -> best 8 electrodes and EOG-only check
python data_processing.py --all-channels
python channel_selection.py
python channel_selection.py --remove-eog   # same search with eye-movement signal regressed out
```

| Step | Output |
|------|--------|
| `data_processing.py` | `processed_data/trials/SubjectN.npz` (one float32 array per trial), `metadata.csv` (subject, trial_id, round, state, label) |
| `feature_extraction.py` | `processed_data/features/features_{1,2,4}s.npz`, `results/metrics/window_selection.csv`, `window_choice.json` |
| `model_training.py` | `results/metrics/summary.csv`, `per_subject_results.csv`, `cv_*.csv`, `chosen_configs.json`, `confusion_matrices.npz`, `test_predictions.csv.gz` |
| `visualization.py` | `results/figures/*.png` |
| `channel_selection.py` | `results/metrics/channel_selection_{greedy,random,test}[_eog_removed].csv`, `results/figures/channel_selection[_eog_removed].png` |
| `subject_diagnostics.py` | `results/metrics/subject_diagnostics.csv`, `diagnostics_correlations.csv`, `results/figures/subject_diagnostics.png` |

### Methods

- **Preprocessing**: 8 electrodes (F3, Fz, F4, P3, P4, Pz, O1, O2), `decimate` 500 -> 125 Hz, 4th-order Butterworth
  band-pass 0.5-45 Hz (`sosfiltfilt`), applied to each trial as one continuous signal.
- **Windowing** (at feature time): 1 s, 2 s and 4 s windows with 50 % overlap are
  compared by cross-validation on rounds 1-3 only; 4 s won in both scenarios.
- **Features** (11 per channel, 88 total): log band power (delta, theta, alpha,
  beta, gamma), log alpha/beta, theta/beta, theta/alpha ratios, log standard
  deviation, skewness, kurtosis.
- **Normalization**: per-subject z-scoring of every feature. The dependent scenario
  uses statistics from the training rounds; in the cross-subject scenario every
  subject, including the held-out one, uses its own statistics (unlabeled calibration).
- **Model selection**: feature selection (none / ANOVA top-30 / RF importance /
  linear correlation / PCA) inside an sklearn `Pipeline` with LDA, logistic
  regression, KNN, random forest, LightGBM (`class_weight='balanced'` where
  supported) or soft-voting ensembles of them. The configuration is chosen with grouped CV on training data only:
  leave-one-round-out over rounds 1-3 for the subject-dependent scenario and
  nested GroupKFold by subject for cross-subject. The test set never picks a configuration.
- **Smoothing**: predicted probabilities are averaged causally over the past ~5 s
  within each trial. Raw-window and smoothed scores are both reported.
- **Evaluation**: subject-dependent (train rounds 1-3, test round 4) and
  leave-one-subject-out; accuracy, balanced accuracy and macro F1, mean +/- std over subjects.

## Results

Mean +/- std across 20 subjects (4 s windows; "smoothed" = 5 s causal probability average).

| Scenario | Baseline | Raw accuracy | Smoothed accuracy | Smoothed balanced acc. | Smoothed macro F1 |
|----------|---------:|-------------:|------------------:|-----------------------:|------------------:|
| Subject-dependent (LightGBM on all subjects' rounds 1-3) | 55.1% | 70.6 +/- 9.7% | **75.1 +/- 10.6%** | 75.6 +/- 10.2% | 0.744 |
| Cross-subject (nested CV: soft-voting ensemble in all 20 folds) | 52.4% | 68.5 +/- 9.6% | **73.0 +/- 11.3%** | 73.0 +/- 11.4% | 0.729 |

**Electrode layout** (`channel_selection.py`): layouts compared with cross-validation
on rounds 1-3 and tested on the untouched round 4 (smoothed accuracy, simpler fixed models):

| Layout | CV calibrated | CV new user | Round 4 calibrated | Round 4 new user | Round 4, eye signal removed |
|--------|------:|------:|------:|------:|------:|
| Attention layout (F3, Fz, F4, C3, C4, Pz, O1, O2) | 69.8% | 64.5% | 70.6% | 71.7% | 70.7% / 72.0% |
| **Final layout (F3, Fz, F4, P3, P4, Pz, O1, O2)** | 71.4% | 69.0% | 73.7% | 73.3% | 73.5% / 72.6% |
| First layout, phases 1-2 (Fz, F3, F4, Cz, C4, P3, T6, O1) | 72.2% | 69.8% | 77.2% | 75.2% | 75.0% / 74.9% |
| Greedy best 8 (T6, FP2, T3, TP8, Cz, P3, Pz, F7) | 77.5% | 76.4% | 81.4% | 79.0% | 76.8% / 76.5% |
| All 29 EEG | 74.6% | 73.7% | 78.7% | 79.3% | 74.0% / 78.8% |
| EOG only (HEOL, HEOR) | 67.5% | 65.5% | 66.0% | 63.3% | - |

The eye-movement channels alone are well above chance, and the highest-scoring layouts
lean on eye, muscle and auditory sites (FP2, F7, T3, T6, TP8): the attentive task is a
lecture video and the relaxed task a scenery video with music, so they differ in more
than attention. The final layout was chosen among research-backed attention sites
(no frontopolar, eye-adjacent or temporal electrodes) by CV on rounds 1-3, and its
results barely change when the eye signal is removed (`--remove-eog`). The NeuroPawn
uses dry electrodes, which work best on hairless sites (the eye/muscle-dominated ones),
so contact quality at P3/P4, Pz and O1/O2 needs checking on the real helmet.

The first version of this project reported 66.9% / 58.0% with the first layout,
picking the best of 20 configurations by test accuracy. Here, no configuration or
layout is chosen on the test set.

## Project structure

```
common.py                # shared paths, z-scoring, smoothing helpers
data_processing.py       # raw EEG -> cleaned continuous trials
feature_extraction.py    # trials -> windowed features + window-length CV
model_training.py        # grouped-CV model selection, training, evaluation
visualization.py         # report figures
subject_diagnostics.py   # per-subject signal strength, typicality, per-round accuracy
channel_selection.py     # greedy search for the best 8 of the 29 EEG electrodes + EOG check
run_pipeline.bat         # runs the main pipeline and diagnostics
report.tex               # report source -> eeg_attention_report.pdf
results/figures/         # figures used by the report
results/metrics/         # CSV / JSON result tables
Papers/                  # reference papers
```

## References

1. Aci et al. (2019) - SVM-based EEG attention classification
2. Wang & Kim (2024) - KNN with feature importance for brain attention detection
3. Liu et al. (2025) - MEMA dataset paper
