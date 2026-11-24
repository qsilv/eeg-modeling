# EEG Attention State Classification

Classical machine learning for classifying focused vs. unfocused mental states
from EEG, restricted to what a budget 8-channel, 125 Hz headset (NeuroPawn)
can record. The report is [`eeg_attention_report.pdf`](eeg_attention_report.pdf)
(source: `report.tex`).

## Dataset

**XJTU-EEG MEMA** (Multi-label EEG for Mental Attention states),
<https://github.com/XJTU-EEG/MEMA>, paper: <https://arxiv.org/abs/2411.09879>.

- 20 subjects x 12 trials (4 rounds x attentive / neutral / relaxed), ~45 min per subject
- The pipeline reads the `For_graph/` text files (32 columns at 500 Hz)
- 8 channels kept: F3, Fz, F4, C3, C4, Pz, O1, O2
- Labels: `a` (attentive) = focused (1); `n` (neutral) and `r` (relaxed) = unfocused (0)

Put the extracted data at `MEMA Dataset/MEMA/For_graph/SubjectN/SubjectN_<state><round>.txt`.
Neither the raw data nor the processed trials are committed: the dataset has no
stated license, so redistribution is not clearly permitted, and the processed
float32 trials are ~190 MB. Everything under `processed_data/` is regenerated
by the pipeline.

## Pipeline

Run from the project root, in order (`run_pipeline.bat` does all four):

```bash
pip install -r requirements.txt
python data_processing.py      # raw .txt -> cleaned continuous trials
python feature_extraction.py   # trials -> windowed features, window-length CV
python model_training.py       # model selection (grouped CV), training, evaluation
python visualization.py        # report figures
python subject_diagnostics.py  # optional: why accuracy differs between subjects
```

| Step | Output |
|------|--------|
| `data_processing.py` | `processed_data/trials/SubjectN.npz` (one float32 array per trial), `metadata.csv` (subject, trial_id, round, state, label) |
| `feature_extraction.py` | `processed_data/features/features_{1,2,4}s.npz`, `results/metrics/window_selection.csv`, `window_choice.json` |
| `model_training.py` | `results/metrics/summary.csv`, `per_subject_results.csv`, `cv_*.csv`, `chosen_configs.json`, `confusion_matrices.npz`, `test_predictions.csv.gz` |
| `visualization.py` | `results/figures/*.png` |
| `subject_diagnostics.py` | `results/metrics/subject_diagnostics.csv`, `diagnostics_correlations.csv`, `results/figures/subject_diagnostics.png` |

### Methods

- **Preprocessing**: 8 channels, `decimate` 500 -> 125 Hz, 4th-order Butterworth
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
  regression, KNN, random forest or LightGBM (`class_weight='balanced'` where
  supported). The configuration is chosen with grouped CV on training data only:
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
| Subject-dependent (LDA, all features) | 55.1% | 74.0 +/- 9.9% | **77.2 +/- 11.1%** | 77.6 +/- 11.4% | 0.767 |
| Cross-subject (nested CV: RF x11, LightGBM x5, LogReg x4) | 52.4% | 68.0 +/- 9.6% | **72.3 +/- 11.3%** | 72.2 +/- 11.4% | 0.721 |

The previous version reported 66.9% / 58.0%, picked as the best of 20
configurations by test accuracy. Here, no configuration is chosen on the test set.

## Project structure

```
common.py                # shared paths, z-scoring, smoothing helpers
data_processing.py       # raw EEG -> cleaned continuous trials
feature_extraction.py    # trials -> windowed features + window-length CV
model_training.py        # grouped-CV model selection, training, evaluation
visualization.py         # report figures
subject_diagnostics.py   # per-subject signal strength, typicality, per-round accuracy
run_pipeline.bat         # runs all four steps
report.tex               # report source -> eeg_attention_report.pdf
results/figures/         # figures used by the report
results/metrics/         # CSV / JSON result tables
Papers/                  # reference papers
```

## References

1. Aci et al. (2019) - SVM-based EEG attention classification
2. Wang & Kim (2024) - KNN with feature importance for brain attention detection
3. Liu et al. (2025) - MEMA dataset paper
