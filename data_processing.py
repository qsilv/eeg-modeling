"""
data_processing.py - step 1: raw EEG -> cleaned continuous trials

Reads every trial file in MEMA's For_graph folder, keeps the 8 electrodes of
the chosen layout (the NeuroPawn's electrodes are movable; the layout was picked
with channel_selection.py), decimates 500 Hz -> 125 Hz and band-passes
0.5-45 Hz. Each trial is stored as one continuous float32 signal; windowing
happens later in feature_extraction.py so the window length can be tuned.

Run with --all-channels to keep all 31 non-reference columns instead (29 EEG +
2 EOG, used by channel_selection.py); that output goes to processed_data/trials_all/.

Output (processed_data/trials/):
    SubjectN.npz    one array per trial, keyed by trial id (e.g. "a1")
    metadata.csv    one row per trial: subject, trial_id, round, state, label,
                    n_samples, duration_s
"""

import argparse
import os
import re
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import signal

DATA_FOLDER = os.path.join('MEMA Dataset', 'MEMA', 'For_graph')
OUTPUT_FOLDER = os.path.join('processed_data', 'trials')
OUTPUT_FOLDER_ALL = os.path.join('processed_data', 'trials_all')

ORIGINAL_HZ = 500
TARGET_HZ = 125
DOWNSAMPLE_FACTOR = ORIGINAL_HZ // TARGET_HZ

# electrode of each of the 32 columns in the For_graph files, from the MEMA
# channel/electrode table. CPz is the reference (constant column); HEOL/HEOR are
# horizontal eye-movement (EOG) channels, not EEG.
COLUMN_ELECTRODES = [
    'FP1', 'FP2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FCz', 'FC3', 'FC4', 'FT7', 'FT8',
    'Cz', 'C3', 'C4', 'T3', 'T4', 'CPz', 'CP3', 'CP4', 'TP7', 'TP8', 'Pz', 'P3',
    'P4', 'T5', 'T6', 'Oz', 'O1', 'O2', 'HEOL', 'HEOR',
]
REFERENCE = 'CPz'

# the 8-electrode layout used by the main pipeline: frontal (F3, Fz, F4),
# parietal (P3, Pz, P4) and occipital (O1, O2) attention sites. It was chosen over
# the same layout with C3/C4 by cross-validation on rounds 1-3 (channel_selection.py).
CHANNEL_NAMES = ['F3', 'Fz', 'F4', 'P3', 'P4', 'Pz', 'O1', 'O2']
CHANNEL_COLUMNS = [COLUMN_ELECTRODES.index(ch) for ch in CHANNEL_NAMES]

# --all-channels: every column except the constant reference (29 EEG + 2 EOG)
ALL_NAMES = [ch for ch in COLUMN_ELECTRODES if ch != REFERENCE]
ALL_COLUMNS = [COLUMN_ELECTRODES.index(ch) for ch in ALL_NAMES]

BANDPASS_HZ = (0.5, 45.0)
FILTER_ORDER = 4

# a = attentive (focused), n = neutral, r = relaxed (both unfocused)
STATE_TO_LABEL = {'a': 1, 'n': 0, 'r': 0}
TRIAL_FILE_RE = re.compile(r'^(Subject\d+)_([anr])([1-4])\.txt$')

SOS = signal.butter(FILTER_ORDER, BANDPASS_HZ, btype='bandpass', fs=TARGET_HZ, output='sos')


def clean_trial(raw):
    """raw: (samples, channels) at 500 Hz -> float32 (samples/4, channels) at 125 Hz."""
    # decimate applies an anti-aliasing low-pass before downsampling
    decimated = signal.decimate(raw, DOWNSAMPLE_FACTOR, axis=0, zero_phase=True)
    filtered = signal.sosfiltfilt(SOS, decimated, axis=0)
    return filtered.astype(np.float32)


def process_subject(subject, columns, output_folder):
    subject_path = os.path.join(DATA_FOLDER, subject)
    trials = {}
    rows = []
    for filename in sorted(os.listdir(subject_path)):
        match = TRIAL_FILE_RE.match(filename)
        if match is None:
            continue
        _, state, round_str = match.groups()
        trial_key = f'{state}{round_str}'

        # usecols returns columns in file order, so reorder to match `columns`
        raw = pd.read_csv(os.path.join(subject_path, filename), header=None,
                          usecols=columns, dtype=np.float64)[columns].values
        clean = clean_trial(raw)
        trials[trial_key] = clean
        rows.append({
            'subject': subject,
            'trial_id': f'{subject}_{trial_key}',
            'round': int(round_str),
            'state': state,
            'label': STATE_TO_LABEL[state],
            'n_samples': clean.shape[0],
            'duration_s': clean.shape[0] / TARGET_HZ,
        })

    np.savez_compressed(os.path.join(output_folder, f'{subject}.npz'), **trials)
    print(f'{subject}: {len(trials)} trials, {sum(r["duration_s"] for r in rows) / 60:.1f} min')
    return rows


def subject_sort_key(name):
    return int(name.replace('Subject', ''))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-channels', action='store_true',
                        help='keep all 31 non-constant columns (for channel_selection.py)')
    args = parser.parse_args()
    if args.all_channels:
        columns, names, output_folder = ALL_COLUMNS, ALL_NAMES, OUTPUT_FOLDER_ALL
    else:
        columns, names, output_folder = CHANNEL_COLUMNS, CHANNEL_NAMES, OUTPUT_FOLDER

    os.makedirs(output_folder, exist_ok=True)
    subjects = sorted((d for d in os.listdir(DATA_FOLDER) if d.startswith('Subject')),
                      key=subject_sort_key)
    print(f'Processing {len(subjects)} subjects from {DATA_FOLDER}')

    start = time.time()
    results = Parallel(n_jobs=4)(delayed(process_subject)(s, columns, output_folder)
                                for s in subjects)
    metadata = pd.DataFrame([row for rows in results for row in rows])
    metadata.to_csv(os.path.join(output_folder, 'metadata.csv'), index=False)

    with open(os.path.join(output_folder, 'info.txt'), 'w') as f:
        f.write(f'fs={TARGET_HZ}\nchannels={",".join(names)}\n'
                f'bandpass={BANDPASS_HZ[0]}-{BANDPASS_HZ[1]} Hz, Butterworth order {FILTER_ORDER}, '
                f'zero-phase (sosfiltfilt)\n')

    print(f'{len(metadata)} trials, {metadata.duration_s.sum() / 60:.0f} min total, '
          f'{time.time() - start:.0f} s')


if __name__ == '__main__':
    main()
