"""
data_processing.py - step 1: raw EEG -> cleaned continuous trials

Reads every trial file in MEMA's For_graph folder, keeps the 8 channels that
match the NeuroPawn headset, decimates 500 Hz -> 125 Hz and band-passes
0.5-45 Hz. Each trial is stored as one continuous float32 signal; windowing
happens later in feature_extraction.py so the window length can be tuned.

Output (processed_data/trials/):
    SubjectN.npz    one array per trial, keyed by trial id (e.g. "a1")
    metadata.csv    one row per trial: subject, trial_id, round, state, label,
                    n_samples, duration_s
"""

import os
import re
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import signal

DATA_FOLDER = os.path.join('MEMA Dataset', 'MEMA', 'For_graph')
OUTPUT_FOLDER = os.path.join('processed_data', 'trials')

ORIGINAL_HZ = 500
TARGET_HZ = 125
DOWNSAMPLE_FACTOR = ORIGINAL_HZ // TARGET_HZ

# column indices in the 32-column For_graph files
CHANNEL_COLUMNS = [2, 3, 4, 12, 14, 23, 26, 28]
CHANNEL_NAMES = ['F3', 'Fz', 'F4', 'C3', 'C4', 'Pz', 'O1', 'O2']

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


def process_subject(subject):
    subject_path = os.path.join(DATA_FOLDER, subject)
    trials = {}
    rows = []
    for filename in sorted(os.listdir(subject_path)):
        match = TRIAL_FILE_RE.match(filename)
        if match is None:
            continue
        _, state, round_str = match.groups()
        trial_key = f'{state}{round_str}'

        raw = pd.read_csv(os.path.join(subject_path, filename), header=None,
                          usecols=CHANNEL_COLUMNS, dtype=np.float64).values
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

    np.savez_compressed(os.path.join(OUTPUT_FOLDER, f'{subject}.npz'), **trials)
    print(f'{subject}: {len(trials)} trials, {sum(r["duration_s"] for r in rows) / 60:.1f} min')
    return rows


def subject_sort_key(name):
    return int(name.replace('Subject', ''))


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    subjects = sorted((d for d in os.listdir(DATA_FOLDER) if d.startswith('Subject')),
                      key=subject_sort_key)
    print(f'Processing {len(subjects)} subjects from {DATA_FOLDER}')

    start = time.time()
    results = Parallel(n_jobs=4)(delayed(process_subject)(s) for s in subjects)
    metadata = pd.DataFrame([row for rows in results for row in rows])
    metadata.to_csv(os.path.join(OUTPUT_FOLDER, 'metadata.csv'), index=False)

    with open(os.path.join(OUTPUT_FOLDER, 'info.txt'), 'w') as f:
        f.write(f'fs={TARGET_HZ}\nchannels={",".join(CHANNEL_NAMES)}\n'
                f'bandpass={BANDPASS_HZ[0]}-{BANDPASS_HZ[1]} Hz, Butterworth order {FILTER_ORDER}, '
                f'zero-phase (sosfiltfilt)\n')

    print(f'{len(metadata)} trials, {metadata.duration_s.sum() / 60:.0f} min total, '
          f'{time.time() - start:.0f} s')


if __name__ == '__main__':
    main()
