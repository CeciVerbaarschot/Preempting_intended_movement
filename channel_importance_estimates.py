#!/usr/bin/env python3
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_DIR = Path("../FlipThatBucket/data")
OUTPUT_DIR = Path(__file__).resolve().parent / "feature_importances"

PARTICIPANTS = {
    "c1": [
        "Session_00637_Lab",
        "Session_00638_Lab",
        "Session_00644_Lab",
    ],
    "p2": [
        "Session_02130_Lab",
        "Session_02198_Lab",
        "Session_02199_Lab",
        "Session_02225_Lab",
        "Session_02227_Lab",
    ],
    "p3": [
        "Session_00197_Home",
        "Session_00200_Home",
        "Session_00203_Home",
    ],
}

PREMOVEMENT_BINS = 100
POSTMOVEMENT_BINS = 50

SEQUENCE_BINS = 10
SEQUENCE_STEP_BINS = 1
BIN_MS = 20
POSITIVE_WINDOW_MS = 600

N_SEEDS = 500
N_ESTIMATORS = 200
MAX_DEPTH = 15

logging.basicConfig(
    format="%(asctime)s; %(levelname)s; %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def create_2d_sequences(
    spikes,
    movement_onset=PREMOVEMENT_BINS,
    sequence_bins=SEQUENCE_BINS,
    bin_ms=BIN_MS,
    step_bins=SEQUENCE_STEP_BINS,
):
    spikes = np.asarray(spikes)
    n_time_bins = spikes.shape[1]

    sequences = []
    onset_horizons_ms = []

    for start in range(
        0,
        n_time_bins - sequence_bins + 1,
        step_bins,
    ):
        stop = start + sequence_bins

        sequences.append(spikes[:, start:stop])
        onset_horizons_ms.append(
            bin_ms * (movement_onset - stop)
        )

    return (
        np.asarray(sequences),
        np.asarray(onset_horizons_ms),
    )


def load_training_trials(session):
    valid_trials = session[
        (session["BlockType"] == "train")
        & (session["PlayerID"] == 1)
        & (session["Onset"] >= PREMOVEMENT_BINS)
        & (session["PostMovement"] >= POSTMOVEMENT_BINS)
    ]

    expected_bins = PREMOVEMENT_BINS + POSTMOVEMENT_BINS
    trials = []

    for row in valid_trials.itertuples(index=False):
        onset = int(row.Onset)
        start = onset - PREMOVEMENT_BINS
        stop = onset + POSTMOVEMENT_BINS

        neural = np.asarray(row.Neural)
        trial = neural[:, start:stop]

        if trial.shape[1] == expected_bins:
            trials.append(trial)

    if not trials:
        return np.empty((0, 0, expected_bins))

    return np.stack(trials)


def build_training_data(trials):
    sequence_parts = []
    label_parts = []

    for trial in trials:
        sequences, horizons_ms = create_2d_sequences(trial)

        onset_indices = np.flatnonzero(horizons_ms == 0)
        if onset_indices.size == 0:
            continue

        stop = onset_indices[0] + 1

        sequences = sequences[:stop]
        labels = (
            horizons_ms[:stop] <= POSITIVE_WINDOW_MS
        ).astype(np.int32)

        sequence_parts.append(sequences)
        label_parts.append(labels)

    if not sequence_parts:
        return np.empty((0, 0)), np.empty(0, dtype=np.int32)

    sequences = np.concatenate(sequence_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)

    # Sum each 200 ms sequence across time, leaving one feature per channel.
    features = sequences.sum(axis=2)

    return features, labels


def train_feature_importances(features, labels, description):
    importances = []

    for seed in tqdm(
        range(N_SEEDS),
        desc=description,
        leave=False,
    ):
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        )
        model.fit(features, labels)
        importances.append(model.feature_importances_)

    return np.asarray(importances)


def process_session(participant, session_name):
    session_path = (
        DATA_DIR
        / participant
        / f"{session_name}.pkl"
    )

    if not session_path.exists():
        logging.warning(
            "Missing file: %s. Skipping.",
            session_path,
        )
        return

    session = pd.read_pickle(session_path)
    trials = load_training_trials(session)

    if trials.shape[0] == 0:
        logging.warning(
            "No valid training trials for %s / %s. Skipping.",
            participant,
            session_name,
        )
        return

    features, labels = build_training_data(trials)

    if features.shape[0] == 0:
        logging.warning(
            "No training sequences for %s / %s. Skipping.",
            participant,
            session_name,
        )
        return

    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        logging.warning(
            "Training labels contain only one class for %s / %s. Skipping.",
            participant,
            session_name,
        )
        return

    logging.info(
        "Training data for %s / %s: X=%s, y=%s, positive rate=%.3f",
        participant,
        session_name,
        features.shape,
        labels.shape,
        labels.mean(),
    )

    importances = train_feature_importances(
        features,
        labels,
        description=f"{participant}/{session_name}",
    )

    output_path = (
        OUTPUT_DIR
        / f"featimp_{participant}_{session_name}.npy"
    )
    np.save(output_path, importances)

    logging.info(
        "Saved: %s; shape=%s",
        output_path.resolve(),
        importances.shape,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for participant, session_names in PARTICIPANTS.items():
        for session_name in session_names:
            process_session(participant, session_name)


if __name__ == "__main__":
    main()