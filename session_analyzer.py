#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd

""" Data is pulled from this script to populate tables and figures that
show statistics about real-time predictions and intention reports. """


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_DIR = Path("../FlipThatBucket/data")

PARTICIPANT = "p2"
SESSION_NAME = "Session_02199_Lab"

BIN_MS = 20
SILENT_WINDOW_BINS = 50

AVAILABLE_SESSIONS = {
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

BLOCK_TYPES = [
    "train",
    "ftb-behavior-test",
    "ftb-mixed-test",
    "ftb-brain-test",
]

BLOCK_LABELS = {
    "train": "Train",
    "ftb-behavior-test": "Validation",
    "ftb-mixed-test": "Mixed",
    "ftb-brain-test": "Brain",
}

SKIPPED_BLOCK_TYPES = {
    "reaction-time",
    "scroll-force-test",
    "scroll-brain-test",
}


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def percentage(numerator, denominator):
    return safe_divide(numerator, denominator) * 100


def first_onset(predictions):
    predictions = np.asarray(predictions).reshape(-1)
    indices = np.flatnonzero(predictions == 1)
    return float(indices[0]) if indices.size else np.nan


def mean_ms(values):
    return np.nanmean(values) * BIN_MS if len(values) else np.nan


def std_ms(values):
    return np.nanstd(values) * BIN_MS if len(values) else np.nan


def new_accumulator():
    return {
        "trials": 0,
        "participant_moves": 0,
        "behavior_moves": 0,
        "behavior_intent": 0,
        "brain_moves": 0,
        "brain_intent": 0,
        "active_trials": 0,
        "active_moves": 0,
        "silent_trials": 0,
        "silent_correct": 0,
        "silent_deltas": [],
        "silent_too_early": 0,
        "too_early_deltas": [],
        "silent_too_late": 0,
        "too_late_deltas": [],
        "silent_no_pred": 0,
    }


def calculate_silent_metrics(player_trials):
    prediction_onsets = np.asarray(
        [first_onset(predictions) for predictions in player_trials["Predictions"]],
        dtype=float,
    )
    action_onsets = player_trials["Onset"].to_numpy(dtype=float)
    deltas = action_onsets - prediction_onsets

    correct = (
        ~np.isnan(deltas)
        & (deltas >= 0)
        & (deltas <= SILENT_WINDOW_BINS)
    )
    too_early = ~np.isnan(deltas) & (deltas > SILENT_WINDOW_BINS)
    too_late = ~np.isnan(deltas) & (deltas < 0)
    no_prediction = np.isnan(deltas)

    return {
        "silent_trials": len(deltas),
        "silent_correct": int(correct.sum()),
        "silent_deltas": deltas[correct].tolist(),
        "silent_too_early": int(too_early.sum()),
        "too_early_deltas": deltas[too_early].tolist(),
        "silent_too_late": int(too_late.sum()),
        "too_late_deltas": deltas[too_late].tolist(),
        "silent_no_pred": int(no_prediction.sum()),
    }


def analyze_block(block_data, block_type):
    total_trials = len(block_data)

    participant_moves = int((block_data["PlayerID"] == 1).sum())

    behavior_mask = block_data["PlayerID"] == 2
    behavior_moves = int(behavior_mask.sum())
    behavior_intent = int(
        (
            behavior_mask
            & block_data["IntentAnswer"].isin([1, 2])
        ).sum()
    )

    brain_mask = block_data["PlayerID"] == 3
    brain_moves = int(brain_mask.sum())
    brain_intent = int(
        (
            brain_mask
            & block_data["IntentAnswer"].isin([1, 2])
        ).sum()
    )

    player_trials = block_data[block_data["PlayerID"] == 1]
    silent = calculate_silent_metrics(player_trials)

    # In brain blocks, an on-time silent prediction is treated as an applied
    # brain-based move with intention = Yes. Other silent outcomes are ignored.
    if block_type == "ftb-brain-test":
        available_trials = max(0, total_trials - brain_moves)
        applied_silent = min(
            silent["silent_correct"],
            available_trials,
        )
        brain_moves += applied_silent
        brain_intent += applied_silent

    include_silent = (
        block_type != "train"
        and block_type != "ftb-brain-test"
    )

    result = {
        "trials": total_trials,
        "participant_moves": participant_moves,
        "behavior_moves": behavior_moves,
        "behavior_intent": behavior_intent,
        "brain_moves": brain_moves,
        "brain_intent": brain_intent,
        "active_trials": (
            total_trials if block_type == "ftb-brain-test" else 0
        ),
        "active_moves": (
            brain_moves if block_type == "ftb-brain-test" else 0
        ),
        "include_silent": include_silent,
        **silent,
    }

    return result


def update_accumulator(accumulator, block_result):
    count_fields = [
        "trials",
        "participant_moves",
        "behavior_moves",
        "behavior_intent",
        "brain_moves",
        "brain_intent",
        "active_trials",
        "active_moves",
    ]

    for field in count_fields:
        accumulator[field] += block_result[field]

    if block_result["include_silent"]:
        silent_count_fields = [
            "silent_trials",
            "silent_correct",
            "silent_too_early",
            "silent_too_late",
            "silent_no_pred",
        ]
        silent_list_fields = [
            "silent_deltas",
            "too_early_deltas",
            "too_late_deltas",
        ]

        for field in silent_count_fields:
            accumulator[field] += block_result[field]

        for field in silent_list_fields:
            accumulator[field].extend(block_result[field])


def print_block_results(block_number, block_type, result):
    label = BLOCK_LABELS[block_type]

    print(f"BLOCK {block_number}: {label} ({result['trials']} trials)")
    print(f"   - Participant:    {result['participant_moves']}")
    print(
        f"   - Behavior-based: {result['behavior_moves']} "
        f"({percentage(result['behavior_intent'], result['behavior_moves']):.2f}% intent)"
    )
    print(
        f"   - Brain-based:    {result['brain_moves']} "
        f"({percentage(result['brain_intent'], result['brain_moves']):.2f}% intent)"
    )

    if block_type == "ftb-brain-test":
        print(
            f"   - Active accuracy: "
            f"{percentage(result['active_moves'], result['active_trials']):.2f}% "
            f"({result['active_moves']}/{result['active_trials']})"
        )

    if result["include_silent"]:
        silent_trials = result["silent_trials"]

        print(f"       - Silent count:     {silent_trials}")
        print(
            f"       - Silent accuracy:  "
            f"{percentage(result['silent_correct'], silent_trials):.2f}%"
        )
        print(
            f"       - Silent timing:    "
            f"{mean_ms(result['silent_deltas']):.2f} ms"
        )
        print(
            f"       - Silent too early: {result['silent_too_early']} "
            f"({percentage(result['silent_too_early'], silent_trials):.2f}%)"
        )
        print(
            f"       - Too early timing: "
            f"{mean_ms(result['too_early_deltas']):.2f} ms"
        )
        print(
            f"       - Silent too late:  {result['silent_too_late']} "
            f"({percentage(result['silent_too_late'], silent_trials):.2f}%)"
        )
        print(
            f"       - Too late timing:  "
            f"{mean_ms(result['too_late_deltas']):.2f} ms"
        )
        print(
            f"       - No prediction:    {result['silent_no_pred']} "
            f"({percentage(result['silent_no_pred'], silent_trials):.2f}%)"
        )

    print()


def print_global_summary(accumulator):
    print("\n-------------------------")
    print("SUMMARY ACROSS ALL BLOCKS")
    print("-------------------------\n")

    print(
        f"- Total participant moves: "
        f"{accumulator['participant_moves']}"
    )
    print(
        f"- Total behavior moves:    {accumulator['behavior_moves']} "
        f"({percentage(accumulator['behavior_intent'], accumulator['behavior_moves']):.2f}% intent)"
    )
    print(
        f"- Total brain moves:       {accumulator['brain_moves']} "
        f"({percentage(accumulator['brain_intent'], accumulator['brain_moves']):.2f}% intent)"
    )
    print(
        f"- Active accuracy:         "
        f"{percentage(accumulator['active_moves'], accumulator['active_trials']):.2f}% "
        f"({accumulator['active_moves']}/{accumulator['active_trials']})"
    )

    silent_trials = accumulator["silent_trials"]

    if not silent_trials:
        print("- No silent data was collected.")
        return

    print(f"- Total silent count:      {silent_trials}")
    print(
        f"- Silent accuracy:         "
        f"{percentage(accumulator['silent_correct'], silent_trials):.2f}% "
        f"({accumulator['silent_correct']}/{silent_trials})"
    )
    print(
        f"- Silent timing:           "
        f"{mean_ms(accumulator['silent_deltas']):.2f} ms"
    )
    print(
        f"- Silent too early:        {accumulator['silent_too_early']} "
        f"({percentage(accumulator['silent_too_early'], silent_trials):.2f}%)"
    )
    print(
        f"- Too early timing:        "
        f"{mean_ms(accumulator['too_early_deltas']):.2f} ms"
    )
    print(
        f"- Silent too late:         {accumulator['silent_too_late']} "
        f"({percentage(accumulator['silent_too_late'], silent_trials):.2f}%)"
    )
    print(
        f"- Too late timing:         "
        f"{mean_ms(accumulator['too_late_deltas']):.2f} ms"
    )
    print(
        f"- No prediction:           {accumulator['silent_no_pred']} "
        f"({percentage(accumulator['silent_no_pred'], silent_trials):.2f}%)"
    )


def print_block_type_summary(block_type_accumulators):
    print("\n--------------------------------")
    print("AGGREGATED RESULTS BY BLOCK TYPE")
    print("--------------------------------\n")

    for block_type in BLOCK_TYPES:
        accumulator = block_type_accumulators[block_type]
        label = BLOCK_LABELS[block_type]

        behavior_moves = accumulator["behavior_moves"]
        brain_moves = accumulator["brain_moves"]
        silent_trials = accumulator["silent_trials"]

        print(f"{label}:")
        print(
            f"   - Behavior-based:  {behavior_moves} "
            f"({percentage(accumulator['behavior_intent'], behavior_moves):.2f}% intent)"
        )
        print(
            f"   - Brain-based:     {brain_moves} "
            f"({percentage(accumulator['brain_intent'], brain_moves):.2f}% intent)"
        )

        if block_type == "ftb-brain-test" and accumulator["active_trials"]:
            print(
                f"   - Active accuracy: "
                f"{percentage(accumulator['active_moves'], accumulator['active_trials']):.2f}%"
            )

        print(f"   - Silent trials:   {silent_trials}")

        if silent_trials:
            print(
                f"   - Silent accuracy: "
                f"{percentage(accumulator['silent_correct'], silent_trials):.2f}%"
            )
            print(
                f"   - Silent timing:   "
                f"{mean_ms(accumulator['silent_deltas']):.2f} "
                f"+/- {std_ms(accumulator['silent_deltas']):.2f} ms"
            )
            print(
                f"   - Silent too early:{accumulator['silent_too_early']} "
                f"({percentage(accumulator['silent_too_early'], silent_trials):.2f}%)"
            )
            print(
                f"   - Too early timing:"
                f"{mean_ms(accumulator['too_early_deltas']):.2f} "
                f"+/- {std_ms(accumulator['too_early_deltas']):.2f} ms"
            )
            print(
                f"   - Silent too late: {accumulator['silent_too_late']} "
                f"({percentage(accumulator['silent_too_late'], silent_trials):.2f}%)"
            )
            print(
                f"   - Too late timing: "
                f"{mean_ms(accumulator['too_late_deltas']):.2f} "
                f"+/- {std_ms(accumulator['too_late_deltas']):.2f} ms"
            )
            print(
                f"   - No prediction:   {accumulator['silent_no_pred']} "
                f"({percentage(accumulator['silent_no_pred'], silent_trials):.2f}%)"
            )
        else:
            print("   - Silent accuracy: n/a")
            print("   - Silent timing:   n/a")
            print("   - Silent too early:0 (n/a)")
            print("   - Too early timing:n/a")
            print("   - Silent too late: 0 (n/a)")
            print("   - Too late timing: n/a")
            print("   - No prediction:   0 (n/a)")

        print()


def validate_session_selection():
    if PARTICIPANT not in AVAILABLE_SESSIONS:
        raise ValueError(
            f"Unknown participant '{PARTICIPANT}'. "
            f"Choose from {list(AVAILABLE_SESSIONS)}."
        )

    if SESSION_NAME not in AVAILABLE_SESSIONS[PARTICIPANT]:
        raise ValueError(
            f"{SESSION_NAME} is not listed for {PARTICIPANT.upper()}."
        )


def main():
    validate_session_selection()

    session_path = DATA_DIR / PARTICIPANT / f"{SESSION_NAME}.pkl"
    session = pd.read_pickle(session_path)

    block_type_accumulators = {
        block_type: new_accumulator()
        for block_type in BLOCK_TYPES
    }
    global_accumulator = new_accumulator()

    print(f"Session: {PARTICIPANT.upper()} / {SESSION_NAME}")
    print("\n-----------------------")
    print("BLOCK BY BLOCK ANALYSIS")
    print("-----------------------\n")

    block_ids = session["BlockID"].drop_duplicates().tolist()

    for block_number, block_id in enumerate(block_ids, start=1):
        block_data = session[
            session["BlockID"] == block_id
        ].reset_index(drop=True)

        if block_data.empty:
            continue

        block_type = block_data.loc[0, "BlockType"]

        if block_type in SKIPPED_BLOCK_TYPES:
            continue

        if block_type not in BLOCK_TYPES:
            print(
                f"[WARN] Skipping unrecognized block type "
                f"'{block_type}' in block {block_number}."
            )
            continue

        block_result = analyze_block(block_data, block_type)

        print_block_results(
            block_number,
            block_type,
            block_result,
        )

        update_accumulator(global_accumulator, block_result)
        update_accumulator(
            block_type_accumulators[block_type],
            block_result,
        )

    print_global_summary(global_accumulator)
    print_block_type_summary(block_type_accumulators)


if __name__ == "__main__":
    main()