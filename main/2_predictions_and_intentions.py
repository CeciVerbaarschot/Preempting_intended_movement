#!/usr/bin/env python3
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from scipy.ndimage import gaussian_filter1d


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_DIR = Path("../../FlipThatBucket/data")
OUTPUT_DIR = Path(__file__).resolve().parent
MAIN_OUTPUT_FILE = OUTPUT_DIR / "figure_2.pdf"
SILENT_OUTPUT_FILE = OUTPUT_DIR / "figure_2_bottom_silent.pdf"
ACTIVE_OUTPUT_FILE = OUTPUT_DIR / "figure_2_bottom_active.pdf"
PRE_ONSET_ZOOM_OUTPUT_FILE = OUTPUT_DIR / "figure_2_pre_onset_mua_zoom.pdf"
PRE_ONSET_DUMBBELL_OUTPUT_FILE = OUTPUT_DIR / "figure_2_pre_onset_mua_dumbbell.pdf"
PRE_ONSET_DUMBBELL_ERROR = "ci"  # "ci" for 95% CI or "sd" for standard deviation

BIN_MS = 20
PREMOVEMENT = 100
POSTMOVEMENT = 25
BASELINE_BINS = 25
N_TIME = PREMOVEMENT + POSTMOVEMENT

MUA_PERCENTILE = 99
MUA_SIGMA = 2
FORCE_STD_FLOOR = None

# Positive-intent MUA departure threshold relative to the initial baseline.
MUA_DEPARTURE_Z = 1
MUA_DEPARTURE_BASELINE_BINS = BASELINE_BINS

# Pre-onset window for brain-based vs behavior-based MUA comparisons.
PRE_ONSET_WINDOW_MS = 200
PRE_ONSET_WINDOW_BINS = max(1, int(round(PRE_ONSET_WINDOW_MS / BIN_MS)))
PRE_ONSET_EFFECTIVE_WINDOW_MS = PRE_ONSET_WINDOW_BINS * BIN_MS

STATS_ALPHA = 0.05
STATS_TABLE_FILE = Path("figure_2_pre_onset_mua_stats.csv")
SAVE_STATS_TABLE = True

TYPE_BEHAVIOR = 2
TYPE_BRAIN = 3

FORCE_BASELINE = 5000

# P2 scroll sessions are excluded because they omit the intention question.
PARTICIPANTS = {
    "p2": [
        "Session_02130_Lab",
        "Session_02198_Lab",
        "Session_02199_Lab",
    ],
    "p3": [
        "Session_00197_Home",
        "Session_00200_Home",
        "Session_00203_Home",
    ],
    "c1": [
        "Session_00637_Lab",
        "Session_00638_Lab",
        "Session_00644_Lab",
    ],
}

PARTICIPANT_ORDER = ["p2", "p3", "c1"]

PARTICIPANT_LABELS = {
    "p2": "P2",
    "p3": "P3",
    "c1": "C1",
}

PALETTE = {
    "p2": "#0072B2",
    "p3": "#009E73",
    "c1": "#CC79A7",
}

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"


def load_session(participant, session_name):
    path = DATA_DIR / participant / f"{session_name}.pkl"
    return pd.read_pickle(path)


def first_one(values):
    indices = np.flatnonzero(np.asarray(values) == 1)
    return int(indices[0]) if indices.size else np.nan


def concatenate_nonempty(arrays, axis=0):
    arrays = [np.asarray(array) for array in arrays if np.asarray(array).size]
    return np.concatenate(arrays, axis=axis) if arrays else np.empty(0)


def extract_aligned_trials(data, alignment_column):
    neural_trials = []
    force_trials = []

    for neural, force, alignment in zip(
        data["Neural"], data["Force"], data[alignment_column]
    ):
        alignment = int(alignment)
        start = alignment - PREMOVEMENT
        stop = alignment + POSTMOVEMENT

        neural_trials.append(np.asarray(neural)[:, start:stop])
        force_trials.append(np.asarray(force)[start:stop])

    if not neural_trials:
        return np.empty(0), np.empty(0)

    return np.stack(neural_trials), np.stack(force_trials)


def load_trial_profiles():
    profiles = {
        participant: {
            "brain": {"neural": [], "force": [], "intent": []},
            "behavioral": {"neural": [], "force": [], "intent": []},
        }
        for participant in PARTICIPANT_ORDER
    }

    for participant in PARTICIPANT_ORDER:
        for session_name in PARTICIPANTS[participant]:
            session = load_session(participant, session_name).copy()
            session["PredOnset"] = session["Predictions"].apply(first_one)

            brain_trials = session[
                session["BlockType"].str.startswith("ftb", na=False)
                & (session["PlayerID"] == 3)
                & (session["PredOnset"] >= PREMOVEMENT)
                & (session["PostMovement"] >= POSTMOVEMENT)
            ].reset_index(drop=True)

            brain_neural, brain_force = extract_aligned_trials(
                brain_trials, "PredOnset"
            )
            profiles[participant]["brain"]["neural"].append(brain_neural)
            profiles[participant]["brain"]["force"].append(brain_force)
            profiles[participant]["brain"]["intent"].append(
                brain_trials["IntentAnswer"].to_numpy() != 0
            )

            behavioral_trials = session[
                session["BlockType"].str.startswith(("ftb", "train"), na=False)
                & (session["PlayerID"] == 2)
                & (session["Onset"] >= PREMOVEMENT)
                & (session["PostMovement"] >= POSTMOVEMENT)
            ].reset_index(drop=True)

            behavioral_neural, behavioral_force = extract_aligned_trials(
                behavioral_trials, "Onset"
            )
            profiles[participant]["behavioral"]["neural"].append(
                behavioral_neural
            )
            profiles[participant]["behavioral"]["force"].append(
                behavioral_force
            )
            profiles[participant]["behavioral"]["intent"].append(
                behavioral_trials["IntentAnswer"].to_numpy() != 0
            )

    return profiles


def group_by_intention(data):
    grouped = {
        "positive_neural": [],
        "negative_neural": [],
        "positive_force": [],
        "negative_force": [],
        "positive_type": [],
        "negative_type": [],
    }

    for session_index in range(len(data["brain"]["intent"])):
        brain_intent = data["brain"]["intent"][session_index]
        behavioral_intent = data["behavioral"]["intent"][session_index]

        brain_neural = data["brain"]["neural"][session_index]
        brain_force = data["brain"]["force"][session_index]
        behavioral_neural = data["behavioral"]["neural"][session_index]
        behavioral_force = data["behavioral"]["force"][session_index]

        brain_positive_neural = brain_neural[brain_intent]
        brain_negative_neural = brain_neural[~brain_intent]
        brain_positive_force = brain_force[brain_intent]
        brain_negative_force = brain_force[~brain_intent]

        behavioral_positive_neural = behavioral_neural[behavioral_intent]
        behavioral_negative_neural = behavioral_neural[~behavioral_intent]
        behavioral_positive_force = behavioral_force[behavioral_intent]
        behavioral_negative_force = behavioral_force[~behavioral_intent]

        grouped["positive_neural"].append(
            concatenate_nonempty(
                [brain_positive_neural, behavioral_positive_neural]
            )
        )
        grouped["negative_neural"].append(
            concatenate_nonempty(
                [brain_negative_neural, behavioral_negative_neural]
            )
        )
        grouped["positive_force"].append(
            concatenate_nonempty([brain_positive_force, behavioral_positive_force])
        )
        grouped["negative_force"].append(
            concatenate_nonempty([brain_negative_force, behavioral_negative_force])
        )

        grouped["positive_type"].append(
            np.concatenate(
                [
                    np.full(brain_positive_neural.shape[0], TYPE_BRAIN),
                    np.full(
                        behavioral_positive_neural.shape[0], TYPE_BEHAVIOR
                    ),
                ]
            )
        )
        grouped["negative_type"].append(
            np.concatenate(
                [
                    np.full(brain_negative_neural.shape[0], TYPE_BRAIN),
                    np.full(
                        behavioral_negative_neural.shape[0], TYPE_BEHAVIOR
                    ),
                ]
            )
        )

    grouped["positive_type"] = concatenate_nonempty(grouped["positive_type"])
    grouped["negative_type"] = concatenate_nonempty(grouped["negative_type"])
    return grouped


def process_mua_by_session(positive_sessions, negative_sessions):
    positive_processed = []
    negative_processed = []

    for positive_trials, negative_trials in zip(
        positive_sessions, negative_sessions
    ):
        def calculate_mua(trials):
            trials = np.asarray(trials)
            if not trials.size:
                return np.empty((0, N_TIME))

            smoothed = gaussian_filter1d(
                trials, sigma=MUA_SIGMA, axis=-1
            )
            return smoothed.sum(axis=1)

        positive_mua = calculate_mua(positive_trials)
        negative_mua = calculate_mua(negative_trials)
        pooled = concatenate_nonempty([positive_mua, negative_mua])

        if not pooled.size:
            continue

        upper_limit = np.percentile(pooled, MUA_PERCENTILE)
        pooled = np.clip(pooled, 0, upper_limit)
        positive_mua = np.clip(positive_mua, 0, upper_limit)
        negative_mua = np.clip(negative_mua, 0, upper_limit)

        mean = pooled.mean()
        std = pooled.std() or 1.0

        positive_processed.append((positive_mua - mean) / std)
        negative_processed.append((negative_mua - mean) / std)

    return (
        concatenate_nonempty(positive_processed),
        concatenate_nonempty(negative_processed),
    )


def process_force_by_session(positive_sessions, negative_sessions):
    positive_processed = []
    negative_processed = []

    for positive_trials, negative_trials in zip(
        positive_sessions, negative_sessions
    ):
        def remove_baseline(trials):
            trials = np.asarray(trials, dtype=float)
            if not trials.size:
                return np.empty((0, N_TIME))

            baseline = trials[:, :BASELINE_BINS].mean(axis=1, keepdims=True)
            return trials - baseline

        positive_force = remove_baseline(positive_trials)
        negative_force = remove_baseline(negative_trials)
        pooled = concatenate_nonempty([positive_force, negative_force])

        if not pooled.size:
            continue

        mean = pooled.mean()
        std = pooled.std()

        if FORCE_STD_FLOOR is not None:
            std = max(std, float(FORCE_STD_FLOOR))
        if std == 0:
            std = 1.0

        positive_processed.append((positive_force - mean) / std)
        negative_processed.append((negative_force - mean) / std)

    return (
        concatenate_nonempty(positive_processed),
        concatenate_nonempty(negative_processed),
    )


def process_profiles(profiles):
    processed = {}

    for participant in PARTICIPANT_ORDER:
        grouped = group_by_intention(profiles[participant])

        positive_neural, negative_neural = process_mua_by_session(
            grouped["positive_neural"], grouped["negative_neural"]
        )
        positive_force, negative_force = process_force_by_session(
            grouped["positive_force"], grouped["negative_force"]
        )

        processed[participant] = {
            "positive": {
                "neural": positive_neural,
                "force": positive_force,
                "type": grouped["positive_type"],
            },
            "negative": {
                "neural": negative_neural,
                "force": negative_force,
                "type": grouped["negative_type"],
            },
        }

    return processed


def calculate_mua_departure_times(processed, prediction_type):
    if prediction_type not in (TYPE_BRAIN, TYPE_BEHAVIOR):
        raise ValueError(
            "prediction_type must be TYPE_BRAIN or TYPE_BEHAVIOR."
        )

    results = []

    for participant in PARTICIPANT_ORDER:
        positive = processed[participant]["positive"]
        prediction_mask = positive["type"] == prediction_type
        prediction_mua = np.asarray(
            positive["neural"][prediction_mask],
            dtype=float,
        )

        if not prediction_mua.size:
            results.append(
                {
                    "participant": PARTICIPANT_LABELS[participant],
                    "ms_relative_to_onset": np.nan,
                    "baseline_mean_z": np.nan,
                    "departure_threshold_z": np.nan,
                }
            )
            continue

        trial_averaged_mua = prediction_mua.mean(axis=0)
        baseline_mean = trial_averaged_mua[
            :MUA_DEPARTURE_BASELINE_BINS
        ].mean()
        departure_threshold = baseline_mean + MUA_DEPARTURE_Z

        # Search the full post-baseline window, including post-onset bins.
        search_start = MUA_DEPARTURE_BASELINE_BINS
        search_stop = N_TIME
        crossing_indices = np.flatnonzero(
            trial_averaged_mua[search_start:search_stop]
            >= departure_threshold
        )

        if crossing_indices.size:
            departure_index = search_start + int(crossing_indices[0])
            ms_relative_to_onset = (
                departure_index - PREMOVEMENT
            ) * BIN_MS
        else:
            ms_relative_to_onset = np.nan

        results.append(
            {
                "participant": PARTICIPANT_LABELS[participant],
                "ms_relative_to_onset": ms_relative_to_onset,
                "baseline_mean_z": baseline_mean,
                "departure_threshold_z": departure_threshold,
            }
        )

    results = pd.DataFrame(results)
    overall_average = results["ms_relative_to_onset"].mean()
    return results, overall_average


def print_mua_departure_times(
    results,
    overall_average,
    prediction_label,
    onset_label,
):
    values = []
    for row in results.itertuples(index=False):
        latency = (
            "n/a"
            if np.isnan(row.ms_relative_to_onset)
            else f"{int(row.ms_relative_to_onset):+d} ms"
        )
        values.append(f"{row.participant}={latency}")

    mean_latency = (
        "n/a"
        if np.isnan(overall_average)
        else f"{overall_average:+.1f} ms"
    )
    print(
        f"{prediction_label} MUA departure ({onset_label}): "
        f"{', '.join(values)}; mean={mean_latency}"
    )


def calculate_force_crossings(profiles, prediction_key):
    """Count positive-intent trials whose raw force exceeds FORCE_BASELINE
    at any point after the aligned onset."""
    results = []

    for participant in PARTICIPANT_ORDER:
        crossed_trials = 0
        total_trials = 0
        pre_onset_force = []

        for force, intent in zip(
            profiles[participant][prediction_key]["force"],
            profiles[participant][prediction_key]["intent"],
        ):
            force = np.asarray(force, dtype=float)
            if not force.size:
                continue

            positive_force = force[np.asarray(intent, dtype=bool)]
            if not positive_force.size:
                continue

            post_onset = positive_force[:, PREMOVEMENT:]
            crossed_trials += int(
                (post_onset.max(axis=1) > FORCE_BASELINE).sum()
            )
            total_trials += post_onset.shape[0]
            pre_onset_force.append(
                positive_force[:, :BASELINE_BINS].ravel()
            )

        results.append(
            {
                "participant": PARTICIPANT_LABELS[participant],
                "crossed_trials": crossed_trials,
                "total_trials": total_trials,
                "percentage": (
                    100 * crossed_trials / total_trials
                    if total_trials
                    else np.nan
                ),
                "median_pre_onset_force": (
                    np.median(concatenate_nonempty(pre_onset_force))
                    if pre_onset_force
                    else np.nan
                ),
            }
        )

    results = pd.DataFrame(results)
    pooled_crossed = int(results["crossed_trials"].sum())
    pooled_total = int(results["total_trials"].sum())
    pooled_percentage = (
        100 * pooled_crossed / pooled_total if pooled_total else np.nan
    )
    return results, pooled_crossed, pooled_total, pooled_percentage


def print_force_crossings(
    results,
    pooled_crossed,
    pooled_total,
    pooled_percentage,
    prediction_label,
    onset_label,
):
    values = []
    for row in results.itertuples(index=False):
        value = (
            "n/a"
            if not row.total_trials
            else f"{row.crossed_trials}/{row.total_trials} ({row.percentage:.1f}%)"
        )
        values.append(f"{row.participant}={value}")

    pooled = (
        "n/a"
        if np.isnan(pooled_percentage)
        else f"{pooled_crossed}/{pooled_total} ({pooled_percentage:.1f}%)"
    )
    print(
        f"{prediction_label} positive-intent force >{FORCE_BASELINE} "
        f"within {POSTMOVEMENT * BIN_MS} ms of {onset_label}: "
        f"{', '.join(values)}; pooled={pooled}"
    )


def calculate_negative_force_crossings(profiles, prediction_key):
    """Count negative-intent trials whose raw force exceeds FORCE_BASELINE
    at any point after the aligned onset."""
    results = []

    for participant in PARTICIPANT_ORDER:
        crossed_trials = 0
        total_trials = 0
        pre_onset_force = []

        for force, intent in zip(
            profiles[participant][prediction_key]["force"],
            profiles[participant][prediction_key]["intent"],
        ):
            force = np.asarray(force, dtype=float)
            if not force.size:
                continue

            negative_force = force[~np.asarray(intent, dtype=bool)]
            if not negative_force.size:
                continue

            post_onset = negative_force[:, PREMOVEMENT:]
            crossed_trials += int(
                (post_onset.max(axis=1) > FORCE_BASELINE).sum()
            )
            total_trials += post_onset.shape[0]
            pre_onset_force.append(
                negative_force[:, :BASELINE_BINS].ravel()
            )

        results.append(
            {
                "participant": PARTICIPANT_LABELS[participant],
                "crossed_trials": crossed_trials,
                "total_trials": total_trials,
                "percentage": (
                    100 * crossed_trials / total_trials
                    if total_trials
                    else np.nan
                ),
                "median_pre_onset_force": (
                    np.median(concatenate_nonempty(pre_onset_force))
                    if pre_onset_force
                    else np.nan
                ),
            }
        )

    results = pd.DataFrame(results)
    pooled_crossed = int(results["crossed_trials"].sum())
    pooled_total = int(results["total_trials"].sum())
    pooled_percentage = (
        100 * pooled_crossed / pooled_total if pooled_total else np.nan
    )
    return results, pooled_crossed, pooled_total, pooled_percentage


def print_negative_force_crossings(
    results,
    pooled_crossed,
    pooled_total,
    pooled_percentage,
    prediction_label,
    onset_label,
):
    values = []
    for row in results.itertuples(index=False):
        value = (
            "n/a"
            if not row.total_trials
            else f"{row.crossed_trials}/{row.total_trials} ({row.percentage:.1f}%)"
        )
        values.append(f"{row.participant}={value}")

    pooled = (
        "n/a"
        if np.isnan(pooled_percentage)
        else f"{pooled_crossed}/{pooled_total} ({pooled_percentage:.1f}%)"
    )
    print(
        f"{prediction_label} negative-intent force >{FORCE_BASELINE} "
        f"within {POSTMOVEMENT * BIN_MS} ms of {onset_label}: "
        f"{', '.join(values)}; pooled={pooled}"
    )


def plot_mean_std(axis, time_s, data, color, label=None):
    data = np.asarray(data)
    if not data.size:
        return

    mean = data.mean(axis=0)
    std = data.std(axis=0, ddof=1) if data.shape[0] > 1 else np.zeros_like(mean)

    axis.plot(time_s, mean, color=color, linewidth=2, label=label)
    axis.fill_between(
        time_s,
        mean - std,
        mean + std,
        color=color,
        alpha=0.2,
        linewidth=0,
    )


def plot_median_iqr(axis, time_s, data, color):
    data = np.asarray(data)
    if not data.size:
        return

    median = np.median(data, axis=0)
    lower = np.percentile(data, 25, axis=0)
    upper = np.percentile(data, 75, axis=0)

    axis.plot(time_s, median, color=color, linewidth=2)
    axis.fill_between(
        time_s,
        lower,
        upper,
        color=color,
        alpha=0.2,
        linewidth=0,
    )


def style_profile_axis(axis, show_prediction_label=False):
    axis.axvline(0, color="red", linewidth=1.2)
    axis.set_xlim(-2.0, 0.5)
    axis.set_ylim(-2, 5)

    if show_prediction_label:
        axis.text(
            -0.05,
            1.0,
            "Prediction\nonset",
            color="red",
            ha="right",
            va="bottom",
            fontsize=10,
            rotation=90,
        )


def draw_prediction_column(axes, processed, column, prediction_type, time_s):
    for participant in PARTICIPANT_ORDER:
        positive = processed[participant]["positive"]
        mask = positive["type"] == prediction_type
        plot_mean_std(
            axes[0, column],
            time_s,
            positive["neural"][mask],
            PALETTE[participant],
            label=PARTICIPANT_LABELS[participant] if column == 0 else None,
        )
        plot_median_iqr(
            axes[1, column],
            time_s,
            positive["force"][mask],
            PALETTE[participant],
        )

        negative = processed[participant]["negative"]
        mask = negative["type"] == prediction_type
        plot_mean_std(
            axes[2, column],
            time_s,
            negative["neural"][mask],
            PALETTE[participant],
        )
        plot_median_iqr(
            axes[3, column],
            time_s,
            negative["force"][mask],
            PALETTE[participant],
        )

    for row in range(4):
        style_profile_axis(
            axes[row, column],
            show_prediction_label=(row == 1 and column == 0),
        )


def plot_main_figure(processed):
    time_s = (
        np.arange(N_TIME) - PREMOVEMENT
    ) * BIN_MS / 1000

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(7.2, 7.0),
        sharex=True,
        sharey=False,
    )

    axes[0, 0].set_title("Brain-based prediction")
    axes[0, 1].set_title("Behavior-based prediction")

    axes[0, 0].set_ylabel("MUA (z-scored)")
    axes[1, 0].set_ylabel("Force (z-scored)")
    axes[2, 0].set_ylabel("MUA (z-scored)")
    axes[3, 0].set_ylabel("Force (z-scored)")

    draw_prediction_column(axes, processed, 0, TYPE_BRAIN, time_s)
    draw_prediction_column(axes, processed, 1, TYPE_BEHAVIOR, time_s)

    axes[0, 0].legend(loc="upper left", frameon=False, fontsize=8)

    for column in range(2):
        axes[3, column].set_xlabel("Time (s)")
        axes[3, column].set_xticks(
            [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5]
        )

    fig.subplots_adjust(hspace=0.22, wspace=0.18)
    fig.tight_layout()
    fig.savefig(MAIN_OUTPUT_FILE, bbox_inches="tight", dpi=500)
    print(f"Saved {MAIN_OUTPUT_FILE}")


def calculate_active_percentages():
    # Hard-coded values are obtained from session_analyzer.py
    data = pd.DataFrame(
        {
            "participant": ["P2", "P3", "C1", "P2", "P3", "C1"],
            "trials": [91, 89, 114, 93, 52, 119],
            "accuracy": [0.890, 0.888, 0.877, 0.940, 0.785, 0.840],
        }
    )
    data["correct_trials"] = data["trials"] * data["accuracy"]

    summary = data.groupby("participant").agg(
        total_trials=("trials", "sum"),
        correct_trials=("correct_trials", "sum"),
    )
    percentages = 100 * summary["correct_trials"] / summary["total_trials"]

    return np.array(
        [percentages["P2"], percentages["P3"], percentages["C1"], 87.5]
    )


def calculate_silent_percentages():
    # Hard-coded values are obtained from session_analyzer.py
    data = pd.DataFrame(
        {
            "participant": ["P2", "P3", "C1", "P2", "P3", "C1"],
            "trials": [41, 38, 40, 92, 101, 106],
            "early_pct": [2.4, 13.2, 27.5, 7.6, 11.9, 12.3],
            "ontime_pct": [95.1, 84.2, 72.5, 85.9, 76.2, 87.7],
        }
    )
    data["early_trials"] = data["trials"] * data["early_pct"] / 100
    data["ontime_trials"] = data["trials"] * data["ontime_pct"] / 100

    summary = data.groupby("participant").agg(
        early_trials=("early_trials", "sum"),
        ontime_trials=("ontime_trials", "sum"),
    )
    percentages = (
        100
        * summary["ontime_trials"]
        / (summary["early_trials"] + summary["ontime_trials"])
    )

    return np.array(
        [percentages["P2"], percentages["P3"], percentages["C1"], 87.7]
    )


def add_rounded_bar(axis, x, y, width, height, color):
    axis.add_patch(
        FancyBboxPatch(
            (x, y - height / 2),
            width,
            height,
            boxstyle=f"round,pad=0,rounding_size={height / 0.9}",
            linewidth=0,
            facecolor=color,
        )
    )


def plot_summary_bars(
    positive_percentages,
    output_file,
    y_label,
    negative_label,
    positive_label,
    negative_color,
    positive_color,
    positive_text_color="black",
    participant_text_color="black",
):
    participant_labels = ["P2", "P3", "C1", "All"]
    negative_percentages = 100 - positive_percentages

    fig, axis = plt.subplots(figsize=(7, 2.2))
    y_positions = np.arange(len(participant_labels))
    bar_height = 0.6

    for index, y_position in enumerate(y_positions):
        negative = negative_percentages[index]
        positive = positive_percentages[index]

        add_rounded_bar(
            axis,
            0,
            y_position,
            negative,
            bar_height,
            negative_color,
        )
        add_rounded_bar(
            axis,
            negative,
            y_position,
            positive,
            bar_height,
            positive_color,
        )

        axis.text(
            negative / 2,
            y_position,
            f"{negative:.1f}%",
            va="center",
            ha="center",
            fontsize=9,
        )
        axis.text(
            negative + positive / 2,
            y_position,
            f"{positive:.1f}%",
            va="center",
            ha="center",
            fontsize=9,
            color=positive_text_color,
        )
        axis.text(
            95,
            y_position,
            participant_labels[index],
            va="center",
            ha="left",
            fontsize=9,
            color=participant_text_color,
        )

    axis.set_xlim(0, 110)
    axis.set_ylim(-0.5, len(y_positions) - 0.5)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_ylabel(y_label, fontsize=10, rotation=90, labelpad=15)

    for spine in axis.spines.values():
        spine.set_visible(False)

    axis.text(
        0.01,
        1.05,
        negative_label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
    )
    axis.text(
        0.90,
        1.05,
        positive_label,
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )

    axis.invert_yaxis()
    fig.savefig(output_file, bbox_inches="tight", dpi=500)
    print(f"Saved {output_file}")


# -----------------------------------------------------------------------------
# Pre-onset MUA: brain-based vs behavior-based predictions
# -----------------------------------------------------------------------------

INTENTION_CONDITIONS = (
    ("positive", "Yes"),
    ("negative", "No"),
)

# Pad the statistics window so Gaussian smoothing matches the plotted traces.
STATS_PAD_BINS = 3 * MUA_SIGMA
STATS_PRE_BINS = PRE_ONSET_WINDOW_BINS + STATS_PAD_BINS
STATS_POST_BINS = STATS_PAD_BINS


def format_p_value(value):
    if value is None or not np.isfinite(value):
        return "n/a"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def format_number(value, decimals=3):
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{decimals}f}"


def significance_label(value, alpha=STATS_ALPHA):
    if value is None or not np.isfinite(value):
        return "n/a"
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < alpha:
        return "*"
    return "n.s."


def bonferroni_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan)
    valid = np.isfinite(p_values)
    n_tests = int(valid.sum())

    if n_tests:
        adjusted[valid] = np.minimum(p_values[valid] * n_tests, 1.0)

    return adjusted


def collect_relaxed_trials(data, alignment_column):
    """Padded pre-onset window and intent flag for every usable trial."""
    neural_windows = []
    intent_flags = []

    for neural, alignment, post_movement, intent in zip(
        data["Neural"],
        data[alignment_column],
        data["PostMovement"],
        data["IntentAnswer"],
    ):
        if not np.isfinite(alignment):
            continue

        neural = np.asarray(neural)
        alignment = int(alignment)
        start = alignment - STATS_PRE_BINS
        stop = alignment + STATS_POST_BINS

        if start < 0:
            continue

        if post_movement < STATS_POST_BINS or stop > neural.shape[1]:
            continue

        neural_windows.append(neural[:, start:stop])
        intent_flags.append(bool(intent != 0))

    if not neural_windows:
        return np.empty(0), np.empty(0, dtype=bool)

    return np.stack(neural_windows), np.asarray(intent_flags, dtype=bool)


def load_relaxed_trial_windows():
    windows = {
        participant: {
            "brain": {"neural": [], "intent": []},
            "behavioral": {"neural": [], "intent": []},
        }
        for participant in PARTICIPANT_ORDER
    }

    for participant in PARTICIPANT_ORDER:
        for session_name in PARTICIPANTS[participant]:
            session = load_session(participant, session_name).copy()
            session["PredOnset"] = session["Predictions"].apply(first_one)

            brain_trials = session[
                session["BlockType"].str.startswith("ftb", na=False)
                & (session["PlayerID"] == 3)
            ].reset_index(drop=True)

            brain_neural, brain_intent = collect_relaxed_trials(
                brain_trials, "PredOnset"
            )
            windows[participant]["brain"]["neural"].append(brain_neural)
            windows[participant]["brain"]["intent"].append(brain_intent)

            behavioral_trials = session[
                session["BlockType"].str.startswith(("ftb", "train"), na=False)
                & (session["PlayerID"] == 2)
            ].reset_index(drop=True)

            behavioral_neural, behavioral_intent = collect_relaxed_trials(
                behavioral_trials, "Onset"
            )
            windows[participant]["behavioral"]["neural"].append(
                behavioral_neural
            )
            windows[participant]["behavioral"]["intent"].append(
                behavioral_intent
            )

    return windows


def calculate_session_z_parameters(profiles, participant):
    """Clip limit, mean and SD per session, taken from the figure's trials."""
    grouped = group_by_intention(profiles[participant])
    parameters = []

    for positive_trials, negative_trials in zip(
        grouped["positive_neural"], grouped["negative_neural"]
    ):
        pooled = []
        for trials in (positive_trials, negative_trials):
            trials = np.asarray(trials)
            if trials.size:
                pooled.append(
                    gaussian_filter1d(
                        trials, sigma=MUA_SIGMA, axis=-1
                    ).sum(axis=1)
                )

        pooled = concatenate_nonempty(pooled)
        if not pooled.size:
            parameters.append(None)
            continue

        upper_limit = np.percentile(pooled, MUA_PERCENTILE)
        pooled = np.clip(pooled, 0, upper_limit)
        parameters.append(
            {
                "upper_limit": float(upper_limit),
                "mean": float(pooled.mean()),
                "std": float(pooled.std()) or 1.0,
            }
        )

    return parameters


def calculate_relaxed_trial_traces(
    windows,
    z_parameters,
    participant,
    source_key,
    positive,
):
    """Z-scored MUA traces for the exact pre-onset trials used in the test."""
    session_windows = windows[participant][source_key]
    traces = []

    for session_index, (neural, intent) in enumerate(
        zip(session_windows["neural"], session_windows["intent"])
    ):
        neural = np.asarray(neural)
        if not neural.size or session_index >= len(z_parameters):
            continue

        parameters = z_parameters[session_index]
        if parameters is None:
            continue

        selected = neural[intent if positive else ~intent]
        if not selected.size:
            continue

        mua = gaussian_filter1d(selected, sigma=MUA_SIGMA, axis=-1).sum(axis=1)
        mua = np.clip(mua, 0, parameters["upper_limit"])
        z_scored = (mua - parameters["mean"]) / parameters["std"]
        traces.append(
            z_scored[
                :, STATS_PAD_BINS:STATS_PAD_BINS + PRE_ONSET_WINDOW_BINS
            ]
        )

    return concatenate_nonempty(traces)


def calculate_relaxed_trial_means(
    windows,
    z_parameters,
    participant,
    source_key,
    positive,
):
    """One value per trial: mean z-scored MUA over the pre-onset window."""
    traces = calculate_relaxed_trial_traces(
        windows,
        z_parameters,
        participant,
        source_key,
        positive,
    )
    if not traces.size:
        return np.empty(0)
    return traces.mean(axis=1)


def plot_pre_onset_mua_zoom(profiles, windows=None):
    """Plot the exact -200-to-0 ms MUA traces entering the statistical test."""
    if windows is None:
        windows = load_relaxed_trial_windows()

    # Test the 10 bins from -200 to -20 ms; keep 0 ms visible for alignment.
    time_s = (
        np.arange(-PRE_ONSET_WINDOW_BINS, 0) * BIN_MS / 1000
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 4.0),
        sharex=True,
        sharey=True,
    )

    axes[0, 0].set_title("Brain-based prediction")
    axes[0, 1].set_title("Behavior-based prediction")
    axes[0, 0].set_ylabel("Intention = Yes\nMUA (z-scored)")
    axes[1, 0].set_ylabel("Intention = No\nMUA (z-scored)")

    source_columns = (("brain", 0), ("behavioral", 1))
    intention_rows = ((True, 0), (False, 1))

    for participant in PARTICIPANT_ORDER:
        z_parameters = calculate_session_z_parameters(profiles, participant)

        for source_key, column in source_columns:
            for positive, row in intention_rows:
                traces = calculate_relaxed_trial_traces(
                    windows,
                    z_parameters,
                    participant,
                    source_key,
                    positive,
                )
                plot_mean_std(
                    axes[row, column],
                    time_s,
                    traces,
                    PALETTE[participant],
                    label=(
                        PARTICIPANT_LABELS[participant]
                        if row == 0 and column == 0
                        else None
                    ),
                )

    for row in range(2):
        for column in range(2):
            axis = axes[row, column]
            axis.axvline(0, color="red", linewidth=1.2)
            axis.set_xlim(-PRE_ONSET_EFFECTIVE_WINDOW_MS / 1000, 0)
            axis.set_ylim(-2, 5)
            axis.set_xticks([-0.2, -0.1, 0.0])

    axes[0, 0].legend(loc="upper left", frameon=False, fontsize=8)
    axes[1, 0].set_xlabel("Time relative to onset (s)")
    axes[1, 1].set_xlabel("Time relative to onset (s)")

    fig.subplots_adjust(hspace=0.22, wspace=0.18)
    fig.tight_layout()
    fig.savefig(PRE_ONSET_ZOOM_OUTPUT_FILE, bbox_inches="tight", dpi=500)
    print(f"Saved {PRE_ONSET_ZOOM_OUTPUT_FILE}")


def summarize_error(values, mode=PRE_ONSET_DUMBBELL_ERROR):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {
            "n": 0,
            "mean": np.nan,
            "lower": np.nan,
            "upper": np.nan,
        }

    mean = float(values.mean())

    if values.size == 1:
        spread = 0.0
    else:
        sd = float(values.std(ddof=1))
        if mode == "sd":
            spread = sd
        else:
            sem = sd / np.sqrt(values.size)
            critical = stats.t.ppf(1 - STATS_ALPHA / 2, values.size - 1)
            spread = critical * sem

    return {
        "n": int(values.size),
        "mean": mean,
        "lower": mean - spread,
        "upper": mean + spread,
    }


def calculate_strict_trial_means(processed, participant, source_key, positive):
    condition_key = "positive" if positive else "negative"
    prediction_type = TYPE_BRAIN if source_key == "brain" else TYPE_BEHAVIOR

    traces = np.asarray(processed[participant][condition_key]["neural"], dtype=float)
    trial_types = np.asarray(processed[participant][condition_key]["type"])

    if not traces.size or not trial_types.size:
        return np.empty(0)

    selected = traces[trial_types == prediction_type]
    if not selected.size:
        return np.empty(0)

    start = PREMOVEMENT - PRE_ONSET_WINDOW_BINS
    stop = PREMOVEMENT
    return selected[:, start:stop].mean(axis=1)


def calculate_relaxed_plot_trial_means(windows, z_parameters, participant, source_key, positive):
    return calculate_relaxed_trial_means(
        windows,
        z_parameters,
        participant,
        source_key,
        positive,
    )


def draw_pre_onset_dumbbell_axis(axis, summary_getter, title):
    x_positions = np.array([0.0, 1.0])
    offsets = np.linspace(-0.06, 0.06, len(PARTICIPANT_ORDER))

    axis.axhline(0, color="0.8", linewidth=1, linestyle="--", zorder=0)

    for offset, participant in zip(offsets, PARTICIPANT_ORDER):
        brain_values = summary_getter(participant, "brain")
        behavior_values = summary_getter(participant, "behavioral")

        brain_summary = summarize_error(brain_values)
        behavior_summary = summarize_error(behavior_values)

        if not np.isfinite(brain_summary["mean"]) or not np.isfinite(behavior_summary["mean"]):
            continue

        x = x_positions + offset
        y = np.array([brain_summary["mean"], behavior_summary["mean"]])

        axis.plot(
            x,
            y,
            color=PALETTE[participant],
            linewidth=1.8,
            marker="o",
            markersize=6,
            label=PARTICIPANT_LABELS[participant],
            zorder=3,
        )

        axis.errorbar(
            x[0],
            brain_summary["mean"],
            yerr=np.array(
                [[brain_summary["mean"] - brain_summary["lower"]],
                 [brain_summary["upper"] - brain_summary["mean"]]]
            ),
            color=PALETTE[participant],
            linewidth=1.2,
            capsize=3,
            zorder=4,
        )
        axis.errorbar(
            x[1],
            behavior_summary["mean"],
            yerr=np.array(
                [[behavior_summary["mean"] - behavior_summary["lower"]],
                 [behavior_summary["upper"] - behavior_summary["mean"]]]
            ),
            color=PALETTE[participant],
            linewidth=1.2,
            capsize=3,
            zorder=4,
        )

    axis.set_title(title)
    axis.set_xlim(-0.35, 1.35)
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["Brain-based", "Behavior-based"])


def plot_pre_onset_dumbbells_strict(processed):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.8), sharey=True)

    def summary_getter_factory(positive):
        def getter(participant, source_key):
            return calculate_strict_trial_means(
                processed,
                participant,
                source_key,
                positive,
            )
        return getter

    draw_pre_onset_dumbbell_axis(
        axes[0],
        summary_getter_factory(True),
        "Intention = Yes",
    )
    draw_pre_onset_dumbbell_axis(
        axes[1],
        summary_getter_factory(False),
        "Intention = No",
    )

    axes[0].set_ylabel("Mean pre-onset MUA (z-scored)")
    axes[0].legend(loc="best", frameon=False, fontsize=8)

    error_label = "95% CI" if PRE_ONSET_DUMBBELL_ERROR == "ci" else "SD"
    fig.suptitle(
        f"Pre-onset MUA summary using main-figure trials (error bars = {error_label})",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(PRE_ONSET_DUMBBELL_STRICT_OUTPUT_FILE, bbox_inches="tight", dpi=500)
    print(f"Saved {PRE_ONSET_DUMBBELL_STRICT_OUTPUT_FILE}")


def plot_pre_onset_dumbbells_relaxed(profiles, windows=None):
    if windows is None:
        windows = load_relaxed_trial_windows()

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.8), sharey=True)

    def summary_getter_factory(positive):
        def getter(participant, source_key):
            z_parameters = calculate_session_z_parameters(profiles, participant)
            return calculate_relaxed_plot_trial_means(
                windows,
                z_parameters,
                participant,
                source_key,
                positive,
            )
        return getter

    draw_pre_onset_dumbbell_axis(
        axes[0],
        summary_getter_factory(True),
        "Intention = Yes",
    )
    draw_pre_onset_dumbbell_axis(
        axes[1],
        summary_getter_factory(False),
        "Intention = No",
    )

    axes[0].set_ylabel("Mean pre-onset MUA (z-scored)")
    axes[0].legend(loc="best", frameon=False, fontsize=8)

    error_label = "95% CI" if PRE_ONSET_DUMBBELL_ERROR == "ci" else "SD"
    fig.suptitle(
        f"Pre-onset MUA summary using relaxed test trials (error bars = {error_label})",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(PRE_ONSET_DUMBBELL_RELAXED_OUTPUT_FILE, bbox_inches="tight", dpi=500)
    print(f"Saved {PRE_ONSET_DUMBBELL_RELAXED_OUTPUT_FILE}")


def draw_stats_dumbbell_axis(
    axis,
    profiles,
    windows,
    z_parameters_by_participant,
    results,
    positive,
    panel_label,
):
    """Draw brain-vs-behavior participant means for the exact test trials."""
    x_positions = np.array([0.0, 1.0])
    offsets = np.linspace(-0.06, 0.06, len(PARTICIPANT_ORDER))
    intention_label = "Yes" if positive else "No"

    plotted = []
    sig_entries = []

    for offset, participant in zip(offsets, PARTICIPANT_ORDER):
        z_parameters = z_parameters_by_participant[participant]

        brain_values = calculate_relaxed_trial_means(
            windows,
            z_parameters,
            participant,
            "brain",
            positive,
        )
        behavior_values = calculate_relaxed_trial_means(
            windows,
            z_parameters,
            participant,
            "behavioral",
            positive,
        )

        brain_summary = summarize_error(brain_values)
        behavior_summary = summarize_error(behavior_values)

        if not (
            np.isfinite(brain_summary["mean"])
            and np.isfinite(behavior_summary["mean"])
        ):
            continue

        result_row = results[
            (results["participant"] == PARTICIPANT_LABELS[participant])
            & (results["intention"] == intention_label)
        ]
        p_corrected = (
            float(result_row.iloc[0]["p_bonferroni"])
            if len(result_row)
            else np.nan
        )
        stars = significance_label(p_corrected)
        sig_entries.append(f"{PARTICIPANT_LABELS[participant]} {stars}")

        plotted.append(
            {
                "participant": participant,
                "offset": offset,
                "brain": brain_summary,
                "behavior": behavior_summary,
            }
        )

    axis.set_title(panel_label)
    axis.set_xlim(-0.35, 1.35)
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["Brain-based", "Behavior-based"])

    if not plotted:
        return None, None, ""

    all_bounds = []
    for item in plotted:
        all_bounds.extend(
            [
                item["brain"]["lower"],
                item["brain"]["upper"],
                item["behavior"]["lower"],
                item["behavior"]["upper"],
            ]
        )
    data_min = float(np.nanmin(all_bounds))
    data_max = float(np.nanmax(all_bounds))
    data_span = max(data_max - data_min, 0.5)

    axis.axhline(0, color="0.8", linewidth=1, linestyle="--", zorder=0)

    for item in plotted:
        participant = item["participant"]
        offset = item["offset"]
        brain_summary = item["brain"]
        behavior_summary = item["behavior"]
        color = PALETTE[participant]

        x = x_positions + offset
        y = np.array([brain_summary["mean"], behavior_summary["mean"]])

        axis.plot(
            x,
            y,
            color=color,
            linewidth=1.8,
            zorder=3,
        )

        axis.plot(
            x,
            y,
            linestyle="None",
            marker="o",
            markersize=6,
            color=color,
            label=PARTICIPANT_LABELS[participant],
            zorder=4,
        )

        axis.errorbar(
            x[0],
            brain_summary["mean"],
            yerr=np.array(
                [[brain_summary["mean"] - brain_summary["lower"]],
                 [brain_summary["upper"] - brain_summary["mean"]]]
            ),
            color=color,
            linewidth=1.2,
            capsize=3,
            zorder=4,
        )
        axis.errorbar(
            x[1],
            behavior_summary["mean"],
            yerr=np.array(
                [[behavior_summary["mean"] - behavior_summary["lower"]],
                 [behavior_summary["upper"] - behavior_summary["mean"]]]
            ),
            color=color,
            linewidth=1.2,
            capsize=3,
            zorder=4,
        )

    panel_sig_text = "Bonferroni-corrected p: " + ", ".join(sig_entries)
    return data_min - 0.10 * data_span, data_max + 0.10 * data_span, panel_sig_text

def plot_pre_onset_dumbbell_stats(profiles, results, windows=None):
    """Dumbbell summary for the exact relaxed trials used in the statistical test."""
    if windows is None:
        windows = load_relaxed_trial_windows()

    z_parameters_by_participant = {
        participant: calculate_session_z_parameters(profiles, participant)
        for participant in PARTICIPANT_ORDER
    }

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 4.8), sharey=True)

    yes_ymin, yes_ymax, yes_sig_text = draw_stats_dumbbell_axis(
        axes[0],
        profiles,
        windows,
        z_parameters_by_participant,
        results,
        positive=True,
        panel_label="Intention = Yes",
    )
    no_ymin, no_ymax, no_sig_text = draw_stats_dumbbell_axis(
        axes[1],
        profiles,
        windows,
        z_parameters_by_participant,
        results,
        positive=False,
        panel_label="Intention = No",
    )

    limits = [
        (ymin, ymax)
        for ymin, ymax in ((yes_ymin, yes_ymax), (no_ymin, no_ymax))
        if ymin is not None and ymax is not None
    ]
    if limits:
        global_ymin = min(limit[0] for limit in limits)
        global_ymax = max(limit[1] for limit in limits)
        for axis in axes:
            axis.set_ylim(global_ymin, global_ymax)

    axes[0].set_ylabel("Mean pre-onset MUA (z-scored)")
    axes[0].legend(loc="best", frameon=False, fontsize=8)

    axes[0].text(
        0.5,
        -0.22,
        yes_sig_text,
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )
    axes[1].text(
        0.5,
        -0.22,
        no_sig_text,
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(PRE_ONSET_DUMBBELL_OUTPUT_FILE, bbox_inches="tight", dpi=500)
    print(f"Saved {PRE_ONSET_DUMBBELL_OUTPUT_FILE}")


def compare_brain_vs_behavior(brain_values, behavior_values):
    brain_values = np.asarray(brain_values, dtype=float)
    behavior_values = np.asarray(behavior_values, dtype=float)

    n_brain = brain_values.size
    n_behavior = behavior_values.size

    result = {
        "n_brain": n_brain,
        "n_behavior": n_behavior,
        "mean_brain": float(brain_values.mean()) if n_brain else np.nan,
        "mean_behavior": (
            float(behavior_values.mean()) if n_behavior else np.nan
        ),
        "sd_brain": float(brain_values.std(ddof=1)) if n_brain > 1 else np.nan,
        "sd_behavior": (
            float(behavior_values.std(ddof=1)) if n_behavior > 1 else np.nan
        ),
        "difference": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "t_statistic": np.nan,
        "df": np.nan,
        "p_value": np.nan,
        "hedges_g": np.nan,
    }

    if n_brain < 2 or n_behavior < 2:
        return result

    variance_brain = brain_values.var(ddof=1)
    variance_behavior = behavior_values.var(ddof=1)
    difference = float(brain_values.mean() - behavior_values.mean())
    result["difference"] = difference

    standard_error = np.sqrt(
        variance_brain / n_brain + variance_behavior / n_behavior
    )

    test_result = stats.ttest_ind(
        brain_values, behavior_values, equal_var=False
    )
    result["t_statistic"] = float(test_result.statistic)
    result["p_value"] = float(test_result.pvalue)

    if standard_error > 0:
        welch_df = (
            variance_brain / n_brain + variance_behavior / n_behavior
        ) ** 2 / (
            (variance_brain / n_brain) ** 2 / (n_brain - 1)
            + (variance_behavior / n_behavior) ** 2 / (n_behavior - 1)
        )
        critical = stats.t.ppf(1 - STATS_ALPHA / 2, welch_df)
        result["df"] = float(welch_df)
        result["ci_low"] = float(difference - critical * standard_error)
        result["ci_high"] = float(difference + critical * standard_error)

    pooled_sd = np.sqrt(
        (
            (n_brain - 1) * variance_brain
            + (n_behavior - 1) * variance_behavior
        )
        / (n_brain + n_behavior - 2)
    )

    if pooled_sd > 0:
        cohens_d = difference / pooled_sd
        correction = 1.0 - 3.0 / (4.0 * (n_brain + n_behavior) - 9.0)
        result["hedges_g"] = float(cohens_d * correction)

    return result


def run_pre_onset_comparisons(profiles, windows=None):
    if windows is None:
        windows = load_relaxed_trial_windows()
    rows = []

    for participant in PARTICIPANT_ORDER:
        z_parameters = calculate_session_z_parameters(profiles, participant)

        for intention_key, intention_label in INTENTION_CONDITIONS:
            positive = intention_key == "positive"

            brain_values = calculate_relaxed_trial_means(
                windows, z_parameters, participant, "brain", positive
            )
            behavior_values = calculate_relaxed_trial_means(
                windows, z_parameters, participant, "behavioral", positive
            )

            row = {
                "participant": PARTICIPANT_LABELS[participant],
                "intention": intention_label,
            }
            row.update(
                compare_brain_vs_behavior(brain_values, behavior_values)
            )
            rows.append(row)

    results = pd.DataFrame(rows)

    p_values = results["p_value"].to_numpy(dtype=float)
    results["p_bonferroni"] = bonferroni_adjust(p_values)
    results["significant"] = results["p_bonferroni"] < STATS_ALPHA
    results["n_tests"] = int(np.isfinite(p_values).sum())

    return results


def print_pre_onset_comparisons(results):
    display = results[
        [
            "participant",
            "intention",
            "t_statistic",
            "df",
            "p_bonferroni",
            "significant",
        ]
    ].copy()
    display["t_statistic"] = display["t_statistic"].map(
        lambda value: format_number(value, 2)
    )
    display["df"] = display["df"].map(lambda value: format_number(value, 1))
    display["p_bonferroni"] = display["p_bonferroni"].map(format_p_value)
    display["significant"] = display["significant"].map(
        {True: "yes", False: "no"}
    )

    print("\nPre-onset MUA: brain vs behavior (Welch; Bonferroni-adjusted p)")
    print(display.to_string(index=False))

    if SAVE_STATS_TABLE:
        results.to_csv(STATS_TABLE_FILE, index=False)
        print(f"Saved: {STATS_TABLE_FILE.resolve()}")


def main():
    profiles = load_trial_profiles()
    processed = process_profiles(profiles)

    brain_departure_results, brain_departure_average = (
        calculate_mua_departure_times(processed, TYPE_BRAIN)
    )
    print_mua_departure_times(
        brain_departure_results,
        brain_departure_average,
        prediction_label="Brain-based",
        onset_label="prediction onset",
    )

    behavior_departure_results, behavior_departure_average = (
        calculate_mua_departure_times(processed, TYPE_BEHAVIOR)
    )
    print_mua_departure_times(
        behavior_departure_results,
        behavior_departure_average,
        prediction_label="Behavior-based",
        onset_label="behavioral onset",
    )

    (
        brain_force_results,
        brain_force_crossed,
        brain_force_total,
        brain_force_percentage,
    ) = calculate_force_crossings(profiles, "brain")
    print_force_crossings(
        brain_force_results,
        brain_force_crossed,
        brain_force_total,
        brain_force_percentage,
        prediction_label="Brain-based",
        onset_label="prediction onset",
    )

    (
        behavior_force_results,
        behavior_force_crossed,
        behavior_force_total,
        behavior_force_percentage,
    ) = calculate_force_crossings(profiles, "behavioral")
    print_force_crossings(
        behavior_force_results,
        behavior_force_crossed,
        behavior_force_total,
        behavior_force_percentage,
        prediction_label="Behavior-based",
        onset_label="behavioral onset",
    )

    (
        brain_negative_force_results,
        brain_negative_force_crossed,
        brain_negative_force_total,
        brain_negative_force_percentage,
    ) = calculate_negative_force_crossings(profiles, "brain")
    print_negative_force_crossings(
        brain_negative_force_results,
        brain_negative_force_crossed,
        brain_negative_force_total,
        brain_negative_force_percentage,
        prediction_label="Brain-based",
        onset_label="prediction onset",
    )

    (
        behavior_negative_force_results,
        behavior_negative_force_crossed,
        behavior_negative_force_total,
        behavior_negative_force_percentage,
    ) = calculate_negative_force_crossings(profiles, "behavioral")
    print_negative_force_crossings(
        behavior_negative_force_results,
        behavior_negative_force_crossed,
        behavior_negative_force_total,
        behavior_negative_force_percentage,
        prediction_label="Behavior-based",
        onset_label="behavioral onset",
    )

    relaxed_windows = load_relaxed_trial_windows()
    pre_onset_results = run_pre_onset_comparisons(profiles, relaxed_windows)
    print_pre_onset_comparisons(pre_onset_results)

    plot_main_figure(processed)
    plot_pre_onset_dumbbell_stats(
        profiles,
        pre_onset_results,
        relaxed_windows,
    )

    plot_summary_bars(
        positive_percentages=calculate_silent_percentages(),
        output_file=SILENT_OUTPUT_FILE,
        y_label="Silent predictions",
        negative_label="Early",
        positive_label="On-time",
        negative_color="#F3E5AB",
        positive_color="#FF8C00",
        positive_text_color="white",
        participant_text_color="white",
    )

    plot_summary_bars(
        positive_percentages=calculate_active_percentages(),
        output_file=ACTIVE_OUTPUT_FILE,
        y_label="Active predictions",
        negative_label="Intention = No",
        positive_label="Intention = Yes",
        negative_color="#999999",
        positive_color="#b1e4faff",
    )

    plt.show()


if __name__ == "__main__":
    main()