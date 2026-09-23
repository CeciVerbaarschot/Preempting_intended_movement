#!/usr/bin/env python3
from pathlib import Path
import re

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
from scipy.stats import binomtest


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_DIR = Path("../../FlipThatBucket/data")
SUBJECTS_DIR = Path("../../FlipThatBucket/Subjects")

OUTPUT_DIR = Path(__file__).resolve().parent
PREMOVEMENT_OUTPUT_FILE = OUTPUT_DIR / "figure_3_clusters_on_premove.pdf"
BASELINE_OUTPUT_FILE = OUTPUT_DIR / "sup_figure_3_clusters_on_baseline.pdf"
ORDERED_SLOPES_OUTPUT_FILE = OUTPUT_DIR / "figure_3_ordered_slopes_example.pdf"
PRETRIAL_PREDICTIONS_OUTPUT_FILE = OUTPUT_DIR / "pretrial_classifier_predictions.csv"

BIN_SECONDS = 0.02
TRIAL_BINS = 50
BASELINE_START_STATE = 3
CLASSIFIER_EARLY_MOVEMENT_EXCLUSION_SECONDS = 1.0
CLASSIFIER_EARLY_MOVEMENT_EXCLUSION_BINS = int(
    round(CLASSIFIER_EARLY_MOVEMENT_EXCLUSION_SECONDS / BIN_SECONDS)
)

SMOOTHING_SIGMA = 2
OUTLIER_Z_THRESHOLD = 3

CLUSTER_SD_THRESHOLD = 2
CLUSTER_ALPHA = 0.05

VALID_BLOCK_TYPES = {
    "train",
    "ftb-behavior-test",
    "ftb-mixed-test",
    "ftb-brain-test",
}

# Classifier FPR excludes training blocks; neural slope analyses do not.
CLASSIFIER_BLOCK_TYPES = {
    "ftb-behavior-test",
    "ftb-mixed-test",
    "ftb-brain-test",
}

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

PARTICIPANT_TITLES = {
    "p2": "Participant P2",
    "p3": "Participant P3",
    "c1": "Participant C1",
}

ORDERED_SLOPES_EXAMPLE_PARTICIPANT = "p3"
ORDERED_SLOPES_EXAMPLE_SESSION_INDEX = 0

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"


def load_session(participant, session_name):
    path = DATA_DIR / participant / f"{session_name}.pkl"
    return pd.read_pickle(path)


def extract_premovement_trials(session):
    trials = []

    participant_trials = session[
        (session["PlayerID"] == 1)
        & session["BlockType"].isin(VALID_BLOCK_TYPES)
    ]

    for row in participant_trials.itertuples(index=False):
        onset = int(row.Onset)
        if onset < TRIAL_BINS:
            continue

        neural = np.asarray(row.Neural)
        segment = neural[:, onset - TRIAL_BINS:onset]

        if segment.shape[1] == TRIAL_BINS:
            trials.append(segment)

    if not trials:
        return np.empty((0, 0, TRIAL_BINS))

    return np.stack(trials)


def process_trials(trials):
    trials = np.asarray(trials, dtype=float)

    if trials.ndim != 3 or trials.shape[0] == 0:
        raise ValueError("No valid neural trials were available for processing.")

    firing_rates = (
        gaussian_filter1d(trials, sigma=SMOOTHING_SIGMA, axis=-1)
        / BIN_SECONDS
    )

    mean_activity = firing_rates.mean(axis=1)
    median = np.median(mean_activity, axis=0)
    mad = np.median(np.abs(mean_activity - median), axis=0)

    robust_scale = 1.4826 * mad
    robust_scale[robust_scale == 0] = np.nan

    robust_z = (mean_activity - median) / robust_scale
    bad_trials = np.any(np.abs(robust_z) > OUTLIER_Z_THRESHOLD, axis=1)

    filtered = firing_rates[~bad_trials]
    if filtered.shape[0] == 0:
        raise ValueError("All neural trials were removed as outliers.")

    excluded_fraction = bad_trials.mean()
    return filtered, excluded_fraction


def calculate_trial_order_slopes(firing_rates):
    n_trials = firing_rates.shape[0]
    if n_trials < 2:
        raise ValueError("At least two trials are required to calculate slopes.")

    trial_order = np.arange(1, n_trials + 1, dtype=float)
    centered_order = trial_order - trial_order.mean()
    denominator = np.sum(centered_order ** 2)

    return np.tensordot(
        centered_order,
        firing_rates,
        axes=(0, 0),
    ) / denominator


def classify_channels(slopes):
    slopes = np.asarray(slopes, dtype=float)

    median = np.nanmedian(slopes)
    mad = np.nanmedian(np.abs(slopes - median))
    # Convert MAD to a robust scale and threshold symmetrically around zero.
    robust_sigma = 1.4826 * mad
    threshold = CLUSTER_SD_THRESHOLD * robust_sigma

    positive_bins = slopes > threshold
    negative_bins = slopes < -threshold

    positive_counts = positive_bins.sum(axis=1)
    negative_counts = negative_bins.sum(axis=1)
    effective_counts = positive_counts + negative_counts

    labels = np.full(slopes.shape[0], "noise", dtype=object)

    for channel in range(slopes.shape[0]):
        n_effective = int(effective_counts[channel])
        if n_effective == 0:
            continue

        n_positive = int(positive_counts[channel])
        n_negative = int(negative_counts[channel])

        if n_positive > n_negative:
            p_value = binomtest(
                n_positive,
                n_effective,
                p=0.5,
                alternative="greater",
            ).pvalue
            if p_value <= CLUSTER_ALPHA:
                labels[channel] = "positive"

        elif n_negative > n_positive:
            p_value = binomtest(
                n_positive,
                n_effective,
                p=0.5,
                alternative="less",
            ).pvalue
            if p_value <= CLUSTER_ALPHA:
                labels[channel] = "negative"

    positive_channels = np.flatnonzero(labels == "positive")
    negative_channels = np.flatnonzero(labels == "negative")
    noise_channels = np.flatnonzero(labels == "noise")

    return positive_channels, negative_channels, noise_channels


def analyze_premovement_activity():
    results = {}

    for participant in PARTICIPANT_ORDER:
        rates = []
        slopes_by_session = []
        positive_indices = []
        negative_indices = []
        noise_indices = []

        for session_name in PARTICIPANTS[participant]:
            session = load_session(participant, session_name)
            trials = extract_premovement_trials(session)
            firing_rates, _ = process_trials(trials)
            slopes = calculate_trial_order_slopes(firing_rates)
            positive, negative, noise = classify_channels(slopes)

            if positive.size == 0 or negative.size == 0:
                raise ValueError(
                    f"{participant.upper()} {session_name} did not contain "
                    "both positive- and negative-slope channel clusters."
                )

            rates.append(firing_rates)
            slopes_by_session.append(slopes)
            positive_indices.append(positive)
            negative_indices.append(negative)
            noise_indices.append(noise)


        results[participant] = {
            "rates": rates,
            "slopes": slopes_by_session,
            "positive_indices": positive_indices,
            "negative_indices": negative_indices,
            "noise_indices": noise_indices,
        }

    return results


def calculate_cluster_zscores(rates, positive_indices, negative_indices):
    positive_sessions = []
    negative_sessions = []

    for session_rates, positive, negative in zip(
        rates,
        positive_indices,
        negative_indices,
    ):
        positive_activity = session_rates[:, positive, :].mean(axis=1)
        negative_activity = session_rates[:, negative, :].mean(axis=1)

        positive_std = positive_activity.std()
        negative_std = negative_activity.std()

        if positive_std == 0 or negative_std == 0:
            raise ValueError("A channel cluster had no activity variance.")

        positive_sessions.append(
            (positive_activity - positive_activity.mean()) / positive_std
        )
        negative_sessions.append(
            (negative_activity - negative_activity.mean()) / negative_std
        )

    return positive_sessions, negative_sessions


def build_cluster_profiles(results, rates_by_participant=None):
    positive_profiles = {}
    negative_profiles = {}

    for participant in PARTICIPANT_ORDER:
        participant_results = results[participant]
        rates = (
            participant_results["rates"]
            if rates_by_participant is None
            else rates_by_participant[participant]
        )

        positive, negative = calculate_cluster_zscores(
            rates,
            participant_results["positive_indices"],
            participant_results["negative_indices"],
        )
        positive_profiles[participant] = positive
        negative_profiles[participant] = negative

    return positive_profiles, negative_profiles


def create_trial_order_colormap(polarity):
    if polarity == "positive":
        end_color = "#CC79A7"
    elif polarity == "negative":
        end_color = "#0072B2"
    else:
        raise ValueError("polarity must be 'positive' or 'negative'.")

    colormap = LinearSegmentedColormap.from_list(
        f"trial_order_{polarity}",
        ["#000000", "#ffffff", end_color],
    )
    scalar_map = ScalarMappable(
        cmap=colormap,
        norm=Normalize(vmin=0, vmax=1),
    )
    scalar_map.set_array(np.linspace(0, 1, 256))
    return colormap, scalar_map


def plot_session_quartiles(
    axis,
    session_profiles,
    polarity,
    quartile_fraction,
    trial_alpha,
    mean_linewidth,
    y_limits,
    title,
    show_legend,
    show_xlabel,
    show_ylabel,
    x_label,
):
    colormap, scalar_map = create_trial_order_colormap(polarity)

    early_trials = []
    late_trials = []
    n_bins = None

    for session_data in session_profiles:
        session_data = np.asarray(session_data)
        if session_data.ndim != 2 or session_data.shape[0] == 0:
            raise ValueError("Each plotted session must contain trial-by-time data.")

        n_trials, n_bins = session_data.shape
        quartile_size = max(1, int(np.floor(n_trials * quartile_fraction)))
        trial_colors = colormap(np.linspace(0, 1, n_trials))

        for trial, color in zip(session_data, trial_colors):
            axis.plot(trial, color=color, alpha=trial_alpha)

        early_trials.append(session_data[:quartile_size])
        late_trials.append(session_data[-quartile_size:])

    if n_bins is None:
        raise ValueError("No session profiles were provided for plotting.")

    overall_early = np.vstack(early_trials).mean(axis=0)
    overall_late = np.vstack(late_trials).mean(axis=0)

    axis.plot(overall_early, color=colormap(0.0), linewidth=mean_linewidth)
    axis.plot(overall_late, color=colormap(1.0), linewidth=mean_linewidth)

    time_s = np.linspace(-1.0, 0.0, n_bins)
    tick_times = [-1.0, -0.75, -0.5, -0.25, 0.0]
    tick_bins = [
        int(np.argmin(np.abs(time_s - tick_time)))
        for tick_time in tick_times
    ]

    axis.set_xticks(tick_bins)
    axis.set_xticklabels([f"{tick_time:.2f}" for tick_time in tick_times])
    axis.set_xlabel(x_label if show_xlabel else "")
    axis.set_ylabel("MUA (z-scored)" if show_ylabel else "")
    axis.set_ylim(*y_limits)
    axis.set_title(title)

    if show_legend:
        axis.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color="gray",
                    linewidth=1,
                    alpha=trial_alpha,
                    label="Single trials",
                ),
                Line2D(
                    [0],
                    [0],
                    color=colormap(0.0),
                    linewidth=mean_linewidth,
                    label="Early trials (P25)",
                ),
                Line2D(
                    [0],
                    [0],
                    color=colormap(1.0),
                    linewidth=mean_linewidth,
                    label="Late trials (P75)",
                ),
            ],
            frameon=True,
            fontsize=8,
            loc="upper left",
        )

    return scalar_map


def plot_cluster_grid(
    positive_profiles,
    negative_profiles,
    x_label,
    positive_ylim=(-3, 6.5),
    negative_ylim=(-3, 5.5),
):
    n_participants = len(PARTICIPANT_ORDER)
    fig = plt.figure(figsize=(9, 6))

    grid = fig.add_gridspec(
        2,
        n_participants + 1,
        width_ratios=[1] * n_participants + [0.035],
        wspace=0.10,
        hspace=0.18,
    )

    positive_axes = [fig.add_subplot(grid[0, 0])]
    negative_axes = [fig.add_subplot(grid[1, 0], sharex=positive_axes[0])]

    for column in range(1, n_participants):
        positive_axis = fig.add_subplot(
            grid[0, column],
            sharey=positive_axes[0],
        )
        negative_axis = fig.add_subplot(
            grid[1, column],
            sharex=positive_axis,
            sharey=negative_axes[0],
        )
        positive_axes.append(positive_axis)
        negative_axes.append(negative_axis)

    positive_colorbar_axis = fig.add_subplot(grid[0, -1])
    negative_colorbar_axis = fig.add_subplot(grid[1, -1])

    positive_scalar_map = None
    negative_scalar_map = None

    for column, participant in enumerate(PARTICIPANT_ORDER):
        positive_scalar_map = plot_session_quartiles(
            axis=positive_axes[column],
            session_profiles=positive_profiles[participant],
            polarity="positive",
            quartile_fraction=0.25,
            trial_alpha=0.25,
            mean_linewidth=1.5,
            y_limits=positive_ylim,
            title=PARTICIPANT_TITLES[participant],
            show_legend=(column == 0),
            show_xlabel=False,
            show_ylabel=(column == 0),
            x_label=x_label,
        )

        negative_scalar_map = plot_session_quartiles(
            axis=negative_axes[column],
            session_profiles=negative_profiles[participant],
            polarity="negative",
            quartile_fraction=0.25,
            trial_alpha=0.25,
            mean_linewidth=1.5,
            y_limits=negative_ylim,
            title="",
            show_legend=(column == 0),
            show_xlabel=True,
            show_ylabel=(column == 0),
            x_label=x_label,
        )

        if column != 0:
            positive_axes[column].tick_params(labelleft=False)
            negative_axes[column].tick_params(labelleft=False)

    for axis in positive_axes:
        axis.tick_params(labelbottom=False)

    positive_colorbar = fig.colorbar(
        positive_scalar_map,
        cax=positive_colorbar_axis,
    )
    positive_colorbar.set_ticks([0, 1])
    positive_colorbar.set_ticklabels(["Early", "Late"])
    positive_colorbar.set_label("Trial order", labelpad=-20)
    positive_colorbar.ax.tick_params(pad=1)

    negative_colorbar = fig.colorbar(
        negative_scalar_map,
        cax=negative_colorbar_axis,
    )
    negative_colorbar.set_ticks([0, 1])
    negative_colorbar.set_ticklabels(["Early", "Late"])
    negative_colorbar.set_label("Trial order", labelpad=-20)
    negative_colorbar.ax.tick_params(pad=1)
    negative_colorbar.ax.invert_yaxis()

    return fig


def plot_ordered_slopes_example(
    premovement_results,
    participant=ORDERED_SLOPES_EXAMPLE_PARTICIPANT,
    session_index=ORDERED_SLOPES_EXAMPLE_SESSION_INDEX,
):
    result = premovement_results[participant]

    slopes = np.asarray(result["slopes"][session_index])
    positive = np.asarray(result["positive_indices"][session_index], dtype=int)
    negative = np.asarray(result["negative_indices"][session_index], dtype=int)
    noise = np.asarray(result["noise_indices"][session_index], dtype=int)

    cluster_order = np.concatenate([positive, negative, noise])

    cmap = LinearSegmentedColormap.from_list(
        "blue_white_pink",
        [
            (0.0, "#0072B2"),
            (0.5, "#FFFFFF"),
            (1.0, "#CC79A7"),
        ],
    )

    vmax = np.nanpercentile(np.abs(slopes), 95)
    if vmax == 0 or not np.isfinite(vmax):
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    n_bins = slopes.shape[1]
    times = np.linspace(-1.0, 0.0, n_bins)
    tick_times = [-1.0, -0.75, -0.5, -0.25, 0.0]
    tick_bins = [
        int(np.argmin(np.abs(times - tick_time)))
        for tick_time in tick_times
    ]

    fig, axis = plt.subplots(figsize=(3, 4.5))
    axis.imshow(
        slopes[cluster_order],
        cmap=cmap,
        aspect="auto",
        norm=norm,
    )

    axis.set_xticks(tick_bins)
    axis.set_xticklabels([f"{tick_time:.2f}" for tick_time in tick_times])
    axis.set_xlabel("Time to action onset (s)")

    tick_positions = []
    tick_labels = []
    tick_colors = []

    start = 0
    if positive.size:
        tick_positions.append(start + positive.size / 2)
        tick_labels.append("Positive")
        tick_colors.append("#CC79A7")
        start += positive.size

    if negative.size:
        tick_positions.append(start + negative.size / 2)
        tick_labels.append("Negative")
        tick_colors.append("#0072B2")
        start += negative.size

    if noise.size:
        tick_positions.append(start + noise.size / 2)
        tick_labels.append("Null")
        tick_colors.append("black")

    axis.set_yticks(tick_positions)
    axis.set_yticklabels(tick_labels, rotation=90, va="center", size=9)

    for label, color in zip(axis.get_yticklabels(), tick_colors):
        label.set_color(color)

    if positive.size:
        axis.axhline(positive.size - 0.5, color="k", linewidth=1)
    if negative.size:
        axis.axhline(positive.size + negative.size - 0.5, color="k", linewidth=1)

    return fig


def extract_run_number(path):
    match = re.search(r"\d+", path.name)
    return int(match.group()) if match else 0


def get_run_files(directory):
    return sorted(
        directory.glob("Run*.mat"),
        key=extract_run_number,
    )


def load_pretrial_observations(run_path):
    with h5py.File(run_path, "r") as file:
        trial_numbers = file["data"]["trial_num"][:].reshape(-1)
        state_numbers = (
            file["data"]["TaskStateMasks"]["state_num"][:].reshape(-1)
        )
        spikes = file["data"]["SpikeCount"][:]
        # ActualPos[15] marks movement; ActualPos[18] identifies the mover.
        is_move = file["data"]["Kinematics"]["ActualPos"][15, :].reshape(-1)
        who_move = file["data"]["Kinematics"]["ActualPos"][18, :].reshape(-1)
        try:
            classifier = file["data"]["Kinematics"]["Control"][6, :].reshape(-1)
        except (KeyError, IndexError):
            # Preserve missing classifier streams as NaN.
            classifier = np.full(trial_numbers.size, np.nan, dtype=float)

    if spikes.ndim != 2:
        raise ValueError(
            f"{run_path}: SpikeCount must be two-dimensional, "
            f"but its shape is {spikes.shape}."
        )

    n_time = trial_numbers.size
    if spikes.shape[1] != n_time and spikes.shape[0] == n_time:
        spikes = spikes.T

    if spikes.shape[1] != n_time:
        raise ValueError(
            f"{run_path}: trial_num contains {n_time} time points, "
            f"but SpikeCount has shape {spikes.shape}."
        )

    spikes = spikes[::5]
    if classifier.size != n_time:
        raise ValueError(
            f"{run_path}: classifier contains {classifier.size} time points, "
            f"but trial_num contains {n_time}."
        )
    if is_move.size != n_time or who_move.size != n_time:
        raise ValueError(
            f"{run_path}: movement streams must contain {n_time} time points; "
            f"got is_move={is_move.size}, who_move={who_move.size}."
        )

    # Neural and classifier windows use separate inclusion rules.
    observations = []
    prediction_windows = []
    classifier_filter_counts = {
        "classifier_candidate_windows_before_movement_filter": 0,
        "excluded_early_participant_movement_windows": 0,
    }

    for trial_number in np.unique(trial_numbers):
        trial_mask = trial_numbers == trial_number
        trial_states = state_numbers[trial_mask]
        state_hits = np.flatnonzero(
            trial_states == BASELINE_START_STATE
        )

        if state_hits.size == 0:
            continue

        start = int(state_hits[0])
        if start < TRIAL_BINS:
            continue

        trial_spikes = spikes[:, trial_mask]
        trial_classifier = classifier[trial_mask]
        trial_is_move = is_move[trial_mask]
        trial_who_move = who_move[trial_mask]

        pretrial = trial_spikes[:, start - TRIAL_BINS:start]
        pretrial_predictions = trial_classifier[start - TRIAL_BINS:start]

        # Classifier-only: retain NaNs and exclude movement within 1 s of trial onset.
        if pretrial_predictions.size == TRIAL_BINS:
            classifier_filter_counts[
                "classifier_candidate_windows_before_movement_filter"
            ] += 1

            movement_hits = np.flatnonzero(trial_is_move == 2)
            exclude_for_early_participant_movement = False
            if movement_hits.size:
                movement_onset = int(movement_hits[0])
                mover = trial_who_move[movement_onset]
                movement_delay_bins = movement_onset - start
                exclude_for_early_participant_movement = (
                    mover == 1
                    and 0 <= movement_delay_bins
                    < CLASSIFIER_EARLY_MOVEMENT_EXCLUSION_BINS
                )

            if exclude_for_early_participant_movement:
                classifier_filter_counts[
                    "excluded_early_participant_movement_windows"
                ] += 1
            else:
                prediction_windows.append(
                    pretrial_predictions.astype(float, copy=True)
                )

        # Neural window keeps the original validity rule.
        if (
            pretrial.shape[1] == TRIAL_BINS
            and not np.isnan(pretrial).any()
        ):
            observations.append(pretrial)

    return observations, prediction_windows, classifier_filter_counts


def load_baseline_trials(participant, session_name):
    run_directory = (
        SUBJECTS_DIR
        / participant.upper()
        / "Motor"
        / session_name
    )
    run_files = get_run_files(run_directory)

    if not run_files:
        raise FileNotFoundError(
            f"No Run*.mat files were found in {run_directory}."
        )

    loaded_by_block = [
        load_pretrial_observations(run_file)
        for run_file in run_files
    ]
    observations_by_block = [item[0] for item in loaded_by_block]
    predictions_by_block = [item[1] for item in loaded_by_block]
    classifier_filter_counts_by_block = [item[2] for item in loaded_by_block]

    session = load_session(participant, session_name)
    block_types = (
        session[["BlockID", "BlockType"]]
        .drop_duplicates(subset="BlockID", keep="first")
        ["BlockType"]
        .tolist()
    )

    if len(observations_by_block) < len(block_types):
        raise ValueError(
            f"{participant.upper()} {session_name}: found "
            f"{len(observations_by_block)} run files but "
            f"{len(block_types)} blocks in the session table."
        )

    baseline_trials = []
    baseline_predictions = []
    classifier_filter_counts = {
        "classifier_candidate_windows_before_movement_filter": 0,
        "excluded_early_participant_movement_windows": 0,
    }

    for block_index, block_type in enumerate(block_types):
        # Neural baseline analysis includes training blocks.
        if block_type in VALID_BLOCK_TYPES:
            baseline_trials.extend(observations_by_block[block_index])

        # Classifier statistics use test blocks only.
        if block_type in CLASSIFIER_BLOCK_TYPES:
            baseline_predictions.extend(predictions_by_block[block_index])
            block_filter_counts = classifier_filter_counts_by_block[block_index]
            for key in classifier_filter_counts:
                classifier_filter_counts[key] += block_filter_counts[key]

    if not baseline_trials:
        raise ValueError(
            f"No valid baseline trials were found for "
            f"{participant.upper()} {session_name}."
        )

    prediction_array = (
        np.stack(baseline_predictions)
        if baseline_predictions
        else np.empty((0, TRIAL_BINS), dtype=float)
    )

    return np.stack(baseline_trials), prediction_array, classifier_filter_counts


def summarize_pretrial_predictions(predictions):
    """Summarize pre-trial classifier output while preserving missing values."""
    predictions = np.asarray(predictions, dtype=float)

    if predictions.ndim != 2 or predictions.shape[1] != TRIAL_BINS:
        raise ValueError(
            f"Classifier predictions must have shape (n_windows, {TRIAL_BINS})."
        )

    total_windows = int(predictions.shape[0])
    total_bins = int(predictions.size)
    nan_mask = np.isnan(predictions)
    non_nan = ~nan_mask

    # Valid windows contain at least one classifier output.
    valid_windows_mask = np.any(non_nan, axis=1)
    valid_windows = int(valid_windows_mask.sum())

    # Track missing classifier data separately from neural inclusion.
    all_nan_windows_mask = np.all(nan_mask, axis=1)
    any_nan_windows_mask = np.any(nan_mask, axis=1)
    all_nan_windows = int(all_nan_windows_mask.sum())
    any_nan_windows = int(any_nan_windows_mask.sum())
    partial_nan_windows = int((any_nan_windows_mask & ~all_nan_windows_mask).sum())

    positive = predictions == 1
    negative = predictions == 0

    windows_with_positive = int(
        np.any(positive, axis=1)[valid_windows_mask].sum()
    )

    positive_bins = int(positive.sum())
    negative_bins = int(negative.sum())
    valid_bins = positive_bins + negative_bins
    nan_bins = int(nan_mask.sum())

    window_positive_pct = (
        100.0 * windows_with_positive / valid_windows
        if valid_windows > 0
        else np.nan
    )
    all_nan_windows_pct = (
        100.0 * all_nan_windows / total_windows
        if total_windows > 0
        else np.nan
    )
    any_nan_windows_pct = (
        100.0 * any_nan_windows / total_windows
        if total_windows > 0
        else np.nan
    )
    partial_nan_windows_pct = (
        100.0 * partial_nan_windows / total_windows
        if total_windows > 0
        else np.nan
    )
    nan_bins_pct = (
        100.0 * nan_bins / total_bins
        if total_bins > 0
        else np.nan
    )

    # False-positive rate is positive bins divided by negative bins.
    false_positive_rate_pct = (
        100.0 * positive_bins / negative_bins
        if negative_bins > 0
        else np.nan
    )

    # NaN bins do not contribute exposure; positive bins are not collapsed into events.
    valid_classifier_seconds = valid_bins * BIN_SECONDS
    false_positive_bins_per_second = (
        positive_bins / valid_classifier_seconds
        if valid_classifier_seconds > 0
        else np.nan
    )
    seconds_per_false_positive_bin = (
        valid_classifier_seconds / positive_bins
        if positive_bins > 0
        else np.inf
    )

    return {
        "total_windows": total_windows,
        "valid_classifier_windows": valid_windows,
        "all_nan_windows": all_nan_windows,
        "all_nan_windows_pct": all_nan_windows_pct,
        "windows_with_any_nan": any_nan_windows,
        "windows_with_any_nan_pct": any_nan_windows_pct,
        "partially_nan_windows": partial_nan_windows,
        "partially_nan_windows_pct": partial_nan_windows_pct,
        "windows_with_positive": windows_with_positive,
        "windows_with_positive_pct": window_positive_pct,
        "total_classifier_bins": total_bins,
        "positive_bins": positive_bins,
        "negative_bins": negative_bins,
        "valid_classifier_bins": valid_bins,
        "nan_bins": nan_bins,
        "nan_bins_pct": nan_bins_pct,
        "false_positive_rate_pct": false_positive_rate_pct,
        "valid_classifier_seconds": valid_classifier_seconds,
        "false_positive_bins_per_second": false_positive_bins_per_second,
        "seconds_per_false_positive_bin": seconds_per_false_positive_bin,
    }


def add_classifier_filter_stats(stats, filter_counts):
    """Attach early-movement exclusion counts to classifier statistics."""
    candidate_windows = int(
        filter_counts["classifier_candidate_windows_before_movement_filter"]
    )
    excluded_windows = int(
        filter_counts["excluded_early_participant_movement_windows"]
    )

    stats = dict(stats)
    stats["classifier_candidate_windows_before_movement_filter"] = candidate_windows
    stats["excluded_early_participant_movement_windows"] = excluded_windows
    stats["excluded_early_participant_movement_pct"] = (
        100.0 * excluded_windows / candidate_windows
        if candidate_windows > 0
        else np.nan
    )
    return stats


def _format_prediction_summary(label, stats):
    return (
        f"{label}: positive windows="
        f"{stats['windows_with_positive']}/{stats['valid_classifier_windows']} "
        f"({stats['windows_with_positive_pct']:.2f}%); "
        f"FPR={stats['false_positive_rate_pct']:.4f}%; "
        f"FP bins/s={stats['false_positive_bins_per_second']:.4f}; "
        f"s/FP={stats['seconds_per_false_positive_bin']:.2f}; "
        f"NaN bins={stats['nan_bins_pct']:.2f}%; "
        f"early exclusions={stats['excluded_early_participant_movement_windows']}/"
        f"{stats['classifier_candidate_windows_before_movement_filter']}"
    )

def load_all_baseline_activity():
    baseline_rates = {}
    session_rows = []
    participant_prediction_windows = {
        participant: [] for participant in PARTICIPANT_ORDER
    }
    participant_filter_counts = {
        participant: {
            "classifier_candidate_windows_before_movement_filter": 0,
            "excluded_early_participant_movement_windows": 0,
        }
        for participant in PARTICIPANT_ORDER
    }
    all_prediction_windows = []
    overall_filter_counts = {
        "classifier_candidate_windows_before_movement_filter": 0,
        "excluded_early_participant_movement_windows": 0,
    }

    for participant in PARTICIPANT_ORDER:
        baseline_rates[participant] = []

        for session_name in PARTICIPANTS[participant]:
            (
                baseline_trials,
                baseline_predictions,
                classifier_filter_counts,
            ) = load_baseline_trials(
                participant,
                session_name,
            )

            # Preserve the existing neural outlier rule.
            firing_rates, _ = process_trials(baseline_trials)
            baseline_rates[participant].append(firing_rates)

            # Classifier statistics are independent of neural outlier removal.
            prediction_stats = summarize_pretrial_predictions(
                baseline_predictions
            )
            prediction_stats = add_classifier_filter_stats(
                prediction_stats,
                classifier_filter_counts,
            )
            session_rows.append({
                "level": "session",
                "participant": participant.upper(),
                "session": session_name,
                **prediction_stats,
            })

            participant_prediction_windows[participant].append(
                baseline_predictions
            )
            all_prediction_windows.append(baseline_predictions)

            for key in classifier_filter_counts:
                participant_filter_counts[participant][key] += (
                    classifier_filter_counts[key]
                )
                overall_filter_counts[key] += classifier_filter_counts[key]


    pooled_rows = []

    print("\nPre-trial classifier summary (pooled)")
    for participant in PARTICIPANT_ORDER:
        participant_windows = np.vstack(
            participant_prediction_windows[participant]
        )
        stats = summarize_pretrial_predictions(participant_windows)
        stats = add_classifier_filter_stats(
            stats,
            participant_filter_counts[participant],
        )
        pooled_rows.append({
            "level": "participant",
            "participant": participant.upper(),
            "session": "ALL",
            **stats,
        })
        print(_format_prediction_summary(
            participant.upper(),
            stats,
        ))

    overall_windows = np.vstack(all_prediction_windows)
    overall_stats = summarize_pretrial_predictions(overall_windows)
    overall_stats = add_classifier_filter_stats(
        overall_stats,
        overall_filter_counts,
    )
    pooled_rows.append({
        "level": "overall",
        "participant": "ALL",
        "session": "ALL",
        **overall_stats,
    })
    print(_format_prediction_summary("OVERALL", overall_stats))

    prediction_table = pd.DataFrame(session_rows + pooled_rows)
    prediction_table.to_csv(PRETRIAL_PREDICTIONS_OUTPUT_FILE, index=False)
    print(f"Saved: {PRETRIAL_PREDICTIONS_OUTPUT_FILE.resolve()}")

    return baseline_rates

def main():
    premovement_results = analyze_premovement_activity()

    premovement_positive, premovement_negative = build_cluster_profiles(
        premovement_results
    )
    premovement_figure = plot_cluster_grid(
        premovement_positive,
        premovement_negative,
        x_label="Time to action onset (s)",
    )
    premovement_figure.savefig(
        PREMOVEMENT_OUTPUT_FILE,
        bbox_inches="tight",
        dpi=500,
    )
    print(f"Saved: {PREMOVEMENT_OUTPUT_FILE.resolve()}")

    ordered_slopes_figure = plot_ordered_slopes_example(
        premovement_results,
    )
    ordered_slopes_figure.savefig(
        ORDERED_SLOPES_OUTPUT_FILE,
        bbox_inches="tight",
        dpi=500,
    )
    print(f"Saved: {ORDERED_SLOPES_OUTPUT_FILE.resolve()}")

    baseline_rates = load_all_baseline_activity()
    baseline_positive, baseline_negative = build_cluster_profiles(
        premovement_results,
        rates_by_participant=baseline_rates,
    )
    baseline_figure = plot_cluster_grid(
        baseline_positive,
        baseline_negative,
        x_label="Time to trial onset (s)",
    )
    baseline_figure.savefig(
        BASELINE_OUTPUT_FILE,
        bbox_inches="tight",
        dpi=500,
    )
    print(f"Saved: {BASELINE_OUTPUT_FILE.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()