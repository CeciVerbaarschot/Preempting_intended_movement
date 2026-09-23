#!/usr/bin/env python3
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter1d


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_DIR = Path("../../FlipThatBucket/data")
OUTPUT_FILE = Path("figure_1.pdf")

BIN_MS = 20
PREMOVEMENT = 100
POSTMOVEMENT = 25
EMG_LOCK = 25
EMG_THRESHOLD = 3

PARTICIPANTS = {
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
    "c1": [
        "Session_00637_Lab",
        "Session_00638_Lab",
        "Session_00644_Lab",
    ],
}

PARTICIPANT_ORDER = ["p2", "p3", "c1"]

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


def calculate_emg_onsets():
    emg_onsets = {}

    for participant, session_names in PARTICIPANTS.items():
        all_emgs = []

        for session_name in session_names:
            session = load_session(participant, session_name)
            reaction_time = session[
                session["BlockType"] == "reaction-time"
            ].reset_index(drop=True)

            for row in reaction_time.itertuples(index=False):
                onset = int(row.Onset)
                all_emgs.append(np.asarray(row.EMG)[onset - EMG_LOCK:onset])

        if not all_emgs:
            raise ValueError(f"No reaction-time EMG trials found for {participant}.")

        mean_emg = np.mean(np.stack(all_emgs), axis=0)
        crossings = np.flatnonzero(mean_emg > EMG_THRESHOLD)

        if not crossings.size:
            raise ValueError(
                f"Mean EMG did not cross the threshold for {participant}."
            )

        onset_ms = (EMG_LOCK - crossings[0]) * BIN_MS
        emg_onsets[participant] = onset_ms / 1000
        print(f"{participant.upper()} EMG onset: {onset_ms} ms")

    return emg_onsets


def load_trial_profiles():
    profiles = {
        participant: {"neural": [], "force": [], "preds": []}
        for participant in PARTICIPANTS
    }

    for participant, session_names in PARTICIPANTS.items():
        for session_name in session_names:
            session = load_session(participant, session_name)

            game = session[
                session["BlockType"].str.startswith("ftb", na=False)
                & ~session["BlockType"].str.contains("-brain-", na=False)
                & (session["PlayerID"] == 1)
                & (session["Onset"] >= PREMOVEMENT)
                & (session["PostMovement"] >= POSTMOVEMENT)
            ].reset_index(drop=True)

            if game.empty:
                continue

            first_predictions = game["Predictions"].apply(first_one)
            deltas = first_predictions - game["Onset"]
            valid_deltas = deltas[
                deltas.notna() & deltas.between(-PREMOVEMENT, POSTMOVEMENT)
            ]

            neural_trials = []
            force_trials = []

            for row in game.itertuples(index=False):
                onset = int(row.Onset)
                start = onset - PREMOVEMENT
                stop = onset + POSTMOVEMENT

                neural_trials.append(np.asarray(row.Neural)[:, start:stop])
                force_trials.append(np.asarray(row.Force)[start:stop])

            profiles[participant]["neural"].append(np.stack(neural_trials))
            profiles[participant]["force"].append(np.stack(force_trials))
            profiles[participant]["preds"].append(valid_deltas.to_numpy())

        if not profiles[participant]["neural"]:
            raise ValueError(f"No valid game trials found for {participant}.")

        profiles[participant]["preds"] = np.concatenate(
            profiles[participant]["preds"]
        )

    return profiles


def process_mua(session_data, percentile=99):
    processed = []

    for trials in session_data:
        profiles = gaussian_filter1d(trials, sigma=2, axis=-1).sum(axis=1)
        upper_limit = np.percentile(profiles, percentile)
        profiles = np.clip(profiles, 0, upper_limit)
        profiles = (profiles - profiles.mean()) / profiles.std()
        processed.append(profiles)

    return np.concatenate(processed, axis=0)


def process_force(session_data):
    processed = []
    baseline_bins = 25

    for trials in session_data:
        baseline = trials[:, :baseline_bins].mean(axis=1, keepdims=True)
        trials = trials - baseline
        trials = (trials - trials.mean()) / trials.std()
        processed.append(trials)

    return np.concatenate(processed, axis=0)


def calculate_summary_statistics(profiles):
    statistics = {}

    for participant in PARTICIPANT_ORDER:
        neural = process_mua(profiles[participant]["neural"])
        force = process_force(profiles[participant]["force"])

        statistics[participant] = {
            "neural_mean": neural.mean(axis=0),
            "neural_std": neural.std(axis=0, ddof=1),
            "force_median": np.median(force, axis=0),
            "force_p25": np.percentile(force, 25, axis=0),
            "force_p75": np.percentile(force, 75, axis=0),
        }

    return statistics


def create_prediction_dataframe(profiles):
    frames = []

    for participant in PARTICIPANT_ORDER:
        frames.append(
            pd.DataFrame(
                {
                    "preds_time": profiles[participant]["preds"] * BIN_MS / 1000,
                    "participant": participant,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def plot_figure(profiles, statistics, prediction_df, emg_onsets):
    time_s = (
        np.arange(PREMOVEMENT + POSTMOVEMENT) - PREMOVEMENT
    ) * BIN_MS / 1000

    fig, axes = plt.subplots(4, 1, figsize=(6, 8), sharex=True)
    ax_a, ax_b, ax_c, ax_d = axes

    # Panel A: example raster from P2
    raster = profiles["p2"]["neural"][1][19, 30:45]
    n_channels = raster.shape[0]

    ax_a.imshow(
        raster,
        cmap="gray_r",
        aspect="auto",
        extent=[time_s[0], time_s[-1], n_channels, 0],
    )
    ax_a.axvline(0, color="red", linewidth=1)
    ax_a.set_yticks([0, n_channels])
    ax_a.set_yticklabels([1, n_channels])
    ax_a.set_ylabel("Subset of\nelectrodes in MC")

    # Panel B: neural activity
    for participant in PARTICIPANT_ORDER:
        values = statistics[participant]
        mean = values["neural_mean"]
        std = values["neural_std"]

        ax_b.plot(
            time_s,
            mean,
            color=PALETTE[participant],
            label=participant.upper(),
            linewidth=2,
        )
        ax_b.fill_between(
            time_s,
            mean - std,
            mean + std,
            color=PALETTE[participant],
            alpha=0.2,
        )

    ax_b.axvline(0, color="red", linewidth=1)
    ax_b.set_ylabel("MUA (z-scored)")
    ax_b.legend(frameon=False, fontsize=8)

    # Panel C: force profiles
    for participant in PARTICIPANT_ORDER:
        values = statistics[participant]

        ax_c.plot(
            time_s,
            values["force_median"],
            color=PALETTE[participant],
            linewidth=2,
        )
        ax_c.fill_between(
            time_s,
            values["force_p25"],
            values["force_p75"],
            color=PALETTE[participant],
            alpha=0.2,
        )

    ax_c.axvline(0, color="red", linewidth=1)
    ax_c.set_ylabel("Force (z-scored)")
    ax_c.annotate(
        "Action onset",
        xy=(-0.15, 0.95),
        xycoords=("data", "axes fraction"),
        xytext=(4, 0),
        textcoords="offset points",
        color="red",
        fontsize=10,
        ha="left",
        va="top",
        rotation=90,
    )

    # Panel D: decoder prediction times
    ax_d.axvspan(-1.0, 0.0, color="lightgray", alpha=0.8, zorder=0)

    sns.histplot(
        data=prediction_df,
        x="preds_time",
        hue="participant",
        bins=50,
        stat="percent",
        common_norm=False,
        multiple="layer",
        palette=PALETTE,
        element="step",
        fill=False,
        alpha=0,
        linewidth=0,
        kde=True,
        kde_kws={"bw_adjust": 0.5},
        line_kws={"lw": 2},
        legend=False,
        ax=ax_d,
    )

    ax_d.axvline(0, color="red", linewidth=1)

    for participant in PARTICIPANT_ORDER:
        ax_d.axvline(
            -emg_onsets[participant],
            color=PALETTE[participant],
            linestyle="--",
            linewidth=2,
            label=participant.upper(),
        )

    ax_d.set_ylabel("Trials (%)")
    ax_d.set_xlabel("Time (s)")
    ax_d.legend(
        frameon=False,
        title="EMG onset",
        fontsize=8,
        loc="upper left",
        title_fontsize=9,
    )
    ax_d.set_xlim(time_s[0], 0.5)
    ax_d.set_xticks([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5])
    ax_d.set_xticklabels(["-2", "-1.5", "-1", "-0.5", "0", "0.5"])

    fig.savefig(OUTPUT_FILE, bbox_inches="tight", dpi=500)
    plt.show()


def main():
    emg_onsets = calculate_emg_onsets()
    profiles = load_trial_profiles()
    statistics = calculate_summary_statistics(profiles)
    prediction_df = create_prediction_dataframe(profiles)
    plot_figure(profiles, statistics, prediction_df, emg_onsets)


if __name__ == "__main__":
    main()