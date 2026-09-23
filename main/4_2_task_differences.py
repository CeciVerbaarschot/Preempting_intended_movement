#!/usr/bin/env python3
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter1d


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_DIR = Path("../../FlipThatBucket/data")
OUTPUT_DIR = Path(__file__).resolve().parent

PEAK_OUTPUT_FILE = OUTPUT_DIR / "figure_4_peak_mua.pdf"
AUC_OUTPUT_FILE = OUTPUT_DIR / "figure_4_integralAUC.pdf"

BIN_SECONDS = 0.02
PREMOVEMENT_BINS = 50
POSTMOVEMENT_BINS = 25
N_TIME_BINS = PREMOVEMENT_BINS + POSTMOVEMENT_BINS

MUA_PERCENTILE = 99
MUA_SMOOTHING_SIGMA = 2

TEST = "mannwhitney"          # "mannwhitney" or "welch_t"
P_CORRECTION = "bonferroni"   # "bonferroni" or "holm"
ALPHA = 0.05
RUN_POSTHOC_ONLY_IF_OMNIBUS_SIG = True
ANNOTATE_ONLY_SIGNIFICANT = False

SIGNIFICANCE_Y_PAD_FRACTION = 0.03
SIGNIFICANCE_STEP_FRACTION = 0.15
SIGNIFICANCE_HEIGHT_FRACTION = 0.015
SIGNIFICANCE_TOP_PAD_FRACTION = 0.20

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
CONDITION_ORDER = ["reaction", "ftb", "scroll"]

CONDITION_LABELS = {
    "reaction": "Reaction",
    "ftb": "Game",
    "scroll": "Scroll",
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


def normalize_block(block):
    block = str(block)

    if block == "train" or block.startswith("ftb"):
        return "ftb"
    if block.startswith("reaction"):
        return "reaction"
    if block.startswith("scroll"):
        return "scroll"
    return "unknown"


def extract_aligned_neural_trials(session):
    neural_trials = []
    block_types = []

    valid_trials = session[
        (session["PlayerID"] == 1)
        & (session["Onset"] >= PREMOVEMENT_BINS)
        & (session["PostMovement"] >= POSTMOVEMENT_BINS)
    ]

    for row in valid_trials.itertuples(index=False):
        onset = int(row.Onset)
        start = onset - PREMOVEMENT_BINS
        stop = onset + POSTMOVEMENT_BINS

        neural = np.asarray(row.Neural)[:, start:stop]
        if neural.shape[1] != N_TIME_BINS:
            continue

        neural_trials.append(neural)
        block_types.append(normalize_block(row.BlockType))

    if not neural_trials:
        return np.empty((0, 0, N_TIME_BINS)), np.empty(0, dtype=str)

    return np.stack(neural_trials), np.asarray(block_types)


def load_trial_data():
    data = {
        participant: {"neural_sessions": [], "blocks": []}
        for participant in PARTICIPANT_ORDER
    }

    for participant in PARTICIPANT_ORDER:
        for session_name in PARTICIPANTS[participant]:
            session = load_session(participant, session_name)
            neural, blocks = extract_aligned_neural_trials(session)

            if neural.shape[0] == 0:
                print(
                    f"{participant.upper()} {session_name}: "
                    "no valid trials found"
                )
                continue

            data[participant]["neural_sessions"].append(neural)
            data[participant]["blocks"].append(blocks)


        if not data[participant]["neural_sessions"]:
            raise ValueError(
                f"No valid neural trials were found for "
                f"{participant.upper()}."
            )

        data[participant]["blocks"] = np.concatenate(
            data[participant]["blocks"]
        )

    return data


def process_mua_sessions(neural_sessions):
    processed = []

    for trials in neural_sessions:
        mua = gaussian_filter1d(
            trials,
            sigma=MUA_SMOOTHING_SIGMA,
            axis=-1,
        ).sum(axis=1)

        upper_limit = np.percentile(mua, MUA_PERCENTILE)
        mua = np.clip(mua, 0, upper_limit)

        mean = mua.mean()
        std = mua.std()
        if std == 0:
            std = 1.0

        processed.append((mua - mean) / std)

    return np.concatenate(processed, axis=0)


def build_processed_data(trial_data):
    processed = {}

    for participant in PARTICIPANT_ORDER:
        neural = process_mua_sessions(
            trial_data[participant]["neural_sessions"]
        )
        blocks = trial_data[participant]["blocks"]

        if neural.shape[0] != blocks.shape[0]:
            raise ValueError(
                f"{participant.upper()}: processed trials and block labels "
                "are not aligned."
            )

        processed[participant] = {
            "neural": neural,
            "blocks": blocks,
        }

    return processed


def calculate_peak_metric(neural):
    premovement = neural[:, :PREMOVEMENT_BINS]
    return np.percentile(premovement, 95, axis=1)


def calculate_auc_metric(neural):
    premovement = neural[:, :PREMOVEMENT_BINS]

    minimum = premovement.min(axis=1, keepdims=True)
    maximum = premovement.max(axis=1, keepdims=True)
    denominator = maximum - minimum

    normalized = np.divide(
        premovement - minimum,
        denominator,
        out=np.zeros_like(premovement, dtype=float),
        where=denominator > 0,
    )

    return np.trapezoid(
        normalized,
        dx=BIN_SECONDS,
        axis=1,
    )


def build_metric_dataframe(processed, metric_function):
    frames = []

    for participant in PARTICIPANT_ORDER:
        values = metric_function(processed[participant]["neural"])
        blocks = processed[participant]["blocks"]

        frames.append(
            pd.DataFrame(
                {
                    "participant": participant,
                    "condition": blocks,
                    "value": values,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def p_to_stars(p_value):
    if np.isnan(p_value):
        return "n/a"
    if p_value < 1e-4:
        return "****"
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def correct_p_values(p_values):
    p_values = np.asarray(p_values, dtype=float)

    if P_CORRECTION == "bonferroni":
        return np.clip(p_values * len(p_values), 0, 1)

    if P_CORRECTION == "holm":
        order = np.argsort(p_values)
        sorted_values = p_values[order]

        adjusted_sorted = np.array(
            [
                (len(p_values) - index) * p_value
                for index, p_value in enumerate(sorted_values)
            ]
        )
        adjusted_sorted = np.maximum.accumulate(adjusted_sorted)
        adjusted_sorted = np.clip(adjusted_sorted, 0, 1)

        adjusted = np.empty_like(adjusted_sorted)
        adjusted[order] = adjusted_sorted
        return adjusted

    raise ValueError(
        "P_CORRECTION must be 'bonferroni' or 'holm'."
    )


def run_pairwise_test(group_a, group_b):
    group_a = np.asarray(group_a, dtype=float)
    group_b = np.asarray(group_b, dtype=float)

    group_a = group_a[~np.isnan(group_a)]
    group_b = group_b[~np.isnan(group_b)]

    if TEST == "mannwhitney":
        result = stats.mannwhitneyu(
            group_a,
            group_b,
            alternative="two-sided",
            method="auto",
        )
        return (
            "U",
            float(result.statistic),
            float(result.pvalue),
            len(group_a),
            len(group_b),
        )

    if TEST == "welch_t":
        result = stats.ttest_ind(
            group_a,
            group_b,
            equal_var=False,
            nan_policy="omit",
        )
        return (
            "t",
            float(result.statistic),
            float(result.pvalue),
            len(group_a),
            len(group_b),
        )

    raise ValueError(
        "TEST must be 'mannwhitney' or 'welch_t'."
    )


def get_present_conditions(participant_data):
    return [
        condition
        for condition in CONDITION_ORDER
        if (participant_data["condition"] == condition).any()
    ]


def run_statistics(dataframe):
    omnibus_rows = []
    pairwise_rows = []

    for participant in PARTICIPANT_ORDER:
        participant_data = dataframe[
            dataframe["participant"] == participant
        ]
        present_conditions = get_present_conditions(participant_data)

        groups = [
            participant_data.loc[
                participant_data["condition"] == condition,
                "value",
            ].dropna().to_numpy()
            for condition in present_conditions
        ]

        if len(groups) >= 2:
            omnibus = stats.kruskal(*groups, nan_policy="omit")
            omnibus_statistic = float(omnibus.statistic)
            omnibus_p = float(omnibus.pvalue)
        else:
            omnibus_statistic = np.nan
            omnibus_p = np.nan

        omnibus_significant = (
            np.isfinite(omnibus_p) and omnibus_p < ALPHA
        )

        omnibus_rows.append(
            {
                "participant": participant.upper(),
                "conditions_tested": ", ".join(
                    CONDITION_LABELS[condition]
                    for condition in present_conditions
                ),
                "n_groups": len(present_conditions),
                "group_sizes": [len(group) for group in groups],
                "test": "Kruskal-Wallis",
                "stat_name": "H",
                "stat": omnibus_statistic,
                "p_raw": omnibus_p,
                "sig": p_to_stars(omnibus_p),
            }
        )

        comparisons = list(combinations(present_conditions, 2))
        participant_pairwise_rows = []
        raw_p_values = []

        for condition_a, condition_b in comparisons:
            group_a = participant_data.loc[
                participant_data["condition"] == condition_a,
                "value",
            ].to_numpy()
            group_b = participant_data.loc[
                participant_data["condition"] == condition_b,
                "value",
            ].to_numpy()

            if (
                RUN_POSTHOC_ONLY_IF_OMNIBUS_SIG
                and not omnibus_significant
            ):
                stat_name = "U" if TEST == "mannwhitney" else "t"
                stat_value = np.nan
                raw_p = np.nan
                n_a = np.sum(~np.isnan(group_a))
                n_b = np.sum(~np.isnan(group_b))
            else:
                (
                    stat_name,
                    stat_value,
                    raw_p,
                    n_a,
                    n_b,
                ) = run_pairwise_test(group_a, group_b)

            raw_p_values.append(raw_p)
            participant_pairwise_rows.append(
                {
                    "participant": participant.upper(),
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "comparison": (
                        f"{CONDITION_LABELS[condition_a]} vs "
                        f"{CONDITION_LABELS[condition_b]}"
                    ),
                    "n_a": n_a,
                    "n_b": n_b,
                    "test": (
                        "Mann–Whitney U"
                        if TEST == "mannwhitney"
                        else "Welch t-test"
                    ),
                    "stat_name": stat_name,
                    "stat": stat_value,
                    "p_raw": raw_p,
                    "omnibus_p": omnibus_p,
                    "omnibus_sig": omnibus_significant,
                }
            )

        raw_p_values = np.asarray(raw_p_values, dtype=float)

        if (
            RUN_POSTHOC_ONLY_IF_OMNIBUS_SIG
            and not omnibus_significant
        ):
            adjusted_p_values = np.full(
                raw_p_values.shape,
                np.nan,
            )
        else:
            adjusted_p_values = correct_p_values(raw_p_values)

        for row, adjusted_p in zip(
            participant_pairwise_rows,
            adjusted_p_values,
        ):
            row["p_adj"] = adjusted_p
            row["sig"] = p_to_stars(adjusted_p)
            row["correction"] = (
                P_CORRECTION
                if np.isfinite(adjusted_p)
                else "not_run"
            )
            pairwise_rows.append(row)

    return pd.DataFrame(omnibus_rows), pd.DataFrame(pairwise_rows)


def print_statistics(title, omnibus, pairwise):
    print(f"\n{title}: omnibus")
    print(
        omnibus[["participant", "stat_name", "stat", "p_raw", "sig"]]
        .to_string(index=False)
    )

    print(f"{title}: pairwise")
    print(
        pairwise[
            ["participant", "comparison", "stat_name", "stat", "p_adj", "sig"]
        ]
        .sort_values(["participant", "comparison"])
        .to_string(index=False)
    )


def order_comparisons_for_plot(
    present_conditions,
    comparisons,
    adjusted_p_values,
):
    x_positions = {
        condition: index
        for index, condition in enumerate(present_conditions)
    }
    ordered = []

    for (condition_a, condition_b), p_value in zip(
        comparisons,
        adjusted_p_values,
    ):
        if np.isnan(p_value):
            continue

        x_a = x_positions[condition_a]
        x_b = x_positions[condition_b]
        if x_b < x_a:
            x_a, x_b = x_b, x_a

        ordered.append(
            (
                x_b - x_a,
                x_a,
                condition_a,
                condition_b,
                float(p_value),
            )
        )

    ordered.sort(key=lambda item: (item[0], item[1]))
    return ordered


def add_significance_bars(
    axis,
    present_conditions,
    comparisons,
    adjusted_p_values,
    y_base,
    step,
    height,
):
    x_positions = {
        condition: index
        for index, condition in enumerate(present_conditions)
    }

    ordered = order_comparisons_for_plot(
        present_conditions,
        comparisons,
        adjusted_p_values,
    )

    if ANNOTATE_ONLY_SIGNIFICANT:
        ordered = [
            comparison
            for comparison in ordered
            if p_to_stars(comparison[4]) != "ns"
        ]

    for level, (_, _, condition_a, condition_b, p_value) in enumerate(
        ordered
    ):
        x_a = x_positions[condition_a]
        x_b = x_positions[condition_b]
        if x_b < x_a:
            x_a, x_b = x_b, x_a

        y = y_base + level * step

        axis.plot(
            [x_a, x_a, x_b, x_b],
            [y, y + height, y + height, y],
            color="black",
            linewidth=1.2,
            alpha=0.8,
            clip_on=False,
        )
        axis.text(
            (x_a + x_b) / 2,
            y + height,
            p_to_stars(p_value),
            ha="center",
            va="bottom",
            fontsize=10,
            alpha=0.8,
        )


def get_plot_comparisons(
    participant,
    participant_data,
    pairwise_results,
):
    present_conditions = get_present_conditions(participant_data)
    comparisons = list(combinations(present_conditions, 2))

    participant_rows = pairwise_results[
        pairwise_results["participant"] == participant.upper()
    ]

    adjusted_p_values = []

    for condition_a, condition_b in comparisons:
        match = participant_rows[
            (participant_rows["condition_a"] == condition_a)
            & (participant_rows["condition_b"] == condition_b)
        ]

        adjusted_p_values.append(
            float(match.iloc[0]["p_adj"])
            if len(match) == 1
            else np.nan
        )

    return present_conditions, comparisons, adjusted_p_values


def plot_metric(
    dataframe,
    pairwise_results,
    y_label,
    output_file,
):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7, 3),
        sharey=True,
    )

    plot_details = {}

    for axis, participant in zip(axes, PARTICIPANT_ORDER):
        participant_data = dataframe[
            dataframe["participant"] == participant
        ]

        (
            present_conditions,
            comparisons,
            adjusted_p_values,
        ) = get_plot_comparisons(
            participant,
            participant_data,
            pairwise_results,
        )

        plot_details[participant] = {
            "conditions": present_conditions,
            "comparisons": comparisons,
            "p_values": adjusted_p_values,
        }

        sns.boxplot(
            data=participant_data,
            x="condition",
            y="value",
            order=present_conditions,
            color=PALETTE[participant],
            ax=axis,
        )

        axis.set_title(participant.upper())
        axis.set_xlabel("")
        axis.set_ylabel(y_label if axis is axes[0] else "")
        axis.set_xticks(range(len(present_conditions)))
        axis.set_xticklabels(
            [
                CONDITION_LABELS[condition]
                for condition in present_conditions
            ],
            fontsize=9,
        )

    y_min, y_max = axes[0].get_ylim()
    y_range = y_max - y_min if y_max > y_min else 1.0

    max_levels = max(
        len(
            order_comparisons_for_plot(
                plot_details[participant]["conditions"],
                plot_details[participant]["comparisons"],
                plot_details[participant]["p_values"],
            )
        )
        for participant in PARTICIPANT_ORDER
    )

    y_base = y_max + SIGNIFICANCE_Y_PAD_FRACTION * y_range
    step = SIGNIFICANCE_STEP_FRACTION * y_range
    height = SIGNIFICANCE_HEIGHT_FRACTION * y_range
    global_top = (
        y_base
        + max(0, max_levels - 1) * step
        + height
        + SIGNIFICANCE_TOP_PAD_FRACTION * y_range
    )

    axes[0].set_ylim(y_min, global_top)

    for axis, participant in zip(axes, PARTICIPANT_ORDER):
        details = plot_details[participant]
        add_significance_bars(
            axis,
            details["conditions"],
            details["comparisons"],
            details["p_values"],
            y_base,
            step,
            height,
        )

    fig.savefig(
        output_file,
        bbox_inches="tight",
        dpi=500,
    )
    print(f"Saved: {output_file.resolve()}")

    return fig


def run_metric_analysis(
    processed,
    metric_function,
    title,
    y_label,
    output_file,
):
    dataframe = build_metric_dataframe(
        processed,
        metric_function,
    )
    omnibus, pairwise = run_statistics(dataframe)
    print_statistics(title, omnibus, pairwise)
    return plot_metric(
        dataframe,
        pairwise,
        y_label,
        output_file,
    )


def main():
    trial_data = load_trial_data()
    processed = build_processed_data(trial_data)

    run_metric_analysis(
        processed=processed,
        metric_function=calculate_peak_metric,
        title="peak MUA",
        y_label="Peak pre-movement MUA (z-scored)",
        output_file=PEAK_OUTPUT_FILE,
    )

    run_metric_analysis(
        processed=processed,
        metric_function=calculate_auc_metric,
        title="integral AUC",
        y_label="Integral AUC",
        output_file=AUC_OUTPUT_FILE,
    )

    plt.show()


if __name__ == "__main__":
    main()