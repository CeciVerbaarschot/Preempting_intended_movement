#!/usr/bin/env python3
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline
import pandas as pd
import statsmodels.formula.api as smf


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = OUTPUT_DIR / "figure_3b.pdf"
REGRESSION_CSV_FILE = OUTPUT_DIR / "figure_3b_regression_results.csv"

SELECT_BLOCK_TYPES = ["val", "mixed"]
EXCLUDE_BLOCK_TYPES = []
SHOW_BLOCKTYPE_LEGEND = True
SMOOTH_POINTS_PER_UNIT = 60

SHOW_LINEAR_FITS = True
SHOW_POOLED_LINEAR_FIT = True
REGRESSION_ALPHA = 0.05

PARTICIPANT_ORDER = ["P2", "P3", "C1"]
PARTICIPANT_COLORS = {
    "P2": "#0072B2",
    "P3": "#009E73",
    "C1": "#CC79A7",
}

PREFERRED_MARKERS = {
    "val": "*",
    "mixed": "o",
    "brain": "^",
    "train": "s",
}
MARKER_SIZES = {"*": 8, "o": 5, "^": 5, "s": 5}
BLOCK_TYPE_LABELS = {
    "val": "Validation",
    "mixed": "Mixed",
    "brain": "Brain",
    "train": "Train",
}
FALLBACK_MARKERS = ["o", "s", "^", "v", "P", "X", "<", ">"]

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"


# (block type, on-time accuracy, timing in ms), from session_analyzer.py
SUMMARIES = {
    "P2": {
        "sess1": [
            ("val", 0.8333, 478),
            ("mixed", 0.7273, 395),
            ("mixed", 0.8333, 168),
            ("brain", 0.95, 0.8421),
            ("mixed", 0.9091, 288),
        ],
        "sess2": [
            ("val", 1, 432.00),
            ("brain", 0.95, 0.9474),
            ("mixed", 0.8, 347.50),
            ("mixed", 1, 364.44),
            ("brain", 0.95, 0.9474),
            ("mixed", 0.7692, 188),
        ],
        "sess3": [
            ("val", 1, 342.86),
            ("mixed", 0.8571, 106.67),
            ("mixed", 1, 117.14),
            ("brain", 0.9, 1),
            ("mixed", 0.8750, 105.71),
            ("brain", 0.9, 1),
            ("mixed", 0.90, 73.33),
        ],
    },
    "P3": {
        "sess1": [
            ("val", 0.9167, 212.73),
            ("mixed", 0.74, 263.33),
            ("brain", 0.7, 1),
            ("mixed", 0.6250, 276),
            ("mixed", 0.60, 166.67),
            ("mixed", 0.5714, 100),
        ],
        "sess2": [
            ("val", 0.8333, 484),
            ("mixed", 0.5, 400),
            ("mixed", 0.8889, 350),
            ("mixed", 0.5714, 195),
            ("brain", 0.90, 0.9444),
            ("mixed", 1, 294.29),
        ],
        "sess3": [
            ("val", 0.7857, 336.36),
            ("mixed", 1, 354.29),
            ("mixed", 0.9091, 220),
            ("brain", 1, 0.7),
            ("mixed", 0.9, 344.44),
            ("mixed", 1, 215),
        ],
    },
    "C1": {
        "sess1": [
            ("val", 0.6, 660),
            ("mixed", 0.70, 485.71),
            ("brain", 1, 1),
            ("mixed", 0.8750, 397.14),
            ("mixed", 1, 342.50),
            ("brain", 1, 0.9),
            ("mixed", 0.9091, 312),
        ],
        "sess2": [
            ("val", 0.8, 581.67),
            ("mixed", 0.70, 240),
            ("brain", 1, 0.75),
            ("mixed", 1, 208.89),
            ("brain", 0.95, 0.9474),
            ("mixed", 1, 113.33),
            ("mixed", 0.8750, 362.86),
            ("mixed", 1, 85),
        ],
        "sess3": [
            ("val", 0.9231, 155),
            ("brain", 1, 0.65),
            ("mixed", 0.75, 204.44),
            ("mixed", 0.8750, 37.14),
            ("brain", 1, 0.8),
            ("mixed", 1, 57.50),
        ],
    },
}


def session_sort_key(session_name):
    digits = "".join(
        character for character in session_name if character.isdigit()
    )
    return int(digits) if digits else 0


def collect_block_types():
    block_types = []

    for participant in PARTICIPANT_ORDER:
        for session_name in sorted(
            SUMMARIES[participant],
            key=session_sort_key,
        ):
            for block_type, _, _ in SUMMARIES[participant][session_name]:
                if block_type not in block_types:
                    block_types.append(block_type)

    return block_types


def resolve_selected_block_types():
    available = collect_block_types()
    excluded = set(EXCLUDE_BLOCK_TYPES)

    if SELECT_BLOCK_TYPES is None:
        selected = [
            block_type
            for block_type in available
            if block_type not in excluded
        ]
    else:
        selected = [
            block_type
            for block_type in SELECT_BLOCK_TYPES
            if block_type in available and block_type not in excluded
        ]

    if not selected:
        raise ValueError(
            "No valid block types were selected. "
            f"Available block types: {available}"
        )

    return selected, available


def build_marker_map(selected_block_types):
    marker_map = {}
    used_markers = set()

    for block_type in selected_block_types:
        if block_type in PREFERRED_MARKERS:
            marker_map[block_type] = PREFERRED_MARKERS[block_type]
            used_markers.add(PREFERRED_MARKERS[block_type])

    fallbacks = iter(
        marker
        for marker in FALLBACK_MARKERS
        if marker not in used_markers
    )

    for block_type in selected_block_types:
        if block_type not in marker_map:
            marker_map[block_type] = next(fallbacks, "o")

    return marker_map


def extract_chronological_series(summary, selected_block_types):
    selected = set(selected_block_types)

    accuracy = []
    timing = []
    block_types = []
    breaks_after = []

    retained_count = 0
    previous_session_count = None

    for session_name in sorted(summary, key=session_sort_key):
        session_rows = [
            row
            for row in summary[session_name]
            if row[0] in selected
        ]
        current_session_count = len(session_rows)

        if (
            previous_session_count is not None
            and previous_session_count > 0
            and current_session_count > 0
            and retained_count > 0
        ):
            breaks_after.append(retained_count)

        for block_type, on_time, timing_ms in session_rows:
            accuracy.append(float(on_time) * 100)
            timing.append(float(timing_ms))
            block_types.append(block_type)
            retained_count += 1

        previous_session_count = current_session_count

    return {
        "x": np.arange(1, retained_count + 1),
        "accuracy": np.asarray(accuracy),
        "timing": np.asarray(timing),
        "block_types": np.asarray(block_types),
        "breaks_after": breaks_after,
    }


def build_regression_dataframe(series_by_participant):
    """Build the long-format trend table; block type is not a model predictor."""
    rows = []

    for participant in PARTICIPANT_ORDER:
        series = series_by_participant[participant]
        for block, accuracy, timing in zip(
            series["x"],
            series["accuracy"],
            series["timing"],
        ):
            rows.append(
                {
                    "participant": participant,
                    "block": float(block),
                    "accuracy": float(accuracy),
                    "timing": float(timing),
                }
            )

    dataframe = pd.DataFrame(rows)
    dataframe["participant"] = pd.Categorical(
        dataframe["participant"],
        categories=PARTICIPANT_ORDER,
        ordered=True,
    )
    return dataframe


def extract_slope_result(model, slope_name="block"):
    """Extract the slope test and 95% CI from a fitted statsmodels OLS model."""
    confidence_interval = model.conf_int(alpha=REGRESSION_ALPHA).loc[slope_name]

    return {
        "n": int(model.nobs),
        "slope": float(model.params[slope_name]),
        "slope_stderr": float(model.bse[slope_name]),
        "slope_ci_low": float(confidence_interval.iloc[0]),
        "slope_ci_high": float(confidence_interval.iloc[1]),
        "t_value": float(model.tvalues[slope_name]),
        "p_value": float(model.pvalues[slope_name]),
        "df_resid": float(model.df_resid),
        "r_squared": float(model.rsquared),
        "adjusted_r_squared": float(model.rsquared_adj),
        "significant": bool(model.pvalues[slope_name] < REGRESSION_ALPHA),
        "model": model,
    }


def fit_participant_regression(dataframe, outcome, participant):
    """Fit outcome ~ block for one participant."""
    participant_data = dataframe.loc[
        dataframe["participant"] == participant,
        ["participant", "block", outcome],
    ].dropna()

    if len(participant_data) < 3 or participant_data["block"].nunique() < 2:
        return None

    model = smf.ols(
        f"{outcome} ~ block",
        data=participant_data,
    ).fit()

    result = extract_slope_result(model)
    result["intercept"] = float(model.params["Intercept"])
    result["x_min"] = float(participant_data["block"].min())
    result["x_max"] = float(participant_data["block"].max())
    return result


def fit_adjusted_common_slope(dataframe, outcome):
    """Fit outcome ~ block + participant with a shared block slope."""
    model_data = dataframe[["participant", "block", outcome]].dropna().copy()

    model = smf.ols(
        f"{outcome} ~ block + C(participant)",
        data=model_data,
    ).fit()

    result = extract_slope_result(model)
    result["x_min"] = float(model_data["block"].min())
    result["x_max"] = float(model_data["block"].max())

    # Draw the common slope through the mean participant-adjusted intercept.
    baseline_data = pd.DataFrame(
        {
            "block": np.zeros(len(PARTICIPANT_ORDER), dtype=float),
            "participant": pd.Categorical(
                PARTICIPANT_ORDER,
                categories=PARTICIPANT_ORDER,
                ordered=True,
            ),
        }
    )
    result["plot_intercept"] = float(model.predict(baseline_data).mean())
    return result


def fit_slope_heterogeneity_test(dataframe, outcome):
    """Test block-by-participant slope heterogeneity with nested OLS models."""
    model_data = dataframe[["participant", "block", outcome]].dropna().copy()

    common_model = smf.ols(
        f"{outcome} ~ block + C(participant)",
        data=model_data,
    ).fit()
    interaction_model = smf.ols(
        f"{outcome} ~ block * C(participant)",
        data=model_data,
    ).fit()

    f_value, p_value, df_difference = interaction_model.compare_f_test(
        common_model
    )

    return {
        "f_value": float(f_value),
        "p_value": float(p_value),
        "df_num": int(round(df_difference)),
        "df_den": int(interaction_model.df_resid),
        "significant": bool(p_value < REGRESSION_ALPHA),
        "common_model": common_model,
        "interaction_model": interaction_model,
    }


def compute_regression_results(series_by_participant):
    """Run pooled, participant-specific, and slope-heterogeneity analyses."""
    dataframe = build_regression_dataframe(series_by_participant)
    regression_results = {"dataframe": dataframe}

    for outcome in ("accuracy", "timing"):
        participant_results = {}
        for participant in PARTICIPANT_ORDER:
            participant_results[participant] = fit_participant_regression(
                dataframe,
                outcome,
                participant,
            )

        regression_results[outcome] = {
            "adjusted_common_slope": fit_adjusted_common_slope(
                dataframe,
                outcome,
            ),
            "participants": participant_results,
            "heterogeneity": fit_slope_heterogeneity_test(
                dataframe,
                outcome,
            ),
        }

    return regression_results


def format_p_value(p_value):
    if p_value < 0.0001:
        return f"{p_value:.3e}"
    return f"{p_value:.4f}"


def build_regression_results_table(regression_results):
    """Return one tidy table containing every inferential regression result."""
    outcome_info = {
        "accuracy": ("On-time percentage", "percentage points/block"),
        "timing": ("Timing", "ms/block"),
    }

    rows = []

    for outcome in ("accuracy", "timing"):
        outcome_label, slope_unit = outcome_info[outcome]
        results = regression_results[outcome]
        adjusted = results["adjusted_common_slope"]

        rows.append(
            {
                "outcome": outcome_label,
                "analysis": "Participant-adjusted common slope",
                "participant": "All",
                "model": "outcome ~ block + C(participant)",
                "test": "OLS slope t-test",
                "n": adjusted["n"],
                "slope": adjusted["slope"],
                "slope_unit": slope_unit,
                "ci_95_low": adjusted["slope_ci_low"],
                "ci_95_high": adjusted["slope_ci_high"],
                "se": adjusted["slope_stderr"],
                "statistic": "t",
                "statistic_value": adjusted["t_value"],
                "df_num_or_resid": adjusted["df_resid"],
                "df_den": np.nan,
                "p_value": adjusted["p_value"],
                "r_squared": adjusted["r_squared"],
                "adjusted_r_squared": adjusted["adjusted_r_squared"],
                "significant": adjusted["significant"],
            }
        )

        for participant in PARTICIPANT_ORDER:
            result = results["participants"][participant]
            if result is None:
                continue

            rows.append(
                {
                    "outcome": outcome_label,
                    "analysis": "Participant-specific slope",
                    "participant": participant,
                    "model": "outcome ~ block",
                    "test": "OLS slope t-test",
                    "n": result["n"],
                    "slope": result["slope"],
                    "slope_unit": slope_unit,
                    "ci_95_low": result["slope_ci_low"],
                    "ci_95_high": result["slope_ci_high"],
                    "se": result["slope_stderr"],
                    "statistic": "t",
                    "statistic_value": result["t_value"],
                    "df_num_or_resid": result["df_resid"],
                    "df_den": np.nan,
                    "p_value": result["p_value"],
                    "r_squared": result["r_squared"],
                    "adjusted_r_squared": result["adjusted_r_squared"],
                    "significant": result["significant"],
                }
            )

        # Test slope heterogeneity across participants.
        heterogeneity = results["heterogeneity"]
        rows.append(
            {
                "outcome": outcome_label,
                "analysis": "Block x participant interaction",
                "participant": "All",
                "model": (
                    "outcome ~ block + C(participant) vs. "
                    "outcome ~ block * C(participant)"
                ),
                "test": "Nested-model F-test",
                "n": int(heterogeneity["interaction_model"].nobs),
                "slope": np.nan,
                "slope_unit": "",
                "ci_95_low": np.nan,
                "ci_95_high": np.nan,
                "se": np.nan,
                "statistic": "F",
                "statistic_value": heterogeneity["f_value"],
                "df_num_or_resid": heterogeneity["df_num"],
                "df_den": heterogeneity["df_den"],
                "p_value": heterogeneity["p_value"],
                "r_squared": heterogeneity["interaction_model"].rsquared,
                "adjusted_r_squared": heterogeneity[
                    "interaction_model"
                ].rsquared_adj,
                "significant": heterogeneity["significant"],
            }
        )

    return pd.DataFrame(rows)


def print_regression_results_table(results_table):
    display = results_table.copy()
    display["slope"] = display["slope"].map(
        lambda value: "" if pd.isna(value) else f"{value:.4f}"
    )
    display["ci_95_low"] = display["ci_95_low"].map(
        lambda value: "" if pd.isna(value) else f"{value:.4f}"
    )
    display["ci_95_high"] = display["ci_95_high"].map(
        lambda value: "" if pd.isna(value) else f"{value:.4f}"
    )
    display["p_value"] = display["p_value"].map(format_p_value)
    display["significant"] = display["significant"].map(
        {True: "yes", False: "no"}
    )

    columns = [
        "outcome",
        "analysis",
        "participant",
        "n",
        "slope",
        "slope_unit",
        "ci_95_low",
        "ci_95_high",
        "p_value",
        "significant",
    ]
    print("\nRegression results")
    print(display[columns].to_string(index=False))


def save_regression_results_csv(results_table, output_file):
    """Save the unrounded statistical results for use as a supplementary table."""
    results_table.to_csv(output_file, index=False, float_format="%.10g")
    print(f"Saved regression-results CSV: {output_file.resolve()}")


def print_regression_results(regression_results):
    """Build and print the consolidated regression table."""
    results_table = build_regression_results_table(regression_results)
    print_regression_results_table(results_table)
    return results_table


def plot_linear_fit(
    axis,
    regression_result,
    color,
    linewidth=1.2,
    linestyle="--",
    alpha=0.9,
    zorder=2,
    use_adjusted_plot_intercept=False,
):
    if regression_result is None:
        return

    fit_x = np.linspace(
        regression_result["x_min"],
        regression_result["x_max"],
        200,
    )

    if use_adjusted_plot_intercept:
        intercept = regression_result["plot_intercept"]
    else:
        intercept = regression_result["intercept"]

    fit_y = intercept + regression_result["slope"] * fit_x

    axis.plot(
        fit_x,
        fit_y,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
    )


def smooth_series(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size <= 1:
        return x, y

    span = x.max() - x.min()
    n_points = int(
        max(2, np.ceil(span * SMOOTH_POINTS_PER_UNIT))
    ) + 1
    smooth_x = np.linspace(x.min(), x.max(), n_points)

    if x.size >= 4:
        try:
            spline = make_interp_spline(x, y, k=3)
            return smooth_x, spline(smooth_x)
        except ValueError:
            pass

    return smooth_x, np.interp(smooth_x, x, y)


def add_session_breaks(x, y, breaks_after):
    disconnected = np.asarray(y, dtype=float).copy()

    for break_index in breaks_after:
        mask = (x > break_index) & (x < break_index + 1)
        disconnected[mask] = np.nan

    return disconnected


def plot_participant_series(
    axis,
    series,
    value_key,
    participant,
    marker_map,
    multiple_block_types,
    regression_result=None,
):
    x = series["x"]
    y = series[value_key]
    color = PARTICIPANT_COLORS[participant]

    if x.size == 0:
        return

    if multiple_block_types:
        for block_type, marker in marker_map.items():
            indices = np.flatnonzero(
                series["block_types"] == block_type
            )
            if indices.size:
                axis.plot(
                    x[indices],
                    y[indices],
                    linestyle="None",
                    marker=marker,
                    color=color,
                    markersize=MARKER_SIZES.get(marker, 5),
                )
    else:
        axis.plot(
            x,
            y,
            linestyle="None",
            marker="o",
            color=color,
            markersize=3.5,
        )

    smooth_x, smooth_y = smooth_series(x, y)
    smooth_y = add_session_breaks(
        smooth_x,
        smooth_y,
        series["breaks_after"],
    )

    axis.plot(
        smooth_x,
        smooth_y,
        color=color,
        linewidth=1.5,
        alpha=0.9,
    )

    if SHOW_LINEAR_FITS:
        plot_linear_fit(
            axis,
            regression_result,
            color=color,
            linewidth=1.1,
            linestyle="--",
            alpha=0.85,
            zorder=2,
        )


def add_block_type_legend(
    figure,
    selected_block_types,
    marker_map,
):
    handles = [
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker=marker_map[block_type],
            color="black",
            markersize=MARKER_SIZES.get(
                marker_map[block_type],
                5,
            ),
            label=BLOCK_TYPE_LABELS.get(
                block_type,
                block_type.capitalize(),
            ),
        )
        for block_type in selected_block_types
    ]

    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.55, 0.59),
        handlelength=0.8,
        handletextpad=0.4,
        columnspacing=1.0,
    )


def plot_figure(
    series_by_participant,
    selected_block_types,
    marker_map,
    regression_results,
):
    multiple_block_types = len(selected_block_types) > 1

    figure, (accuracy_axis, timing_axis) = plt.subplots(
        2,
        1,
        figsize=(6, 3.2),
        sharex=True,
        constrained_layout=True,
    )

    for participant in PARTICIPANT_ORDER:
        plot_participant_series(
            accuracy_axis,
            series_by_participant[participant],
            "accuracy",
            participant,
            marker_map,
            multiple_block_types,
            regression_results["accuracy"]["participants"][participant],
        )

    if SHOW_POOLED_LINEAR_FIT:
        plot_linear_fit(
            accuracy_axis,
            regression_results["accuracy"]["adjusted_common_slope"],
            color="black",
            linewidth=1.5,
            linestyle=":",
            alpha=0.95,
            zorder=2.5,
            use_adjusted_plot_intercept=True,
        )

    accuracy_axis.set_ylabel("On-time (%)")
    accuracy_axis.set_ylim(0, 110)

    for participant in PARTICIPANT_ORDER:
        plot_participant_series(
            timing_axis,
            series_by_participant[participant],
            "timing",
            participant,
            marker_map,
            multiple_block_types,
            regression_results["timing"]["participants"][participant],
        )

    if SHOW_POOLED_LINEAR_FIT:
        plot_linear_fit(
            timing_axis,
            regression_results["timing"]["adjusted_common_slope"],
            color="black",
            linewidth=1.5,
            linestyle=":",
            alpha=0.95,
            zorder=2.5,
            use_adjusted_plot_intercept=True,
        )

    timing_axis.set_ylabel("Timing (ms)")
    timing_axis.set_xlabel("Block number (chronological)")

    maximum_blocks = max(
        len(series["x"])
        for series in series_by_participant.values()
    )
    maximum_blocks = max(maximum_blocks, 1)

    padding = 0.35
    timing_axis.set_xlim(
        1 - padding,
        maximum_blocks + padding,
    )
    timing_axis.set_xticks(
        np.arange(1, maximum_blocks + 1)
    )

    label_positions = {
        "P2": (0.95, 0.42),
        "P3": (0.95, 0.27),
        "C1": (0.948, 0.12),
    }
    for participant, position in label_positions.items():
        accuracy_axis.text(
            *position,
            participant,
            color=PARTICIPANT_COLORS[participant],
            transform=accuracy_axis.transAxes,
            va="center",
        )

    if multiple_block_types and SHOW_BLOCKTYPE_LEGEND:
        add_block_type_legend(
            figure,
            selected_block_types,
            marker_map,
        )

    return figure



def main():
    selected_block_types, _ = (
        resolve_selected_block_types()
    )
    marker_map = build_marker_map(selected_block_types)

    series_by_participant = {}

    for participant in PARTICIPANT_ORDER:
        series = extract_chronological_series(
            SUMMARIES[participant],
            selected_block_types,
        )
        series_by_participant[participant] = series


    regression_results = compute_regression_results(
        series_by_participant
    )
    results_table = print_regression_results(regression_results)
    save_regression_results_csv(
        results_table,
        REGRESSION_CSV_FILE,
    )

    figure = plot_figure(
        series_by_participant,
        selected_block_types,
        marker_map,
        regression_results,
    )
    figure.savefig(
        OUTPUT_FILE,
        bbox_inches="tight",
        dpi=500,
    )
    print(f"Saved: {OUTPUT_FILE.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()