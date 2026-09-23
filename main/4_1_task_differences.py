#!/usr/bin/env python3
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats
from scipy.ndimage import gaussian_filter1d


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_DIR = Path("../../FlipThatBucket/data")
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = OUTPUT_DIR / "figure_4b.pdf"
FORCE_ANOVA_STATS_FILE = OUTPUT_DIR / "figure_4b_force_anova_stats.csv"
FORCE_PAIRWISE_STATS_FILE = OUTPUT_DIR / "figure_4b_force_pairwise_stats.csv"

PREMOVEMENT_BINS = 50
POSTMOVEMENT_BINS = 25
BIN_SECONDS = 0.02

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

TASK_ORDER = ["reaction", "ftb", "scroll"]

TASK_COLORS = {
    "reaction": "#7F7F7F",
    "ftb": "#0072B2",
    "scroll": "#56B4E9",
}

TASK_LABELS = {
    "reaction": "Reaction time",
    "ftb": "Flip-that-Bucket",
    "scroll": "Scroll",
}

MUA_PERCENTILE = 99
MUA_SMOOTHING_SIGMA = 2
FORCE_BASELINE_BINS = 25

# -----------------------------------------------------------------------------
# Onset analysis
# -----------------------------------------------------------------------------
# Fit a flat -> slow-rise -> fast-rise piecewise-linear model to trial-averaged
# MUA. Breakpoints are exhaustively searched on the 20 ms grid up to the peak.
NORMALISE_BEFORE_FIT = True       # fit the min-max (0-1) trace shown in panel 2
BREAKPOINT_FIT_START = -1.00      # s, start of the fit window
BREAKPOINT_PEAK_SEARCH_START = -0.30  # s, earliest allowed location of the peak
BREAKPOINT_FIT_END = None         # s, hard end of fit window; None -> use the peak
BREAKPOINT_TAU_STEP = BIN_SECONDS  # s, break-points are searched on the bin grid
# Require at least 100 ms per segment.
BREAKPOINT_MIN_SEGMENT_BINS = 5
BREAKPOINT_REQUIRE_INCREASING = True  # enforce 0 < b1 < b2 (slow then fast)

PLOT_BREAKPOINT_MARKERS = True
PLOT_SEGMENTED_FITS = False       # thin black line showing the fitted model
PLOT_ONSET_LINES = False          # optional vertical lines at the break-points
SLOW_MARKER = "o"                 # baseline -> slow rise
FAST_MARKER = "D"                 # -> fast rise
MARKER_SIZE = 7.5
MARKER_NEUTRAL = "#BFBFBF"        # legend swatch colour

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"


def normalize_block(block):
    block = str(block)

    if block == "train" or block.startswith("ftb"):
        return "ftb"
    if block.startswith("reaction"):
        return "reaction"
    if block.startswith("scroll"):
        return "scroll"
    return "unknown"


def mua_3d_processing(session_list, percentile=MUA_PERCENTILE):
    processed = []

    for session in session_list:
        profiles = np.sum(
            gaussian_filter1d(session, sigma=MUA_SMOOTHING_SIGMA, axis=-1),
            axis=1,
        )

        upper_limit = np.percentile(profiles, percentile)
        profiles = np.clip(profiles, 0, upper_limit)

        mean = profiles.mean()
        std = profiles.std()
        if std == 0:
            std = 1.0

        processed.append((profiles - mean) / std)

    return np.concatenate(processed, axis=0)


def force_2d_processing(session_list, std_floor=None):
    processed = []

    for session in session_list:
        baseline = session[:, :FORCE_BASELINE_BINS].mean(axis=1, keepdims=True)
        session = session - baseline

        mean = session.mean()
        std = session.std()

        if std_floor is not None:
            std = max(std, std_floor)
        if std == 0:
            std = 1.0

        processed.append((session - mean) / std)

    return np.concatenate(processed, axis=0)


def load_session_trials(session_path, pre_bins, post_bins, player_id=1):
    session = pd.read_pickle(session_path)

    player_actions = session[
        (session["PlayerID"] == player_id)
        & (session["Onset"] >= pre_bins)
        & (session["PostMovement"] >= post_bins)
    ].reset_index(drop=True)

    total_bins = pre_bins + post_bins

    neural_trials = []
    force_trials = []
    blocks = []

    for row in player_actions.itertuples(index=False):
        onset = int(row.Onset)
        start = onset - pre_bins
        stop = onset + post_bins

        neural = np.asarray(row.Neural)[:, start:stop]
        force = np.asarray(row.Force)[start:stop]

        if neural.shape[-1] != total_bins or force.shape[-1] != total_bins:
            continue

        neural_trials.append(neural)
        force_trials.append(force)
        blocks.append(normalize_block(row.BlockType))

    if not neural_trials:
        return (
            np.empty((0, 0, total_bins)),
            np.empty((0, total_bins)),
            np.empty(0, dtype=object),
        )

    return (
        np.stack(neural_trials, axis=0),
        np.stack(force_trials, axis=0),
        np.asarray(blocks, dtype=object),
    )


def mean_and_std(data_2d):
    data_2d = np.asarray(data_2d)
    mean = np.nanmean(data_2d, axis=0)

    if data_2d.shape[0] > 1:
        std = np.nanstd(data_2d, axis=0, ddof=1)
    else:
        std = np.zeros_like(mean)

    return mean, std


def minmax_1d(values, eps=1e-12):
    values = np.asarray(values, dtype=float)
    low = np.nanmin(values)
    high = np.nanmax(values)
    return (values - low) / (high - low + eps)


def load_all_sessions():
    neural_sessions = []
    force_sessions = []
    block_sessions = []

    for participant, session_names in PARTICIPANTS.items():
        for session_name in session_names:
            session_path = DATA_DIR / participant / f"{session_name}.pkl"
            neurals, forces, blocks = load_session_trials(
                session_path,
                PREMOVEMENT_BINS,
                POSTMOVEMENT_BINS,
                player_id=1,
            )

            if neurals.shape[0] == 0:
                print(f"[WARN] No valid trials found for {participant}/{session_name}")
                continue

            neural_sessions.append(neurals)
            force_sessions.append(forces)
            block_sessions.append(blocks)


    if not block_sessions:
        raise RuntimeError("No sessions or trials were loaded. Check paths and filters.")

    return neural_sessions, force_sessions, block_sessions


def _format_p_value(value):
    if not np.isfinite(value):
        return "n/a"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def _significance_code(value):
    """Publication-style significance code for a p-value."""
    if not np.isfinite(value):
        return "n/a"
    if value < 0.0001:
        return "****"
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return "ns"


def _bonferroni_adjust(p_values):
    """Bonferroni-adjust a one-dimensional array of p-values."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan, dtype=float)
    valid = np.isfinite(p_values)
    n_tests = int(valid.sum())

    if n_tests:
        adjusted[valid] = np.minimum(p_values[valid] * n_tests, 1.0)

    return adjusted


def _welch_anova(groups):
    """Welch one-way ANOVA for independent groups with unequal variances."""
    cleaned = [np.asarray(group, dtype=float) for group in groups]
    cleaned = [group[np.isfinite(group)] for group in cleaned]

    if len(cleaned) < 2 or any(group.size < 2 for group in cleaned):
        return np.nan, np.nan, np.nan, np.nan

    n = np.asarray([group.size for group in cleaned], dtype=float)
    means = np.asarray([group.mean() for group in cleaned], dtype=float)
    variances = np.asarray([group.var(ddof=1) for group in cleaned], dtype=float)

    # Welch weights require nonzero within-group variance.
    if np.any(variances <= 0) or not np.all(np.isfinite(variances)):
        return np.nan, np.nan, np.nan, np.nan

    k = len(cleaned)
    weights = n / variances
    total_weight = weights.sum()
    weighted_mean = np.sum(weights * means) / total_weight

    numerator = np.sum(weights * (means - weighted_mean) ** 2) / (k - 1)
    correction_sum = np.sum(
        ((1.0 - weights / total_weight) ** 2) / (n - 1.0)
    )
    correction = 1.0 + (2.0 * (k - 2.0) / (k ** 2 - 1.0)) * correction_sum

    f_statistic = numerator / correction
    df1 = float(k - 1)
    df2 = float((k ** 2 - 1.0) / (3.0 * correction_sum))
    p_value = float(stats.f.sf(f_statistic, df1, df2))
    return float(f_statistic), df1, df2, p_value


def report_post_onset_force(forces, tasks, time_axis):
    """Report 0-to-200-ms force summaries and task comparisons."""
    window_mask = (time_axis >= 0.0) & (time_axis <= 0.200 + 1e-12)
    trial_window_means = np.nanmean(forces[:, window_mask], axis=1)

    task_values = {}
    task_summaries = {}
    print("\nMean z-scored force, 0-200 ms")

    for task in TASK_ORDER:
        task_mask = tasks == task
        if not np.any(task_mask):
            continue

        values = trial_window_means[task_mask]
        values = values[np.isfinite(values)]
        task_values[task] = values

        mean_force = float(values.mean()) if values.size else np.nan
        sd_force = float(values.std(ddof=1)) if values.size > 1 else np.nan
        task_summaries[task] = {
            "mean": mean_force,
            "sd": sd_force,
            "n": int(values.size),
        }
        label = TASK_LABELS.get(task, task)
        print(f"{label}: {mean_force:.3f} ± {sd_force:.3f} (n={values.size})")

    available_tasks = [task for task in TASK_ORDER if task in task_values]
    f_statistic, df1, df2, p_value = _welch_anova(
        [task_values[task] for task in available_tasks]
    )
    if np.isfinite(f_statistic):
        print(
            f"Welch ANOVA: F({df1:.0f}, {df2:.2f})={f_statistic:.3f}, "
            f"p={_format_p_value(p_value)}"
        )
    else:
        print("Welch ANOVA: unavailable")

    anova_row = {
        "Test": "Welch one-way ANOVA",
        "Reaction time mean": task_summaries.get("reaction", {}).get("mean", np.nan),
        "Reaction time SD": task_summaries.get("reaction", {}).get("sd", np.nan),
        "Reaction time n": task_summaries.get("reaction", {}).get("n", 0),
        "Flip-that-Bucket mean": task_summaries.get("ftb", {}).get("mean", np.nan),
        "Flip-that-Bucket SD": task_summaries.get("ftb", {}).get("sd", np.nan),
        "Flip-that-Bucket n": task_summaries.get("ftb", {}).get("n", 0),
        "Scroll mean": task_summaries.get("scroll", {}).get("mean", np.nan),
        "Scroll SD": task_summaries.get("scroll", {}).get("sd", np.nan),
        "Scroll n": task_summaries.get("scroll", {}).get("n", 0),
        "Statistic (F)": f_statistic,
        "DF1": df1,
        "DF2": df2,
        "P-value (raw)": p_value,
        "Significance": _significance_code(p_value),
    }
    pd.DataFrame([anova_row]).to_csv(FORCE_ANOVA_STATS_FILE, index=False)

    pairwise_results = []
    for first_index in range(len(available_tasks)):
        for second_index in range(first_index + 1, len(available_tasks)):
            task_a = available_tasks[first_index]
            task_b = available_tasks[second_index]
            values_a = task_values[task_a]
            values_b = task_values[task_b]

            if values_a.size < 2 or values_b.size < 2:
                pairwise_results.append(
                    {
                        "task_a": task_a,
                        "task_b": task_b,
                        "mean_a": float(values_a.mean()) if values_a.size else np.nan,
                        "mean_b": float(values_b.mean()) if values_b.size else np.nan,
                        "difference": np.nan,
                        "t": np.nan,
                        "df": np.nan,
                        "p": np.nan,
                    }
                )
                continue

            mean_a = float(values_a.mean())
            mean_b = float(values_b.mean())
            mean_difference = mean_a - mean_b
            variance_a = float(values_a.var(ddof=1))
            variance_b = float(values_b.var(ddof=1))
            standard_error_sq = variance_a / values_a.size + variance_b / values_b.size

            if standard_error_sq <= 0:
                t_statistic = np.nan
                df = np.nan
                raw_p = np.nan
            else:
                standard_error = np.sqrt(standard_error_sq)
                t_statistic = mean_difference / standard_error
                df = standard_error_sq ** 2 / (
                    (variance_a / values_a.size) ** 2 / (values_a.size - 1)
                    + (variance_b / values_b.size) ** 2 / (values_b.size - 1)
                )
                raw_p = float(2.0 * stats.t.sf(abs(t_statistic), df))

            pairwise_results.append(
                {
                    "task_a": task_a,
                    "task_b": task_b,
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    "difference": mean_difference,
                    "t": float(t_statistic),
                    "df": float(df),
                    "p": raw_p,
                }
            )

    adjusted_p_values = _bonferroni_adjust(
        [result["p"] for result in pairwise_results]
    )
    pairwise_table_rows = []

    for result, adjusted_p in zip(pairwise_results, adjusted_p_values):
        label_a = TASK_LABELS.get(result["task_a"], result["task_a"])
        label_b = TASK_LABELS.get(result["task_b"], result["task_b"])
        pairwise_table_rows.append(
            {
                "Task 1": label_a,
                "Task 2": label_b,
                "Task 1 mean": result["mean_a"],
                "Task 2 mean": result["mean_b"],
                "Difference": result["difference"],
                "Statistic (t)": result["t"],
                "DF": result["df"],
                "P-value (raw)": result["p"],
                "P-value (adjusted)": adjusted_p,
                "Significance": _significance_code(adjusted_p),
            }
        )

        if not np.isfinite(result["t"]):
            print(f"{label_a} vs {label_b}: unavailable")
            continue

        print(
            f"{label_a} vs {label_b}: Δ={result['difference']:+.3f}, "
            f"t({result['df']:.2f})={result['t']:.3f}, "
            f"p_adj={_format_p_value(adjusted_p)}"
        )

    pd.DataFrame(pairwise_table_rows).to_csv(FORCE_PAIRWISE_STATS_FILE, index=False)
    print(f"Saved: {FORCE_ANOVA_STATS_FILE.resolve()}")
    print(f"Saved: {FORCE_PAIRWISE_STATS_FILE.resolve()}")


def compute_task_summaries(neurals, forces, tasks):
    mua_mean = {}
    mua_std = {}
    mua_mean_normalized = {}

    force_median = {}
    force_q25 = {}
    force_q75 = {}

    present_tasks = []

    for task in TASK_ORDER:
        mask = tasks == task
        if not np.any(mask):
            continue

        present_tasks.append(task)

        mua_mean[task], mua_std[task] = mean_and_std(neurals[mask])
        mua_mean_normalized[task] = minmax_1d(mua_mean[task])

        force_subset = forces[mask]
        force_median[task] = np.nanmedian(force_subset, axis=0)
        force_q25[task] = np.nanpercentile(force_subset, 25, axis=0)
        force_q75[task] = np.nanpercentile(force_subset, 75, axis=0)

    return {
        "present_tasks": present_tasks,
        "mua_mean": mua_mean,
        "mua_std": mua_std,
        "mua_mean_normalized": mua_mean_normalized,
        "force_median": force_median,
        "force_q25": force_q25,
        "force_q75": force_q75,
    }


# -----------------------------------------------------------------------------
# Break-point fitting
# -----------------------------------------------------------------------------
def _resolve_fit_window(time_axis, trace, fit_start, fit_end, peak_search_start):
    """Resolve the fit window, ending at the rising-phase peak when unspecified."""
    time_axis = np.asarray(time_axis, dtype=float)
    trace = np.asarray(trace, dtype=float)

    if fit_end is None:
        candidates = np.where(time_axis >= peak_search_start)[0]
        if candidates.size == 0:
            candidates = np.arange(time_axis.size)
        peak_index = candidates[int(np.nanargmax(trace[candidates]))]
        fit_end = float(time_axis[peak_index])

    mask = (time_axis >= fit_start - 1e-12) & (time_axis <= fit_end + 1e-12)
    return mask, float(fit_end)


def _tau_grid(t, min_segment_bins, tau_step, bin_width):
    low = t[0] + (min_segment_bins - 1) * bin_width
    high = t[-1] - (min_segment_bins - 1) * bin_width
    if high <= low:
        return None
    n_grid = max(int(np.round((high - low) / tau_step)) + 1, 3)
    return np.linspace(low, high, n_grid)


def _model_curve(t, fit):
    """Piecewise-linear prediction of the fitted model."""
    return (
        fit["baseline_level"]
        + fit["slope_slow"] * np.maximum(0.0, t - fit["tau_slow"])
        + (fit["slope_fast"] - fit["slope_slow"]) * np.maximum(0.0, t - fit["tau_fast"])
    )


def fit_breakpoints(
    time_axis,
    trace,
    fit_start=BREAKPOINT_FIT_START,
    fit_end=BREAKPOINT_FIT_END,
    peak_search_start=BREAKPOINT_PEAK_SEARCH_START,
    tau_step=BREAKPOINT_TAU_STEP,
    min_segment_bins=BREAKPOINT_MIN_SEGMENT_BINS,
    require_increasing_slopes=BREAKPOINT_REQUIRE_INCREASING,
):
    """Fit the two-breakpoint piecewise-linear model by exhaustive grid search."""
    time_axis = np.asarray(time_axis, dtype=float)
    trace = np.asarray(trace, dtype=float)

    mask, resolved_end = _resolve_fit_window(
        time_axis, trace, fit_start, fit_end, peak_search_start
    )
    t = time_axis[mask]
    y = trace[mask]
    if t.size < (3 * min_segment_bins + 1) or not np.all(np.isfinite(y)):
        return None

    bin_width = float(np.median(np.diff(time_axis)))
    taus = _tau_grid(t, min_segment_bins, tau_step, bin_width)
    if taus is None or taus.size < 2:
        return None

    n_time = t.size
    x = np.maximum(0.0, t[None, :] - taus[:, None])
    x_mean = x.mean(axis=1, keepdims=True)
    xc = x - x_mean
    y_mean = float(y.mean())
    yc = y - y_mean

    gram = xc @ xc.T
    h = xc @ yc
    syy = float(yc @ yc)

    diag = np.diag(gram)
    suu, svv, suv = diag[:, None], diag[None, :], gram
    suy, svy = h[:, None], h[None, :]

    det = suu * svv - suv ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        beta_u = (svv * suy - suv * svy) / det
        beta_v = (suu * svy - suv * suy) / det
        sse = syy - beta_u * suy - beta_v * svy

    n_le = np.searchsorted(t, taus, side="right")
    ones = np.ones((1, taus.size), dtype=int)
    seg1 = n_le[:, None] * ones
    seg2 = n_le[None, :] - n_le[:, None]
    seg3 = n_time - n_le[None, :] * ones.T

    # Retain breakpoint-search counts for diagnostics.
    ordered_pairs = taus[None, :] > taus[:, None]

    segment_length_valid = (
        ordered_pairs
        & (seg1 >= min_segment_bins)
        & (seg2 >= min_segment_bins)
        & (seg3 >= min_segment_bins)
    )

    numerically_valid = (
        np.isfinite(det)
        & (np.abs(det) > 1e-12)
        & np.isfinite(sse)
    )

    valid = segment_length_valid & numerically_valid

    n_breakpoint_locations = int(taus.size)
    n_ordered_pairs = int(np.count_nonzero(ordered_pairs))
    n_segment_valid_pairs = int(np.count_nonzero(segment_length_valid))
    n_estimable_pairs = int(np.count_nonzero(valid))

    if require_increasing_slopes:
        valid &= (beta_u > 0) & (beta_v > 0)

    n_valid_pairs = int(np.count_nonzero(valid))

    if not np.any(valid):
        return None

    sse = np.where(valid, sse, np.inf)
    i, j = np.unravel_index(int(np.argmin(sse)), sse.shape)

    fit = {
        "tau_slow": float(taus[i]),
        "tau_fast": float(taus[j]),
        "slope_slow": float(beta_u[i, j]),
        "slope_fast": float(beta_u[i, j] + beta_v[i, j]),
        "baseline_level": float(
            y_mean - beta_u[i, j] * x_mean[i, 0] - beta_v[i, j] * x_mean[j, 0]
        ),
        "sse": float(sse[i, j]),
        "r_squared": float(1.0 - sse[i, j] / syy) if syy > 0 else np.nan,
        "n_obs": int(n_time),
        "fit_start": float(t[0]),
        "fit_end": resolved_end,
        "fit_time": t,
        "n_breakpoint_locations": n_breakpoint_locations,
        "n_ordered_pairs": n_ordered_pairs,
        "n_segment_valid_pairs": n_segment_valid_pairs,
        "n_estimable_pairs": n_estimable_pairs,
        "n_valid_pairs": n_valid_pairs,
    }
    fit["fit_values"] = _model_curve(t, fit)
    return fit


def run_onset_analysis(tasks, time_axis, summaries):
    """Fit the break-point model to each task's averaged profile."""
    results = {}
    trace_key = "mua_mean_normalized" if NORMALISE_BEFORE_FIT else "mua_mean"

    for task in summaries["present_tasks"]:
        fit = fit_breakpoints(time_axis, summaries[trace_key][task])
        if fit is None:
            print(f"[WARN] Break-point fit failed for task '{task}'.")
            continue

        fit["n_trials"] = int(np.count_nonzero(tasks == task))

        bin_width = float(np.median(np.diff(time_axis)))
        edge = (BREAKPOINT_MIN_SEGMENT_BINS + 1) * bin_width
        flags = []
        if fit["tau_slow"] <= fit["fit_start"] + edge:
            flags.append("departure pinned to window start")
        if fit["tau_fast"] >= fit["fit_end"] - edge:
            flags.append("fast rise pinned to window end")
        if (fit["tau_fast"] - fit["tau_slow"]) <= BREAKPOINT_MIN_SEGMENT_BINS * bin_width:
            flags.append("slow segment at its minimum length")
        if fit["r_squared"] < 0.80:
            flags.append("low R2 (<0.80)")
        fit["flags"] = flags

        results[task] = fit

    return results


def report_onset_results(results):
    if not results:
        return

    print("\nMUA piecewise-linear fit")
    for task in TASK_ORDER:
        fit = results.get(task)
        if fit is None:
            continue

        label = TASK_LABELS.get(task, task)
        summary = (
            f"{label}: departure={1000.0 * fit['tau_slow']:.0f} ms, "
            f"fast rise={1000.0 * fit['tau_fast']:.0f} ms, "
            f"R²={fit['r_squared']:.3f}, n={fit['n_trials']}"
        )
        if fit["flags"]:
            summary += f"; check={'; '.join(fit['flags'])}"
        print(summary)


def plot_figure(summaries, time_axis, onsets=None):
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(5.4, 6.4),
        sharex=True,
        gridspec_kw={"hspace": 0.15},
    )

    present_tasks = summaries["present_tasks"]
    onsets = onsets or {}

    # Panel 1: MUA mean ± SD.
    axis = axes[0]
    for task in present_tasks:
        axis.plot(
            time_axis,
            summaries["mua_mean"][task],
            linewidth=2.2,
            color=TASK_COLORS[task],
            label=TASK_LABELS[task],
        )
        axis.fill_between(
            time_axis,
            summaries["mua_mean"][task] - summaries["mua_std"][task],
            summaries["mua_mean"][task] + summaries["mua_std"][task],
            color=TASK_COLORS[task],
            alpha=0.18,
            linewidth=0,
        )
    axis.axvline(0.0, color="red", linewidth=1.2)
    axis.set_ylabel("MUA (z-scored)")
    axis.legend(loc="upper left", fontsize=9, frameon=False)

    # Panel 2: normalized MUA with fitted breakpoints.
    axis = axes[1]
    for task in present_tasks:
        axis.plot(
            time_axis,
            summaries["mua_mean_normalized"][task],
            linewidth=2.2,
            color=TASK_COLORS[task],
        )
    axis.axvline(0.0, color="red", linewidth=1.2)
    axis.set_ylabel("MUA (norm. 0–1)")

    if PLOT_SEGMENTED_FITS:
        for task in present_tasks:
            fit = onsets.get(task)
            if fit is None:
                continue
            axis.plot(
                fit["fit_time"],
                fit["fit_values"],
                color="black",
                linewidth=0.9,
                alpha=0.7,
                zorder=3,
            )

    if PLOT_ONSET_LINES:
        for task in present_tasks:
            fit = onsets.get(task)
            if fit is None:
                continue
            for tau in (fit["tau_slow"], fit["tau_fast"]):
                axis.axvline(tau, color=TASK_COLORS[task], linestyle="--",
                             linewidth=1.0, alpha=0.7, zorder=1)

    if PLOT_BREAKPOINT_MARKERS:
        trace_key = "mua_mean_normalized"
        for task in present_tasks:
            fit = onsets.get(task)
            if fit is None:
                continue
            for tau, marker in ((fit["tau_slow"], SLOW_MARKER),
                                (fit["tau_fast"], FAST_MARKER)):
                if PLOT_SEGMENTED_FITS:
                    marker_y = float(_model_curve(np.array([tau]), fit)[0])
                else:
                    # Place markers on the plotted MUA trace.
                    marker_y = float(
                        np.interp(tau, time_axis, summaries[trace_key][task])
                    )
                axis.plot(
                    [tau],
                    [marker_y],
                    linestyle="none",
                    marker=marker,
                    markersize=MARKER_SIZE,
                    markerfacecolor=TASK_COLORS[task],
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                    zorder=5,
                    clip_on=False,
                )

        handles = [
            Line2D([], [], linestyle="none", marker=SLOW_MARKER,
                   markersize=MARKER_SIZE, markerfacecolor=MARKER_NEUTRAL,
                   markeredgecolor="black", markeredgewidth=0.8,
                   label="Slow-rise onset"),
            Line2D([], [], linestyle="none", marker=FAST_MARKER,
                   markersize=MARKER_SIZE, markerfacecolor=MARKER_NEUTRAL,
                   markeredgecolor="black", markeredgewidth=0.8,
                   label="Fast-rise onset"),
        ]
        axis.legend(handles=handles, loc="upper left", fontsize=8.5, frameon=False,
                    handletextpad=0.4, labelspacing=0.3)

    # Panel 3: force median + IQR.
    axis = axes[2]
    for task in present_tasks:
        axis.plot(
            time_axis,
            summaries["force_median"][task],
            linewidth=2.2,
            color=TASK_COLORS[task],
        )
        axis.fill_between(
            time_axis,
            summaries["force_q25"][task],
            summaries["force_q75"][task],
            color=TASK_COLORS[task],
            alpha=0.15,
            linewidth=0,
        )
    axis.axvline(0.0, color="red", linewidth=1.2)
    axis.set_ylabel("Force (z-scored)")
    axis.set_xlabel("Time (s)")
    axis.text(
        0.62,
        0.36,
        "Action onset",
        color="red",
        transform=axis.transAxes,
        fontsize=10,
        rotation=90,
    )

    for axis in axes:
        axis.spines["top"].set_visible(True)
        axis.spines["right"].set_visible(True)

    axes[2].set_xlim(-1.0, 0.5)
    axes[2].set_xticks([-1.0, -0.5, 0.0, 0.5])
    axes[2].set_xticklabels(["-1", "-0.5", "0", "0.5"])

    return fig


def main():
    neural_sessions, force_sessions, block_sessions = load_all_sessions()

    tasks = np.concatenate(block_sessions, axis=0)
    neurals = mua_3d_processing(neural_sessions)
    forces = force_2d_processing(force_sessions)

    if neurals.shape[0] != tasks.shape[0] or forces.shape[0] != tasks.shape[0]:
        raise RuntimeError(
            "Alignment mismatch between processed neural trials, force trials, "
            "and task labels."
        )

    summaries = compute_task_summaries(neurals, forces, tasks)

    total_bins = PREMOVEMENT_BINS + POSTMOVEMENT_BINS
    time_axis = (np.arange(total_bins) - PREMOVEMENT_BINS) * BIN_SECONDS

    report_post_onset_force(forces, tasks, time_axis)

    onsets = run_onset_analysis(tasks, time_axis, summaries)
    report_onset_results(onsets)

    figure = plot_figure(summaries, time_axis, onsets=onsets)
    figure.savefig(OUTPUT_FILE, bbox_inches="tight", dpi=500)
    print(f"Saved: {OUTPUT_FILE.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()