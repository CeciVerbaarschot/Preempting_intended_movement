#!/usr/bin/env python3
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
IMPORTANCE_DIR = SCRIPT_DIR / "feature_importances"

PARTICIPANT = "c1"
REGION = "both"          # "motor", "sensory", or "both"
PEDESTAL = "both"        # "lat", "med", or "both"
ANNOTATE_CHANNELS = False

COLORMAP = "RdPu"
OUTPUT_DPI = 600
OUTPUT_FILE = SCRIPT_DIR / f"{PARTICIPANT.lower()}_array_importance.png"

PARTICIPANT_SESSIONS = {
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

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"


def get_array_maps(participant):
    participant = participant.upper()

    p2_motor = np.array(
        [
            [np.nan, np.nan, 42, 58, 3, 13, 27, 97, np.nan, np.nan],
            [np.nan, 34, 44, 57, 4, 19, 29, 98, 107, np.nan],
            [33, 36, 51, 62, 7, 10, 31, 99, 108, 117],
            [35, 38, 53, 60, 5, 12, 18, 100, 109, 119],
            [37, 40, 50, 59, 6, 23, 22, 101, 110, 121],
            [39, 43, 46, 64, 9, 25, 24, 102, 111, 123],
            [41, 47, 56, 61, 17, 21, 26, 103, 113, 125],
            [45, 49, 55, 63, 15, 14, 28, 104, 112, 127],
            [np.nan, 48, 54, 2, 8, 16, 30, 105, 115, np.nan],
            [np.nan, np.nan, 52, 1, 11, 20, 32, 106, np.nan, np.nan],
        ],
        dtype=float,
    )

    shared_motor = np.array(
        [
            [np.nan, 38, 50, 59, 6, 23, 22, 101, 111, np.nan],
            [33, 40, 46, 64, 9, 25, 24, 102, 113, 128],
            [35, 43, 56, 61, 17, 21, 26, 103, 112, 114],
            [37, 47, 55, 63, 15, 14, 28, 104, 115, 116],
            [39, 49, 54, 2, 8, 16, 30, 105, 117, 118],
            [41, 48, 52, 1, 11, 20, 32, 106, 119, 120],
            [45, 42, 58, 3, 13, 27, 97, 107, 121, 122],
            [34, 44, 57, 4, 19, 29, 99, 108, 123, 124],
            [36, 51, 62, 7, 10, 31, 98, 109, 125, 126],
            [np.nan, 53, 60, 5, 12, 18, 100, 110, 127, np.nan],
        ],
        dtype=float,
    )

    shared_sensory = np.array(
        [
            [65, np.nan, 72, np.nan, 85, 91],
            [np.nan, 77, np.nan, 81, np.nan, 92],
            [67, np.nan, 74, np.nan, 87, np.nan],
            [np.nan, 79, np.nan, 82, np.nan, 93],
            [69, np.nan, 76, np.nan, 88, np.nan],
            [np.nan, 66, np.nan, 84, np.nan, 94],
            [71, np.nan, 78, np.nan, 89, np.nan],
            [np.nan, 68, np.nan, 83, np.nan, 96],
            [73, np.nan, 80, np.nan, 90, np.nan],
            [75, 70, np.nan, 86, np.nan, 95],
        ],
        dtype=float,
    )

    p2_sensory_lateral = np.array(
        [
            [65, np.nan, 72, np.nan, 85, 91],
            [np.nan, 77, np.nan, 81, np.nan, 92],
            [67, np.nan, 74, np.nan, 87, np.nan],
            [np.nan, 79, np.nan, 82, np.nan, 94],
            [69, np.nan, 76, np.nan, 88, np.nan],
            [np.nan, 66, np.nan, 84, np.nan, 93],
            [71, np.nan, 78, np.nan, 89, np.nan],
            [np.nan, 68, np.nan, 83, np.nan, 96],
            [73, np.nan, 80, np.nan, 90, np.nan],
            [75, 70, np.nan, 86, np.nan, 95],
        ],
        dtype=float,
    )

    p2_sensory_medial = np.array(
        [
            [65, np.nan, 72, np.nan, 85, 91],
            [np.nan, 77, np.nan, 81, np.nan, 92],
            [67, np.nan, 74, np.nan, 87, np.nan],
            [np.nan, np.nan, np.nan, 82, np.nan, 94],
            [69, 79, 76, np.nan, 88, np.nan],
            [np.nan, 66, np.nan, 84, np.nan, 93],
            [71, np.nan, 78, np.nan, 89, np.nan],
            [np.nan, 68, np.nan, 83, np.nan, 96],
            [73, np.nan, 80, np.nan, 90, np.nan],
            [75, 70, np.nan, 86, np.nan, 95],
        ],
        dtype=float,
    )

    if participant == "P2":
        return {
            "motor": p2_motor,
            "sens_lat": p2_sensory_lateral,
            "sens_med": p2_sensory_medial,
        }

    if participant in {"P3", "C1", "C2"}:
        return {
            "motor": shared_motor,
            "sens_lat": shared_sensory,
            "sens_med": shared_sensory,
        }

    raise ValueError("Participant must be P2, P3, C1, or C2.")


def get_bank_offsets(participant):
    participant = participant.upper()

    if participant in {"P2", "P3"}:
        return {"lat": 0, "med": 128}

    if participant in {"C1", "C2"}:
        return {"lat": 128, "med": 0}

    raise ValueError("Participant must be P2, P3, C1, or C2.")


def load_mean_feature_importance(participant):
    participant = participant.lower()

    if participant not in PARTICIPANT_SESSIONS:
        raise ValueError(
            f"No session list is configured for participant '{participant}'."
        )

    session_importances = []

    for session_name in PARTICIPANT_SESSIONS[participant]:
        path = (
            IMPORTANCE_DIR
            / f"featimp_{participant}_{session_name}.npy"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Feature-importance file not found: {path}"
            )

        importances = np.load(path)

        if importances.ndim != 2 or importances.shape[1] != 256:
            raise ValueError(
                f"{path} must have shape (seeds, 256), "
                f"but has shape {importances.shape}."
            )

        session_importances.append(importances)

        print(
            f"Loaded {path.name}: "
            f"{importances.shape[0]} seeds"
        )

    combined = np.concatenate(session_importances, axis=0)
    return combined.mean(axis=0)


def build_heatmap(weights, layout, bank_offset):
    heatmap = np.full(layout.shape, np.nan, dtype=float)

    for row in range(layout.shape[0]):
        for column in range(layout.shape[1]):
            channel = layout[row, column]

            if np.isnan(channel):
                continue

            absolute_index = int(channel) - 1 + bank_offset
            heatmap[row, column] = weights[absolute_index]

    return heatmap


def get_panels(participant, region, pedestal):
    maps = get_array_maps(participant)

    if region == "both":
        return [
            ("motor", "lat", maps["motor"]),
            ("motor", "med", maps["motor"]),
            ("sensory", "lat", maps["sens_lat"]),
            ("sensory", "med", maps["sens_med"]),
        ]

    if region not in {"motor", "sensory"}:
        raise ValueError("REGION must be 'motor', 'sensory', or 'both'.")

    if pedestal == "both":
        pedestals = ["lat", "med"]
    elif pedestal in {"lat", "med"}:
        pedestals = [pedestal]
    else:
        raise ValueError("PEDESTAL must be 'lat', 'med', or 'both'.")

    return [
        (
            region,
            current_pedestal,
            maps["motor"]
            if region == "motor"
            else maps[f"sens_{current_pedestal}"],
        )
        for current_pedestal in pedestals
    ]


def get_figure_layout(region, n_panels):
    if region == "both":
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(11, 10),
        )
        return figure, axes.ravel()

    width_per_panel = 5.5 if region == "motor" else 3.8
    figure, axes = plt.subplots(
        1,
        n_panels,
        figsize=(width_per_panel * n_panels, 5),
        squeeze=False,
    )
    return figure, axes.ravel()


def plot_utah_heatmap(
    weights,
    participant,
    region="both",
    pedestal="both",
    colormap=COLORMAP,
    annotate=False,
):
    weights = np.asarray(weights, dtype=float).reshape(-1)

    if weights.size != 256:
        raise ValueError(
            f"Expected a 256-channel vector, but received {weights.size} values."
        )

    participant = participant.upper()
    offsets = get_bank_offsets(participant)
    panels = get_panels(participant, region, pedestal)

    heatmaps = [
        build_heatmap(
            weights,
            layout,
            offsets[current_pedestal],
        )
        for _, current_pedestal, layout in panels
    ]

    finite_values = np.concatenate(
        [
            heatmap[np.isfinite(heatmap)]
            for heatmap in heatmaps
            if np.isfinite(heatmap).any()
        ]
    )

    if finite_values.size == 0:
        raise ValueError("No finite feature-importance values were available.")

    vmin = float(finite_values.min())
    vmax = float(finite_values.max())

    if np.isclose(vmin, vmax):
        padding = 1e-12 if vmax == 0 else 0.01 * abs(vmax)
        vmin -= padding
        vmax += padding

    figure, axes = get_figure_layout(region, len(panels))
    images = []

    for axis, panel, heatmap in zip(axes, panels, heatmaps):
        current_region, current_pedestal, layout = panel

        image = axis.imshow(
            np.ma.masked_invalid(heatmap),
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            interpolation="none",
            aspect="equal",
        )
        images.append(image)

        pedestal_label = (
            "Lateral"
            if current_pedestal == "lat"
            else "Medial"
        )
        axis.set_title(
            f"{participant} {current_region.title()} — "
            f"{pedestal_label}"
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlim(-0.5, layout.shape[1] - 0.5)
        axis.set_ylim(layout.shape[0] - 0.5, -0.5)

        for spine in axis.spines.values():
            spine.set_linewidth(1.5)

        if annotate:
            bank_offset = offsets[current_pedestal]

            for row in range(layout.shape[0]):
                for column in range(layout.shape[1]):
                    channel = layout[row, column]

                    if np.isnan(channel):
                        continue

                    absolute_channel = int(channel) + bank_offset
                    axis.text(
                        column,
                        row,
                        str(absolute_channel),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black",
                    )

    colorbar = figure.colorbar(
        images[0],
        ax=axes.tolist(),
        fraction=0.046,
        pad=0.04,
    )
    colorbar.set_label(
        "Feature importance",
        rotation=270,
        labelpad=20,
        fontsize=15,
    )
    colorbar.ax.tick_params(labelsize=15)

    return figure


def main():
    participant = PARTICIPANT.lower()
    mean_importance = load_mean_feature_importance(participant)

    figure = plot_utah_heatmap(
        mean_importance,
        participant=participant,
        region=REGION,
        pedestal=PEDESTAL,
        colormap=COLORMAP,
        annotate=ANNOTATE_CHANNELS,
    )
    figure.savefig(
        OUTPUT_FILE,
        bbox_inches="tight",
        dpi=OUTPUT_DPI,
    )

    print(f"Saved: {OUTPUT_FILE.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()