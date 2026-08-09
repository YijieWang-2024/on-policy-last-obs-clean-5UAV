"""Plot the 700 m x 700 m UAV layout and communication ranges.

Edit the constants in the CONFIGURATION section, then run:

    python plot_uav_communication_layout.py

The PNG is written next to this script unless ``--output`` is provided.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# ---------------------------------------------------------------------------
# CONFIGURATION: change these values to explore another layout.
# ---------------------------------------------------------------------------
MAP_SIZE = 700.0
LOWER_UAV_POSITIONS = [
    (110.0, 220.0),
    (220.0, 220.0),
    # (200.0, 140.0),
    (330.0, 220.0),
    (440.0, 220.0),
    # (400.0, 140.0),
]
UPPER_UAV_POSITION = (600.0, 225.0)
UPPER_UAV_POSITION = (400.0, 500.0)
UPPER_UAV_POSITION = (450.0, 550.0)
UPPER_UAV_POSITION = (400.0, 450.0)
COMMUNICATION_RADII = (260.0, 520.0, 780.0)


def visible_uavs(center: tuple[float, float], radius: float,
                 positions: list[tuple[float, float]]) -> list[int]:
    """Return one-based UAV indices visible from ``center``."""
    cx, cy = center
    return [
        index + 1
        for index, (x, y) in enumerate(positions)
        if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    ]


def plot_layout(
    map_size: float = MAP_SIZE,
    lower_positions: list[tuple[float, float]] | None = None,
    upper_position: tuple[float, float] = UPPER_UAV_POSITION,
    communication_radii: tuple[float, ...] = COMMUNICATION_RADII,
    output_path: Path | None = None,
    show: bool = False,
) -> Path:
    """Create and save the layout figure, returning the output path."""
    if lower_positions is None:
        lower_positions = LOWER_UAV_POSITIONS
    if output_path is None:
        output_path = Path(__file__).with_name("uav_communication_layout.png")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("UAV initial positions and communication ranges")
    ax.set_xticks(range(0, int(map_size) + 1, 100))
    ax.set_yticks(range(0, int(map_size) + 1, 100))
    ax.grid(True, linewidth=0.6, alpha=0.35)

    # The largest circle is intentionally clipped by the 700 m map boundary.
    range_colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    for radius, color in zip(communication_radii, range_colors):
        circle = Circle(
            upper_position,
            radius,
            fill=False,
            linestyle="--",
            linewidth=1.8,
            color=color,
            label=f"R = {radius:g} m",
            clip_on=True,
        )
        ax.add_patch(circle)

    lower_x, lower_y = zip(*lower_positions)
    ax.scatter(
        lower_x,
        lower_y,
        s=70,
        color="tab:blue",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label="lower UAVs",
    )
    ax.scatter(
        [upper_position[0]],
        [upper_position[1]],
        s=90,
        color="tab:red",
        edgecolor="white",
        linewidth=0.8,
        zorder=6,
        label="upper UAV",
    )

    for index, (x, y) in enumerate(lower_positions, start=1):
        ax.annotate(
            f"U{index} ({x:g},{y:g})",
            (x, y),
            xytext=(5, 7),
            textcoords="offset points",
            fontsize=9,
        )
    ax.annotate(
        f"U{len(lower_positions) + 1} ({upper_position[0]:g},{upper_position[1]:g})",
        upper_position,
        xytext=(7, 8),
        textcoords="offset points",
        fontsize=9,
    )

    # Put the exact membership at each radius in the figure instead of relying
    # only on visual inspection.
    membership_lines = [
        f"R={radius:g} m: lower UAVs "
        f"{visible_uavs(upper_position, radius, lower_positions)}"
        for radius in communication_radii
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(membership_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.82},
    )
    ax.legend(loc="lower right", fontsize=9)

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("uav_communication_layout.png"),
        help="PNG output path (default: next to this script)",
    )
    parser.add_argument(
        "--show",
        action="store_false",
        help="Also open an interactive Matplotlib window",
    )
    args = parser.parse_args()
    output_path = plot_layout(output_path=args.output, show=args.show)
    print(f"Saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
