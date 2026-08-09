"""Plot the largest square fully covered by four equal UAV disks.

For four UAVs with coverage radius R, the optimal symmetric arrangement is a
2 x 2 grid. Each UAV is placed at the center of one quarter-square, so the
largest fully covered square has side length 2 * sqrt(2) * R.

Run:

    python plot_four_uav_square_coverage.py
    python plot_four_uav_square_coverage.py --radius 120 --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle
from matplotlib.colors import to_rgba


DEFAULT_RADIUS = 120.0
DEFAULT_OUTPUT = Path(__file__).with_name("four_uav_max_square_coverage.png")

UAV_COLORS = (
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
)


def max_square_side(radius: float) -> float:
    """Return the largest side length fully covered by four radius-R disks."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    return 2.0 * np.sqrt(2.0) * radius


def max_square_uav_positions(radius: float) -> list[tuple[float, float]]:
    """Return the four 2 x 2 grid-center positions for the maximum square."""
    side = max_square_side(radius)
    quarter = side / 4.0
    return [
        (quarter, quarter),
        (3.0 * quarter, quarter),
        (quarter, 3.0 * quarter),
        (3.0 * quarter, 3.0 * quarter),
    ]


def plot_layout(
    radius: float = DEFAULT_RADIUS,
    output_path: Path = DEFAULT_OUTPUT,
    show: bool = False,
) -> Path:
    """Create and save the four-UAV square coverage schematic."""
    side = max_square_side(radius)
    positions = max_square_uav_positions(radius)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)

    # Draw the four quarter-squares first. Their union is the target square.
    quarter_side = side / 2.0
    for index, ((x, y), color) in enumerate(
        zip(positions, UAV_COLORS),
        start=1,
    ):
        cell_x = 0.0 if x < side / 2.0 else quarter_side
        cell_y = 0.0 if y < side / 2.0 else quarter_side
        ax.add_patch(
            Rectangle(
                (cell_x, cell_y),
                quarter_side,
                quarter_side,
                facecolor=to_rgba(color, 0.08),
                edgecolor=to_rgba(color, 0.40),
                linewidth=1.0,
                zorder=0,
            )
        )

    # Coverage disks. The axes limits clip the parts outside the target square.
    for index, ((x, y), color) in enumerate(
        zip(positions, UAV_COLORS),
        start=1,
    ):
        ax.add_patch(
            Circle(
                (x, y),
                radius,
                facecolor=to_rgba(color, 0.18),
                edgecolor=to_rgba(color, 0.90),
                linewidth=2.0,
                clip_on=True,
                zorder=2,
            )
        )
        ax.scatter(
            [x],
            [y],
            s=105,
            color=color,
            edgecolor="white",
            linewidth=1.4,
            zorder=5,
        )
        ax.annotate(
            f"U{index} ({x:.1f}, {y:.1f})",
            (x, y),
            xytext=(7, 8),
            textcoords="offset points",
            fontsize=9,
            color="#222222",
            zorder=6,
        )

    # The 2 x 2 construction is shown explicitly.
    ax.axvline(
        side / 2.0,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        alpha=0.72,
        zorder=4,
    )
    ax.axhline(
        side / 2.0,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        alpha=0.72,
        zorder=4,
    )

    # All nine 2 x 2 grid vertices are at distance <= R from at least one UAV.
    grid_points = np.array(
        [
            (x, y)
            for x in (0.0, side / 2.0, side)
            for y in (0.0, side / 2.0, side)
        ]
    )
    ax.scatter(
        grid_points[:, 0],
        grid_points[:, 1],
        s=20,
        color="#303030",
        zorder=7,
        label="2 x 2 cell vertices",
    )
    ax.scatter(
        [side / 2.0],
        [side / 2.0],
        s=58,
        color="#303030",
        edgecolor="white",
        linewidth=0.9,
        zorder=8,
    )
    ax.annotate(
        "four coverage boundaries meet\n(farthest-point distance = 120 m)",
        (side / 2.0, side / 2.0),
        xytext=(10, -42),
        textcoords="offset points",
        fontsize=9,
        color="#303030",
        ha="left",
        va="top",
        arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.9},
        zorder=9,
    )

    # Target-square outline is drawn last so the boundary remains visible.
    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            side,
            side,
            fill=False,
            edgecolor="#202020",
            linewidth=2.4,
            zorder=10,
        )
    )

    region_handles = [
        Patch(
            facecolor=to_rgba(color, 0.18),
            edgecolor=to_rgba(color, 0.90),
            label=f"U{index}: R = {radius:g} m",
        )
        for index, color in enumerate(UAV_COLORS, start=1)
    ]
    construction_handles = [
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle="--",
            linewidth=1.2,
            label="2 x 2 construction boundary",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color="#303030",
            markersize=5,
            label="cell vertices",
        ),
    ]
    ax.legend(
        handles=region_handles + construction_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=8.5,
        framealpha=0.94,
    )

    tick_step = 50 if side <= 400 else 100
    ticks = np.arange(0.0, side + tick_step, tick_step)
    ticks = ticks[ticks <= side]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(0.0, side)
    ax.set_ylim(0.0, side)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        "Maximum square fully covered by four UAVs\n"
        f"coverage radius R = {radius:g} m, side length L = {side:.3f} m"
    )
    ax.grid(True, linewidth=0.6, alpha=0.28, zorder=0)

    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    side = max_square_side(args.radius)
    output_path = plot_layout(
        radius=args.radius,
        output_path=args.output,
        show=args.show,
    )
    print(f"Maximum square side: {side:.6f} m")
    print(f"Saved: {output_path.resolve()}")
    for index, (x, y) in enumerate(
        max_square_uav_positions(args.radius),
        start=1,
    ):
        print(f"U{index}: ({x:.6f}, {y:.6f})")


if __name__ == "__main__":
    main()
