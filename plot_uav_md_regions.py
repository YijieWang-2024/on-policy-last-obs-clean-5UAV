"""Plot two MD source regions and five UAV coverage disks.

The default figure uses:

* a 1100 m x 1100 m square map;
* a lower-left MD region from (0, 0) to (220, 220);
* an upper-right MD region from (220, 220) to (1100, 1100);
* five UAVs with a 120 m coverage radius.

Run with the deterministic random layout:

    python plot_uav_md_regions.py

Pass fixed UAV coordinates from the command line:

    python plot_uav_md_regions.py --positions \
        "100,100;300,180;500,500;800,800;1000,1000"

You can also set UAV_POSITIONS below to a list of five (x, y) tuples.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle
from matplotlib.colors import to_rgba


# ---------------------------------------------------------------------------
# Configuration: edit these values when preparing a fixed scenario.
# ---------------------------------------------------------------------------
MAP_SIZE = 700.0
MD_REGION_SIZE = 220.0
COVERAGE_RADIUS = 120.0
DEFAULT_SEED = 20260806

# Keep None for reproducible random positions. To fix the positions, replace
# None with exactly five tuples, for example:
# UAV_POSITIONS = [
#     (100.0, 100.0),
#     (300.0, 180.0),
#     (500.0, 500.0),
#     (800.0, 800.0),
#     (1000.0, 1000.0),
# ]
UAV_POSITIONS: list[tuple[float, float]] | None = None

DEFAULT_OUTPUT = Path(__file__).with_name("uav_md_two_regions_layout.png")

UAV_COLORS = (
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#B07AA1",
)


def parse_positions(value: str) -> list[tuple[float, float]]:
    """Parse five positions in the form x,y;x,y;..."""
    positions: list[tuple[float, float]] = []
    try:
        for item in value.split(";"):
            x_text, y_text = item.split(",")
            positions.append((float(x_text), float(y_text)))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "positions must look like 'x1,y1;x2,y2;...'"
        ) from exc

    if len(positions) != 5:
        raise argparse.ArgumentTypeError(
            f"exactly five UAV positions are required, got {len(positions)}"
        )
    return positions


def resolve_uav_positions(
    map_size: float,
    seed: int,
    positions: list[tuple[float, float]] | None,
    margin: float = 0.0,
) -> list[tuple[float, float]]:
    """Return fixed positions or reproducible random positions."""
    if positions is None:
        rng = np.random.default_rng(seed)
        margin = min(max(margin, 0.0), map_size / 2.0)
        sampled = rng.uniform(margin, map_size - margin, size=(5, 2))
        positions = [(float(x), float(y)) for x, y in sampled]

    if len(positions) != 5:
        raise ValueError("exactly five UAV positions are required")

    for index, (x, y) in enumerate(positions, start=1):
        if not (0.0 <= x <= map_size and 0.0 <= y <= map_size):
            raise ValueError(
                f"U{index} position ({x}, {y}) is outside the map "
                f"[0, {map_size}] x [0, {map_size}]"
            )
    return [(float(x), float(y)) for x, y in positions]


def plot_layout(
    map_size: float = MAP_SIZE,
    md_region_size: float = MD_REGION_SIZE,
    coverage_radius: float = COVERAGE_RADIUS,
    uav_positions: list[tuple[float, float]] | None = UAV_POSITIONS,
    seed: int = DEFAULT_SEED,
    output_path: Path = DEFAULT_OUTPUT,
    show: bool = False,
) -> Path:
    """Create and save the MD-region/UAV coverage figure."""
    if map_size <= 0:
        raise ValueError("map_size must be positive")
    if not 0 < md_region_size < map_size:
        raise ValueError("md_region_size must be between 0 and map_size")
    if coverage_radius <= 0:
        raise ValueError("coverage_radius must be positive")

    positions = resolve_uav_positions(
        map_size,
        seed,
        uav_positions,
        margin=coverage_radius,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)

    # Two disjoint square regions where MDs are generated.
    small_region = Rectangle(
        (0.0, 0.0),
        md_region_size,
        md_region_size,
        facecolor="#F6BD8B",
        edgecolor="#C66A1B",
        linewidth=2.0,
        alpha=0.45,
        zorder=1,
    )
    large_region = Rectangle(
        (md_region_size, md_region_size),
        map_size - md_region_size,
        map_size - md_region_size,
        facecolor="#9CC5E8",
        edgecolor="#2F6FA3",
        linewidth=2.0,
        alpha=0.30,
        zorder=1,
    )
    ax.add_patch(small_region)
    ax.add_patch(large_region)

    # Outer map boundary and the two region corner coordinates.
    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            map_size,
            map_size,
            fill=False,
            edgecolor="#333333",
            linewidth=2.2,
            zorder=4,
        )
    )

    # Coverage disks are drawn before UAV markers so the markers remain clear.
    for index, ((x, y), color) in enumerate(
        zip(positions, UAV_COLORS),
        start=1,
    ):
        ax.add_patch(
            Circle(
                (x, y),
                coverage_radius,
                facecolor=to_rgba(color, 0.16),
                edgecolor=to_rgba(color, 0.88),
                linewidth=1.8,
                zorder=2,
                clip_on=True,
            )
        )
        ax.scatter(
            [x],
            [y],
            s=95,
            color=color,
            edgecolor="white",
            linewidth=1.2,
            zorder=5,
        )
        ax.annotate(
            f"U{index} ({x:.0f}, {y:.0f})",
            (x, y),
            xytext=(7, 8),
            textcoords="offset points",
            fontsize=9,
            color="#222222",
            zorder=6,
        )

    # Region labels make the intended MD-generation layout explicit.
    ax.text(
        md_region_size / 2,
        md_region_size / 2,
        f"MD region A\n0–{md_region_size:g} m\n(small MD region)",
        ha="center",
        va="center",
        fontsize=10,
        color="#7A3F0A",
        zorder=3,
    )
    large_center = md_region_size + (map_size - md_region_size) / 2
    ax.text(
        large_center,
        large_center,
        f"MD region B\n({md_region_size:g}, {md_region_size:g})–"
        f"({map_size:g}, {map_size:g})\n(large MD region)",
        ha="center",
        va="center",
        fontsize=11,
        color="#17476B",
        zorder=3,
    )

    region_handles = [
        Patch(
            facecolor=small_region.get_facecolor(),
            edgecolor=small_region.get_edgecolor(),
            label=f"MD region A: {md_region_size:g} x {md_region_size:g} m",
        ),
        Patch(
            facecolor=large_region.get_facecolor(),
            edgecolor=large_region.get_edgecolor(),
            label=(
                f"MD region B: {map_size - md_region_size:g} x "
                f"{map_size - md_region_size:g} m"
            ),
        ),
    ]
    coverage_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=8,
            label=f"U{index} + {coverage_radius:g} m coverage",
        )
        for index, color in enumerate(UAV_COLORS, start=1)
    ]
    ax.legend(
        handles=region_handles + coverage_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=8.5,
        framealpha=0.92,
    )

    tick_step = 100 if map_size <= 1200 else 200
    ticks = np.arange(0.0, map_size + tick_step, tick_step)
    ticks = ticks[ticks <= map_size]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(0.0, map_size)
    ax.set_ylim(0.0, map_size)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"Two MD source regions and five UAV coverage disks\n"
        f"map={map_size:g} x {map_size:g} m, coverage radius={coverage_radius:g} m"
    )
    ax.grid(True, linewidth=0.6, alpha=0.28, zorder=0)

    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    # if show:
    plt.show()
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-size", type=float, default=MAP_SIZE)
    parser.add_argument("--md-region-size", type=float, default=MD_REGION_SIZE)
    parser.add_argument("--coverage-radius", type=float, default=COVERAGE_RADIUS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--positions",
        type=parse_positions,
        default=UAV_POSITIONS,
        help="five fixed positions: x1,y1;x2,y2;...;x5,y5",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    output_path = plot_layout(
        map_size=args.map_size,
        md_region_size=args.md_region_size,
        coverage_radius=args.coverage_radius,
        uav_positions=args.positions,
        seed=args.seed,
        output_path=args.output,
        show=args.show,
    )
    print(f"Saved: {output_path.resolve()}")
    for index, (x, y) in enumerate(
        resolve_uav_positions(
            args.map_size,
            args.seed,
            args.positions,
            margin=args.coverage_radius,
        ),
        start=1,
    ):
        print(f"U{index}: ({x:.3f}, {y:.3f})")


if __name__ == "__main__":
    main()
