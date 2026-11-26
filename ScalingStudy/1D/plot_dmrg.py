"""Plot the DMRG bond-dimension baseline from a coverage report."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, ScalarFormatter


PHASES = (("2", "Superfluid", "#0072B2"), ("3.275", "BKT", "#D55E00"), ("8", "Mott insulator", "#009E73"))
SIZES = (16, 24, 32, 48, 64)


def plot(report, output):
    figure, axis = plt.subplots(figsize=(6.2, 4.2), layout="constrained")
    phase_handles = []
    for coupling, label, color in PHASES:
        phase_handles.append(Line2D([], [], color=color, marker="o", linestyle="none", label=label))
        for length in SIZES:
            row = report.get(f"L{length}-U{coupling}-n6", {})
            if row.get("status") != "crossing":
                continue
            bound = not row.get("largest_failing_cap_below_crossing") or row.get("smaller_unconverged_caps")
            axis.scatter(length, row["chi_to_target"], s=48, linewidth=1.5,
                         facecolors="none" if bound else color, edgecolors=color, zorder=3)
    guide_lengths = [16, 64]
    axis.plot(guide_lengths, [96 * (length / 16)**0.744 for length in guide_lengths],
              linestyle="--", color="#6C7076", lw=1.3, label=r"$L^{0.744}$ guide")
    handles = phase_handles + axis.get_legend_handles_labels()[0]
    handles.append(Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor="#303336",
                          linestyle="none", label="Upper bound only"))
    axis.legend(handles=handles, frameon=False, loc="upper left", fontsize=9)
    axis.set(xscale="log", yscale="log", xlabel="Ring length $L$",
             ylabel=r"Smallest passing achieved bond dimension $\chi$")
    axis.xaxis.set_major_locator(FixedLocator(SIZES))
    axis.xaxis.set_major_formatter(ScalarFormatter())
    axis.yaxis.set_major_locator(FixedLocator([8, 16, 32, 64, 128, 256, 512, 1024]))
    axis.yaxis.set_major_formatter(ScalarFormatter())
    axis.set_xlim(14, 73)
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(output)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(json.loads(args.report.read_text()), args.output)


if __name__ == "__main__":
    main()
