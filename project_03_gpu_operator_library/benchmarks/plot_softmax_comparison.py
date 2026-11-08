"""Plot saved graph-replay medians; this does not execute GPU workloads."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator


PROJECT = Path(__file__).resolve().parents[1]
WIDTHS = (128, 512, 1024, 2048, 4096, 8192, 16384)
ROWS = (37, 256, 4096)
COLORS = {"pytorch": "#2563EB", "online": "#C45508", "online_registers": "#C45508"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("online", "online_registers"),
                        default="online", help="Custom variant to compare with PyTorch")
    parser.add_argument(
        "--input", type=Path,
        default=PROJECT / "results/2026-10-31-softmax-128k/combined_summary.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT / "notes/05-softmax-comparison/figures",
    )
    args = parser.parse_args()
    variants = ("pytorch", args.variant)
    retained = args.variant == "online_registers"
    custom_label = "Retained Online" if retained else "Online"
    description = "retained Online" if retained else "non-retained Online"
    file_label = "retained" if retained else "online"
    with args.input.open(newline="") as stream:
        records = list(csv.DictReader(stream))
    values = {}
    for record in records:
        row_count, width = int(record["rows"]), int(record["dims"])
        if (record["mode"] == "graph" and record["variant"] in variants
                and row_count in ROWS and width in WIDTHS):
            key = (row_count, width, record["variant"])
            assert key not in values, f"Duplicate measurement: {key}"
            expected_run = ("2026-10-31-softmax-128k" if width == 16384
                            else "2026-10-27-softmax-sweep")
            assert record["experiment"] == expected_run
            value = float(record["median_us"])
            assert value > 0
            values[key] = value
    expected = {(r, w, v) for r in ROWS for w in WIDTHS for v in variants}
    assert set(values) == expected, "Incomplete or unexpected shape coverage"
    args.output.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#CBD5E1", "axes.labelcolor": "#334155",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "svg.fonttype": "none",
    })
    for row_count in ROWS:
        fig, ax = plt.subplots(figsize=(11, 6.4), facecolor="white")
        fig.subplots_adjust(left=0.105, right=0.965, bottom=0.235, top=0.79)
        fig.text(0.105, 0.935, f"Softmax · {row_count:,} rows", fontsize=23,
                 fontweight="bold", color="#0F172A")
        fig.text(0.105, 0.88, f"PyTorch vs {description}  |  FP32  |  H100 80 GB",
                 fontsize=12, color="#475569")
        ax.set_xscale("linear")
        ax.set_yscale("linear")
        ax.set_xlim(0, 17500)
        # Match limits between the two variant plots at each row count, while
        # avoiding a 400-us scale compressing the small-row workloads.
        ax.set_ylim(0, {37: 25, 256: 40, 4096: 400}[row_count])
        ax.xaxis.set_major_locator(FixedLocator([0, 2048, 4096, 8192, 12288, 16384]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        ax.minorticks_off()
        ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("Columns per row — shape[1]", labelpad=12)
        ax.set_ylabel("Median time per call (µs)", labelpad=12)
        for variant, label, marker in [("pytorch", "PyTorch", "o"),
                                        (args.variant, custom_label, "s")]:
            ys = [values[row_count, width, variant] for width in WIDTHS]
            ax.plot(WIDTHS, ys, color=COLORS[variant], marker=marker,
                    linewidth=2.4, markersize=6, label=label)
            other = args.variant if variant == "pytorch" else "pytorch"
            for width, value in zip(WIDTHS, ys):
                # Small widths are close together on a genuinely numeric linear
                # axis. Keep their markers but omit overlapping value labels.
                if width < 2048:
                    continue
                above = value >= values[row_count, width, other]
                ax.annotate(f"{value:.2f}", (width, value),
                            xytext=(0, 9 if above else -11),
                            textcoords="offset points", ha="center",
                            va="bottom" if above else "top", fontsize=9,
                            color=COLORS[variant],
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.6))
        ax.legend(loc="upper left", frameon=False, ncol=2)
        fig.text(0.105, 0.105,
                 "CUDA Graph replay · median of nine 500-call batch averages · lower is better",
                 fontsize=10, color="#475569")
        fig.text(0.105, 0.058,
                 "128–8,192: earlier run; 16,384: extension run. Lines connect measured widths.",
                 fontsize=9, color="#64748B")
        stem = args.output / f"softmax-pytorch-{file_label}-rows-{row_count}"
        fig.savefig(stem.with_suffix(".png"), dpi=180)
        fig.savefig(stem.with_suffix(".svg"))
        plt.close(fig)
        print(f"Saved {stem.name}.png and .svg (14 verified measurements)")


if __name__ == "__main__":
    main()
