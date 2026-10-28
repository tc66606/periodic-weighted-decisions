"""Regenerate the two-panel resource comparison from released CSV records.

This is a display-only adaptation of the established resource plotting code.
No fitted quantities, parameter selection, or new simulations are performed.
Run from the scientific archive: python paper_figures/render_resource_frontier.py
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique(rows, **keys):
    matched = [row for row in rows if all(row[key] == value for key, value in keys.items())]
    if len(matched) != 1:
        raise ValueError(f"Expected one resource row for {keys}; found {len(matched)}")
    return matched[0]


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=here.parent)
    parser.add_argument("--output", type=Path, default=here)
    parser.add_argument("--qa-tools", type=Path,
                        help="Optional directory containing the panel alignment auditor")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_names = ("controls.csv", "resources.csv")
    tables = {}
    for name in source_names:
        with (args.data_root / name).open(encoding="utf-8", newline="") as stream:
            tables[name] = list(csv.DictReader(stream))
    chosen = [
        unique(tables["controls.csv"], tag="discovery_n68", response="nodes", route="standard_sign"),
        unique(tables["controls.csv"], tag="discovery_n68", response="nodes", route="sine_sign"),
        unique(tables["resources.csv"], tag="discovery_n68", response="nodes", route="randomized", S="4"),
        unique(tables["resources.csv"], tag="discovery_n68", response="nodes", route="padded", S="4"),
    ]
    records = []
    for index, row in enumerate(chosen):
        control = index < 2
        records.append({
            "source": source_names[0 if control else 1],
            "tag": row["tag"], "response": row["response"], "route": row["route"],
            "P": int(row["P"]), "K": int(row["K"]),
            "S": None if control else int(row["S"]),
            "T": int(row["T_count" if control else "T_worst"]),
            "maximum_consecutive_oracle_calls": int(row["maximum_sequential_label_calls" if control else "longest_chain"]),
            "allocated_logical_qubits": int(row["allocated_width" if control else "width"]),
        })
    # These guards tie the display to the previously certified comparison.
    assert [row["T"] for row in records] == [725638197, 483761943, 241883741, 241887769]
    assert [row["maximum_consecutive_oracle_calls"] for row in records] == [16383, 32767, 16383, 16383]
    assert [row["allocated_logical_qubits"] for row in records] == [586, 587, 586, 588]
    labels = ["Sign QPE\n3 runs", "Sine QPE\n1 run", "Rand. grid\n1 run", "Zero pad.\n1 run"]
    colors = ["#8993a0", "#48769b", "#008678", "#b87932"]
    values = [[row["T"] / 1_000_000 for row in records],
              [row["maximum_consecutive_oracle_calls"] for row in records]]
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.8,
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.30, 2.65))
    fig.subplots_adjust(left=.07, right=.99, bottom=.20, top=.81, wspace=.27)
    titles = ["Total T count (millions)", "Maximum consecutive oracle calls"]
    for index, (ax, vals, title) in enumerate(zip(axes, values, titles)):
        bars = ax.bar(range(4), vals, color=colors, width=.66)
        ax.set_xticks(range(4), labels, fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_title(title, fontsize=8.8, pad=10)
        ax.text(-.12, 1.11, chr(ord("a") + index), transform=ax.transAxes,
                fontweight="bold", fontsize=9, ha="left", va="center")
        ax.set_ylim(0, 880 if index == 0 else 40000)
        ax.yaxis.set_major_locator(MultipleLocator(200 if index == 0 else 10000))
        if index == 1:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:,.0f}"))
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, value + max(vals)*.025,
                    f"{value:.1f}" if index == 0 else f"{value:,}",
                    ha="center", va="bottom", fontsize=8)
    fig.canvas.draw()
    if args.qa_tools:
        sys.path.insert(0, str(args.qa_tools))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(
            fig, json_out=args.output / "resource_frontier.alignment.json",
            overlay_svg=args.output / "resource_frontier.alignment.svg",
            tolerance_pt=1.5, gutter_tolerance_pt=1.5,
            require_panel_labels=True, strict=True)
    fig.savefig(args.output / "resource_frontier.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(args.output / "resource_frontier.svg")
    fig.savefig(args.output / "resource_frontier.png", dpi=600)
    plt.close(fig)
    result = {
        "figure": "Figure 3", "source_sha256": {name: sha256(args.data_root/name) for name in source_names},
        "source_row_counts": {name: len(rows) for name, rows in tables.items()},
        "selection": "Principal n=68 cubic-response comparison; four routes, S=4 for randomized and padded routes",
        "records": records,
        "panels": ["Complete public T count in millions", "Maximum consecutive oracle calls"],
        "width_data": "Retained in manuscript Table 2 and in the records above; not plotted in Figure 3",
        "statistics": "Deterministic certified resource counts; no sampling intervals",
        "matplotlib_version": matplotlib.__version__,
    }
    (args.output / "figure_values.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
