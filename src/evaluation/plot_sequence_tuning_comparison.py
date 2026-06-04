"""Plot baseline vs tuned sequence model metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
ASSETS_DIR = PROJECT_ROOT / "docs" / "assets"
MODEL_PAIRS = {
    "GRU": ("gru", "gru_tuned"),
    "LSTM": ("lstm", "lstm_tuned"),
}
METRICS = [
    ("smape", "SMAPE", "lower is better"),
    ("pod_accuracy", "Pod accuracy", "higher is better"),
    ("under_provisioning_rate", "Under-provisioning rate", "lower is better"),
    ("over_provisioning_rate", "Over-provisioning rate", "lower is better"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create baseline vs tuned sequence model metric plot.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=ASSETS_DIR / "sequence_tuning_metric_comparison.png")
    return parser.parse_args()


def load_metric(results_dir: Path, model: str) -> dict[str, float]:
    path = results_dir / f"{model}_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Metric file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return {name: float(data[name]) for name, _, _ in METRICS}


def build_metric_frame(results_dir: Path) -> pd.DataFrame:
    rows = []
    for family, (baseline, tuned) in MODEL_PAIRS.items():
        for label, model in [("Baseline", baseline), ("Tuned", tuned)]:
            metrics = load_metric(results_dir, model)
            rows.append({"family": family, "label": label, "model": model, **metrics})
    return pd.DataFrame(rows)


def plot_metric_frame(frame: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    colors = {"Baseline": "#2563eb", "Tuned": "#f97316"}

    for axis, (metric, title, direction) in zip(axes, METRICS):
        pivot = frame.pivot(index="family", columns="label", values=metric).loc[["GRU", "LSTM"]]
        pivot[["Baseline", "Tuned"]].plot(
            kind="bar",
            ax=axis,
            color=[colors["Baseline"], colors["Tuned"]],
            width=0.72,
            legend=False,
        )
        axis.set_title(f"{title} ({direction})")
        axis.set_xlabel("")
        axis.set_ylabel(metric)
        axis.tick_params(axis="x", rotation=0)
        axis.grid(axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Sequence Model Tuning: Baseline vs Tuned", fontsize=14, y=0.99)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frame = build_metric_frame(args.results_dir)
    plot_metric_frame(frame, args.output)
    print(f"Saved sequence tuning metric comparison plot to {args.output}")


if __name__ == "__main__":
    main()
