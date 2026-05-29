"""Plot one-year holdout overview for all candidate models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAMES = ("prophet", "sarima", "gru", "lstm", "prophet_tuned", "openevolve_prophet")
MODEL_COLORS = {
    "prophet": "#2563eb",
    "sarima": "#9333ea",
    "gru": "#059669",
    "lstm": "#dc2626",
    "prophet_tuned": "#f97316",
    "openevolve_prophet": "#0891b2",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for holdout overview plotting."""
    parser = argparse.ArgumentParser(description="Create a full-holdout overview plot for all candidate models.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "docs" / "assets" / "holdout_year_overview.png"),
        help="Output image path.",
    )
    return parser.parse_args()


def prediction_path(model: str) -> Path:
    """Build the prediction CSV path for a model."""
    return PROJECT_ROOT / "data" / "predictions" / f"{model}_predictions.csv"


def load_model_predictions(model: str) -> pd.DataFrame:
    """Load prediction rows for a model."""
    path = prediction_path(model)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    df = pd.read_csv(path, parse_dates=["ds"])
    required_columns = {"ds", "actual", "predicted", "actual_pods", "predicted_pods"}
    missing = required_columns.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{model} prediction file is missing required columns: {missing_text}")
    if df.empty:
        raise ValueError(f"{model} prediction file is empty.")

    return df.sort_values("ds").reset_index(drop=True)


def build_daily_overview() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build daily mean traffic and pod data across all models."""
    traffic_parts: list[pd.DataFrame] = []
    pod_parts: list[pd.DataFrame] = []
    actual_added = False

    for model in MODEL_NAMES:
        df = load_model_predictions(model).set_index("ds")
        if not actual_added:
            traffic_parts.append(df[["actual"]])
            pod_parts.append(df[["actual_pods"]])
            actual_added = True

        traffic_parts.append(df[["predicted"]].rename(columns={"predicted": model}))
        pod_parts.append(df[["predicted_pods"]].rename(columns={"predicted_pods": model}))

    traffic = pd.concat(traffic_parts, axis=1).resample("1D").mean()
    pods = pd.concat(pod_parts, axis=1).resample("1D").mean()
    return traffic, pods


def plot_overview(traffic: pd.DataFrame, pods: pd.DataFrame, output_path: Path) -> None:
    """Render and save full-holdout traffic and pod overview charts."""
    fig, (traffic_ax, pod_ax) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    traffic_ax.plot(traffic.index, traffic["actual"], label="Actual traffic", color="#111827", linewidth=2.2)
    for model in MODEL_NAMES:
        traffic_ax.plot(
            traffic.index,
            traffic[model],
            label=f"{model.upper()} predicted",
            color=MODEL_COLORS[model],
            linewidth=1.4,
            alpha=0.85,
        )

    traffic_ax.set_title("Full holdout traffic overview")
    traffic_ax.set_ylabel("Daily mean request rate")
    traffic_ax.grid(alpha=0.25)
    traffic_ax.legend(ncol=3, loc="upper right")

    pod_ax.plot(pods.index, pods["actual_pods"], label="Actual pods", color="#111827", linewidth=2.0)
    for model in MODEL_NAMES:
        pod_ax.plot(
            pods.index,
            pods[model],
            label=f"{model.upper()} predicted pods",
            color=MODEL_COLORS[model],
            linewidth=1.3,
            alpha=0.85,
        )

    pod_ax.set_title("Full holdout pod decision overview")
    pod_ax.set_ylabel("Daily mean pods")
    pod_ax.set_xlabel("Holdout date")
    pod_ax.grid(alpha=0.25)
    pod_ax.legend(ncol=3, loc="upper right")

    fig.autofmt_xdate()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    """Run one-year holdout overview plotting from the command line."""
    args = parse_args()
    output_path = Path(args.output)
    traffic, pods = build_daily_overview()
    plot_overview(traffic, pods, output_path)
    print(f"Saved holdout overview plot to {output_path}")


if __name__ == "__main__":
    main()
