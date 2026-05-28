"""Plot holdout traffic and pod comparison for a selected model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "prophet"
DEFAULT_DAYS = 30
MODEL_NAMES = ("prophet", "sarima", "gru", "lstm", "prophet_tuned")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for holdout comparison plotting."""
    parser = argparse.ArgumentParser(
        description="Create a holdout comparison plot for actual/predicted traffic and pods."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name used in prediction file names. Use 'all' to plot every candidate model.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help="Number of days to plot from the holdout prediction period.",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Optional plot start timestamp. Defaults to the highest-traffic window.",
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="Optional prediction CSV path. Only valid for a single model.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output image path. Only valid for a single model.",
    )
    args = parser.parse_args()

    if args.days <= 0:
        parser.error("--days must be a positive integer.")
    if args.model != "all" and args.model not in MODEL_NAMES:
        parser.error(f"--model must be one of {', '.join(MODEL_NAMES)} or all.")
    if args.model == "all" and args.predictions:
        parser.error("--predictions cannot be used with --model all.")
    if args.model == "all" and args.output:
        parser.error("--output cannot be used with --model all.")

    return args


def load_predictions(path: Path) -> pd.DataFrame:
    """Load and validate prediction rows used by the comparison plot."""
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    df = pd.read_csv(path, parse_dates=["ds"])
    required_columns = {"ds", "actual", "predicted", "actual_pods", "predicted_pods"}
    missing = required_columns.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Prediction file is missing required columns: {missing_text}")
    if df.empty:
        raise ValueError("Prediction file is empty.")

    return df.sort_values("ds").reset_index(drop=True)


def select_window(df: pd.DataFrame, start: str | None, days: int) -> pd.DataFrame:
    """Select a fixed holdout window from prediction rows."""
    start_at = pd.Timestamp(start) if start else find_high_traffic_window_start(df, days)
    end_at = start_at + pd.Timedelta(days=days)
    window = df[(df["ds"] >= start_at) & (df["ds"] < end_at)].copy()
    if window.empty:
        raise ValueError(f"No prediction rows found between {start_at} and {end_at}.")
    return window.reset_index(drop=True)


def find_high_traffic_window_start(df: pd.DataFrame, days: int) -> pd.Timestamp:
    """Find the start timestamp for the highest average actual-traffic window."""
    if len(df) < 2:
        return df["ds"].iloc[0]

    median_step = df["ds"].diff().dropna().median()
    if pd.isna(median_step) or median_step <= pd.Timedelta(0):
        raise ValueError("Prediction timestamps must be sorted and have a positive interval.")

    rows_per_day = max(1, round(pd.Timedelta(days=1) / median_step))
    window_size = min(len(df), rows_per_day * days)
    rolling_mean = df["actual"].rolling(window=window_size, min_periods=window_size).mean()
    end_index = int(rolling_mean.idxmax())
    start_index = max(0, end_index - window_size + 1)
    return df["ds"].iloc[start_index]


def shade_pod_mismatch(ax: plt.Axes, window: pd.DataFrame) -> None:
    """Shade under- and over-provisioned intervals on a pod comparison axis."""
    if len(window) > 1:
        default_width = window["ds"].iloc[1] - window["ds"].iloc[0]
    else:
        default_width = pd.Timedelta(hours=1)

    for index, row in window.iterrows():
        start_at = row["ds"]
        end_at = window["ds"].iloc[index + 1] if index + 1 < len(window) else start_at + default_width
        actual_pods = row["actual_pods"]
        predicted_pods = row["predicted_pods"]

        if predicted_pods < actual_pods:
            ax.axvspan(start_at, end_at, color="#ef4444", alpha=0.12, linewidth=0)
        elif predicted_pods > actual_pods:
            ax.axvspan(start_at, end_at, color="#3b82f6", alpha=0.10, linewidth=0)


def plot_holdout_comparison(window: pd.DataFrame, model: str, output_path: Path) -> None:
    """Render and save traffic and pod holdout comparison charts."""
    fig, (traffic_ax, pod_ax) = plt.subplots(
        2,
        1,
        figsize=(13, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    traffic_ax.plot(window["ds"], window["actual"], label="Actual traffic", color="#111827", linewidth=1.8)
    traffic_ax.plot(window["ds"], window["predicted"], label="Predicted traffic", color="#2563eb", linewidth=1.6)
    start_label = window["ds"].iloc[0].strftime("%Y-%m-%d")
    end_label = window["ds"].iloc[-1].strftime("%Y-%m-%d")
    traffic_ax.set_title(f"{model.capitalize()} high-traffic holdout prediction ({start_label} to {end_label})")
    traffic_ax.set_ylabel("Request rate")
    traffic_ax.grid(alpha=0.25)
    traffic_ax.legend(loc="upper right")

    shade_pod_mismatch(pod_ax, window)
    pod_ax.step(window["ds"], window["actual_pods"], where="post", label="Actual pods", color="#111827")
    pod_ax.step(window["ds"], window["predicted_pods"], where="post", label="Predicted pods", color="#2563eb")
    pod_ax.set_title("Pod decision comparison")
    pod_ax.set_ylabel("Pods")
    pod_ax.set_xlabel("Holdout timestamp")
    pod_ax.grid(alpha=0.25)
    pod_ax.legend(
        handles=[
            *pod_ax.get_legend_handles_labels()[0],
            Patch(color="#ef4444", alpha=0.12, label="Under-provisioning"),
            Patch(color="#3b82f6", alpha=0.10, label="Over-provisioning"),
        ],
        loc="upper right",
    )

    fig.autofmt_xdate()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_prediction_path(model: str) -> Path:
    """Build the default prediction CSV path for a model."""
    return PROJECT_ROOT / "data" / "predictions" / f"{model}_predictions.csv"


def build_output_path(model: str) -> Path:
    """Build the default documentation image path for a model."""
    return PROJECT_ROOT / "docs" / "assets" / f"{model}_holdout_comparison.png"


def plot_model(model: str, start: pd.Timestamp | str | None, days: int, predictions: str | None, output: str | None) -> None:
    """Create a holdout comparison plot for one model."""
    prediction_path = Path(predictions) if predictions else build_prediction_path(model)
    output_path = Path(output) if output else build_output_path(model)

    df = load_predictions(prediction_path)
    window = select_window(df, str(start) if start is not None else None, days)
    plot_holdout_comparison(window, model, output_path)
    print(f"{model}: selected window {window['ds'].iloc[0]} to {window['ds'].iloc[-1]}")
    print(f"{model}: saved holdout comparison plot to {output_path}")


def main() -> None:
    """Run holdout comparison plotting from the command line."""
    args = parse_args()

    if args.model == "all":
        base_df = load_predictions(build_prediction_path(DEFAULT_MODEL))
        start = pd.Timestamp(args.start) if args.start else find_high_traffic_window_start(base_df, args.days)
        for model in MODEL_NAMES:
            plot_model(model, start, args.days, predictions=None, output=None)
        return

    plot_model(args.model, args.start, args.days, args.predictions, args.output)


if __name__ == "__main__":
    main()
