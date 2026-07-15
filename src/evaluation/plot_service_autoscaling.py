"""Plot service-level autoscaling experiment outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "service_autoscaling" / "onprem_bnpl" / "results"
ASSETS_DIR = PROJECT_ROOT / "docs" / "assets"
SERVICE_ORDER = ["payment", "auth", "limit_scoring", "batch", "admin"]
SERVICE_LABELS = {
    "auth": "Auth",
    "payment": "Payment",
    "batch": "Batch",
    "admin": "Admin",
    "limit_scoring": "Limit Scoring",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create service autoscaling plots.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=ASSETS_DIR)
    parser.add_argument("--days", type=int, default=30)
    return parser.parse_args()


def _load_csv(path: Path, date_columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required result file not found: {path}")
    return pd.read_csv(path, parse_dates=date_columns or [])


def _service_color(service: str) -> str:
    return {
        "payment": "#dc2626",
        "auth": "#2563eb",
        "limit_scoring": "#7c3aed",
        "batch": "#f97316",
        "admin": "#059669",
    }[service]


def plot_service_traffic_overview(predictions: pd.DataFrame, output_path: Path) -> None:
    daily = (
        predictions.set_index("ds")
        .groupby("service")[["actual", "predicted"]]
        .resample("1D")
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(len(SERVICE_ORDER), 1, figsize=(13, 12), sharex=True)
    for ax, service in zip(axes, SERVICE_ORDER):
        rows = daily[daily["service"] == service]
        ax.plot(rows["ds"], rows["actual"], color="#111827", linewidth=1.5, label="Actual")
        ax.plot(rows["ds"], rows["predicted"], color=_service_color(service), linewidth=1.3, label="Predicted")
        ax.set_ylabel(SERVICE_LABELS[service])
        ax.grid(alpha=0.25)
    axes[0].set_title("Service-level GRU traffic prediction overview")
    axes[-1].set_xlabel("Holdout date")
    axes[0].legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_service_metric_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    metrics = metrics.set_index("service").loc[SERVICE_ORDER].reset_index()
    x = range(len(metrics))
    width = 0.22

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar([value - width for value in x], metrics["under_provisioning_rate"], width, label="Under")
    ax.bar(x, metrics["over_provisioning_rate"], width, label="Over")
    ax.bar([value + width for value in x], metrics["pod_accuracy"], width, label="Pod accuracy")
    ax.set_xticks(list(x), [SERVICE_LABELS[service] for service in metrics["service"]])
    ax.set_ylim(0, 1)
    ax.set_title("Service autoscaling metric comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_onprem_adjustment(onprem: pd.DataFrame, output_path: Path, days: int) -> None:
    start_at = onprem["ds"].min()
    window = onprem[onprem["ds"] < start_at + pd.Timedelta(days=days)].copy()
    daily = (
        window.set_index("ds")
        .groupby("service")[["predicted_pods", "onprem_adjusted_pods"]]
        .resample("1D")
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(13, 6))
    bottom_pred = None
    bottom_adjusted = None
    dates = sorted(daily["ds"].unique())
    pred_matrix = []
    adjusted_matrix = []
    for service in SERVICE_ORDER:
        rows = daily[daily["service"] == service].set_index("ds").reindex(dates).fillna(0)
        pred_matrix.append(rows["predicted_pods"].to_numpy())
        adjusted_matrix.append(rows["onprem_adjusted_pods"].to_numpy())

    for service, values in zip(SERVICE_ORDER, pred_matrix):
        ax.bar(dates, values, bottom=bottom_pred, color=_service_color(service), alpha=0.25, width=0.8)
        bottom_pred = values if bottom_pred is None else bottom_pred + values
    for service, values in zip(SERVICE_ORDER, adjusted_matrix):
        ax.plot(dates, values if bottom_adjusted is None else bottom_adjusted + values, color=_service_color(service), linewidth=1.3)
        bottom_adjusted = values if bottom_adjusted is None else bottom_adjusted + values

    ax.set_title("On-prem pod budget adjustment")
    ax.set_ylabel("Daily mean pods")
    ax.set_xlabel("Holdout date")
    ax.grid(axis="y", alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.days <= 0:
        raise ValueError("--days must be a positive integer")

    predictions = _load_csv(args.results_dir / "service_predictions.csv", ["ds"])
    metrics = _load_csv(args.results_dir / "service_gru_metrics.csv")
    onprem = _load_csv(args.results_dir / "onprem_scaling_decisions.csv", ["ds"])

    plot_service_traffic_overview(predictions, args.output_dir / "onprem_bnpl_traffic_overview.png")
    plot_service_metric_comparison(metrics, args.output_dir / "onprem_bnpl_metric_comparison.png")
    plot_onprem_adjustment(onprem, args.output_dir / "onprem_bnpl_pod_adjustment.png", args.days)
    print(f"Saved service autoscaling plots to {args.output_dir}")


if __name__ == "__main__":
    main()
