"""Compare model metrics and select the best forecasting model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
PLOTS_DIR = PROJECT_ROOT / "experiments" / "plots"
MODEL_NAMES = ["prophet", "sarima", "gru", "lstm"]
METRIC_COLUMNS = [
    "smape",
    "pod_accuracy",
    "under_provisioning_rate",
    "over_provisioning_rate",
]
SORT_COLUMNS = [
    "under_provisioning_rate",
    "smape",
    "pod_accuracy",
    "over_provisioning_rate",
]
SORT_ASCENDING = [True, True, False, True]


def load_model_metrics(results_dir: Path = RESULTS_DIR, model_names: list[str] | None = None) -> pd.DataFrame:
    """Load model metric JSON files into one comparison DataFrame."""
    model_names = model_names or MODEL_NAMES
    rows = []
    missing = []

    for model_name in model_names:
        path = results_dir / f"{model_name}_metrics.json"
        if not path.exists():
            missing.append(path)
            continue

        with path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)

        row = {"model": metrics.get("model", model_name)}
        for metric in METRIC_COLUMNS:
            if metric not in metrics:
                raise ValueError(f"{path} is missing required metric: {metric}")
            row[metric] = float(metrics[metric])
        rows.append(row)

    if missing:
        missing_list = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Missing model metric files. Run each model training script first:\n"
            f"{missing_list}"
        )

    return pd.DataFrame(rows)


def rank_models(metrics: pd.DataFrame) -> pd.DataFrame:
    """Rank models by autoscaling-first criteria."""
    ranked = metrics.sort_values(SORT_COLUMNS, ascending=SORT_ASCENDING).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def write_comparison_outputs(
    ranked: pd.DataFrame,
    results_dir: Path = RESULTS_DIR,
    plots_dir: Path = PLOTS_DIR,
) -> dict[str, Path]:
    """Write comparison CSV, best-model JSON, and comparison plot."""
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = results_dir / "comparison_results.csv"
    best_model_path = results_dir / "best_model.json"
    plot_path = plots_dir / "model_comparison.png"

    ranked.to_csv(comparison_path, index=False)

    best = ranked.iloc[0].to_dict()
    best["selection_rule"] = (
        "Lowest under_provisioning_rate, then lowest smape, "
        "highest pod_accuracy, lowest over_provisioning_rate."
    )
    with best_model_path.open("w", encoding="utf-8") as file:
        json.dump(best, file, indent=2, ensure_ascii=False)

    plot_comparison(ranked, plot_path)
    return {
        "comparison_results": comparison_path,
        "best_model": best_model_path,
        "plot": plot_path,
    }


def plot_comparison(ranked: pd.DataFrame, output_path: Path) -> None:
    """Create a compact metric comparison chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.flatten()

    for axis, metric in zip(axes, METRIC_COLUMNS):
        axis.bar(ranked["model"], ranked[metric])
        axis.set_title(metric)
        axis.set_ylim(0, max(1.0, ranked[metric].max() * 1.15))
        axis.tick_params(axis="x", rotation=20)

    fig.suptitle("Model Comparison Metrics", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare model metric JSON files.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        metrics = load_model_metrics(args.results_dir)
        ranked = rank_models(metrics)
        outputs = write_comparison_outputs(ranked, args.results_dir, args.plots_dir)
    except (FileNotFoundError, ValueError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error

    print("Model comparison completed.")
    print(f"- comparison_results: {outputs['comparison_results']}")
    print(f"- best_model: {outputs['best_model']}")
    print(f"- plot: {outputs['plot']}")
    print(f"- selected_model: {ranked.iloc[0]['model']}")


if __name__ == "__main__":
    main()
