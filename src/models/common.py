"""Shared helpers for model training scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.evaluation.metrics import evaluate_predictions
from src.evaluation.pod_policy import DEFAULT_POD_POLICY, required_pods


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "traffic.csv"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
MODELS_DIR = PROJECT_ROOT / "models"

FEATURE_COLUMNS = ["is_monsoon", "typhoon_index", "hour", "day_of_week", "month"]
TARGET_COLUMN = "y"
TIMESTAMP_COLUMN = "ds"


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    return parser


def load_traffic_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run `python src/data/generate_dummy_data.py` first."
        )

    df = pd.read_csv(path, parse_dates=[TIMESTAMP_COLUMN])
    required = {TIMESTAMP_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    return df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)


def split_train_holdout(
    df: pd.DataFrame,
    holdout_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < holdout_ratio < 1:
        raise ValueError("holdout_ratio must be between 0 and 1")

    split_index = int(len(df) * (1 - holdout_ratio))
    if split_index <= 0 or split_index >= len(df):
        raise ValueError("holdout split produced an empty train or holdout set")

    train = df.iloc[:split_index].copy()
    holdout = df.iloc[split_index:].copy()
    return train, holdout


def build_prediction_frame(
    holdout: pd.DataFrame,
    predicted: Iterable[float],
    model_name: str,
) -> pd.DataFrame:
    result = holdout[[TIMESTAMP_COLUMN, TARGET_COLUMN]].copy()
    result = result.rename(columns={TARGET_COLUMN: "actual"})
    result["predicted"] = list(predicted)
    result["actual_pods"] = required_pods(result["actual"].to_numpy(), DEFAULT_POD_POLICY)
    result["predicted_pods"] = required_pods(result["predicted"].to_numpy(), DEFAULT_POD_POLICY)
    result["model"] = model_name
    return result


def save_model_outputs(
    model_name: str,
    predictions: pd.DataFrame,
    extra_metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    prediction_path = PREDICTIONS_DIR / f"{model_name}_predictions.csv"
    metrics_path = RESULTS_DIR / f"{model_name}_metrics.json"

    predictions.to_csv(prediction_path, index=False)
    metrics = evaluate_predictions(predictions["actual"], predictions["predicted"])
    output = {
        "model": model_name,
        "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)),
        **metrics,
    }
    if extra_metrics:
        output.update(extra_metrics)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    return output
