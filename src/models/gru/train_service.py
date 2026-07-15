"""Train service-level GRU models and build autoscaling decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_predictions
from src.evaluation.service_pod_policy import (
    add_service_pod_columns,
    build_onprem_scaling_decisions,
    load_service_policies,
)
from src.models.common import (
    MODEL_FEATURE_COLUMNS,
    MODELS_DIR,
    PROJECT_ROOT,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    build_model_feature_frame,
    split_train_holdout,
)
from src.models.sequence_model import (
    SequenceRegressor,
    make_sequences,
    recursive_holdout_forecast,
    scale_values,
    train_sequence_regressor,
)
from src.optimization.autoscaling_score import pod_change_rate, severe_under_provisioning_rate


SERVICE_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "onprem_bnpl_service_traffic.csv"
SERVICE_EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "service_autoscaling" / "onprem_bnpl"
SERVICE_RESULTS_DIR = SERVICE_EXPERIMENT_DIR / "results"
SERVICE_CONFIG_PATH = SERVICE_EXPERIMENT_DIR / "configs" / "service_policies.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train service-level GRU autoscaling models.")
    parser.add_argument("--data-path", type=Path, default=SERVICE_DATA_PATH)
    parser.add_argument("--policy-path", type=Path, default=SERVICE_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=SERVICE_RESULTS_DIR)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=0.0)
    parser.add_argument("--onprem-pod-budget", type=int, default=26)
    args = parser.parse_args()

    if args.sequence_length <= 0:
        parser.error("--sequence-length must be a positive integer")
    if args.epochs <= 0:
        parser.error("--epochs must be a positive integer")
    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer")
    if args.hidden_size <= 0:
        parser.error("--hidden-size must be a positive integer")
    if not 0 < args.learning_rate < 1:
        parser.error("--learning-rate must be greater than 0 and less than 1")
    if args.onprem_pod_budget <= 0:
        parser.error("--onprem-pod-budget must be a positive integer")
    return args


def load_service_traffic(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run `python -m src.data.generate_service_dummy_data` first."
        )

    df = pd.read_csv(path, parse_dates=[TIMESTAMP_COLUMN])
    required = {
        TIMESTAMP_COLUMN,
        "service",
        TARGET_COLUMN,
        "is_monsoon",
        "typhoon_index",
        "hour",
        "day_of_week",
        "month",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df.sort_values(["service", TIMESTAMP_COLUMN]).reset_index(drop=True)


def _train_one_service(
    service_df: pd.DataFrame,
    service: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object]]:
    import torch

    service_df = service_df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    train, holdout = split_train_holdout(service_df, args.holdout_ratio)
    columns = [TARGET_COLUMN, *MODEL_FEATURE_COLUMNS]
    model_frame = service_df[[TARGET_COLUMN]].join(build_model_feature_frame(service_df))

    train_values = model_frame.iloc[: len(train)][columns].to_numpy(dtype=float)
    all_values = model_frame[columns].to_numpy(dtype=float)
    scaled_train, scaling = scale_values(train_values)
    scaled_all, _ = scale_values(all_values, scaling)
    train_x, train_y = make_sequences(scaled_train, args.sequence_length)

    model = SequenceRegressor.gru(
        input_size=len(columns),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    train_sequence_regressor(
        model,
        train_x,
        train_y,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
    )

    scaled_predictions = recursive_holdout_forecast(
        model,
        scaled_all,
        holdout_start=len(train),
        sequence_length=args.sequence_length,
    )
    target_mean = scaling.mean[0]
    target_std = scaling.std[0]
    predicted = np.maximum((scaled_predictions * target_std) + target_mean, 0)

    predictions = holdout[[TIMESTAMP_COLUMN, TARGET_COLUMN]].rename(columns={TARGET_COLUMN: "actual"})
    predictions["predicted"] = predicted
    predictions["service"] = service
    predictions["model"] = "service_gru"

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"service_gru_{service}.pt"
    torch.save(model.state_dict(), model_path)

    metrics = {
        "service": service,
        "model": "service_gru",
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
    }
    return predictions, metrics


def _priority_weighted_under_provisioning(predictions: pd.DataFrame, policies) -> float:
    weighted_under = 0.0
    weight_total = 0.0
    for row in predictions.itertuples(index=False):
        priority = policies[row.service].priority
        weight = 1 / priority
        weighted_under += weight * int(row.predicted_pods < row.actual_pods)
        weight_total += weight
    return float(weighted_under / weight_total) if weight_total else 0.0


def build_service_metrics(predictions: pd.DataFrame, policies) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for service, rows in predictions.groupby("service", sort=True):
        policy = policies[service]
        base_metrics = evaluate_predictions(rows["actual"], rows["predicted"], policy)
        actual_pods = rows["actual_pods"].to_numpy()
        predicted_pods = rows["predicted_pods"].to_numpy()
        records.append(
            {
                "service": service,
                **base_metrics,
                "severe_under_provisioning_rate": severe_under_provisioning_rate(actual_pods, predicted_pods),
                "pod_change_rate": pod_change_rate(predicted_pods),
                "priority": policy.priority,
            }
        )
    return pd.DataFrame.from_records(records)


def main() -> None:
    import torch

    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)

    policies = load_service_policies(args.policy_path if args.policy_path.exists() else None)
    df = load_service_traffic(args.data_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prediction_frames = []
    model_records = []
    for service, service_df in df.groupby("service", sort=True):
        predictions, metrics = _train_one_service(service_df, service, args)
        prediction_frames.append(predictions)
        model_records.append(metrics)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = add_service_pod_columns(predictions, policies)
    predictions = predictions.sort_values([TIMESTAMP_COLUMN, "service"]).reset_index(drop=True)

    service_metrics = build_service_metrics(predictions, policies)
    onprem = build_onprem_scaling_decisions(predictions, args.onprem_pod_budget, policies)

    predictions.to_csv(args.output_dir / "service_predictions.csv", index=False)
    service_metrics.to_csv(args.output_dir / "service_gru_metrics.csv", index=False)
    onprem.to_csv(args.output_dir / "onprem_scaling_decisions.csv", index=False)

    summary = {
        "model": "service_gru",
        "services": sorted(predictions["service"].unique().tolist()),
        "prediction_path": str((args.output_dir / "service_predictions.csv").relative_to(PROJECT_ROOT)),
        "metric_path": str((args.output_dir / "service_gru_metrics.csv").relative_to(PROJECT_ROOT)),
        "onprem_decision_path": str((args.output_dir / "onprem_scaling_decisions.csv").relative_to(PROJECT_ROOT)),
        "priority_weighted_under_provisioning_rate": _priority_weighted_under_provisioning(predictions, policies),
        "sequence_length": args.sequence_length,
        "epochs": args.epochs,
        "hidden_size": args.hidden_size,
        "learning_rate": args.learning_rate,
        "onprem_pod_budget": args.onprem_pod_budget,
        "model_records": model_records,
    }
    with (args.output_dir / "service_gru_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(summary)


if __name__ == "__main__":
    main()
