"""Train and evaluate the OpenEvolve-optimized Prophet model recipe."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from src.models.common import (
    add_common_args,
    build_prediction_frame,
    load_traffic_data,
    save_model_outputs,
    split_train_holdout,
)
from src.optimization.autoscaling_score import evaluate_autoscaling_predictions


MODEL_NAME = "openevolve_prophet"
SOURCE_PROGRAM = "experiments/openevolve/prophet_model/openevolve_output_smoke2/best/best_program.py"
SOURCE_PROGRAM_ID = "c2225f8e-ac6e-4ef3-908d-28622cb22e2e"


def validate_source_program() -> Path:
    """Verify that the recorded OpenEvolve source program is available."""
    source_path = Path(SOURCE_PROGRAM)
    if not source_path.is_file():
        raise FileNotFoundError(
            f"OpenEvolve source program is missing or unreadable: {source_path}. "
            "Run OpenEvolve first or update SOURCE_PROGRAM before saving provenance metadata."
        )
    return source_path


def _ensure_ascii_tbb_path() -> None:
    """Expose CmdStan's TBB DLL through an ASCII-only path for Windows UTF-8 mode."""
    try:
        import prophet as prophet_package
    except ImportError:
        return

    prophet_dir = Path(prophet_package.__file__).resolve().parent
    matches = list((prophet_dir / "stan_model").glob("**/tbb.dll"))
    if not matches:
        return

    target_dir = Path(tempfile.gettempdir()) / "prophet_tbb_ascii"
    target_dir.mkdir(parents=True, exist_ok=True)
    source_dir = matches[0].parent
    for dll_path in source_dir.glob("*.dll"):
        shutil.copy2(dll_path, target_dir / dll_path.name)

    target = str(target_dir)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if target not in path_parts:
        os.environ["PATH"] = target + os.pathsep + os.environ.get("PATH", "")


def prepare_features(frame):
    """Create the OpenEvolve-selected Prophet regressors."""
    validate_input_columns(frame)
    prepared = frame.copy()
    prepared["is_peak_hour"] = prepared["hour"].isin([8, 9, 10, 18, 19, 20]).astype(int)
    prepared["is_weekend"] = prepared["day_of_week"].isin([5, 6]).astype(int)
    prepared["monsoon_typhoon"] = prepared["is_monsoon"] * prepared["typhoon_index"]
    prepared["hour_sin"] = np.sin(2 * np.pi * prepared["hour"] / 24)
    prepared["hour_cos"] = np.cos(2 * np.pi * prepared["hour"] / 24)
    return prepared


def validate_input_columns(frame) -> None:
    """Validate base columns required by the selected feature recipe."""
    required = {"hour", "day_of_week", "is_monsoon", "typhoon_index"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns for feature engineering: {missing}")

    hour = np.asarray(frame["hour"], dtype=float)
    day_of_week = np.asarray(frame["day_of_week"], dtype=float)
    if not np.all(np.isfinite(hour)) or not np.all((0 <= hour) & (hour <= 23)):
        raise ValueError("Column 'hour' must contain numeric values in [0, 23]")
    if not np.all(np.isfinite(day_of_week)) or not np.all((0 <= day_of_week) & (day_of_week <= 6)):
        raise ValueError("Column 'day_of_week' must contain numeric values in [0, 6]")


def candidate_regressors() -> list[str]:
    """Return the OpenEvolve-selected Prophet regressor columns."""
    return [
        "is_monsoon",
        "typhoon_index",
        "is_peak_hour",
        "is_weekend",
        "monsoon_typhoon",
    ]


def build_model():
    """Build the OpenEvolve-optimized Prophet model."""
    from prophet import Prophet

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="additive",
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=10.0,
        holidays_prior_scale=1.0,
        changepoint_range=0.8,
        stan_backend="CMDSTANPY",
    )
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    return model


def train_and_predict(train, holdout) -> np.ndarray:
    """Train the optimized Prophet recipe and return holdout predictions."""
    _ensure_ascii_tbb_path()
    train_features = prepare_features(train)
    holdout_features = prepare_features(holdout)
    regressors = candidate_regressors()

    model = build_model()
    for regressor in regressors:
        model.add_regressor(regressor)

    prophet_train = train_features[["ds", "y", *regressors]]
    prophet_holdout = holdout_features[["ds", *regressors]]
    model.fit(prophet_train)
    forecast = model.predict(prophet_holdout)
    return np.maximum(forecast["yhat"].to_numpy(dtype=float), 0)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Train OpenEvolve-optimized Prophet model.")
    add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_program_path = validate_source_program()
    df = load_traffic_data(args.data_path)
    train, holdout = split_train_holdout(df, args.holdout_ratio)

    predicted = train_and_predict(train, holdout)
    predictions = build_prediction_frame(holdout, predicted, MODEL_NAME)
    autoscaling_metrics = evaluate_autoscaling_predictions(predictions["actual"], predictions["predicted"])
    metrics = save_model_outputs(
        MODEL_NAME,
        predictions,
        {
            "train_rows": len(train),
            "holdout_rows": len(holdout),
            "features": candidate_regressors(),
            "source_program": str(source_program_path),
            "source_program_id": SOURCE_PROGRAM_ID,
            "openevolve_penalty_score": autoscaling_metrics["penalty_score"],
            "openevolve_combined_score": autoscaling_metrics["combined_score"],
            "severe_under_provisioning_rate": autoscaling_metrics["severe_under_provisioning_rate"],
            "pod_change_rate": autoscaling_metrics["pod_change_rate"],
            "prophet_params": {
                "seasonality_mode": "additive",
                "changepoint_prior_scale": 0.1,
                "seasonality_prior_scale": 10.0,
                "holidays_prior_scale": 1.0,
                "changepoint_range": 0.8,
                "custom_seasonalities": [{"name": "monthly", "period": 30.5, "fourier_order": 5}],
            },
        },
    )
    print(metrics)


if __name__ == "__main__":
    main()
