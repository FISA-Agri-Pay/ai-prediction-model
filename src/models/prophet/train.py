"""Train and evaluate the Prophet traffic forecasting model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.models.common import (
    FEATURE_COLUMNS,
    add_common_args,
    build_prediction_frame,
    load_traffic_data,
    save_model_outputs,
    split_train_holdout,
)


MODEL_NAME = "prophet"


def build_model(params: dict[str, object] | None = None):
    from prophet import Prophet

    params = params or {}
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode=params.get("seasonality_mode", "additive"),
        changepoint_prior_scale=params.get("changepoint_prior_scale", 0.05),
        seasonality_prior_scale=params.get("seasonality_prior_scale", 10.0),
        holidays_prior_scale=params.get("holidays_prior_scale", 10.0),
        changepoint_range=params.get("changepoint_range", 0.8),
    )
    for feature in ["is_monsoon", "typhoon_index"]:
        model.add_regressor(feature)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Prophet model.")
    add_common_args(parser)
    parser.add_argument(
        "--params-path",
        type=Path,
        default=None,
        help="Optional JSON file with Prophet parameters.",
    )
    return parser.parse_args()


def load_params(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    args = parse_args()
    df = load_traffic_data(args.data_path)
    train, holdout = split_train_holdout(df, args.holdout_ratio)

    params = load_params(args.params_path)
    model = build_model(params)
    model.fit(train[["ds", "y", "is_monsoon", "typhoon_index"]])
    forecast = model.predict(holdout[["ds", "is_monsoon", "typhoon_index"]])

    predictions = build_prediction_frame(holdout, forecast["yhat"].clip(lower=0), MODEL_NAME)
    metrics = save_model_outputs(
        MODEL_NAME,
        predictions,
        {
            "train_rows": len(train),
            "holdout_rows": len(holdout),
            "features": [feature for feature in FEATURE_COLUMNS if feature in ["is_monsoon", "typhoon_index"]],
            "prophet_params": params or "default",
        },
    )
    print(metrics)


if __name__ == "__main__":
    main()
