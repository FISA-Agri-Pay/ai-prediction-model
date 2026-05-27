"""Train and evaluate the Prophet traffic forecasting model."""

from __future__ import annotations

import argparse

from src.models.common import (
    FEATURE_COLUMNS,
    add_common_args,
    build_prediction_frame,
    load_traffic_data,
    save_model_outputs,
    split_train_holdout,
)


MODEL_NAME = "prophet"


def build_model():
    from prophet import Prophet

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="additive",
    )
    for feature in ["is_monsoon", "typhoon_index"]:
        model.add_regressor(feature)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Prophet model.")
    add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_traffic_data(args.data_path)
    train, holdout = split_train_holdout(df, args.holdout_ratio)

    model = build_model()
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
        },
    )
    print(metrics)


if __name__ == "__main__":
    main()

