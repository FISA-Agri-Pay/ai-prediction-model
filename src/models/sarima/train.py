"""Train and evaluate the SARIMA traffic forecasting model."""

from __future__ import annotations

import argparse

from src.models.common import (
    MODEL_FEATURE_COLUMNS,
    add_common_args,
    build_model_feature_frame,
    build_prediction_frame,
    load_traffic_data,
    save_model_outputs,
    split_train_holdout,
)


MODEL_NAME = "sarima"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SARIMA model.")
    add_common_args(parser)
    parser.add_argument("--order", default="1,1,1", help="ARIMA order as p,d,q")
    parser.add_argument("--seasonal-order", default="1,0,1,24", help="Seasonal order as P,D,Q,s")
    return parser.parse_args()


def parse_order(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(","))


def main() -> None:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    args = parse_args()
    df = load_traffic_data(args.data_path)
    feature_frame = build_model_feature_frame(df)
    train, holdout = split_train_holdout(df, args.holdout_ratio)
    features = MODEL_FEATURE_COLUMNS
    train_exog = feature_frame.iloc[: len(train)][features]
    holdout_exog = feature_frame.iloc[len(train) :][features]

    model = SARIMAX(
        train["y"],
        exog=train_exog,
        order=parse_order(args.order),
        seasonal_order=parse_order(args.seasonal_order),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    forecast = fitted.forecast(steps=len(holdout), exog=holdout_exog).clip(lower=0)

    predictions = build_prediction_frame(holdout, forecast, MODEL_NAME)
    metrics = save_model_outputs(
        MODEL_NAME,
        predictions,
        {
            "train_rows": len(train),
            "holdout_rows": len(holdout),
            "features": features,
            "order": args.order,
            "seasonal_order": args.seasonal_order,
        },
    )
    print(metrics)


if __name__ == "__main__":
    main()

