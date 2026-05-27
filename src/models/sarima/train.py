"""Train and evaluate the SARIMA traffic forecasting model."""

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
    train, holdout = split_train_holdout(df, args.holdout_ratio)
    features = FEATURE_COLUMNS

    model = SARIMAX(
        train["y"],
        exog=train[features],
        order=parse_order(args.order),
        seasonal_order=parse_order(args.seasonal_order),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    forecast = fitted.forecast(steps=len(holdout), exog=holdout[features]).clip(lower=0)

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

