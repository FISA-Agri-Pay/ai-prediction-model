import numpy as np
from prophet import Prophet

def validate_input_columns(frame):
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

def prepare_features(frame):
    """Create candidate regressors for Prophet from the common traffic data."""
    validate_input_columns(frame)
    prepared = frame.copy()
    prepared["is_peak_hour"] = np.where((prepared["hour"] >= 8) & (prepared["hour"] <= 10) | 
                                        (prepared["hour"] >= 18) & (prepared["hour"] <= 20), 1, 0)
    prepared["is_weekend"] = np.where(prepared["day_of_week"].isin([5, 6]), 1, 0)
    prepared["monsoon_typhoon"] = prepared["is_monsoon"] * prepared["typhoon_index"]
    prepared["hour_sin"] = np.sin(2 * np.pi * prepared["hour"] / 24)
    prepared["hour_cos"] = np.cos(2 * np.pi * prepared["hour"] / 24)
    return prepared

def candidate_regressors():
    """Return feature columns that should be added as Prophet regressors."""
    return ["is_monsoon", "typhoon_index", "is_peak_hour", "is_weekend", "monsoon_typhoon"]

def build_model():
    """Build the candidate Prophet model."""
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="additive",
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        changepoint_range=0.8,
        stan_backend="CMDSTANPY",
    )
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    return model

def transform_target(values):
    """Transform y before fitting Prophet."""
    return np.asarray(values, dtype=float)

def inverse_transform_predictions(values):
    """Invert the target transformation for Prophet yhat values."""
    return np.asarray(values, dtype=float)

def run_forecast(train, holdout):
    """Train the evolved Prophet model and return one prediction per holdout row."""
    train_features = prepare_features(train)
    holdout_features = prepare_features(holdout)
    regressors = list(candidate_regressors())

    required_columns = {"ds", "y", *regressors}
    missing_train = sorted(required_columns - set(train_features.columns))
    missing_holdout = sorted(({"ds", *regressors}) - set(holdout_features.columns))
    if missing_train or missing_holdout:
        raise ValueError(
            f"Missing columns for Prophet recipe. train={missing_train}, holdout={missing_holdout}"
        )
    
    model = build_model()
    for regressor in regressors:
        model.add_regressor(regressor)

    prophet_train = train_features[["ds", *regressors]].copy()
    prophet_train["y"] = transform_target(train_features["y"].to_numpy(dtype=float))
    prophet_train = prophet_train[["ds", "y", *regressors]]
    prophet_holdout = holdout_features[["ds", *regressors]]
    model.fit(prophet_train)
    
    forecast = model.predict(prophet_holdout)
    predicted = inverse_transform_predictions(forecast["yhat"].astype(float))
    
    return np.maximum(predicted, 0)
