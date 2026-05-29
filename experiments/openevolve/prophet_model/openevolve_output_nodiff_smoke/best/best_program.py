import numpy as np
from prophet import Prophet

def prepare_features(frame):
    """Create candidate regressors for Prophet from the common traffic data."""
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
    
    model = build_model()
    model.fit(train_features)
    
    forecast = model.predict(holdout_features)
    predicted = inverse_transform_predictions(forecast["yhat"].astype(float))
    
    return np.maximum(predicted, 0)