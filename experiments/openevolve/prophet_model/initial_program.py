"""Initial Prophet model recipe for OpenEvolve.

OpenEvolve should mutate only the EVOLVE block. The fixed `run_forecast`
function trains the evolved Prophet recipe and validates the holdout forecast.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np


def _ensure_ascii_tbb_path():
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


# EVOLVE-BLOCK-START
def prepare_features(frame):
    """Create candidate regressors for Prophet from the common traffic data.

    Keep this function lightweight: add simple numeric columns derived from
    existing fields, then return the copied frame.
    Original input columns are: ds, y, is_monsoon, typhoon_index, hour,
    day_of_week, and month. Create any additional columns here before adding
    them to candidate_regressors().
    """
    prepared = frame.copy()
    prepared["is_peak_hour"] = prepared["hour"].isin([8, 9, 10, 18, 19, 20]).astype(int)
    prepared["is_weekend"] = prepared["day_of_week"].isin([5, 6]).astype(int)
    prepared["monsoon_typhoon"] = prepared["is_monsoon"] * prepared["typhoon_index"]
    prepared["hour_sin"] = np.sin(2 * np.pi * prepared["hour"] / 24)
    prepared["hour_cos"] = np.cos(2 * np.pi * prepared["hour"] / 24)
    return prepared


def candidate_regressors():
    """Return feature columns that should be added as Prophet regressors.

    Every returned column must exist after prepare_features(frame) runs.
    """
    return [
        "is_monsoon",
        "typhoon_index",
        "is_peak_hour",
        "is_weekend",
        "monsoon_typhoon",
    ]


def build_model():
    """Build the candidate Prophet model.

    Keep stan_backend="CMDSTANPY" for this Windows evaluation environment.
    """
    from prophet import Prophet

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.03,
        seasonality_prior_scale=8.0,
        holidays_prior_scale=1.0,
        changepoint_range=0.9,
        stan_backend="CMDSTANPY",
    )
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    return model


def transform_target(values):
    """Transform y before fitting Prophet."""
    return np.asarray(values, dtype=float)


def inverse_transform_predictions(values):
    """Invert the target transform for Prophet yhat values."""
    return np.asarray(values, dtype=float)


# EVOLVE-BLOCK-END


def run_forecast(train, holdout):
    """Train the evolved Prophet recipe and return one prediction per holdout row."""
    _ensure_ascii_tbb_path()
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
    if not np.all(np.isfinite(prophet_train["y"].to_numpy(dtype=float))):
        raise ValueError("Transformed target contains NaN or infinite values")

    prophet_train = prophet_train[["ds", "y", *regressors]]
    prophet_holdout = holdout_features[["ds", *regressors]]

    model.fit(prophet_train)
    forecast = model.predict(prophet_holdout)
    predicted = inverse_transform_predictions(forecast["yhat"].to_numpy(dtype=float))
    predicted = np.asarray(predicted, dtype=float)

    if predicted.shape != (len(holdout),):
        raise ValueError(f"run_forecast must return {len(holdout)} predictions")
    if not np.all(np.isfinite(predicted)):
        raise ValueError("Forecast contains NaN or infinite values")
    return np.maximum(predicted, 0)
