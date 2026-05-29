"""Tune Prophet hyperparameters with Optuna for autoscaling metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.evaluation.metrics import evaluate_predictions
from src.models.common import (
    FEATURE_COLUMNS,
    RESULTS_DIR,
    add_common_args,
    build_prediction_frame,
    load_traffic_data,
    save_model_outputs,
    split_train_holdout,
)
from src.models.prophet.train import build_model


BASE_MODEL_NAME = "prophet"
TUNED_MODEL_NAME = "prophet_tuned"
PROPHET_FEATURES = ["is_monsoon", "typhoon_index"]
BEST_PARAMS_PATH = RESULTS_DIR / "prophet_best_params.json"
TRIALS_PATH = RESULTS_DIR / "prophet_tuning_trials.csv"
SUMMARY_PATH = RESULTS_DIR / "prophet_tuning_summary.json"

if TYPE_CHECKING:
    import optuna


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Prophet hyperparameter tuning."""
    parser = argparse.ArgumentParser(description="Tune Prophet parameters with Optuna.")
    add_common_args(parser)
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel Optuna trials. Start with 2 for Prophet on local CPU.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Optuna sampler.")
    parser.add_argument(
        "--smape-weight",
        type=float,
        default=0.1,
        help="Penalty weight for SMAPE in the objective score.",
    )
    parser.add_argument(
        "--over-provisioning-weight",
        type=float,
        default=0.2,
        help="Penalty weight for over-provisioning rate in the objective score.",
    )
    args = parser.parse_args()

    if args.trials <= 0:
        parser.error("--trials must be a positive integer.")
    if args.n_jobs <= 0:
        parser.error("--n-jobs must be a positive integer.")
    if args.smape_weight < 0:
        parser.error("--smape-weight must be non-negative.")
    if args.over_provisioning_weight < 0:
        parser.error("--over-provisioning-weight must be non-negative.")

    return args


def suggest_params(trial: "optuna.Trial") -> dict[str, object]:
    """Suggest Prophet parameter values for one Optuna trial."""
    return {
        "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True),
        "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale", 0.01, 20.0, log=True),
        "holidays_prior_scale": trial.suggest_float("holidays_prior_scale", 0.01, 20.0, log=True),
        "changepoint_range": trial.suggest_float("changepoint_range", 0.75, 0.95),
        "seasonality_mode": trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"]),
    }


def autoscaling_objective_score(
    metrics: dict[str, float],
    smape_weight: float = 0.1,
    over_provisioning_weight: float = 0.2,
) -> float:
    """Score metrics with under-provisioning as the primary objective."""
    return (
        metrics["under_provisioning_rate"]
        + smape_weight * metrics["smape"]
        + over_provisioning_weight * metrics["over_provisioning_rate"]
    )


def evaluate_prophet_params(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    params: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Train Prophet with params and evaluate holdout predictions."""
    model = build_model(params)
    model.fit(train[["ds", "y", *PROPHET_FEATURES]])
    forecast = model.predict(holdout[["ds", *PROPHET_FEATURES]])
    predictions = build_prediction_frame(holdout, forecast["yhat"].clip(lower=0), TUNED_MODEL_NAME)
    metrics = evaluate_predictions(predictions["actual"], predictions["predicted"])
    return predictions, metrics


def objective_factory(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    smape_weight: float,
    over_provisioning_weight: float,
):
    """Build an Optuna objective function bound to the train/holdout data."""
    import optuna

    def objective(trial: "optuna.Trial") -> float:
        try:
            params = suggest_params(trial)
            _, metrics = evaluate_prophet_params(train, holdout, params)
            score = autoscaling_objective_score(metrics, smape_weight, over_provisioning_weight)

            for name, value in metrics.items():
                trial.set_user_attr(name, value)
            trial.set_user_attr("objective_score", score)
            return score
        except optuna.TrialPruned:
            raise
        except Exception as error:
            trial.set_user_attr("error", str(error))
            raise

    return objective


def write_tuning_outputs(study: "optuna.Study", best_metrics: dict[str, object]) -> None:
    """Persist tuning trial history, best params, and summary files."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state")).to_csv(
        TRIALS_PATH,
        index=False,
    )

    best_params = dict(study.best_params)
    with BEST_PARAMS_PATH.open("w", encoding="utf-8") as file:
        json.dump(best_params, file, indent=2, ensure_ascii=False)

    summary = {
        "best_trial": study.best_trial.number,
        "best_objective_score": study.best_value,
        "best_params_path": str(BEST_PARAMS_PATH.relative_to(RESULTS_DIR.parent.parent)),
        "trials_path": str(TRIALS_PATH.relative_to(RESULTS_DIR.parent.parent)),
        "tuned_metrics": best_metrics,
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)


def main() -> None:
    """Run Prophet Optuna tuning and save tuned model evaluation outputs."""
    import optuna

    args = parse_args()
    df = load_traffic_data(args.data_path)
    train, holdout = split_train_holdout(df, args.holdout_ratio)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        objective_factory(train, holdout, args.smape_weight, args.over_provisioning_weight),
        n_trials=args.trials,
        n_jobs=args.n_jobs,
        catch=(Exception,),
    )

    best_predictions, best_metric_values = evaluate_prophet_params(train, holdout, study.best_params)
    best_metrics = save_model_outputs(
        TUNED_MODEL_NAME,
        best_predictions,
        {
            "base_model": BASE_MODEL_NAME,
            "train_rows": len(train),
            "holdout_rows": len(holdout),
            "features": [feature for feature in FEATURE_COLUMNS if feature in PROPHET_FEATURES],
            "prophet_params": dict(study.best_params),
            "objective_score": autoscaling_objective_score(
                best_metric_values,
                args.smape_weight,
                args.over_provisioning_weight,
            ),
            "objective_formula": (
                "under_provisioning_rate "
                f"+ {args.smape_weight} * smape "
                f"+ {args.over_provisioning_weight} * over_provisioning_rate"
            ),
        },
    )
    write_tuning_outputs(study, best_metrics)
    print(best_metrics)


if __name__ == "__main__":
    main()
