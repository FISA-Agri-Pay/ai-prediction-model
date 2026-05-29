"""OpenEvolve evaluator for Prophet model recipe optimization."""

from __future__ import annotations

import json
import os
import sys
import traceback
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.common import DATA_PATH, load_traffic_data, split_train_holdout
from src.optimization.autoscaling_score import evaluate_autoscaling_predictions

try:
    from openevolve.evaluation_result import EvaluationResult
except ImportError:  # Allows local smoke tests before OpenEvolve is installed.

    class EvaluationResult:  # type: ignore[no-redef]
        def __init__(self, metrics, artifacts=None):
            self.metrics = metrics
            self.artifacts = artifacts or {}


def _load_program(program_path: str | Path):
    """Load candidate code while tolerating Windows-local encoded temp files."""
    source_path = Path(program_path)
    raw_source = source_path.read_bytes()
    source = None
    decode_errors = []
    for encoding in ("utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"):
        try:
            source = raw_source.decode(encoding)
            break
        except UnicodeDecodeError as error:
            decode_errors.append(f"{encoding}: {error}")
    if source is None:
        raise UnicodeDecodeError(
            "candidate_source",
            raw_source,
            0,
            min(len(raw_source), 1),
            "; ".join(decode_errors),
        )

    program = types.ModuleType("openevolve_prophet_candidate")
    program.__file__ = str(source_path)
    exec(compile(source, str(source_path), "exec"), program.__dict__)
    return program


def _maybe_limit_rows(train, holdout):
    """Optionally limit rows for faster exploratory evolution runs."""
    train_tail_rows = int(os.getenv("OPENEVOLVE_TRAIN_TAIL_ROWS", "0"))
    holdout_head_rows = int(os.getenv("OPENEVOLVE_HOLDOUT_HEAD_ROWS", "0"))

    if train_tail_rows > 0:
        train = train.tail(train_tail_rows).copy()
    if holdout_head_rows > 0:
        holdout = holdout.head(holdout_head_rows).copy()
    return train, holdout


def _load_data():
    df = load_traffic_data(DATA_PATH)
    train, holdout = split_train_holdout(df, holdout_ratio=0.2)
    return _maybe_limit_rows(train, holdout)


def _artifact_summary(metrics: dict[str, float], program_path: str | Path) -> dict[str, str]:
    return {
        "candidate_metrics": json.dumps(metrics, indent=2, sort_keys=True),
        "program_path": str(program_path),
        "score_direction": "OpenEvolve should maximize combined_score; penalty_score is lower-is-better.",
    }


def evaluate(program_path):
    """Evaluate an evolved Prophet model recipe.

    The candidate program must expose `run_forecast(train, holdout)`, which
    trains Prophet and returns non-negative traffic predictions for holdout.
    """
    try:
        program = _load_program(program_path)
        if not hasattr(program, "run_forecast"):
            return EvaluationResult(
                metrics={
                    "combined_score": 0.0,
                    "penalty_score": 999.0,
                    "error": "Missing run_forecast function",
                },
                artifacts={"suggestion": "Keep the fixed run_forecast(train, holdout) function."},
            )

        train, holdout = _load_data()
        predicted = program.run_forecast(train.copy(), holdout.copy())
        metrics = evaluate_autoscaling_predictions(holdout["y"], predicted)
        metrics["train_rows"] = float(len(train))
        metrics["holdout_rows"] = float(len(holdout))

        return EvaluationResult(
            metrics=metrics,
            artifacts=_artifact_summary(metrics, program_path),
        )
    except Exception as error:
        error_details = traceback.format_exc()
        debug_path = Path(__file__).with_name("evaluator_last_error.txt")
        debug_path.write_text(error_details, encoding="utf-8")
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "penalty_score": 999.0,
                "error": str(error),
                "error_type": type(error).__name__,
            },
            artifacts={
                "error_type": type(error).__name__,
                "traceback": error_details,
            },
        )


if __name__ == "__main__":
    result = evaluate(Path(__file__).with_name("initial_program.py"))
    print(json.dumps(result.metrics, indent=2, ensure_ascii=False, sort_keys=True))
