"""OpenEvolve evaluator for Prophet model recipe optimization."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
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


DEFAULT_CHILD_TIMEOUT_SECONDS = 300
DEFAULT_CHILD_MEMORY_MB = 2048


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


def _parse_env_int(var_name: str, default: int = 0) -> int:
    """Parse an integer env var without letting bad local config crash evaluation."""
    raw_value = os.getenv(var_name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return int(raw_value.strip())
    except ValueError:
        print(
            f"Warning: ignoring invalid {var_name}={raw_value!r}; using {default}.",
            file=sys.stderr,
        )
        return default


def _maybe_limit_rows(train, holdout):
    """Optionally limit rows for faster exploratory evolution runs."""
    train_tail_rows = _parse_env_int("OPENEVOLVE_TRAIN_TAIL_ROWS", 0)
    holdout_head_rows = _parse_env_int("OPENEVOLVE_HOLDOUT_HEAD_ROWS", 0)

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


def _error_result(error: Exception | str, artifacts: dict[str, str] | None = None) -> EvaluationResult:
    error_message = str(error)
    error_type = type(error).__name__ if isinstance(error, Exception) else "EvaluationError"
    return EvaluationResult(
        metrics={
            "combined_score": 0.0,
            "penalty_score": 999.0,
            "error": error_message,
            "error_type": error_type,
        },
        artifacts=artifacts or {},
    )


def _evaluation_result_to_dict(result: EvaluationResult) -> dict[str, object]:
    return {
        "metrics": result.metrics,
        "artifacts": result.artifacts,
    }


def _evaluation_result_from_dict(payload: dict[str, object]) -> EvaluationResult:
    return EvaluationResult(
        metrics=payload.get("metrics", {}),
        artifacts=payload.get("artifacts", {}),
    )


def _child_env() -> dict[str, str]:
    allowed_keys = {
        "PATH",
        "PATHEXT",
        "APPDATA",
        "HOMEDRIVE",
        "HOMEPATH",
        "LOCALAPPDATA",
        "PYTHONIOENCODING",
        "PYTHONPATH",
        "PYTHONUTF8",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "VIRTUAL_ENV",
        "WINDIR",
        "OPENEVOLVE_TRAIN_TAIL_ROWS",
        "OPENEVOLVE_HOLDOUT_HEAD_ROWS",
    }
    env = {key: value for key, value in os.environ.items() if key.upper() in allowed_keys}
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    python_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) if not python_path else str(PROJECT_ROOT) + os.pathsep + python_path
    return env


def _posix_resource_limiter(timeout_seconds: int):
    if os.name == "nt":
        return None

    def limit_resources() -> None:
        try:
            import resource

            memory_mb = max(_parse_env_int("OPENEVOLVE_EVALUATOR_MEMORY_MB", DEFAULT_CHILD_MEMORY_MB), 256)
            memory_bytes = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_CPU, (timeout_seconds, timeout_seconds + 5))
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        except Exception:
            os._exit(125)

    return limit_resources


def _run_in_child_process(program_path: str | Path) -> EvaluationResult:
    timeout_seconds = _parse_env_int("OPENEVOLVE_EVALUATOR_TIMEOUT_SECONDS", DEFAULT_CHILD_TIMEOUT_SECONDS)
    timeout_seconds = max(timeout_seconds, 1)

    result_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
            result_path = Path(temp_file.name)

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child-evaluate",
            str(program_path),
            "--result-path",
            str(result_path),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            env=_child_env(),
            preexec_fn=_posix_resource_limiter(timeout_seconds),
            text=True,
            timeout=timeout_seconds,
        )
        if result_path.exists() and result_path.stat().st_size > 0:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            return _evaluation_result_from_dict(payload)

        error = completed.stderr.strip() or completed.stdout.strip() or f"child exited with {completed.returncode}"
        return _error_result(error, {"child_returncode": str(completed.returncode)})
    except subprocess.TimeoutExpired as error:
        return _error_result(
            f"Candidate evaluation exceeded {timeout_seconds}s timeout",
            {"timeout_seconds": str(timeout_seconds), "stdout": error.stdout or "", "stderr": error.stderr or ""},
        )
    except Exception as error:
        return _error_result(error, {"traceback": traceback.format_exc()})
    finally:
        if result_path is not None:
            result_path.unlink(missing_ok=True)


def _evaluate_in_process(program_path):
    """Evaluate a candidate inside the child process boundary."""
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
        return _error_result(
            error,
            {
                "error_type": type(error).__name__,
                "traceback": error_details,
            },
        )


def evaluate(program_path):
    """Evaluate an evolved Prophet model recipe in a separate Python process.

    The candidate program must expose `run_forecast(train, holdout)`, which
    trains Prophet and returns non-negative traffic predictions for holdout.
    """
    return _run_in_child_process(program_path)


if __name__ == "__main__":
    if "--child-evaluate" in sys.argv:
        candidate_index = sys.argv.index("--child-evaluate") + 1
        result_index = sys.argv.index("--result-path") + 1
        result = _evaluate_in_process(sys.argv[candidate_index])
        Path(sys.argv[result_index]).write_text(
            json.dumps(_evaluation_result_to_dict(result), ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        result = evaluate(Path(__file__).with_name("initial_program.py"))
        print(json.dumps(result.metrics, indent=2, ensure_ascii=False, sort_keys=True))
