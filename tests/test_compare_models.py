import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.evaluation.compare_models import (
    MODEL_NAMES,
    load_model_metrics,
    rank_models,
    write_comparison_outputs,
)

try:
    import matplotlib  # noqa: F401
except ModuleNotFoundError:
    matplotlib = None


class CompareModelsTest(unittest.TestCase):
    def write_metric(self, directory: Path, model: str, **overrides):
        metrics = {
            "model": model,
            "smape": 0.2,
            "pod_accuracy": 0.8,
            "under_provisioning_rate": 0.1,
            "over_provisioning_rate": 0.1,
        }
        metrics.update(overrides)
        with (directory / f"{model}_metrics.json").open("w", encoding="utf-8") as file:
            json.dump(metrics, file)

    def test_load_model_metrics_requires_all_models(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(FileNotFoundError, "Missing model metric files"):
                load_model_metrics(Path(temp_dir))

    def test_rank_models_prioritizes_under_provisioning(self):
        metrics = pd.DataFrame(
            [
                {
                    "model": "low_smape_high_under",
                    "smape": 0.1,
                    "pod_accuracy": 0.9,
                    "under_provisioning_rate": 0.2,
                    "over_provisioning_rate": 0.0,
                },
                {
                    "model": "higher_smape_low_under",
                    "smape": 0.2,
                    "pod_accuracy": 0.7,
                    "under_provisioning_rate": 0.0,
                    "over_provisioning_rate": 0.1,
                },
            ]
        )

        ranked = rank_models(metrics)

        self.assertEqual(ranked.iloc[0]["model"], "higher_smape_low_under")

    def test_rank_models_uses_pod_accuracy_before_over_provisioning(self):
        metrics = pd.DataFrame(
            [
                {
                    "model": "lower_over_lower_accuracy",
                    "smape": 0.2,
                    "pod_accuracy": 0.7,
                    "under_provisioning_rate": 0.0,
                    "over_provisioning_rate": 0.0,
                },
                {
                    "model": "higher_over_higher_accuracy",
                    "smape": 0.2,
                    "pod_accuracy": 0.9,
                    "under_provisioning_rate": 0.0,
                    "over_provisioning_rate": 0.1,
                },
            ]
        )

        ranked = rank_models(metrics)

        self.assertEqual(ranked.iloc[0]["model"], "higher_over_higher_accuracy")

    @unittest.skipIf(matplotlib is None, "matplotlib is not installed")
    def test_write_comparison_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            for model in MODEL_NAMES:
                self.write_metric(temp_path, model)

            ranked = rank_models(load_model_metrics(temp_path))
            outputs = write_comparison_outputs(ranked, temp_path, temp_path)

            self.assertTrue(outputs["comparison_results"].exists())
            self.assertTrue(outputs["best_model"].exists())
            self.assertTrue(outputs["plot"].exists())


if __name__ == "__main__":
    unittest.main()
