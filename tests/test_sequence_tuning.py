import unittest

import pandas as pd

from src.models.sequence_tune import (
    autoscaling_objective_score,
    evaluate_sequence_params,
    normalize_params,
    suggest_params,
    tuned_model_name,
)

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = None


class SequenceTuningTest(unittest.TestCase):
    def test_autoscaling_objective_prioritizes_under_provisioning_with_penalties(self):
        metrics = {
            "smape": 0.5,
            "under_provisioning_rate": 0.1,
            "over_provisioning_rate": 0.2,
        }

        score = autoscaling_objective_score(metrics, smape_weight=0.1, over_provisioning_weight=0.2)

        self.assertAlmostEqual(score, 0.19)

    def test_tuned_model_name_uses_suffix(self):
        self.assertEqual(tuned_model_name("gru"), "gru_tuned")
        self.assertEqual(tuned_model_name("lstm"), "lstm_tuned")

    def test_normalize_params_adds_default_dropout_when_absent(self):
        params = normalize_params({"num_layers": 1})

        self.assertEqual(params["dropout"], 0.0)

    def test_suggest_params_disables_dropout_for_single_layer_models(self):
        class SingleLayerTrial:
            def suggest_int(self, name, low, high):
                return 1 if name == "num_layers" else low

            def suggest_categorical(self, name, choices):
                return choices[0]

            def suggest_float(self, name, low, high, log=False):
                if name == "dropout":
                    raise AssertionError("dropout should not be suggested for a single recurrent layer")
                return low

        params = suggest_params(SingleLayerTrial())

        self.assertEqual(params["num_layers"], 1)
        self.assertEqual(params["dropout"], 0.0)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_evaluate_sequence_params_returns_predictions_and_metrics(self):
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2024-01-01", periods=30, freq="h"),
                "y": [float(20 + index) for index in range(30)],
                "is_monsoon": [0] * 30,
                "typhoon_index": [0.0] * 30,
                "hour": [index % 24 for index in range(30)],
                "day_of_week": [index % 7 for index in range(30)],
                "month": [1] * 30,
            }
        )
        params = {
            "sequence_length": 4,
            "hidden_size": 4,
            "learning_rate": 0.01,
            "epochs": 1,
            "batch_size": 0,
            "num_layers": 1,
            "dropout": 0.0,
            "weight_decay": 0.0,
            "gradient_clip": 0.0,
        }

        predictions, metrics, _ = evaluate_sequence_params("gru", df.iloc[:24], df.iloc[24:], params, seed=42)

        self.assertEqual(len(predictions), 6)
        self.assertIn("under_provisioning_rate", metrics)


if __name__ == "__main__":
    unittest.main()
