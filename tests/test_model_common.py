import unittest

import pandas as pd

from src.models.common import build_prediction_frame, save_model_outputs, split_train_holdout


class ModelCommonTest(unittest.TestCase):
    def test_split_train_holdout_keeps_chronological_order(self):
        df = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=10, freq="h"), "y": range(10)})

        train, holdout = split_train_holdout(df, holdout_ratio=0.2)

        self.assertEqual(len(train), 8)
        self.assertEqual(len(holdout), 2)
        self.assertLess(train["ds"].max(), holdout["ds"].min())

    def test_build_prediction_frame_adds_pod_columns(self):
        holdout = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=2, freq="h"), "y": [10, 40]})

        result = build_prediction_frame(holdout, [10, 20], "test_model")

        self.assertEqual(
            list(result.columns),
            ["ds", "actual", "predicted", "actual_pods", "predicted_pods", "model"],
        )
        self.assertEqual(result["model"].unique().tolist(), ["test_model"])

    def test_save_model_outputs_rejects_canonical_key_collisions(self):
        holdout = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=2, freq="h"), "y": [10, 40]})
        predictions = build_prediction_frame(holdout, [10, 20], "test_model")

        with self.assertRaisesRegex(ValueError, "extra_metrics cannot override"):
            save_model_outputs("test_model", predictions, {"model": "other"})


if __name__ == "__main__":
    unittest.main()
