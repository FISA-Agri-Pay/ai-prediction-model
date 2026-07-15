import unittest
from unittest.mock import patch

import pandas as pd

from experiments.openevolve.prophet_model.evaluator import _parse_env_int
from src.models.prophet.openevolve_train import (
    candidate_regressors,
    prepare_features,
    validate_input_columns,
    validate_source_program,
)


class OpenEvolveProphetTest(unittest.TestCase):
    def test_candidate_regressors_exist_after_feature_preparation(self):
        frame = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2024-01-01 08:00:00"]),
                "y": [10.0],
                "is_monsoon": [1],
                "typhoon_index": [0.5],
                "hour": [8],
                "day_of_week": [0],
                "month": [1],
            }
        )

        prepared = prepare_features(frame)

        for regressor in candidate_regressors():
            self.assertIn(regressor, prepared.columns)

    def test_prepare_features_rejects_missing_base_columns(self):
        frame = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2024-01-01 08:00:00"]),
                "y": [10.0],
                "is_monsoon": [1],
                "typhoon_index": [0.5],
            }
        )

        with self.assertRaisesRegex(ValueError, "Missing columns"):
            validate_input_columns(frame)

    def test_prepare_features_rejects_invalid_hour(self):
        frame = pd.DataFrame(
            {
                "ds": pd.to_datetime(["2024-01-01 08:00:00"]),
                "y": [10.0],
                "is_monsoon": [1],
                "typhoon_index": [0.5],
                "hour": [24],
                "day_of_week": [0],
                "month": [1],
            }
        )

        with self.assertRaisesRegex(ValueError, "hour"):
            prepare_features(frame)

    def test_parse_env_int_uses_default_for_invalid_values(self):
        with patch.dict("os.environ", {"OPENEVOLVE_TRAIN_TAIL_ROWS": "not-an-int"}):
            self.assertEqual(_parse_env_int("OPENEVOLVE_TRAIN_TAIL_ROWS", 0), 0)

    def test_source_program_provenance_exists(self):
        self.assertTrue(validate_source_program().is_file())


if __name__ == "__main__":
    unittest.main()
