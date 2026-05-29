import unittest

import pandas as pd

from src.models.prophet.openevolve_train import candidate_regressors, prepare_features


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


if __name__ == "__main__":
    unittest.main()
