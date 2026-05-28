import unittest

from src.models.prophet.tune import autoscaling_objective_score


class ProphetTuningTest(unittest.TestCase):
    def test_autoscaling_objective_prioritizes_under_provisioning_with_penalties(self):
        metrics = {
            "smape": 0.5,
            "under_provisioning_rate": 0.1,
            "over_provisioning_rate": 0.2,
        }

        score = autoscaling_objective_score(metrics, smape_weight=0.1, over_provisioning_weight=0.2)

        self.assertAlmostEqual(score, 0.19)


if __name__ == "__main__":
    unittest.main()
