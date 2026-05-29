import unittest

from src.optimization.autoscaling_score import (
    autoscaling_penalty_score,
    evaluate_autoscaling_predictions,
    fitness_from_penalty,
    pod_change_rate,
    severe_under_provisioning_rate,
)


class AutoscalingScoreTest(unittest.TestCase):
    def test_pod_change_rate_counts_adjacent_changes(self):
        self.assertAlmostEqual(pod_change_rate([1, 1, 2, 2, 3]), 0.5)

    def test_severe_under_provisioning_rate_counts_gaps_greater_than_one(self):
        self.assertAlmostEqual(severe_under_provisioning_rate([3, 4, 2], [1, 3, 2]), 1 / 3)

    def test_fitness_from_penalty_is_higher_for_lower_penalty(self):
        self.assertGreater(fitness_from_penalty(0.1), fitness_from_penalty(0.5))

    def test_evaluate_autoscaling_predictions_includes_combined_score(self):
        metrics = evaluate_autoscaling_predictions([10, 30, 50], [10, 20, 60])

        self.assertIn("penalty_score", metrics)
        self.assertIn("combined_score", metrics)
        self.assertAlmostEqual(metrics["combined_score"], fitness_from_penalty(metrics["penalty_score"]))

    def test_autoscaling_penalty_score_matches_weighted_formula(self):
        metrics = {
            "under_provisioning_rate": 0.1,
            "smape": 0.5,
            "over_provisioning_rate": 0.2,
            "severe_under_provisioning_rate": 0.3,
            "pod_change_rate": 0.4,
        }

        self.assertAlmostEqual(autoscaling_penalty_score(metrics), 0.24)


if __name__ == "__main__":
    unittest.main()
