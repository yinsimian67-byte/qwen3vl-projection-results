import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate import candidate_diagnostics, evaluate, validate_predictions


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.gt = [{"qid": 1, "vid": "video_0_150", "duration": 150,
                    "relevant_windows": [[10, 20], [40, 50]]}]
        self.pred = [{"qid": 1, "vid": "video_0_150",
                      "pred_relevant_windows": [[10, 20, 0.9], [80, 90, 0.8], [100, 110, 0.7]]}]

    def test_rejects_duplicate_missing_extra_qids(self):
        for predictions in (self.pred * 2, [], [dict(self.pred[0], qid=2)]):
            with self.assertRaises(ValueError):
                validate_predictions(predictions, self.gt)

    def test_rejects_illegal_spans_scores_count_and_identity(self):
        for window in ([20, 10, 0.9], [-1, 10, 0.9], [140, 151, 0.9], [10, 20, float("nan")], [10, 10, 1]):
            broken = copy.deepcopy(self.pred)
            broken[0]["pred_relevant_windows"][0] = window
            with self.assertRaises(ValueError):
                validate_predictions(broken, self.gt)
        for update in ({"vid": "wrong"}, {"pred_relevant_windows": [[10, 20, 1]]},
                       {"pred_relevant_windows": [[10, 20, 0], [80, 90, 1], [100, 110, 0]]}):
            with self.assertRaises(ValueError):
                validate_predictions([self.pred[0] | update], self.gt)

    def test_multigt_coverage_differs_from_query_hit(self):
        metrics = candidate_diagnostics(self.pred, self.gt)
        self.assertEqual(metrics["QueryHit@3/tIoU0.5"], 100)
        self.assertEqual(metrics["GT-Coverage@3/tIoU0.5"], 50)

    def test_fixed_pool_accepts_permutation_but_rejects_boundary_change(self):
        reranked = copy.deepcopy(self.pred)
        reranked[0]["pred_relevant_windows"] = [[80, 90, 3], [10, 20, 2], [100, 110, 1]]
        validate_predictions(reranked, self.gt, self.pred)
        reranked[0]["pred_relevant_windows"][0][0] = 79
        with self.assertRaisesRegex(ValueError, "candidate pool"):
            validate_predictions(reranked, self.gt, self.pred)

    def test_official_metrics_multigt_and_empty_length_bins(self):
        result = evaluate(self.pred, self.gt)
        self.assertEqual(result["brief"]["MR-full-R1@0.5"], 100)
        self.assertEqual(result["brief"]["MR-full-mAP"], 50)
        self.assertIsNone(result["detailed"]["long"]["MR-mAP"])
        reranked = copy.deepcopy(self.pred)
        reranked[0]["pred_relevant_windows"] = [[80, 90, 3], [10, 20, 2], [100, 110, 1]]
        result = evaluate(reranked, self.gt, self.pred)
        self.assertEqual(result["brief"]["MR-full-R1@0.5"], 0)
        self.assertEqual(result["brief"]["MR-full-mAP"], 25)


if __name__ == "__main__":
    unittest.main()
