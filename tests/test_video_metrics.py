import unittest

import numpy as np

from egohtr_phmr_eval.video_metrics import compute_window_metrics


class VideoMetricTests(unittest.TestCase):
    def test_rigidly_transformed_sequence_has_zero_window_errors(self):
        rng = np.random.default_rng(7)
        gt = rng.normal(size=(8, 25, 3)) * 0.2
        angle = 0.3
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])
        pred = (gt - np.array([1.0, -2.0, 0.5])) @ rotation.T
        result = compute_window_metrics(pred, gt, np.arange(8), window_frames=8,
                                       initial_alignment_frames=2)
        self.assertEqual(len(result), 1)
        self.assertLess(result[0]["wa_mpjpe_mm"], 1e-8)
        self.assertLess(result[0]["w_mpjpe_mm"], 1e-8)

    def test_discontinuities_do_not_stitch_windows(self):
        rng = np.random.default_rng(3)
        pred = rng.normal(size=(6, 25, 3))
        result = compute_window_metrics(pred, pred.copy(), [10, 11, 12, 20, 21, 22],
                                       window_frames=100, initial_alignment_frames=2)
        self.assertEqual(len(result), 2)
        self.assertEqual([(x["start_frame"], x["end_frame"]) for x in result],
                         [(10, 12), (20, 22)])

    def test_too_short_tail_is_excluded(self):
        pred = np.zeros((1, 25, 3))
        result = compute_window_metrics(pred, pred, [0], window_frames=100,
                                       initial_alignment_frames=2)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
