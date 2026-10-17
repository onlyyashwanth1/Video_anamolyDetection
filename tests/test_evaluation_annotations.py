"""Regression checks for annotation endpoints without loading any models."""

import unittest

import numpy as np

from anomaly_detection.evaluation.dataset import Annotation, frame_labels


class AnnotationEndpointTests(unittest.TestCase):
    def test_native_small_overruns_preserve_available_labels(self):
        for overrun in (1, 2):
            annotation = Annotation("example", "Arson", ((2, 4), (8, 10 + overrun)))
            with self.assertLogs("anomaly_detection.evaluation.dataset", level="WARNING"):
                labels = frame_labels(annotation, 10, 30)
            np.testing.assert_array_equal(labels, [0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
            self.assertEqual(annotation.ranges[-1][1], 10 + overrun)

    def test_valid_inclusive_endpoint(self):
        labels = frame_labels(Annotation("example", "Arson", ((10, 10),)), 10, 30)
        np.testing.assert_array_equal(labels, [0] * 9 + [1])

    def test_large_overrun_fails(self):
        with self.assertRaises(ValueError):
            frame_labels(Annotation("example", "Arson", ((8, 13),)), 10, 30)

    def test_interval_entirely_outside_video_fails(self):
        with self.assertRaises(ValueError):
            frame_labels(Annotation("example", "Arson", ((11, 12),)), 10, 30)

    def test_resampled_overrun_fails(self):
        with self.assertRaises(ValueError):
            frame_labels(Annotation("example", "Arson", ((1, 6),)), 30, 30, 5)
