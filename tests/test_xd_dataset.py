import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from anomaly_detection.evaluation.xd_dataset import load_dataset, read_split
from anomaly_detection.evaluation.dataset import frame_labels


class XDDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for folder, names in [('Normal_Videos', ['Movie.2001_label_A']),
                              ('Testing_Video', ['Other.2002_label_A', 'Other.2002_label_B1'])]:
            (self.root / folder).mkdir()
            for name in names:
                (self.root / folder / (name + '.mp4')).touch()
        (self.root / 'train_split.txt').write_text('Movie.2001_label_A 0 0\n')
        (self.root / 'test_split.txt').write_text('Other.2002_label_A 0 0\nOther.2002_label_B1 1 1\n')
        (self.root / 'annotations.txt').write_text('Other.2002_label_B1.mp4 0 2\n')

    def test_zero_frame_and_normal_labels(self):
        cap = MagicMock()
        cap.get.side_effect = [5, 30, 5, 30]
        with patch('cv2.VideoCapture', return_value=cap):
            normal, testing, annotations = load_dataset(self.root)
        self.assertEqual(len(normal), 1)
        self.assertEqual(len(testing), 2)
        self.assertEqual(frame_labels(annotations['Other.2002_label_B1'], 5, 30).tolist(), [1, 1, 1, 0, 0])
        self.assertEqual(frame_labels(annotations['Other.2002_label_A'], 5, 30).sum(), 0)

    def test_missing_anomaly_annotation_rejected(self):
        (self.root / 'annotations.txt').write_text('')
        with self.assertRaisesRegex(ValueError, 'annotations do not match'):
            load_dataset(self.root)

    def test_duplicate_split_rejected(self):
        (self.root / 'train_split.txt').write_text('Movie.2001_label_A 0 0\nMovie.2001_label_A 0 0\n')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            read_split(self.root / 'train_split.txt')
