"""Strict adapter for the prepared XDViolence folder and its split lists.

Source intervals use zero-based inclusive native frame indices, matching the
local XD dataset loader. Convert to the shared one-based inclusive contract.
"""

from pathlib import Path

from .dataset import Annotation, check_disjoint, discover_videos, frame_labels


def video_id(name):
    # Movie names contain dots even when no video extension is present.
    return name[:-4] if name.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')) else name


def read_split(path):
    records = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        parts = line.rsplit(maxsplit=2)
        if len(parts) != 3 or parts[1] not in ('0', '1'):
            raise ValueError(f'Invalid XD split row: {line}')
        name = video_id(parts[0])
        if name in records:
            raise ValueError(f'Duplicate XD split video: {name}')
        records[name] = int(parts[1])
    if not records:
        raise ValueError(f'Empty XD split: {path}')
    return records


def load_dataset(root):
    root = Path(root)
    train, test = read_split(root / 'train_split.txt'), read_split(root / 'test_split.txt')
    if set(train) & set(test):
        raise ValueError('XD training and test split overlap')
    normal = discover_videos(root / 'Normal_Videos')
    testing = discover_videos(root / 'Testing_Video')
    if set(normal) != {n for n, label in train.items() if label == 0}:
        raise ValueError('XD normal folder does not match normal training split')
    if set(testing) != set(test):
        raise ValueError('XD test folder does not match test split')
    check_disjoint(normal, testing)
    annotations = {}
    for line in (root / 'annotations.txt').read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        name = video_id(parts[0])
        values = list(map(int, parts[1:]))
        if name in annotations or not values or len(values) % 2:
            raise ValueError(f'Invalid/duplicate XD annotation: {line}')
        ranges = []
        for start, end in zip(values[::2], values[1::2]):
            if not 0 <= start <= end:
                raise ValueError(f'Invalid XD interval: {line}')
            ranges.append((start + 1, end + 1))
        annotations[name] = Annotation(name, 'Violence', tuple(ranges))
    if set(annotations) != {n for n, label in test.items() if label == 1}:
        raise ValueError('XD annotations do not match anomalous test split')
    for name, label in test.items():
        if label == 0:
            annotations[name] = Annotation(name, 'Normal', ())
    # Check every label boundary before spending time building the memory bank.
    import cv2
    for name, path in testing.items():
        cap = cv2.VideoCapture(str(path))
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
        finally:
            cap.release()
        frame_labels(annotations[name], count, fps)
    return normal, testing, annotations
