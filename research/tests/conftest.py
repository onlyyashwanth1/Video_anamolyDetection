"""Test setup for the frozen research scripts.

Adds src/ to the path like tests/conftest.py. The research code was written for the
package layout before the restructure (see research/README.md), so on the current
tree these tests are skipped with a message instead of failing at import. They run
from the snapshot commit a3cb86e (git worktree add ../vad_snapshot a3cb86e).
"""

import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

OLD_LAYOUT = importlib.util.find_spec("anomaly_detection.evaluation.window_aggregation") is not None

if not OLD_LAYOUT:
    collect_ignore_glob = ["test_*.py"]


def pytest_report_header(config):
    if not OLD_LAYOUT:
        return ("research/tests skipped: they need the pre-restructure package layout; "
                "run them from the a3cb86e snapshot (see research/README.md)")
    return None
