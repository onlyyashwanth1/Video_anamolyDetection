"""Makes `anomaly_detection` importable from tests/ without needing `pip install -e .` first."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
