#!/usr/bin/env python3
"""Resumable single-host CUHK Avenue evaluation with one worker per GPU.

Expects the prepared UCF-style layout written by scripts/prepare_avenue.py.
"""

from evaluate_ucf_multi_gpu import main


if __name__ == "__main__":
    main(dataset="Avenue")
