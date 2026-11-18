#!/usr/bin/env python3
"""Resumable single-host ShanghaiTech Campus evaluation with one worker per GPU.

Expects the prepared UCF-style layout written by scripts/prepare_shanghaitech.py.
"""

from evaluate_ucf_multi_gpu import main


if __name__ == "__main__":
    main(dataset="ShanghaiTech")
