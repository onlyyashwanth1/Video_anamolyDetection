#!/usr/bin/env python3
"""Resumable single-host XD-Violence evaluation with one worker per GPU."""

from evaluate_ucf_multi_gpu import main


if __name__ == "__main__":
    main(dataset="XD-Violence")
