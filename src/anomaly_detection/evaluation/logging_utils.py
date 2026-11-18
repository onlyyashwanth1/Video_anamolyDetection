"""Readable evaluation logs in the coordinator and spawned GPU processes."""

import logging


def configure_evaluation_logging(*, worker=False, include_logger=False):
    """Keep progress at INFO while preserving HTTP warnings and errors."""
    fields = "%(asctime)s "
    if worker:
        fields += "%(process)d "
    fields += "%(levelname)s "
    if include_logger:
        fields += "%(name)s: "
    logging.basicConfig(level=logging.INFO, format=fields + "%(message)s")
    # HTTPX emits an INFO record for every model POST. Apply in each spawned
    # process too, since multiprocessing spawn does not inherit logger levels.
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
