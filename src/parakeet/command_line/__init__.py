#
# parakeet.command_line.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import logging
import logging.config
from parakeet.command_line._export import *  # noqa
from parakeet.command_line._run import *  # noqa
from parakeet.command_line._main import *  # noqa

# Get the logger
logger = logging.getLogger(__name__)


def configure_logging():
    """
    Configure the logging

    """

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": True,
            "handlers": {
                "stream": {
                    "level": "DEBUG",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                }
            },
            "loggers": {
                "parakeet": {
                    "handlers": ["stream"],
                    "level": "DEBUG",
                    "propagate": True,
                }
            },
        }
    )
