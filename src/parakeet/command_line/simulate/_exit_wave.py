#
# parakeet.command_line.simulate.exit_wave.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
from __future__ import annotations

import logging
import time
import parakeet.io
import parakeet.command_line
import parakeet.config
import parakeet.microscope
import parakeet.sample
import parakeet.scan
import parakeet.simulate
from argparse import ArgumentParser


__all__ = ["exit_wave"]


# Get the logger
logger = logging.getLogger(__name__)


def get_description():
    """
    Get the program description

    """
    return "Simulate the exit wave from the sample"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the parakeet.simulate.exit_wave parser

    """

    # Initialise the parser
    if parser is None:
        parser = ArgumentParser(description=get_description())

    # Add some command line arguments
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        dest="config",
        required=True,
        help="The yaml file to configure the simulation",
    )
    parser.add_argument(
        "-s",
        "--sample",
        type=str,
        default="sample.h5",
        dest="sample",
        help="The filename for the sample",
    )
    parser.add_argument(
        "-e",
        "--exit_wave",
        type=str,
        default="exit_wave.h5",
        dest="exit_wave",
        help="The filename for the exit wave",
    )
    parser.add_argument(
        "-d",
        "--device",
        choices=["cpu", "gpu"],
        default=None,
        dest="device",
        help="Choose the device to use",
    )
    parser.add_argument(
        "--cluster.max_workers",
        type=int,
        default=None,
        dest="cluster_max_workers",
        help="The maximum number of worker processes",
    )
    parser.add_argument(
        "--cluster.method",
        type=str,
        choices=["sge"],
        default=None,
        dest="cluster_method",
        help="The cluster method to use",
    )

    return parser


def exit_wave_impl(args):
    """
    Simulate the exit wave from the sample

    """

    # Get the start time
    start_time = time.time()

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Do the work
    parakeet.simulate.exit_wave(
        args.config,
        args.sample,
        args.exit_wave,
        device=args.device,
        cluster_method=args.cluster_method,
        cluster_max_workers=args.cluster_max_workers,
    )

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))


def exit_wave(args: list[str] = None):
    """
    Simulate the exit wave from the sample

    """
    exit_wave_impl(get_parser().parse_args(args=args))
