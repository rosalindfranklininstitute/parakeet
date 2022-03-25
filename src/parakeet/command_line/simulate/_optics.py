#
# parakeet.command_line.simulate.optics.py
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


__all__ = ["optics"]


# Get the logger
logger = logging.getLogger(__name__)


def get_description():
    """
    Get the program description

    """
    return "Simulate the optics and infinite dose image"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the parakeet.simulate.optics parser

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
    parser.add_argument(
        "-e",
        "--exit_wave",
        type=str,
        default="exit_wave.h5",
        dest="exit_wave",
        help="The filename for the exit wave",
    )
    parser.add_argument(
        "-o",
        "--optics",
        type=str,
        default="optics.h5",
        dest="optics",
        help="The filename for the optics",
    )

    return parser


def optics_impl(args):
    """
    Simulate the optics

    """

    # Get the start time
    start_time = time.time()

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Do the work
    parakeet.simulate.optics(
        args.config,
        args.exit_wave,
        args.optics,
        device=args.device,
        cluster_method=args.cluster_method,
        cluster_max_workers=args.cluster_max_workers,
    )

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))


def optics(args: list[str] = None):
    """
    Simulate the optics

    """
    optics_impl(get_parser().parse_args(args=args))
