#
# parakeet.command_line.run.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#


import logging
import time
import parakeet.io
import parakeet.command_line
import parakeet.config
import parakeet.sample
from argparse import ArgumentParser
from typing import List


__all__ = ["run"]


# Get the logger
logger = logging.getLogger(__name__)


def get_description():
    """
    Get the program description

    """
    return "Run full simulation experiment"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the parakeet.sample.new parser

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
        "-o",
        "--optics",
        type=str,
        default="optics.h5",
        dest="optics",
        help="The filename for the optics",
    )
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        default="image.h5",
        dest="image",
        help="The filename for the image",
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
        "--nproc",
        type=int,
        default=None,
        dest="nproc",
        help="The number of processes to use",
    )
    parser.add_argument(
        "--gpu_id",
        type=lambda x: [int(item) for item in x.split(",")],
        default=None,
        dest="gpu_id",
        help="The GPU ids (must match number of processors)",
    )
    parser.add_argument(
        "--steps",
        type=str,
        choices=[
            "all",
            "sample",
            "sample.new",
            "sample.add_molecules",
            "simulate",
            "simulate.exit_wave",
            "simulate.optics",
            "simulate.image",
        ],
        nargs="+",
        default=None,
        dest="steps",
        help="Which simulation steps to run",
    )

    return parser


def run_impl(args):
    """
    Run the whole simulation experiment

    """

    # Get the start time
    start_time = time.time()

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Do the work
    parakeet.run(
        args.config,
        args.sample,
        args.exit_wave,
        args.optics,
        args.image,
        args.device,
        args.nproc,
        args.gpu_id,
        args.steps,
    )

    # Print output
    logger.info("Time taken: %.1f seconds" % (time.time() - start_time))


def run(args: List[str] = None):
    """
    Run the whole simulation experiment

    """
    run_impl(get_parser().parse_args(args=args))
