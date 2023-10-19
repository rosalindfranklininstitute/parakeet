#
# parakeet.command_line.simulate.cbed.py
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
import parakeet.microscope
import parakeet.sample
import parakeet.scan
import parakeet.simulate
from argparse import ArgumentParser
from typing import List


__all__ = ["cbed"]


# Get the logger
logger = logging.getLogger(__name__)


def get_description():
    """
    Get the program description

    """
    return "Simulate the image from the sample"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the parakeet.simulate.cbed parser

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

    return parser


def cbed_impl(args):
    """
    Simulate the image from the sample

    """

    # Get the start time
    start_time = time.time()

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Do the work
    parakeet.simulate.cbed(
        args.config,
        args.sample,
        args.image,
        device=args.device,
        nproc=args.nproc,
        gpu_id=args.gpu_id,
    )

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))


def cbed(args: List[str] = None):
    """
    Simulate the image from the sample

    """
    cbed_impl(get_parser().parse_args(args=args))
