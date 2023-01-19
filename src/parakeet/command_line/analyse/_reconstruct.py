#
# parakeet.command_line.analyse.reconstruct.py
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
import parakeet.analyse
import parakeet.io
import parakeet.command_line
import parakeet.config
import parakeet.microscope
import parakeet.sample
from argparse import ArgumentParser
from typing import List


__all__ = ["reconstruct"]


# Get the logger
logger = logging.getLogger(__name__)


def get_description():
    """
    Get the program description

    """
    return "Reconstruct the volume"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the parakeet.analyse.reconstruct parser

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
        "-i",
        "--image",
        type=str,
        default="image.mrc",
        dest="image",
        help="The filename for the image",
    )
    parser.add_argument(
        "-r",
        "--rec",
        type=str,
        default="rec.mrc",
        dest="rec",
        help="The filename for the reconstruction",
    )

    return parser


def reconstruct_impl(args):
    """
    Reconstruct the volume

    """

    # Get the start time
    start_time = time.time()

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Do the reconstruction
    parakeet.analyse.reconstruct(args.config, args.image, args.rec, args.device)

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))


def reconstruct(args: List[str] = None):
    """
    Reconstruct the volume

    """
    reconstruct_impl(get_parser().parse_args(args=args))
