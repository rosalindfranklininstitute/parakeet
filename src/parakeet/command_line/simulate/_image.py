#
# parakeet.command_line.simulate.image.py
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


__all__ = ["image"]


# Get the logger
logger = logging.getLogger(__name__)


def get_description():
    """
    Get the program description

    """
    return "Simulate the noisy detector image"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the parakeet.simulate.image parser

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

    return parser


def image_impl(args):
    """
    Simulate the image with noise

    """

    # Get the start time
    start_time = time.time()

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Do the work
    parakeet.simulate.image(args.config, args.optics, args.image)

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))


def image(args: List[str] = None):
    """
    Simulate the image with noise

    """
    image_impl(get_parser().parse_args(args=args))
