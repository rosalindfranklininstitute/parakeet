#
# parakeet.command_line.analyse.correct.py
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


__all__ = ["correct"]


# Get the logger
logger = logging.getLogger(__name__)


def get_description():
    """
    Get the program description

    """
    return "3D CTF correction of the images"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the parakeet.analyse.correct parser

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
        "-i",
        "--image",
        type=str,
        default="image.mrc",
        dest="image",
        help="The filename for the image",
    )
    parser.add_argument(
        "-cr",
        "--corrected",
        type=str,
        default="corrected_image.mrc",
        dest="corrected",
        help="The filename for the corrected image",
    )
    parser.add_argument(
        "-ndf",
        "--num-defocus",
        type=int,
        default=None,
        dest="num_defocus",
        help="Number of defoci that correspond to different depths through for which the sample will be 3D CTF corrected",
    )
    parser.add_argument(
        "-d",
        "--device",
        choices=["cpu", "gpu"],
        default=None,
        dest="device",
        help="Choose the device to use",
    )

    return parser


def correct_impl(args):
    """
    Correct the images using 3D CTF correction

    """

    # Get the start time
    start_time = time.time()

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Do the work
    parakeet.analyse.correct(
        args.config, args.image, args.corrected, args.num_defocus, args.device
    )

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))


def correct(args: List[str] = None):
    """
    Correct the images using 3D CTF correction

    """
    correct_impl(get_parser().parse_args(args=args))
