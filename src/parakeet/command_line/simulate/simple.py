#
# parakeet.command_line.simulate.simple.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import argparse
import logging
import time
import parakeet.io
import parakeet.command_line
import parakeet.config
import parakeet.microscope
import parakeet.sample
import parakeet.scan
import parakeet.simulate

# Get the logger
logger = logging.getLogger(__name__)


def get_parser():
    """
    Get the parakeet.simulate.simple parser

    """

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Simulate the whole process")

    # Add some command line arguments
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        dest="config",
        help="The yaml file to configure the simulation",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.h5",
        dest="output",
        help="The filename for the output",
    )
    parser.add_argument(
        "atoms",
        type=str,
        default=None,
        nargs="?",
        help="The filename for the input atoms",
    )

    return parser


def simple():
    """
    Simulate the image

    """

    # Get the start time
    start_time = time.time()

    # Get parser
    parser = get_parser()

    # Parse the arguments
    args = parser.parse_args()

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Do the work
    parakeet.simulate.simple(args.config, args.atoms, args.output)

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))
