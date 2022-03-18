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
    Get the parakeet.simulate.image parser

    """

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Simulate the noisy detector image")

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


def image(args=None):
    """
    Simulate the image with noise

    """

    # Get the start time
    start_time = time.time()

    # Get parser
    parser = get_parser()

    # Parse the arguments
    args = parser.parse_args(args=args)

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Do the work
    parakeet.simulate.image(args.config, args.optics, args.image)

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))
