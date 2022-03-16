#
# parakeet.command_line.sample.show.py
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
import parakeet.io
import parakeet.command_line
import parakeet.config
import parakeet.sample

# Get the logger
logger = logging.getLogger(__name__)


def get_parser():
    """
    Get the parakeet.sample.show parser

    """
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Create an ice sample and save it")

    # Add some command line arguments
    parser.add_argument(
        "-s",
        "--sample",
        type=str,
        default="sample.h5",
        dest="sample",
        help="The filename for the sample file",
    )

    return parser


def show():
    """
    Show the sample information

    """
    # Get the parser
    parser = get_parser()

    # Parse the arguments
    args = parser.parse_args()

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Create the sample
    sample = parakeet.sample.load(args.sample)
    logger.info(sample.info())
