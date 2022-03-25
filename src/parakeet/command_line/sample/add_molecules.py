#
# parakeet.command_line.sample.add_molecules.py
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
import parakeet.sample

# Get the logger
logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    """
    Get the add molecules parser

    """

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Add molecules to the sample model")

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
        "-s",
        "--sample",
        type=str,
        default="sample.h5",
        dest="sample",
        help="The filename for the sample file",
    )

    return parser


def add_molecules(args=None):
    """
    Add molecules to the sample

    """
    st = time.time()

    # Get the add_molecules parser
    parser = get_parser()

    # Parse the arguments
    args = parser.parse_args(args=args)

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Do the work
    parakeet.sample.add_molecules(args.config, args.sample)

    # Print output
    logger.info("Time taken: %.1f seconds" % (time.time() - st))
