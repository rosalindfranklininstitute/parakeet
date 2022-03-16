#
# parakeet.command_line.config.show.py
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
import parakeet.config
import parakeet.command_line

# Get the logger
logger = logging.getLogger(__name__)


def get_parser():
    """
    Get the parser for the parakeet.config.show command

    """

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Show the configuration")

    # Add some command line arguments
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        dest="config",
        help="The yaml file to configure the simulation",
    )

    return parser


def show():
    """
    Show the full configuration

    """

    # Get the show parser
    parser = get_parser()

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Parse the arguments
    config = parakeet.config.load(parser.parse_args().config)

    # Print some options
    parakeet.config.show(parser.parse_args().config, full=True)
