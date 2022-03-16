#
# parakeet.command_line.config.new.py
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
    Get the parser for the parakeet.config.new command

    """

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Generate a new comfig file")

    # Add some command line arguments
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        dest="config",
        help="The yaml file to configure the simulation",
    )

    parser.add_argument(
        "-f",
        "--full",
        type=bool,
        default=False,
        dest="full",
        help="Generate a file with the full configuration specification",
    )

    return parser


def new():
    """
    Show the full configuration

    """

    # Get the parser
    parser = get_parser()

    # Parse the arguments
    args = parser.parse_args()

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Parse the arguments
    parakeet.config.new(filename=args.config, full=args.full)
