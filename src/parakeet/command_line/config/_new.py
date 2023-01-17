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


import logging
import parakeet.config
import parakeet.command_line
from argparse import ArgumentParser
from typing import List

# Get the logger
logger = logging.getLogger(__name__)


__all__ = ["new"]


def get_description():
    """
    Get the program description

    """
    return "Generate a new config file"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the parser for the parakeet.config.new command

    """

    # Initialise the parser
    if parser is None:
        parser = ArgumentParser(description=get_description())

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


def new_impl(args):
    """
    Create a new configuration

    """

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Parse the arguments
    parakeet.config.new(filename=args.config, full=args.full)


def new(args: List[str] = None):
    """
    Create a new configuration

    """
    new_impl(get_parser().parse_args(args=args))
