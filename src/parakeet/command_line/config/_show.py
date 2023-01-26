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


import logging
import parakeet.config
import parakeet.command_line
from argparse import ArgumentParser
from typing import List

# Get the logger
logger = logging.getLogger(__name__)


__all__ = ["show"]


def get_description():
    """
    Get the program description

    """
    return "Show the configuration"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the parser for the parakeet.config.show command

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
        help="The yaml file to configure the simulation",
    )

    # Add some command line arguments
    parser.add_argument(
        "-s",
        "--schema",
        type=str,
        default=None,
        dest="schema",
        help=(
            "Show the config schema. \n"
            "To show full scheme type '-s .'. \n"
            "To show the schema for a specific section type e.g. '-s /definitions/Simulation'"
        ),
    )

    return parser


def show_impl(args):
    """
    Show the full configuration

    """

    # Parse the arguments
    config = parakeet.config.load(args.config)

    # Print some options
    print(parakeet.config.show(config, full=True, schema=args.schema))


def show(args: List[str] = None):
    """
    Show the full configuration

    """
    show_impl(get_parser().parse_args(args=args))
