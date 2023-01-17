#
# parakeet.command_line.config.edit.py
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


__all__ = ["edit"]


def get_description():
    """
    Get the program description

    """
    return "Edit the configuration"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the parser for the parakeet.config.edit command

    """

    # Initialise the parser
    if parser is None:
        parser = ArgumentParser(description=get_description())

    # Add some command line arguments
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        dest="input",
        required=True,
        help="The input yaml file to configure the simulation",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        dest="output",
        required=True,
        help="The output yaml file to configure the simulation",
    )

    parser.add_argument(
        "-s",
        type=str,
        default="",
        dest="config",
        required=True,
        help="The configuration string",
    )

    return parser


def edit_impl(args):
    """
    Edit the configuration

    """

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Call internally
    config = parakeet.config.edit(args.input, args.output, args.config)

    # Print the config
    parakeet.config.show(config, full=True)


def edit(args: List[str] = None):
    """
    Edit the configuration

    """
    edit_impl(get_parser().parse_args(args=args))
