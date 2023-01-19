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


import logging
import parakeet.io
import parakeet.command_line
import parakeet.config
import parakeet.sample
from argparse import ArgumentParser
from typing import List


__all__ = ["show"]


# Get the logger
logger = logging.getLogger(__name__)


def get_description():
    """
    Get the program description

    """
    return "Print details about the sample model"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the parakeet.sample.show parser

    """

    # Initialise the parser
    if parser is None:
        parser = ArgumentParser(description=get_description())

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


def show_impl(args):
    """
    Show the sample information

    """

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Create the sample
    sample = parakeet.sample.load(args.sample)
    logger.info(sample.info())


def show(args: List[str] = None):
    """
    Show the sample information

    """
    show_impl(get_parser().parse_args(args=args))
