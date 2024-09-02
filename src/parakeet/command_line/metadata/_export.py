#
# parakeet.command_line.metadata._export.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#


import logging
import parakeet.metadata
import parakeet.command_line
from argparse import ArgumentParser
from typing import List

# Get the logger
logger = logging.getLogger(__name__)


__all__ = ["export"]


def get_description():
    """
    Get the program description

    """
    return "Export the metadata to star files for downstream processing"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the parser for the parakeet.metadata.export command

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
        "-s",
        "--sample",
        type=str,
        default="sample.h5",
        dest="sample",
        help="The filename for the sample file",
    )

    parser.add_argument(
        "-i",
        "--image",
        type=str,
        default="image.h5",
        dest="image",
        help="The filename for the image file",
    )

    parser.add_argument(
        "--directory",
        type=str,
        default=".",
        dest="directory",
        help="The directory to export to",
    )

    parser.add_argument(
        "--relion",
        type=bool,
        default=True,
        dest="relion",
        help="Export the relion metadata",
    )

    return parser


def export_impl(args):
    """
    Export the metadata

    """

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Parse the arguments
    parakeet.metadata.export(
        args.config, args.sample, args.image, args.directory, args.relion
    )


def export(args: List[str] = None):
    """
    Export the metadata

    """
    export_impl(get_parser().parse_args(args=args))
