#
# parakeet.command_line.pdb._get.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#


import logging
import logging.config
import os
import shutil
from argparse import ArgumentParser
import parakeet.data
from typing import List


__all__ = ["get"]


# Get the logger
logger = logging.getLogger(__name__)


def get_description():
    """
    Get the program description

    """
    return "Get a PDB file"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the parser for parakeet.pdb.get

    """

    # Initialise the parser
    if parser is None:
        parser = ArgumentParser(description=get_description())

    # Add an argument for the PDB ID
    parser.add_argument(
        type=str,
        default=None,
        dest="id",
        help="The PDB ID",
    )

    # Add argument for directory output
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default=".",
        dest="directory",
        help="The directory to save to",
    )

    return parser


def get_impl(args):
    """
    Get the PDB file

    """

    # Get the filename
    filename = parakeet.data.get_pdb(args.id)

    # Copy the file to the directory
    shutil.copyfile(filename, os.path.join(args.directory, os.path.basename(filename)))


def get(args: List[str] = None):
    """
    Read the given PDB file and show the atom positions

    """
    return get_impl(get_parser().parse_args(args=args))
