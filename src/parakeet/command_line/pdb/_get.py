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
from __future__ import annotations

import gemmi
import logging
import logging.config
import os
import profet
from pypdb.clients.pdb.pdb_client import PDBFileType
from argparse import ArgumentParser


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
        default=None,
        dest="directory",
        help="The directory to save to",
    )

    return parser


def get_pdb(pdb):
    """
    Get the pdb

    """

    # Get the fetcher
    fetcher = profet.Fetcher("pdb")

    # Download the data
    filedata = None
    filename = None
    for filetype in ["pdb", "cif"]:
        filename, filedata = fetcher.get_file(pdb, filetype=filetype)
        if filedata is not None:
            break
    else:
        raise RuntimeError("No PDB or CIF file found")

    # Return filedata
    return filename, filedata


def get_impl(args):
    """
    Get the PDB file

    """

    # Get the filename and filedata
    filename, filedata = get_pdb(args.id)

    # Construct the path
    filepath = os.path.join(args.directory, filename)

    # Save the file
    if filepath is not None and filedata is not None:
        with open(filepath, "w") as outfile:
            outfile.write(filedata)


def get(args: list[str] = None):
    """
    Read the given PDB file and show the atom positions

    """
    return get_impl(get_parser().parse_args(args=args))
