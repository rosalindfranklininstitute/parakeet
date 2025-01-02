#
# parakeet.command_line.pdb._read.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#


import gemmi
import logging
import logging.config
import parakeet.io
import parakeet.config
import parakeet.sample
from argparse import ArgumentParser
from typing import List


__all__ = ["read"]


# Get the logger
logger = logging.getLogger(__name__)


def get_description():
    """
    Get the program description

    """
    return "Read a PDB file"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the parser for parakeet.pdb.read

    """

    # Initialise the parser
    if parser is None:
        parser = ArgumentParser(description=get_description())

    # Add an argument for the filename
    parser.add_argument(
        type=str,
        default=None,
        dest="filename",
        help="The path to the PDB file",
    )

    return parser


def read_impl(args):
    """
    Read the given PDB file and show the atom positions

    """

    # Check a filename has been given
    if args.filename is None:
        raise RuntimeError("filename is not set")

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Read the structure
    structure = gemmi.read_structure(args.filename)

    # Iterate through atoms
    prefix = " " * 4
    logger.info("Structure: %s" % structure.name)
    for model in structure:
        logger.info("%sModel: %s" % (prefix, str(model)))
        for chain in model:
            logger.info("%sChain: %s" % (prefix * 2, chain.name))
            for residue in chain:
                logger.info("%sResidue: %s" % (prefix * 3, residue.name))
                for atom in residue:
                    logger.info(
                        "%sAtom: %s, %f, %f, %f, %f, %f, %f"
                        % (
                            prefix * 4,
                            atom.element.name,
                            atom.pos.x,
                            atom.pos.y,
                            atom.pos.z,
                            atom.occ,
                            atom.charge,
                            parakeet.sample.get_atom_sigma(atom),
                        )
                    )


def read(args: List[str] = None):
    """
    Read the given PDB file and show the atom positions

    """
    return read_impl(get_parser().parse_args(args=args))
