#
# parakeet.command_line.read_pdb.py
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
import parakeet.io
import parakeet.config
import parakeet.sample
from argparse import ArgumentParser


__all__ = ["read_pdb"]


# Get the logger
logger = logging.getLogger(__name__)


def get_description():
    """
    Get the program description

    """
    return "Read a PDB file"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the parser for parakeet.read_pdb

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


def read_pdb_impl(args):
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
        logger.info("%sModel: %s" % (prefix, model.name))
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


def read_pdb(args: list[str] = None):
    """
    Read the given PDB file and show the atom positions

    """
    return read_pdb_impl(get_parser().parse_args(args=args))
