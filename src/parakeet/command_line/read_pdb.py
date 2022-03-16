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
import argparse
import gemmi
import logging
import logging.config
import parakeet.io
import parakeet.config
import parakeet.sample

# Get the logger
logger = logging.getLogger(__name__)


def get_parser():
    """
    Get the parser for parakeet.read_pdb

    """

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Read a PDB file")

    # Add an argument for the filename
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        default=None,
        dest="filename",
        help="The path to the PDB file",
    )

    return parser


def read_pdb():
    """
    Read the given PDB file and show the atom positions

    """

    # Get the parser
    parser = get_parser()

    # Parse the arguments
    args = parser.parse_args()

    # Check a filename has been given
    if args.filename is None:
        parser.print_help()
        exit(0)

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
