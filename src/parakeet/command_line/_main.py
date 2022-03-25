#
# parakeet.command_line.main.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
from __future__ import annotations

import logging
import parakeet.config
import parakeet.command_line
import parakeet.command_line.config as config
import parakeet.command_line.sample as sample
import parakeet.command_line.simulate as simulate
import parakeet.command_line.analyse as analyse
from argparse import ArgumentParser


__all__ = ["main"]


# Get the logger
logger = logging.getLogger(__name__)


def add_config_command(parser: ArgumentParser):
    """
    Add the config sub command

    """

    # Add some sub commands
    subparsers = parser.add_subparsers(
        dest="config_command", help="The parakeet config sub commands"
    )

    # Add the config commands
    config._new.get_parser(
        subparsers.add_parser("new", help=config._new.get_description())
    )
    config._edit.get_parser(
        subparsers.add_parser("edit", help=config._edit.get_description())
    )
    config._show.get_parser(
        subparsers.add_parser("show", help=config._show.get_description())
    )

    # Return parser
    return parser


def add_sample_command(parser: ArgumentParser):
    """
    Add the sample sub command

    """

    # Add some sub commands
    subparsers = parser.add_subparsers(
        dest="sample_command", help="The parakeet sample sub commands"
    )

    # Add the sample commands
    sample._new.get_parser(
        subparsers.add_parser("new", help=sample._new.get_description())
    )
    sample._add_molecules.get_parser(
        subparsers.add_parser(
            "add_molecules",
            help=sample._add_molecules.get_description(),
        )
    )
    sample._mill.get_parser(
        subparsers.add_parser("mill", help=sample._mill.get_description())
    )
    sample._sputter.get_parser(
        subparsers.add_parser("sputter", help=sample._sputter.get_description())
    )
    sample._show.get_parser(
        subparsers.add_parser("show", help=sample._show.get_description())
    )

    # Return parser
    return parser


def add_simulate_command(parser: ArgumentParser):
    """
    Add the simulate sub command

    """

    # Add some sub commands
    subparsers = parser.add_subparsers(
        dest="simulate_command", help="The parakeet simulate sub commands"
    )

    # Add simulate commands
    simulate._potential.get_parser(
        subparsers.add_parser(
            "potential",
            help=simulate._potential.get_description(),
        )
    )
    simulate._exit_wave.get_parser(
        subparsers.add_parser(
            "exit_wave",
            help=simulate._exit_wave.get_description(),
        )
    )
    simulate._optics.get_parser(
        subparsers.add_parser("optics", help=simulate._optics.get_description())
    )
    simulate._image.get_parser(
        subparsers.add_parser("image", help=simulate._image.get_description())
    )
    simulate._ctf.get_parser(
        subparsers.add_parser("ctf", help=simulate._ctf.get_description())
    )

    # Return parser
    return parser


def add_analyse_command(parser: ArgumentParser):
    """
    Add the analyse sub command

    """
    # Add some sub commands
    subparsers = parser.add_subparsers(
        dest="analyse_command", help="The parakeet analyse sub commands"
    )

    # Add analyse commands
    analyse._reconstruct.get_parser(
        subparsers.add_parser(
            "reconstruct",
            help=analyse._reconstruct.get_description(),
        )
    )
    analyse._correct.get_parser(
        subparsers.add_parser("correct", help=analyse._correct.get_description())
    )
    analyse._average_particles.get_parser(
        subparsers.add_parser(
            "average_particles",
            help=analyse._average_particles.get_description(),
        )
    )
    analyse._average_all_particles.get_parser(
        subparsers.add_parser(
            "average_all_particles",
            help=analyse._average_all_particles.get_description(),
        )
    )
    analyse._extract.get_parser(
        subparsers.add_parser("extract", help=analyse._extract.get_description())
    )
    analyse._refine.get_parser(
        subparsers.add_parser("refine", help=analyse._refine.get_description())
    )

    # Return parser
    return parser


def get_parser() -> ArgumentParser:
    """
    Get the parser for the parakeet.config.new command

    """

    # Create the argument parser
    parser = ArgumentParser(description="Generate a new comfig file")

    # Add some sub commands
    subparsers = parser.add_subparsers(dest="command", help="The parakeet sub commands")

    # Add the "parakeet config" command
    add_config_command(
        subparsers.add_parser(
            "config", help="Commands to manipulate configuration files"
        )
    )

    # Add the "parakeet sample" command
    add_sample_command(
        subparsers.add_parser("sample", help="Commands to manipulate the sample files")
    )

    # Add the "parakeet simulate" command
    add_simulate_command(
        subparsers.add_parser("simulate", help="Commands to simulate the TEM images")
    )

    # Add the "parakeet analyse" command
    add_analyse_command(
        subparsers.add_parser("analyse", help="Commands to analyse the simulated data")
    )

    # Add the parakeet export command
    parakeet.command_line._export.get_parser(
        subparsers.add_parser(
            "export",
            help=parakeet.command_line._export.get_description(),
        )
    )

    # Add the parakeet read_pdb command
    parakeet.command_line._read_pdb.get_parser(
        subparsers.add_parser(
            "read_pdb",
            help=parakeet.command_line._read_pdb.get_description(),
        )
    )

    # Add the parakeet run command
    parakeet.command_line._run.get_parser(
        subparsers.add_parser(
            "run",
            help=parakeet.command_line._run.get_description(),
        )
    )

    # Return parser
    return parser


def config_main(parser, args):
    """
    Perform the parakeet config action

    """
    {
        None: lambda x: parser.print_help(),
        "new": config._new.new_impl,
        "edit": config._edit.edit_impl,
        "show": config._show.show_impl,
    }[args.config_command](args)


def sample_main(parser, args):
    """
    Perform the parakeet sample action

    """
    {
        None: lambda x: parser.print_help(),
        "new": sample._new.new_impl,
        "show": sample._show.show_impl,
        "add_molecules": sample._add_molecules.add_molecules_impl,
        "mill": sample._mill.mill_impl,
        "sputter": sample._sputter.sputter_impl,
    }[args.sample_command](args)


def simulate_main(parser, args):
    """
    Perform the parakeet simulate action

    """
    {
        None: lambda x: parser.print_help(),
        "potential": simulate._potential.potential_impl,
        "exit_wave": simulate._exit_wave.exit_wave_impl,
        "optics": simulate._optics.optics_impl,
        "image": simulate._image.image_impl,
        "ctf": simulate._ctf.ctf_impl,
    }[args.simulate_command](args)


def analyse_main(parser, args):
    """
    Perform the parakeet analyse action

    """
    {
        None: lambda x: parser.print_help(),
        "correct": analyse._correct.correct_impl,
        "reconstruct": analyse._reconstruct.reconstruct_impl,
        "average_particles": analyse._average_particles.average_particles_impl,
        "average_all_particles": analyse._average_all_particles.average_all_particles_impl,
        "extract": analyse._extract.extract_impl,
        "refine": analyse._refine.refine_impl,
    }[args.analyse_command](args)


def export_main(parser, args):
    """
    Perform the parakeet export action

    """
    parakeet.command_line._export.export_impl(args)


def read_pdb_main(parser, args):
    """
    Perform the parakeet export action

    """
    parakeet.command_line._read_pdb.read_pdb_impl(args)


def run_main(parser, args):
    """
    Perform the parakeet run action

    """
    parakeet.command_line._run.run_impl(args)


def get_subparser(parser, command):
    """
    Helper function to get the relevant sub parser

    """
    if command is None:
        return None
    for sp in parser._subparsers._group_actions:
        p = sp.choices.get(command, None)
        if p is not None:
            return p
    raise RuntimeError("Parser for %s not found" % command)


def main():
    """
    Parakeet as a single command line program

    """

    # Get the parser
    parser = get_parser()

    # Parse the arguments
    args = parser.parse_args()

    # Call the top level command
    {
        None: lambda a, b: parser.print_help(),
        "config": config_main,
        "sample": sample_main,
        "simulate": simulate_main,
        "analyse": analyse_main,
        "export": export_main,
        "read_pdb": read_pdb_main,
        "run": run_main,
    }[args.command](get_subparser(parser, args.command), args)
