#
# parakeet.command_line.config.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import argparse
import logging
import yaml
import parakeet.config
import parakeet.command_line

# Get the logger
logger = logging.getLogger(__name__)


def get_show_parser():
    """
    Get the parser for the parakeet.config.show command

    """

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Show the configuration")

    # Add some command line arguments
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        dest="config",
        help="The yaml file to configure the simulation",
    )

    return parser


def show():
    """
    Show the full configuration

    """

    # Get the show parser
    parser = get_show_parser()

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Parse the arguments
    config = parakeet.config.load(parser.parse_args().config)

    # Print some options
    parakeet.config.show(config, full=True)


def get_new_parser():
    """
    Get the parser for the parakeet.config.new command

    """

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Generate a new comfig file")

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
        "-f",
        "--full",
        type=bool,
        default=False,
        dest="full",
        help="Generate a file with the full configuration specification",
    )

    return parser


def new():
    """
    Show the full configuration

    """

    # Get the parser
    parser = get_new_parser()

    # Parse the arguments
    args = parser.parse_args()

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Parse the arguments
    parakeet.config.new(filename=args.config, full=args.full)


def get_edit_parser():
    """
    Get the parser for the parakeet.config.edit command

    """

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Edit the configuration")

    # Add some command line arguments
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="config.yaml",
        dest="input",
        help="The input yaml file to configure the simulation",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="config_new.yaml",
        dest="output",
        help="The output yaml file to configure the simulation",
    )

    parser.add_argument(
        "-s", type=str, default="", dest="config", help="The configuration string"
    )

    return parser


def edit():
    """
    Edit the configuration

    """

    # Get the edit parser
    parser = get_edit_parser()

    # Parse the command line
    args = parser.parse_args()

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Parse the arguments
    config = parakeet.config.load(args.input, yaml.safe_load(args.config))

    # Print the config
    parakeet.config.show(config, full=True)

    # Save the config
    with open(args.output, "w") as outfile:
        yaml.safe_dump(config.dict(), outfile)
