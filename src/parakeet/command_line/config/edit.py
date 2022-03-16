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
import argparse
import logging
import yaml
import parakeet.config
import parakeet.command_line

# Get the logger
logger = logging.getLogger(__name__)


def get_parser():
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
    parser = get_parser()

    # Parse the command line
    args = parser.parse_args()

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Parse the arguments
    config = parakeet.config.load(args.input)

    # Merge the dictionaries
    d1 = config.dict(exclude_unset=True)
    d2 = yaml.safe_load(args.config)
    d = parakeet.config.deepmerge(d1, d2)

    # Load the new configuration
    config = parakeet.config.load(d)

    # Print the config
    parakeet.config.show(config, full=True)

    # Save the config
    parakeet.config.save(config, args.output, exclude_unset=True)
