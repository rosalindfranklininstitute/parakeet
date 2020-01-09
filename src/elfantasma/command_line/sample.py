#
# elfantasma.command_line.sample.py
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
import elfantasma.io
import elfantasma.command_line
import elfantasma.config
import elfantasma.sample

# Get the logger
logger = logging.getLogger(__name__)


def new():
    """
    Create an ice sample and save it

    """
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Create an ice sample and save it")

    # Add some command line arguments
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        dest="config",
        help="The yaml file to configure the simulation",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="sample.h5",
        dest="output",
        help="The filename for the sample file",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Configure some basic logging
    elfantasma.command_line.configure_logging()

    # Load the configuration
    config = elfantasma.config.load(args.config)

    # Print some options
    elfantasma.config.show(config)

    # Create the sample
    logger.info(f"Writing sample to {args.output}")
    sample = elfantasma.sample.new(args.output, **config["sample"])


def modify():
    pass


def show():
    pass
