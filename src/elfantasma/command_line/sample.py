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
import time
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
    st = time.time()

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
        "-s",
        "--sample",
        type=str,
        default="sample.h5",
        dest="sample",
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
    logger.info(f"Writing sample to {args.sample}")
    sample = elfantasma.sample.new(args.sample, **config["sample"])
    logger.info("Time taken: %.1f seconds" % (time.time() - st))


def add_molecules():
    """
    Add molecules to the sample

    """
    st = time.time()

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
        "-s",
        "--sample",
        type=str,
        default="sample.h5",
        dest="sample",
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
    logger.info(f"Writing sample to {args.sample}")
    sample = elfantasma.sample.add_molecules(args.sample, **config["sample"])
    logger.info("Time taken: %.1f seconds" % (time.time() - st))


def mill():
    pass


def show():
    """
    Show the sample information

    """
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Create an ice sample and save it")

    # Add some command line arguments
    parser.add_argument(
        "-s",
        "--sample",
        type=str,
        default="sample.h5",
        dest="sample",
        help="The filename for the sample file",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Configure some basic logging
    elfantasma.command_line.configure_logging()

    # Create the sample
    sample = elfantasma.sample.load(args.sample)
    logger.info(sample.info())
