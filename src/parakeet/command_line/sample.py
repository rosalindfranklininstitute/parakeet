#
# parakeet.command_line.sample.py
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
import parakeet.io
import parakeet.command_line
import parakeet.config
import parakeet.sample

# Get the logger
logger = logging.getLogger(__name__)


def get_new_parser():
    """
    Get the parakeet.sample.new parser

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
        "-s",
        "--sample",
        type=str,
        default="sample.h5",
        dest="sample",
        help="The filename for the sample file",
    )

    return parser


def new(args=None):
    """
    Create an ice sample and save it

    """
    st = time.time()

    # Get the parser
    parser = get_new_parser()

    # Parse the arguments
    args = parser.parse_args(args=args)

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Load the configuration
    config = parakeet.config.load(args.config)

    # Print some options
    parakeet.config.show(config)

    # Create the sample
    logger.info(f"Writing sample to {args.sample}")
    parakeet.sample.new(args.sample, **config.sample.dict())
    logger.info("Time taken: %.1f seconds" % (time.time() - st))


def get_add_molecules_parser():
    """
    Get the add molecules parser

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
        "-s",
        "--sample",
        type=str,
        default="sample.h5",
        dest="sample",
        help="The filename for the sample file",
    )

    return parser


def add_molecules(args=None):
    """
    Add molecules to the sample

    """
    st = time.time()

    # Get the add_molecules parser
    parser = get_add_molecules_parser()

    # Parse the arguments
    args = parser.parse_args(args=args)

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Load the configuration
    config = parakeet.config.load(args.config)

    # Print some options
    parakeet.config.show(config)

    # Create the sample
    logger.info(f"Writing sample to {args.sample}")
    parakeet.sample.add_molecules(args.sample, **config.sample.dict())
    logger.info("Time taken: %.1f seconds" % (time.time() - st))


def get_mill_parser():
    """
    Get the mill parser

    """

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Mill the sample")

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

    return parser


def mill(args=None):
    """
    Mill to the shape of the sample

    """
    st = time.time()

    # Get the mill parser
    parser = get_mill_parser()

    # Parse the arguments
    args = parser.parse_args(args=args)

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Load the configuration
    config = parakeet.config.load(args.config)

    # Print some options
    parakeet.config.show(config)

    # Create the sample
    logger.info(f"Writing sample to {args.sample}")
    parakeet.sample.mill(args.sample, **config.sample.dict())
    logger.info("Time taken: %.1f seconds" % (time.time() - st))


def get_sputter_parser():
    """
    Get the sputter parser

    """
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Sputter the sample")

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

    return parser


def sputter(args=None):
    """
    Sputter the sample

    """
    st = time.time()

    # Get the sputter parser
    parser = get_sputter_parser()

    # Parse the arguments
    args = parser.parse_args(args=args)

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Load the configuration
    config = parakeet.config.load(args.config)

    # Print some options
    parakeet.config.show(config)

    # Create the sample
    logger.info(f"Writing sample to {args.sample}")
    parakeet.sample.sputter(args.sample, **config.sample.sputter.dict())
    logger.info("Time taken: %.1f seconds" % (time.time() - st))


def get_show_parser():
    """
    Get the parakeet.sample.show parser

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

    return parser


def show():
    """
    Show the sample information

    """
    # Get the parser
    parser = get_show_parser()

    # Parse the arguments
    args = parser.parse_args()

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Create the sample
    sample = parakeet.sample.load(args.sample)
    logger.info(sample.info())
