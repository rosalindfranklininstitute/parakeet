#
# parakeet.command_line.sample.mill.py
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


def get_parser():
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


def mill_internal(config_file, sample):
    """
    Mill to the shape of the sample

    """
    # Load the configuration
    config = parakeet.config.load(config_file)

    # Print some options
    parakeet.config.show(config)

    # Create the sample
    logger.info(f"Writing sample to {sample}")
    parakeet.sample.mill(sample, **config.sample.dict())


def mill(args=None):
    """
    Mill to the shape of the sample

    """
    st = time.time()

    # Get the mill parser
    parser = get_parser()

    # Parse the arguments
    args = parser.parse_args(args=args)

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Do work
    mill_internal(args.config, args.sample)

    # Print output
    logger.info("Time taken: %.1f seconds" % (time.time() - st))
