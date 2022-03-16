#
# parakeet.command_line.analyse.refine.py
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
import parakeet.analyse
import parakeet.io
import parakeet.command_line
import parakeet.config
import parakeet.microscope
import parakeet.sample

# Get the logger
logger = logging.getLogger(__name__)


def get_parser():
    """
    Get the parakeet.analyse.refine parser

    """

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Refine map and model")

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
        help="The filename for the sample",
    )
    parser.add_argument(
        "-r",
        "--rec",
        type=str,
        default="half1.mrc",
        dest="rec",
        help="The filename for the reconstruction",
    )

    return parser


def refine(args=None):
    """
    Refine against the model

    """

    # Get the start time
    start_time = time.time()

    # Get the parser
    parser = get_parser()

    # Parse the arguments
    args = parser.parse_args(args=args)

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Load the full configuration
    config = parakeet.config.load(args.config)

    # Print some options
    parakeet.config.show(config)

    # Do the sub tomogram averaging
    parakeet.analyse.refine(args.sample, args.rec)

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))
