#
# parakeet.command_line.analyse.average_extracted_particles.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#


import logging
import time
import parakeet.analyse
import parakeet.io
import parakeet.command_line
import parakeet.config
import parakeet.microscope
import parakeet.sample
from argparse import ArgumentParser
from typing import List


__all__ = ["average_extracted_particles"]


# Get the logger
logger = logging.getLogger(__name__)


def get_description():
    """
    Get the program description

    """
    return "Perform sub tomogram averaging"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the parakeet.analyse.average_extracted_particles parser

    """

    # Initialise the parser
    if parser is None:
        parser = ArgumentParser(description=get_description())

    # Add some command line arguments
    parser.add_argument(
        "-p",
        "--particles",
        type=str,
        default="particles.h5",
        dest="particles",
        help="The filename for the particles",
    )
    parser.add_argument(
        "-h1",
        "--half1",
        type=str,
        default="half1.mrc",
        dest="half1",
        help="The filename for the particle average",
    )
    parser.add_argument(
        "-h2",
        "--half2",
        type=str,
        default="half2.mrc",
        dest="half2",
        help="The filename for the particle average",
    )
    parser.add_argument(
        "-n",
        "--num_particles",
        type=int,
        default=0,
        dest="num_particles",
        help="The number of particles to use",
    )

    return parser


def average_extracted_particles_impl(args):
    """
    Perform sub tomogram averaging

    """

    # Get the start time
    start_time = time.time()

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Do the work
    parakeet.analyse.average_extracted_particles(
        args.particles,
        args.half1,
        args.half2,
        args.num_particles,
    )

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))


def average_extracted_particles(args: List[str] = None):
    """
    Perform sub tomogram averaging

    """
    average_extracted_particles_impl(get_parser().parse_args(args=args))
