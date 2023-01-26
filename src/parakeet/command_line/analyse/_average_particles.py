#
# parakeet.command_line.analyse.average_particles.py
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


__all__ = ["average_particles"]


# Get the logger
logger = logging.getLogger(__name__)


def get_description():
    """
    Get the program description

    """
    return "Perform sub tomogram averaging"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the parakeet.analyse.average_particles parser

    """

    # Initialise the parser
    if parser is None:
        parser = ArgumentParser(description=get_description())

    # Add some command line arguments
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        dest="config",
        required=True,
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
        default="rec.mrc",
        dest="rec",
        help="The filename for the reconstruction",
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
        "-psz",
        "--particle_size",
        type=int,
        default=0,
        dest="particle_size",
        help="The size of the particles extracted (px)",
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


def average_particles_impl(args):
    """
    Perform sub tomogram averaging

    """

    # Get the start time
    start_time = time.time()

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Do the work
    parakeet.analyse.average_particles(
        args.config,
        args.sample,
        args.rec,
        args.half1,
        args.half2,
        args.particle_size,
        args.num_particles,
    )

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))


def average_particles(args: List[str] = None):
    """
    Perform sub tomogram averaging

    """
    average_particles_impl(get_parser().parse_args(args=args))
