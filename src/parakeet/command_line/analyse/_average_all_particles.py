#
# parakeet.command_line.analyse.average_all_particles.py
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


__all__ = ["average_all_particles"]


# Get the logger
logger = logging.getLogger(__name__)


def get_description():
    """
    Get the program description

    """
    return "Perform sub tomogram averaging"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the parakeet.analyse.average_all_particles parser

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
        "-avm",
        "--average_map",
        type=str,
        default="average_map.mrc",
        dest="average",
        help="The filename for the particle average",
    )
    parser.add_argument(
        "-psz",
        "--particle_size",
        type=lambda x: tuple(x.split(":")),
        default=None,
        action="extend",
        nargs="+",
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


def average_all_particles_impl(args):
    """
    Perform sub tomogram averaging

    """

    # Get the start time
    start_time = time.time()

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Get the particle size as a dictionary. If no particle size is set then
    # set it to 0. Otherwise, parse the "name:size" strings. If the length is 1
    # and it is an integer then set as integer, otherwise convert into a
    # dictionary
    if args.particle_size is None:
        args.particle_size = 0
    else:
        args.particle_size = [
            (x[0], int(x[1])) if len(x) == 2 else int(x[0]) for x in args.particle_size
        ]
        if len(args.particle_size) == 1 and isinstance(args.particle_size[0], int):
            args.particle_size = args.particle_size[0]
        else:
            args.particle_size = dict(args.particle_size)

    # Do the work
    parakeet.analyse.average_all_particles(
        args.config,
        args.sample,
        args.rec,
        args.average,
        args.particle_size,
        args.num_particles,
    )

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))


def average_all_particles(args: List[str] = None):
    """
    Perform sub tomogram averaging

    """
    average_all_particles_impl(get_parser().parse_args(args=args))
