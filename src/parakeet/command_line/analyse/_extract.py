#
# parakeet.command_line.analyse.extract.py
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


__all__ = ["extract"]


# Get the logger
logger = logging.getLogger(__name__)


def get_description():
    """
    Get the program description

    """
    return "Perform sub tomogram extraction"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the parakeet.analyse.extract parser

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
        "-pm",
        "--particle_map",
        type=str,
        default="particle_map.h5",
        dest="particles",
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
        "-psm",
        "--particle_sampling",
        type=int,
        default=1,
        dest="particle_sampling",
        help="The sampling of the particle volume (factor of 2)",
    )

    return parser


def extract_impl(args):
    """
    Perform sub tomogram extraction

    """

    # Get the start time
    start_time = time.time()

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Do the work
    parakeet.analyse.extract(
        args.config,
        args.sample,
        args.rec,
        args.particles,
        args.particle_size,
        args.particle_sampling,
    )

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))


def extract(args: List[str] = None):
    """
    Perform sub tomogram extraction

    """
    extract_impl(get_parser().parse_args(args=args))
