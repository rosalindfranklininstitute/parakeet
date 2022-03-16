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
    Get the parakeet.analyse.average_particles parser

    """

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Perform sub tomogram averaging")

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
        help="The size of for the particles extracted",
    )
    parser.add_argument(
        "-n",
        "--num_particles",
        type=int,
        default=0,
        dest="num_particles",
        help="The nunber of particles to use",
    )

    return parser


def average_particles_internal(
    config_file, sample, rec, half1, half2, particle_size, num_particles
):
    """
    Perform sub tomogram averaging

    """

    # Load the full configuration
    config = parakeet.config.load(config_file)

    # Print some options
    parakeet.config.show(config)

    # Do the sub tomogram averaging
    parakeet.analyse.average_particles(
        config.scan.dict(),
        sample,
        rec,
        half1,
        half2,
        particle_size,
        num_particles,
    )


def average_particles(args=None):
    """
    Perform sub tomogram averaging

    """

    # Get the start time
    start_time = time.time()

    # Get the parser
    parser = get_parser()

    # Parse the arguments
    args = parser.parse_args(args=args)

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Do the work
    average_particles_internal(
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
