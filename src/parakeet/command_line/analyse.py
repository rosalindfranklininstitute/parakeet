#
# parakeet.command_line.analyse.py
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


def reconstruct(args=None):
    """
    Reconstruct the volume

    """

    # Get the start time
    start_time = time.time()

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Reconstruct the volume")

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
        "-d",
        "--device",
        choices=["cpu", "gpu"],
        default=None,
        dest="device",
        help="Choose the device to use",
    )
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        default="image.mrc",
        dest="image",
        help="The filename for the image",
    )
    parser.add_argument(
        "-r",
        "--rec",
        type=str,
        default="rec.mrc",
        dest="rec",
        help="The filename for the reconstruction",
    )

    # Parse the arguments
    args = parser.parse_args(args=args)

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Set the command line args in a dict
    command_line = {}
    if args.device is not None:
        command_line["device"] = args.device

    # Load the full configuration
    config = parakeet.config.load(args.config, command_line)

    # Print some options
    parakeet.config.show(config)

    # Create the microscope
    microscope = parakeet.microscope.new(**config["microscope"])

    # Do the reconstruction
    parakeet.analyse.reconstruct(
        args.image,
        args.rec,
        microscope=microscope,
        simulation=config["simulation"],
        device=config["device"],
    )

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))


def correct(args=None):
    """
    Correct the images using 3D CTF correction

    """

    # Get the start time
    start_time = time.time()

    # Create the argument parser
    parser = argparse.ArgumentParser(description="3D CTF correction of the images")

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
        "-d",
        "--device",
        choices=["cpu", "gpu"],
        default=None,
        dest="device",
        help="Choose the device to use",
    )
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        default="image.mrc",
        dest="image",
        help="The filename for the image",
    )
    parser.add_argument(
        "-cr",
        "--corrected",
        type=str,
        default="corrected_image.mrc",
        dest="corrected",
        help="The filename for the corrected image",
    )

    # Parse the arguments
    args = parser.parse_args(args=args)

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Set the command line args in a dict
    command_line = {}
    if args.device is not None:
        command_line["device"] = args.device

    # Load the full configuration
    config = parakeet.config.load(args.config, command_line)

    # Print some options
    parakeet.config.show(config)

    # Create the microscope
    microscope = parakeet.microscope.new(**config["microscope"])

    # Do the reconstruction
    parakeet.analyse.correct(
        args.image,
        args.corrected,
        microscope=microscope,
        simulation=config["simulation"],
        device=config["device"],
    )

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))


def average_particles(args=None):
    """
    Perform sub tomogram averaging

    """

    # Get the start time
    start_time = time.time()

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

    # Parse the arguments
    args = parser.parse_args(args=args)

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Set the command line args in a dict
    command_line = {}

    # Load the full configuration
    config = parakeet.config.load(args.config, command_line)

    # Print some options
    parakeet.config.show(config)

    # Do the sub tomogram averaging
    parakeet.analyse.average_particles(
        config["scan"],
        args.sample,
        args.rec,
        args.half1,
        args.half2,
        args.particle_size,
        args.num_particles,
    )

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))


def average_all_particles(args=None):
    """
    Perform sub tomogram averaging

    """

    # Get the start time
    start_time = time.time()

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
        type=int,
        default=0,
        dest="particle_size",
        help="The size of for the particles extracted",
    )

    # Parse the arguments
    args = parser.parse_args(args=args)

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Set the command line args in a dict
    command_line = {}

    # Load the full configuration
    config = parakeet.config.load(args.config, command_line)

    # Print some options
    parakeet.config.show(config)

    # Do the sub tomogram averaging
    parakeet.analyse.average_all_particles(
        config["scan"], args.sample, args.rec, args.average, args.particle_size
    )

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))


def extract_particles(args=None):
    """
    Perform sub tomogram extraction

    """

    # Get the start time
    start_time = time.time()

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Perform sub tomogram extraction")

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
        help="The size of for the particles extracted",
    )

    # Parse the arguments
    args = parser.parse_args(args=args)

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Set the command line args in a dict
    command_line = {}

    # Load the full configuration
    config = parakeet.config.load(args.config, command_line)

    # Print some options
    parakeet.config.show(config)

    # Do the sub tomogram averaging
    parakeet.analyse.extract_particles(
        config["scan"], args.sample, args.rec, args.particles, args.particle_size
    )

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))


def refine(args=None):
    """
    Refine against the model

    """

    # Get the start time
    start_time = time.time()

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

    # Parse the arguments
    args = parser.parse_args(args=args)

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Set the command line args in a dict
    command_line = {}

    # Load the full configuration
    config = parakeet.config.load(args.config, command_line)

    # Print some options
    parakeet.config.show(config)

    # Do the sub tomogram averaging
    parakeet.analyse.refine(args.sample, args.rec)

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))
