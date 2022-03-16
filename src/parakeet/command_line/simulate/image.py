#
# parakeet.command_line.simulate.image.py
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
import numpy
import time
import parakeet.io
import parakeet.command_line
import parakeet.config
import parakeet.microscope
import parakeet.sample
import parakeet.scan
import parakeet.simulation

# Get the logger
logger = logging.getLogger(__name__)


def get_parser():
    """
    Get the parakeet.simulate.image parser

    """

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Simulate the image")

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
        "-o",
        "--optics",
        type=str,
        default="optics.h5",
        dest="optics",
        help="The filename for the optics",
    )
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        default="image.h5",
        dest="image",
        help="The filename for the image",
    )

    return parser


def image_internal(config_file, optics, image):
    """
    Simulate the image with noise

    """

    # Load the full configuration
    config = parakeet.config.load(config_file)

    # Print some options
    parakeet.config.show(config)

    # Create the microscope
    microscope = parakeet.microscope.new(**config.microscope.dict())

    # Create the exit wave data
    logger.info(f"Loading sample from {optics}")
    optics = parakeet.io.open(optics)

    # Create the scan
    scan = parakeet.scan.new(
        angles=optics.angle, positions=optics.position[:, 1], **config.scan.dict()
    )

    # Create the simulation
    simulation = parakeet.simulation.image(
        microscope=microscope,
        optics=optics,
        scan=scan,
        device=config.device,
        simulation=config.simulation.dict(),
        cluster=config.cluster.dict(),
    )

    # Create the writer
    logger.info(f"Opening file: {image}")
    writer = parakeet.io.new(
        image,
        shape=simulation.shape,
        pixel_size=simulation.pixel_size,
        dtype=numpy.float32,
    )

    # Run the simulation
    simulation.run(writer)


def image(args=None):
    """
    Simulate the image with noise

    """

    # Get the start time
    start_time = time.time()

    # Get parser
    parser = get_parser()

    # Parse the arguments
    args = parser.parse_args(args=args)

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Do the work
    image_internal(args.config, args.optics, args.image)

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))
