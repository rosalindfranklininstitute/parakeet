#
# parakeet.command_line.simulate.optics.py
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
    Get the parakeet.simulate.optics parser

    """

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Simulate the optics")

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
        "--cluster.max_workers",
        type=int,
        default=None,
        dest="cluster_max_workers",
        help="The maximum number of worker processes",
    )
    parser.add_argument(
        "--cluster.method",
        type=str,
        choices=["sge"],
        default=None,
        dest="cluster_method",
        help="The cluster method to use",
    )
    parser.add_argument(
        "-e",
        "--exit_wave",
        type=str,
        default="exit_wave.h5",
        dest="exit_wave",
        help="The filename for the exit wave",
    )
    parser.add_argument(
        "-o",
        "--optics",
        type=str,
        default="optics.h5",
        dest="optics",
        help="The filename for the optics",
    )

    return parser


def optics_internal(
    config_file,
    exit_wave,
    optics,
    device="gpu",
    cluster_method=None,
    cluster_max_workers=1,
):
    """
    Simulate the optics

    """

    # Load the full configuration
    config = parakeet.config.load(config_file)

    # Set the device in a dict
    if device is not None:
        config.device = device
    if cluster_max_workers is not None:
        config.cluster.max_workers = cluster_max_workers
    if cluster_method is not None:
        config.cluster.method = cluster_method

    # Print some options
    parakeet.config.show(config)

    # Create the microscope
    microscope = parakeet.microscope.new(**config.microscope.dict())

    # Create the exit wave data
    logger.info(f"Loading sample from {exit_wave}")
    exit_wave = parakeet.io.open(exit_wave)

    # Create the scan
    scan = parakeet.scan.new(
        angles=exit_wave.angle, positions=exit_wave.position[:, 1], **config.scan.dict()
    )

    # Create the simulation
    simulation = parakeet.simulation.optics(
        microscope=microscope,
        exit_wave=exit_wave,
        scan=scan,
        device=config.device,
        simulation=config.simulation.dict(),
        sample=config.sample.dict(),
        cluster=config.cluster.dict(),
    )

    # Create the writer
    logger.info(f"Opening file: {optics}")
    writer = parakeet.io.new(
        optics,
        shape=simulation.shape,
        pixel_size=simulation.pixel_size,
        dtype=numpy.float32,
    )

    # Run the simulation
    simulation.run(writer)


def optics(args=None):
    """
    Simulate the optics

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
    optics_internal(
        args.config,
        args.exit_wave,
        args.optics,
        args.device,
        args.cluster_method,
        args.cluster_max_workers,
    )

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))
