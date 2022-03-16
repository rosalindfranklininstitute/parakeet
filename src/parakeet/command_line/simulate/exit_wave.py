#
# parakeet.command_line.simulate.exit_wave.py
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
from math import pi

# Get the logger
logger = logging.getLogger(__name__)


def get_parser():
    """
    Get the parakeet.simulate.exit_wave parser

    """

    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="Simulate the exit wave from the sample"
    )

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
        "-s",
        "--sample",
        type=str,
        default="sample.h5",
        dest="sample",
        help="The filename for the sample",
    )
    parser.add_argument(
        "-e",
        "--exit_wave",
        type=str,
        default="exit_wave.h5",
        dest="exit_wave",
        help="The filename for the exit wave",
    )

    return parser


def exit_wave_internal(
    config_file,
    sample,
    exit_wave,
    device="gpu",
    cluster_method=None,
    cluster_max_workers=1,
):
    """
    Simulate the exit wave from the sample

    """

    # Load the full configuration
    config = parakeet.config.load(config_file)

    # Set the command line args in a dict
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

    # Create the sample
    logger.info(f"Loading sample from {sample}")
    sample = parakeet.sample.load(sample)

    # Create the scan
    if config.scan.step_pos == "auto":
        radius = sample.shape_radius
        config.scan.step_pos = config.scan.step_angle * radius * pi / 180.0
    scan = parakeet.scan.new(**config.scan.dict())

    # Create the simulation
    simulation = parakeet.simulation.exit_wave(
        microscope=microscope,
        sample=sample,
        scan=scan,
        device=config.device,
        simulation=config.simulation.dict(),
        cluster=config.cluster.dict(),
    )

    # Create the writer
    logger.info(f"Opening file: {exit_wave}")
    writer = parakeet.io.new(
        exit_wave,
        shape=simulation.shape,
        pixel_size=simulation.pixel_size,
        dtype=numpy.complex64,
    )

    # Run the simulation
    simulation.run(writer)


def exit_wave(args=None):
    """
    Simulate the exit wave from the sample

    """

    # Get the start time
    start_time = time.time()

    # Get exit wave parser
    parser = get_parser()

    # Parse the arguments
    args = parser.parse_args(args=args)

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Do the work
    exit_wave_internal(
        args.config,
        args.sample,
        args.exit_wave,
        args.device,
        args.cluster_method,
        args.cluster_max_workers,
    )

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))
