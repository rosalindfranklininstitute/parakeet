#
# parakeet.command_line.simulate.projected_potential.py
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
    Get the parser for parakeet.simulate.projected_potential

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

    return parser


def projected_potential_internal(
    config_file, sample, device="gpu", cluster_method=None, cluster_max_workers=1
):
    """
    Simulate the projected potential from the sample

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
    if scan.positions[-1] > sample.containing_box[1][0]:
        raise RuntimeError("Scan goes beyond sample containing box")

    # Create the simulation
    simulation = parakeet.simulation.projected_potential(
        microscope=microscope,
        sample=sample,
        scan=scan,
        device=config.device,
        simulation=config.simulation,
        cluster=config.cluster,
    )

    # Run the simulation
    simulation.run()


def projected_potential(args=None):
    """
    Simulate the projected potential from the sample

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
    projected_potential_internal(
        args.config,
        args.sample,
        args.device,
        args.cluster_method,
        args.cluster_max_workers,
    )

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))
