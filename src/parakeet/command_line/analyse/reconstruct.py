#
# parakeet.command_line.analyse.reconstruct.py
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
    Get the parakeet.analyse.reconstruct parser

    """

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

    return parser


def reconstruct_internal(config_file, image, rec, device="gpu"):
    """
    Reconstruct the volume

    """

    # Load the full configuration
    config = parakeet.config.load(config_file)

    # Set the device
    if device is not None:
        config.device = device

    # Print some options
    parakeet.config.show(config)

    # Create the microscope
    microscope = parakeet.microscope.new(**config.microscope.dict())

    # Do the reconstruction
    parakeet.analyse.reconstruct(
        image,
        rec,
        microscope=microscope,
        simulation=config.simulation.dict(),
        device=config.device,
    )


def reconstruct(args=None):
    """
    Reconstruct the volume

    """

    # Get the start time
    start_time = time.time()

    # Get the parser
    parser = get_parser()

    # Parse the arguments
    args = parser.parse_args(args=args)

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Do the reconstruction
    reconstruct_internal(args.config, args.image, args.rec, args.device)

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))
