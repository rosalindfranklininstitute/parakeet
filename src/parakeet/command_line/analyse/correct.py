#
# parakeet.command_line.analyse.correct.py
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
    Get the parakeet.analyse.correct parser

    """

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
    parser.add_argument(
        "-ndf",
        "--num-defocus",
        type=int,
        default=None,
        dest="num_defocus",
        help="Number of defoci that correspond to different depths through for which the sample will be 3D CTF corrected",
    )

    return parser


def correct_internal(config_file, image, corrected, num_defocus=1, device="gpu"):
    """
    Correct the images using 3D CTF correction

    """

    # Load the full configuration
    config = parakeet.config.load(config_file)

    # Set the command line args in a dict
    if device is not None:
        config.device = device

    # Print some options
    parakeet.config.show(config)

    # Create the microscope
    microscope = parakeet.microscope.new(**config.microscope.dict())

    # Do the reconstruction
    parakeet.analyse.correct(
        image,
        corrected,
        microscope=microscope,
        simulation=config.simulation.dict(),
        num_defocus=num_defocus,
        device=config.device,
    )


def correct(args=None):
    """
    Correct the images using 3D CTF correction

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
    correct_internal(
        args.config, args.image, args.corrected, args.num_defocus, args.device
    )

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))
