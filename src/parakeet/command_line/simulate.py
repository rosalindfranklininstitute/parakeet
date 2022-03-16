#
# parakeet.command_line.simulate.py
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


def get_projected_potential_parser():
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


def projected_potential(args=None):
    """
    Simulate the projected potential from the sample

    """

    # Get the start time
    start_time = time.time()

    # Get parser
    parser = get_projected_potential_parser()

    # Parse the arguments
    args = parser.parse_args(args=args)

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Load the full configuration
    config = parakeet.config.load(args.config)

    # Set the command line args in a dict
    if args.device is not None:
        config.device = args.device
    if args.cluster_max_workers is not None:
        config.cluster.max_workers = args.cluster_max_workers
    if args.cluster_method is not None:
        config.cluster.method = args.cluster_method

    # Print some options
    parakeet.config.show(config)

    # Create the microscope
    microscope = parakeet.microscope.new(**config.microscope.dict())

    # Create the sample
    logger.info(f"Loading sample from {args.sample}")
    sample = parakeet.sample.load(args.sample)

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

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))


def get_exit_wave_parser():
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


def exit_wave(args=None):
    """
    Simulate the exit wave from the sample

    """

    # Get the start time
    start_time = time.time()

    # Get exit wave parser
    parser = get_exit_wave_parser()

    # Parse the arguments
    args = parser.parse_args(args=args)

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Load the full configuration
    config = parakeet.config.load(args.config)

    # Set the command line args in a dict
    if args.device is not None:
        config.device = args.device
    if args.cluster_max_workers is not None:
        config.cluster.max_workers = args.cluster_max_workers
    if args.cluster_method is not None:
        config.cluster.method = args.cluster_method

    # Print some options
    parakeet.config.show(config)

    # Create the microscope
    microscope = parakeet.microscope.new(**config.microscope.dict())

    # Create the sample
    logger.info(f"Loading sample from {args.sample}")
    sample = parakeet.sample.load(args.sample)

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
    logger.info(f"Opening file: {args.exit_wave}")
    writer = parakeet.io.new(
        args.exit_wave,
        shape=simulation.shape,
        pixel_size=simulation.pixel_size,
        dtype=numpy.complex64,
    )

    # Run the simulation
    simulation.run(writer)

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))


def get_optics_parser():
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


def optics(args=None):
    """
    Simulate the optics

    """

    # Get the start time
    start_time = time.time()

    # Get the parser
    parser = get_optics_parser()

    # Parse the arguments
    args = parser.parse_args(args=args)

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Load the full configuration
    config = parakeet.config.load(args.config)

    # Set the command line args in a dict
    if args.device is not None:
        config.device = args.device
    if args.cluster_max_workers is not None:
        config.cluster.max_workers = args.cluster_max_workers
    if args.cluster_method is not None:
        config.cluster.method = args.cluster_method

    # Print some options
    parakeet.config.show(config)

    # Create the microscope
    microscope = parakeet.microscope.new(**config.microscope.dict())

    # Create the exit wave data
    logger.info(f"Loading sample from {args.exit_wave}")
    exit_wave = parakeet.io.open(args.exit_wave)

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
    logger.info(f"Opening file: {args.optics}")
    writer = parakeet.io.new(
        args.optics,
        shape=simulation.shape,
        pixel_size=simulation.pixel_size,
        dtype=numpy.float32,
    )

    # Run the simulation
    simulation.run(writer)

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))


def get_ctf_parser():
    """
    Get the parakeet.simulate.ctf parser

    """

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Simulate the ctf")

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
        type=str,
        default="ctf.h5",
        dest="output",
        help="The filename for the output",
    )

    return parser


def ctf(args=None):
    """
    Simulate the ctf

    """

    # Get the start time
    start_time = time.time()

    # Get the parser
    parser = get_parser()

    # Parse the arguments
    args = parser.parse_args(args=args)

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Load the full configuration
    config = parakeet.config.load(args.config)

    # Print some options
    parakeet.config.show(config)

    # Create the microscope
    microscope = parakeet.microscope.new(**config.microscope.dict())

    # Create the simulation
    simulation = parakeet.simulation.ctf(
        microscope=microscope, simulation=config.simulation.dict()
    )

    # Create the writer
    logger.info(f"Opening file: {args.output}")
    writer = parakeet.io.new(
        args.output,
        shape=simulation.shape,
        pixel_size=simulation.pixel_size,
        dtype=numpy.complex64,
    )

    # Run the simulation
    simulation.run(writer)

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))


def get_image_parser():
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


def image(args=None):
    """
    Simulate the image with noise

    """

    # Get the start time
    start_time = time.time()

    # Get parser
    parser = get_image_parser()

    # Parse the arguments
    args = parser.parse_args(args=args)

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Load the full configuration
    config = parakeet.config.load(args.config)

    # Print some options
    parakeet.config.show(config)

    # Create the microscope
    microscope = parakeet.microscope.new(**config.microscope.dict())

    # Create the exit wave data
    logger.info(f"Loading sample from {args.optics}")
    optics = parakeet.io.open(args.optics)

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
    logger.info(f"Opening file: {args.image}")
    writer = parakeet.io.new(
        args.image,
        shape=simulation.shape,
        pixel_size=simulation.pixel_size,
        dtype=numpy.float32,
    )

    # Run the simulation
    simulation.run(writer)

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))


def get_simple_parser():
    """
    Get the parakeet.simulate.simple parser

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
        "--output",
        type=str,
        default="output.h5",
        dest="output",
        help="The filename for the output",
    )
    parser.add_argument(
        "atoms",
        type=str,
        default=None,
        nargs="?",
        help="The filename for the input atoms",
    )

    return parser


def simple():
    """
    Simulate the image

    """

    # Get the start time
    start_time = time.time()

    # Get parser
    parser = get_simple_parser()

    # Parse the arguments
    args = parser.parse_args()

    # Configure some basic logging
    parakeet.command_line.configure_logging()

    # Load the full configuration
    config = parakeet.config.load(args.config)

    # Print some options
    parakeet.config.show(config)

    # Create the microscope
    microscope = parakeet.microscope.new(**config.microscope.dict())

    # Create the exit wave data
    logger.info(f"Loading sample from {args.atoms}")
    atoms = parakeet.sample.AtomData.from_text_file(args.atoms)

    # Create the simulation
    simulation = parakeet.simulation.simple(
        microscope=microscope,
        atoms=atoms,
        device=config.device,
        simulation=config.simulation,
    )

    # Create the writer
    logger.info(f"Opening file: {args.output}")
    writer = parakeet.io.new(
        args.output,
        shape=simulation.shape,
        pixel_size=simulation.pixel_size,
        dtype=numpy.complex64,
    )

    # Run the simulation
    simulation.run(writer)

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))
