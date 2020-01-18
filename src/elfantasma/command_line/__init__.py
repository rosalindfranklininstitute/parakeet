#
# elfantasma.command_line.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import argparse
import gemmi
import logging
import numpy
import time
import elfantasma.io
import elfantasma.config
import elfantasma.microscope
import elfantasma.freeze
import elfantasma.sample
import elfantasma.scan
import elfantasma.simulation

# Get the logger
logger = logging.getLogger(__name__)


def configure_logging():
    """
    Configure the logging

    """
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": True,
            "handlers": {
                "stream": {
                    "level": "DEBUG",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                }
            },
            "loggers": {
                "elfantasma": {
                    "handlers": ["stream"],
                    "level": "DEBUG",
                    "propagate": True,
                }
            },
        }
    )


def main():
    """
    The main interface to elfantasma

    """

    # Get the start time
    start_time = time.time()

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Generate EM phantoms")

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
        "-o",
        "--output",
        type=str,
        default=None,
        dest="output",
        help="The filename for the simulation results",
    )
    parser.add_argument(
        "-p",
        "--phantom",
        choices=[
            "4v5d",
            "ribosomes_in_lamella",
            "ribosomes_in_cylinder",
            "single_ribosome_in_ice",
            "custom",
        ],
        default=None,
        dest="phantom",
        help="Choose the phantom to generate",
    )
    parser.add_argument(
        "--freeze",
        type=bool,
        default=None,
        dest="freeze",
        help="Freeze the sample in vitreous ice",
    )
    parser.add_argument(
        "--beam.flux",
        type=float,
        default=None,
        dest="beam_flux",
        help="The beam flux (None means infinite normalized)",
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
        "--sample.custom.filename",
        type=str,
        default=None,
        dest="sample_custom_filename",
        help="Choose the phantom to generate",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Configure some basic logging
    configure_logging()

    # Set the command line args in a dict
    command_line = {}
    if args.device is not None:
        command_line["device"] = args.device
    if args.output is not None:
        command_line["output"] = args.output
    if args.phantom is not None:
        command_line["phantom"] = args.phantom
    if args.freeze is not None:
        command_line["freeze"] = args.freeze
    if args.beam_flux is not None:
        command_line["microscope"] = {"beam": {"flux": args.beam_flux}}
    if args.cluster_max_workers is not None or args.cluster_method is not None:
        command_line["cluster"] = {}
    if args.cluster_max_workers is not None:
        command_line["cluster"]["max_workers"] = args.cluster_max_workers
    if args.cluster_method is not None:
        command_line["cluster"]["method"] = args.cluster_method
    if args.sample_custom_filename is not None:
        command_line["sample"] = {"custom": {"filename": args.sample_custom_filename}}

    # Load the full configuration
    config = elfantasma.config.load(args.config, command_line)

    # Print some options
    elfantasma.config.show(config)

    # Create the microscope
    microscope = elfantasma.microscope.new(**config["microscope"])

    # Create the sample
    sample = elfantasma.sample.new(config["phantom"], **config["sample"])

    # Create the scan
    scan = elfantasma.scan.new(**config["scan"])

    # Create the simulation
    simulation = elfantasma.simulation.new(
        microscope=microscope,
        sample=sample,
        scan=scan,
        device=config["device"],
        simulation=config["simulation"],
        cluster=config["cluster"],
    )

    # Create the writer
    logger.info(f"Opening file: {config['output']}")
    writer = elfantasma.io.new(config["output"], shape=simulation.shape)

    # Run the simulation
    simulation.run(writer)

    # Write some timing stats
    logger.info("Time taken: %.2f seconds" % (time.time() - start_time))


def show_config_main():
    """
    Show the full configuration

    """
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Show the configuration")

    # Add some command line arguments
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        dest="config",
        help="The yaml file to configure the simulation",
    )

    # Configure some basic logging
    configure_logging()

    # Parse the arguments
    config = elfantasma.config.load(parser.parse_args().config)

    # Print some options
    elfantasma.config.show(config, full=True)


def export(argv=None):
    """
    Convert the input file type to a different file type

    """
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Read a PDB file")

    # Add an argument for the filename
    parser.add_argument("filename", type=str, default=None, help="The input filename")

    # Add an argument for the filename
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        required=True,
        dest="output",
        help="The output filename",
    )
    parser.add_argument(
        "--transpose",
        type=bool,
        default=False,
        dest="transpose",
        help="Transpose the data",
    )
    parser.add_argument(
        "--rotation_range",
        type=str,
        default=None,
        dest="rotation_range",
        help="Select a rotation range",
    )

    # Parse the arguments
    args = parser.parse_args(argv)

    # Configure some basic logging
    configure_logging()

    # Read the input
    logger.info(f"Reading data from {args.filename}")
    reader = elfantasma.io.open(args.filename)

    # Get the shape and indices to read
    if args.rotation_range is not None:
        args.rotation_range = tuple(map(int, args.rotation_range.split(",")))
        indices = []
        for i in range(reader.shape[0]):
            angle = reader.angle[i]
            if angle >= args.rotation_range[0] and angle < args.rotation_range[1]:
                indices.append(i)
            else:
                logger.info(f"    Skipping image {i} because angle is out of range")
    else:
        indices = list(range(reader.shape[0]))

    # Set the dataset shape
    shape = (len(indices), reader.data.shape[1], reader.data.shape[2])

    # Create the write
    logger.info(f"Writing data to {args.output}")
    writer = elfantasma.io.new(args.output, shape=shape, dtype=reader.data.dtype.name)

    # If converting to images, determine min and max
    if writer.is_image_writer:
        logger.info("Computing min and max of dataset:")
        min_image = []
        max_image = []
        for i in indices:
            min_image.append(numpy.min(reader.data[i, :, :]))
            max_image.append(numpy.max(reader.data[i, :, :]))
            logger.info(
                "    Reading image %d: min/max: %.2f/%.2f"
                % (i, min_image[-1], max_image[-1])
            )
        writer.vmin = min(min_image)
        writer.vmax = max(max_image)
        logger.info("Min: %f" % writer.vmin)
        logger.info("Max: %f" % writer.vmax)

    # Write the data
    for j, i in enumerate(indices):
        logger.info(f"    Copying image {i} -> image {j}")
        image = reader.data[i, :, :]
        angle = reader.angle[i]
        position = reader.position[i]
        if args.transpose:
            image = image.T
            position = (position[1], position[0], position[2])
        writer.data[j, :, :] = image
        writer.angle[j] = angle
        writer.position[j] = position


def read_pdb():
    """
    Read the given PDB file and show the atom positions

    """

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Read a PDB file")

    # Add an argument for the filename
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        default=None,
        dest="filename",
        help="The path to the PDB file",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Check a filename has been given
    if args.filename == None:
        parser.print_help()
        exit(0)

    # Configure some basic logging
    configure_logging()

    # Read the structure
    structure = gemmi.read_structure(args.filename)

    # Iterate through atoms
    prefix = " " * 4
    logger.info("Structure: %s" % structure.name)
    for model in structure:
        logger.info("%sModel: %s" % (prefix, model.name))
        for chain in model:
            logger.info("%sChain: %s" % (prefix * 2, chain.name))
            for residue in chain:
                logger.info("%sResidue: %s" % (prefix * 3, residue.name))
                for atom in residue:
                    logger.info(
                        "%sAtom: %s, %f, %f, %f, %f, %f, %f"
                        % (
                            prefix * 4,
                            atom.element.name,
                            atom.pos.x,
                            atom.pos.y,
                            atom.pos.z,
                            atom.occ,
                            atom.charge,
                            elfantasma.sample.get_atom_sigma(atom),
                        )
                    )
