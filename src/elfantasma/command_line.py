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
import numpy
import time
import elfantasma.io
import elfantasma.config
import elfantasma.sample
import elfantasma.scan
import elfantasma.simulation


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

    # Set the command line args in a dict
    command_line = {}
    if args.device is not None:
        command_line["device"] = args.device
    if args.output is not None:
        command_line["output"] = args.output
    if args.phantom is not None:
        command_line["phantom"] = args.phantom
    if args.beam_flux is not None:
        command_line["beam"] = {"flux": args.beam_flux}
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

    # Create the sample
    sample = elfantasma.sample.new(config["phantom"], **config["sample"])

    # Create the scan
    scan = elfantasma.scan.new(**config["scan"])

    # Create the simulation
    simulation = elfantasma.simulation.new(
        sample,
        scan,
        device=config["device"],
        beam=config["beam"],
        detector=config["detector"],
        simulation=config["simulation"],
        cluster=config["cluster"],
    )

    # Create the writer
    print(f"Opening file: {config['output']}")
    writer = elfantasma.io.new(config["output"], shape=simulation.shape)

    # Run the simulation
    simulation.run(writer)

    # Write some timing stats
    print("Time taken: %.2f seconds" % (time.time() - start_time))


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

    # Parse the arguments
    config = elfantasma.config.load(parser.parse_args().config)

    # Print some options
    elfantasma.config.show(config, full=True)


def convert():
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
        "--electrons_per_pixel",
        type=float,
        default=None,
        dest="beam_flux",
        help="Multiply data by this value and get Poisson pixel counts",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Read the input
    print(f"Reading data from {args.filename}")
    reader = elfantasma.io.open(args.filename)

    # Create the write
    print(f"Writing data to {args.output}")
    writer = elfantasma.io.new(args.output, shape=reader.data.shape)

    # If converting to images, determine min and max
    if writer.is_image_writer:
        print("Computing min and max of dataset:")
        min_image = []
        max_image = []
        for i in range(reader.shape[0]):
            min_image.append(numpy.min(reader.data[i, :, :]))
            max_image.append(numpy.max(reader.data[i, :, :]))
            print(
                "    Reading image %d: min/max: %.2f/%.2f"
                % (i, min_image[-1], max_image[-1])
            )
        writer.vmin = min(min_image)
        writer.vmax = max(max_image)
        if args.electrons_per_pixel is not None:
            writer.vmin *= args.electons_per_pixel
            writer.vmax *= args.electons_per_pixel
        print("Min: %f" % writer.vmin)
        print("Max: %f" % writer.vmax)

    # Write the data
    for i in range(reader.shape[0]):
        print(f"    Copying image {i}")
        if args.electrons_per_pixel is None:
            image = reader.data[i, :, :]
        else:
            image = numpy.random.poisson(electrons_per_pixel * reader.data[i, :, :])
        writer.data[i, :, :] = image
        writer.angle[i] = reader.angle[i]
        writer.position[i] = reader.position[i]


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

    # Read the structure
    structure = gemmi.read_structure(args.filename)

    # Iterate through atoms
    prefix = " " * 4
    print("Structure: %s" % structure.name)
    for model in structure:
        print("%sModel: %s" % (prefix, model.name))
        for chain in model:
            print("%sChain: %s" % (prefix * 2, chain.name))
            for residue in chain:
                print("%sResidue: %s" % (prefix * 3, residue.name))
                for atom in residue:
                    print(
                        "%sAtom: %s, %f, %f, %f, %f, %f"
                        % (
                            prefix * 4,
                            atom.element.name,
                            atom.pos.x,
                            atom.pos.y,
                            atom.pos.z,
                            atom.occ,
                            atom.charge,
                        )
                    )


def create_sample():
    """
    Create a sample and save it

    """
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Create a sample and save it")

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
        default="sample.pickle",
        dest="output",
        help="The filename for the sample file",
    )
    parser.add_argument(
        "-p",
        "--phantom",
        choices=[
            "4v5d",
            "ribosomes_in_lamella",
            "ribosomes_in_cylinder",
            "single_ribosome_in_ice",
        ],
        default=None,
        dest="phantom",
        help="Choose the phantom to generate",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Set the command line args in a dict
    command_line = {}
    if args.phantom is not None:
        command_line["phantom"] = args.phantom

    # Load the configuration
    config = elfantasma.config.load(args.config, command_line)

    # Print some options
    elfantasma.config.show(config)

    # Create the sample
    sample = elfantasma.sample.new(config["phantom"], **config["sample"])

    # Write the sample to file
    print(f"Writing sample to {args.output}")
    sample.as_file(args.output)


if __name__ == "__main__":
    main()
