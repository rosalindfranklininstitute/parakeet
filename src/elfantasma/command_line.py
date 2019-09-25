#
# elfantasma.command_line.py
#
# Copyright (C) 2019 Diamond Light Source
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import argparse
import gemmi
import yaml
import elfantasma.io
import elfantasma.config
import elfantasma.sample
import elfantasma.scan
import elfantasma.simulation


def load_config(args):
    """
    Load the configuration from the various inputs

    Args:
        args (object): The command line arguments

    Returns:
        dict: The configuration dictionary

    """

    # Get the command line arguments
    command_line = dict(
        (k, v) for k, v in vars(args).items() if k != "config" and v != None
    )

    # If the yaml configuration is set then merge the configuration
    if args.config:
        with open(args.config) as infile:
            config_file = yaml.safe_load(infile)
    else:
        config_file = {}

    # Get the configuration
    config = elfantasma.config.deepmerge(
        elfantasma.config.default_config(),
        elfantasma.config.deepmerge(config_file, command_line),
    )

    # Return the config
    return config


def show_config(config, full=False):
    """
    Print the command line arguments

    Args:
        config (object): The configuration object

    """
    if full == False:
        config = elfantasma.config.difference(
            elfantasma.config.default_config(), config
        )
    print("Configuration:")
    print(
        "\n".join([f"    {line}" for line in yaml.dump(config, indent=4).split("\n")])
    )


def main():
    """
    The main interface to elfantasma

    """
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Generate EM phantoms")

    # Add some command line arguments
    parser.add_argument(
        "-c,--config",
        type=str,
        default=None,
        dest="config",
        help="The yaml file to configure the simulation",
    )
    parser.add_argument(
        "-d,--device",
        choices=["cpu", "gpu"],
        default=None,
        dest="device",
        help="Choose the device to use",
    )
    parser.add_argument(
        "-o,--output",
        type=str,
        default=None,
        dest="output",
        help="The filename for the simulation results",
    )
    parser.add_argument(
        "-p,--phantom",
        choices=["4v5d"],
        default=None,
        dest="phantom",
        help="Choose the phantom to generate",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        dest="max_workers",
        help="The maximum number of worker processes",
    )
    parser.add_argument(
        "--mp_method",
        type=str,
        choices=["multiprocessing", "sge"],
        default=None,
        dest="mp_method",
        help="The multiprocessing method to use",
    )

    # Parse the arguments
    config = load_config(parser.parse_args())

    # Print some options
    show_config(config)

    # Create the sample
    sample = elfantasma.sample.create_sample(config["phantom"])

    # Create the scan
    scan = elfantasma.scan.create_scan(**config["scan"])

    # Create the simulation
    simulation = elfantasma.simulation.create_simulation(
        sample,
        scan,
        device=config["device"],
        beam=config["beam"],
        detector=config["detector"],
        simulation=config["simulation"],
        multiprocessing={
            "mp_method": config["mp_method"],
            "max_workers": config["max_workers"],
        },
    )

    # Create the writer
    writer = elfantasma.io.Writer()

    # Run the simulation
    simulation.run(writer)

    # Write the simulated data to file
    print(f"Writing data to {config['output']}")
    writer.as_file(config["output"])


def show_config_main():
    """
    Show the full configuration

    """
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Show the configuration")

    # Add some command line arguments
    parser.add_argument(
        "-c,--config",
        type=str,
        default=None,
        dest="config",
        help="The yaml file to configure the simulation",
    )

    # Parse the arguments
    config = load_config(parser.parse_args())

    # Print some options
    show_config(config, full=True)


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
        "-o,--output",
        type=str,
        default=None,
        required=True,
        dest="output",
        help="The output filename",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Read the input
    print(f"Reading data from {args.filename}")
    reader = elfantasma.io.Reader.from_file(args.filename)

    # Create the write
    print(f"Writing data to {args.output}")
    writer = elfantasma.io.Writer(shape=reader.data.shape)
    writer.data[:, :, :] = reader.data[:, :, :]
    writer.angle[:] = reader.angle[:]
    writer.as_file(args.output)


def read_pdb():
    """
    Read the given PDB file and show the atom positions

    """

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Read a PDB file")

    # Add an argument for the filename
    parser.add_argument(
        "-f,--filename",
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


if __name__ == "__main__":
    main()
