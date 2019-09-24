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
import os.path
import argparse
import pickle
import gemmi
import numpy
import elfantasma.io
import elfantasma.sample
# import elfantasma.simulation


def show_input(args):
    """
    Print the command line arguments

    Args:
        args (object): The arguments object

    """
    print("Command line arguments:")
    for key, value in vars(args).items():
        print(f"    {key} = {value}")


def write_pickle(simulation, filename):
    """
    Write the simulated data to a python pickle file

    Args:
        simulated (object): The simulation object
        filename (str): The output filename

    """
    with open(filename, "wb") as outfile:
        pickle.dump(simulation.asdict(), outfile, protocol=2)


def main():
    """
    The main interface to elfantasma

    """
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Generate EM phantoms")

    # Add some command line arguments
    parser.add_argument(
        "-d,--device",
        choices=["cpu", "gpu"],
        default="gpu",
        dest="device",
        help="Choose the device to use",
    )
    parser.add_argument(
        "-o,--output",
        type=str,
        default="output.pickle",
        dest="output",
        help="The filename for the simulation results",
    )
    parser.add_argument(
        "-p,--phantom",
        choices=["4v5d"],
        default="4v5d",
        dest="phantom",
        help="Choose the phantom to generate",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Print some options
    show_input(args)

    # Create the sample
    sample = elfantasma.sample.create_sample(args.phantom)

    # Create the simulation
    simulation = elfantasma.simulation.create_simulation(sample, args.device)

    # Run the simulation
    simulation.run()

    # Write the simulated data to file
    write_file(simulation, args.output)

def convert():
    """
    Convert the input file type to a different file type

    """
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Read a PDB file")

    # Add an argument for the filename
    parser.add_argument(
        "filename",
        type=str,
        default=None,
        help="The input filename",
    )
    
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
    writer.data[:,:,:] = reader.data[:,:,:]
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


# if __name__ == "__main__":
#     main()




if __name__ == "__main__":

    writer = elfantasma.io.Writer(shape=(10, 100, 100))
    for i in range(10):
        image = numpy.random.random(100*100)
        image.shape = (100,100)
        writer.data[i,:,:] = image
        writer.angle[i] = i
    writer.as_file("hello.h5")

    
    reader = elfantasma.io.Reader.from_file("hello.h5")
    print(reader.data)

