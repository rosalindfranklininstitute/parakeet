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
import h5py
import mrcfile
import numpy
import elfantasma.sample
import elfantasma.simulation


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


def write_mrcfile(simulation, filename):
    """
    Write the simulated data to a mrc file

    Args:
        simulated (object): The simulation object
        filename (str): The output filename

    """
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(simulation.data)


def write_nexus(simulation, filename):
    """
    Write the simulated data to a nexus file

    Args:
        simulated (object): The simulation object
        filename (str): The output filename

    """
    with h5py.File(filename, "w") as outfile:

        # Create the entry
        entry = outfile.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry["definition"] = "NXtomo"

        # Create the instrument
        instrument = entry.create_group("instrument")
        instrument.attrs["NX_class"] = "NXinstrument"

        # Create the detector
        detector = instrument.create_group("detector")
        detector.attrs["NX_class"] = "NXdetector"
        detector["data"] = simulation.data
        detector["image_key"] = numpy.zeros(shape=(len(simulation.output_multislice),))

        # Create the sample
        sample = entry.create_group("sample")
        sample.attrs["NX_class"] = "NXsample"
        sample["name"] = "elfantasma-simulation"
        sample["rotation_angle"] = simulation.angles

        # Create the data
        data = entry.create_group("data")
        data["data"] = detector["data"]
        data["rotation_angle"] = sample["rotation_angle"]
        data["image_key"] = detector["image_key"]


def write_file(simulation, filename):
    """
    Write the simulated data to file

    Args:
        simulated (object): The simulation object
        filename (str): The output filename

    """
    print(f"Saving simulation results to {filename}")
    extension = os.path.splitext(filename)[1]
    if extension in [".p", ".pkl", ".pickle"]:
        write_pickle(simulation, filename)
    elif extension in [".mrc"]:
        write_mrcfile(simulation, filename)
    elif extension in [".h5", ".hdf5", ".nx", ".nxs", ".nexus", "nxtomo"]:
        write_nexus(simulation, filename)
    else:
        raise RuntimeError(f"File with unknown extension: {filename}")


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
