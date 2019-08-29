#
# elfantasma.phantom.py
#
# Copyright (C) 2019 Diamond Light Source
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import gemmi
import numpy
import elfantasma.data


class Sample(object):
    """
    A class to wrap the sample information

    """

    def __init__(self, atom_data, length_x, length_y, length_z):
        """
        Initialise the sample

        Args:
            atom_data (list): A list of tuples containing atom data
            length_x (float): The x size of the sample in A
            length_y (float): The y size of the sample in A
            length_z (float): The z size of the sample in A

        """
        self.atom_data = atom_data
        self.length_x = length_x
        self.length_y = length_y
        self.length_z = length_z


def create_4v5d_sample():
    """
    Create a sample with a Ribosome in Ice

    Returns:
        object: The atom data

    """
    print("Creating sample 4v5d")

    # Get the filename of the 4v5d.cif file
    filename = elfantasma.data.get_path("4v5d.cif")

    # Read the structure
    structure = gemmi.read_structure(filename)

    # Iterate through atoms and create the atom data
    element = []
    x = []
    y = []
    z = []
    occ = []
    charge = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    element.append(atom.element.atomic_number)
                    x.append(atom.pos.x)
                    y.append(atom.pos.y)
                    z.append(atom.pos.z)
                    occ.append(atom.occ)
                    charge.append(atom.charge)

    # Cast to numpy array
    x = numpy.array(x, dtype=numpy.float64)
    y = numpy.array(y, dtype=numpy.float64)
    z = numpy.array(z, dtype=numpy.float64)

    # Get the min and max atom positions
    min_x = min(x)
    min_y = min(y)
    min_z = min(z)
    max_x = max(x)
    max_y = max(y)
    max_z = max(z)

    # Set the total size of the sample
    length_x = 1.0 * (max_x - min_x)
    length_y = 1.0 * (max_y - min_y)
    length_z = 1.0 * (max_z - min_z)

    # Translate the structure so that it is centred in the sample
    x += (length_x - (max_x + min_x)) / 2.0
    y += (length_y - (max_y + min_y)) / 2.0
    z += (length_z - (max_z + min_z)) / 2.0

    # Create sigma and region
    sigma = [0.085 for number in element]  # From multem HRTEM example
    region = [0 for number in element]

    # Print some sample information
    print("Sample information:")
    print("  # atoms: %d" % len(element))
    print("  Min x:   %.2f" % min(x))
    print("  Min y:   %.2f" % min(y))
    print("  Min z:   %.2f" % min(z))
    print("  Max x:   %.2f" % max(x))
    print("  Max y:   %.2f" % max(y))
    print("  Max z:   %.2f" % max(z))
    print("  Len x:   %.2f" % length_x)
    print("  Len y:   %.2f" % length_y)
    print("  Len z:   %.2f" % length_z)

    # Return the atom data
    return Sample(
        list(zip(element, x, y, z, sigma, occ, region, charge)),
        length_x,
        length_y,
        length_z,
    )


def create_sample(name):
    """
    Create the sample

    Args:
        name (str): The name of the sample

    Returns:
        object: The test sample

    """
    return {"4v5d": create_4v5d_sample}[name]()
