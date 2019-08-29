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
import elfantasma.data
import numpy


def create_4v5d_sample():
    """
    Create a sample with a Ribosome in Ice

    Returns:
        object: The atom data

    """

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

    # Get the minimum position
    min_x = min(x)
    min_y = min(y)
    min_z = min(z)

    # Translate the structure so that all coordinates are positive
    x -= min_x
    y -= min_y
    z -= min_z

    # Create sigma and region
    sigma = [0.085 for z in element]  # From multem HRTEM example
    region = [0 for z in element]

    # Return the atom data
    return zip(element, x, y, z, sigma, occ, region, charge)


def create_sample(name):
    """
    Create the sample

    Args:
        name (str): The name of the sample

    Returns:
        object: The test sample

    """
    return {"4v5d": create_4v5d_sample}[name]()
