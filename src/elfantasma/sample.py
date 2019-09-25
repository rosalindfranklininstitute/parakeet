#
# elfantasma.phantom.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import gemmi
import numpy
import scipy.spatial.transform
import elfantasma.data
from math import acos, cos, pi, sin


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


def create_4v5d_sample(length_x=None, length_y=None, length_z=None):
    """
    Create a sample with a Ribosome

    If the size of the sample is not specified then the sample will be 2x the
    size of the ribosome and the ribosome will be positioned in the centre

    Args:
        length_x (float): The X size of the sample (units: A)
        length_y (float): The Y size of the sample (units: A)
        length_z (float): The Z size of the sample (units: A)

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
    if length_x is None:
        length_x = 2.0 * (max_x - min_x)
    if length_y is None:
        length_y = 2.0 * (max_y - min_y)
    if length_z is None:
        length_z = 2.0 * (max_z - min_z)
    assert length_x > (max_x - min_x)
    assert length_y > (max_y - min_y)
    assert length_z > (max_z - min_z)

    # Translate the structure so that it is centred in the sample
    x += (length_x - (max_x + min_x)) / 2.0
    y += (length_y - (max_y + min_y)) / 2.0
    z += (length_z - (max_z + min_z)) / 2.0

    # Create sigma and region
    sigma = [0.085 for number in element]  # From multem HRTEM example
    region = [0 for number in element]

    # Print some sample information
    print("Sample information:")
    print("    # atoms: %d" % len(element))
    print("    Min x:   %.2f" % min(x))
    print("    Min y:   %.2f" % min(y))
    print("    Min z:   %.2f" % min(z))
    print("    Max x:   %.2f" % max(x))
    print("    Max y:   %.2f" % max(y))
    print("    Max z:   %.2f" % max(z))
    print("    Len x:   %.2f" % length_x)
    print("    Len y:   %.2f" % length_y)
    print("    Len z:   %.2f" % length_z)

    # Return the atom data
    return Sample(
        list(zip(element, x, y, z, sigma, occ, region, charge)),
        length_x,
        length_y,
        length_z,
    )


def distribute_cubes_randomly(length_x, length_y, length_z, size, n, max_tries=100):
    """
    Find n random non overlapping positions for cubes within a volume

    Args:
        length_x (float): The X size of the box
        length_y (float): The Y size of the box
        length_z (float): The Z size of the box
        size (float): The size of the cube
        n (int): The number of cubes
        max_tries (int): The maximum tries per cube

    Returns:
        list: A list of centre positions

    """

    # Check if the cube overlaps with any other
    def overlapping(positions, q):
        for p in positions:
            p0 = p - size / 2
            p1 = p + size / 2
            q0 = q - size / 2
            q1 = q + size / 2
            if not (
                q0[0] > p1[0]
                or q1[0] < p0[0]
                or q0[1] > p1[1]
                or q1[1] < p0[1]
                or q0[2] > p1[2]
                or q1[2] < p0[2]
            ):
                return True
        return False

    # The bounds to search in
    lower = (size / 2, size / 2, size / 2)
    upper = (length_x - size / 2, length_y - size / 2, length_z - size / 2)
    assert lower[0] < upper[0]
    assert lower[1] < upper[1]
    assert lower[2] < upper[2]

    # Loop until we have added enough cubes
    positions = []
    num_tries = 0
    while len(positions) < n:
        q = numpy.random.uniform(lower, upper)
        if len(positions) == 0 or not overlapping(positions, q):
            positions.append(q)
        num_tries += 1
        if num_tries > max_tries:
            raise RuntimeError(f"Unable to place {n} cubes")

    # Return the cube positions
    return positions


def random_uniform_rotation():
    """
    Return a uniform rotation vector sampled on a sphere

    Returns:
        vector: The rotation vector

    """
    u1, u2, u3 = numpy.random.uniform(0, 1, 3)
    theta = acos(2 * u1 - 1)
    phi = 2 * pi * u2
    x = sin(theta) * cos(phi)
    y = sin(theta) * sin(phi)
    z = cos(theta)
    vector = numpy.array((x, y, z))
    vector /= numpy.linalg.norm(vector)
    vector *= 2 * pi * u3
    return vector


def create_ribosomes_in_lamella_sample(
    length_x=None, length_y=None, length_z=None, number_of_ribosomes=4
):
    """
    Create a sample with some Ribosomes in ice

    If the size of the sample is not specified then the sample will be 2x the
    size of the ribosome and the ribosome will be positioned in the centre

    Args:
        length_x (float): The X size of the sample (units: A)
        length_y (float): The Y size of the sample (units: A)
        length_z (float): The Z size of the sample (units: A)
        number_of_ribosomes (int): The number of ribosomes to place

    Returns:
        object: The atom data

    """
    print("Creating sample: ribosomes_in_lamella")

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

    # Compute the size of the box around the ribosomes
    size_x = max_x - min_x
    size_y = max_y - min_y
    size_z = max_z - min_z

    # Recentre coords on zero (needed or rotation)
    x -= (max_x + min_x) / 2.0
    y -= (max_y + min_y) / 2.0
    z -= (max_z + min_z) / 2.0

    # Set the total size of the sample
    assert length_x > size_x
    assert length_y > size_y
    assert length_z > size_z

    # Make a numpy array of coords
    coords = numpy.array(list(zip(x, y, z)))

    # Put the ribosomes in the sample
    sample_element = []
    sample_x = []
    sample_y = []
    sample_z = []
    sample_occ = []
    sample_charge = []
    cube_size = max(size_x, size_y, size_z)
    for position in distribute_cubes_randomly(
        length_x, length_y, length_z, cube_size, number_of_ribosomes
    ):

        # Get a random rotation
        vector = random_uniform_rotation()

        # Create the scipy rotation object and apply the rotation
        rotation = scipy.spatial.transform.Rotation.from_rotvec(vector)
        transformed_coords = rotation.apply(coords)

        # Now apply the translation
        transformed_coords += position

        x, y, z = zip(*transformed_coords)
        sample_element.extend(element)
        sample_x.extend(x)
        sample_y.extend(y)
        sample_z.extend(z)
        sample_occ.extend(occ)
        sample_charge.extend(charge)

    # Create sigma and region
    sample_sigma = [0.085 for number in sample_element]  # From multem HRTEM example
    sample_region = [0 for number in sample_element]

    # Print some sample information
    print("Sample information:")
    print("    # atoms: %d" % len(sample_element))
    print("    Min x:   %.2f" % min(sample_x))
    print("    Min y:   %.2f" % min(sample_y))
    print("    Min z:   %.2f" % min(sample_z))
    print("    Max x:   %.2f" % max(sample_x))
    print("    Max y:   %.2f" % max(sample_y))
    print("    Max z:   %.2f" % max(sample_z))
    print("    Len x:   %.2f" % length_x)
    print("    Len y:   %.2f" % length_y)
    print("    Len z:   %.2f" % length_z)

    # Return the atom data
    return Sample(
        list(
            zip(
                sample_element,
                sample_x,
                sample_y,
                sample_z,
                sample_sigma,
                sample_occ,
                sample_region,
                sample_charge,
            )
        ),
        length_x,
        length_y,
        length_z,
    )


def create_sample(name, **kwargs):
    """
    Create the sample

    Args:
        name (str): The name of the sample

    Returns:
        object: The test sample

    """
    return {
        "4v5d": create_4v5d_sample,
        "ribosomes_in_lamella": create_ribosomes_in_lamella_sample,
    }[name](**kwargs[name])
