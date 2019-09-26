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
import pickle
import scipy.spatial.transform
import elfantasma.data
from math import acos, cos, pi, sin

import numpy.random


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


def distribute_boxes_uniformly(length_x, length_y, length_z, boxes, max_tries=1000):
    """
    Find n random non overlapping positions for cubes within a volume

    Args:
        length_x (float): The X size of the box
        length_y (float): The Y size of the box
        length_z (float): The Z size of the box
        boxes (float): The list of boxes
        max_tries (int): The maximum tries per cube

    Returns:
        list: A list of centre positions

    """

    # Check if the cube overlaps with any other
    def overlapping(positions, box_sizes, q, q_size):
        for p, p_size in zip(positions, box_sizes):
            p0 = p - p_size / 2
            p1 = p + p_size / 2
            q0 = q - q_size / 2
            q1 = q + q_size / 2
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

    # Loop until we have added the boxes
    positions = []
    box_sizes = []
    for box_size in boxes:

        # The bounds to search in
        lower = box_size / 2
        upper = numpy.array((length_x, length_y, length_z)) - box_size / 2
        assert lower[0] < upper[0]
        assert lower[1] < upper[1]
        assert lower[2] < upper[2]

        # Try to add the box
        num_tries = 0
        while True:
            q = numpy.random.uniform(lower, upper)
            if len(positions) == 0 or not overlapping(
                positions, box_sizes, q, box_size
            ):
                positions.append(q)
                box_sizes.append(box_size)
                break
            num_tries += 1
            if num_tries > max_tries:
                raise RuntimeError(f"Unable to place cube of size {box_size}")

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

    # Recentre the coords on zero
    def recentre(coords):

        # Get the min and max atom positions
        min_coords = numpy.min(coords, axis=0)
        max_coords = numpy.max(coords, axis=0)

        # Compute the size of the box around the ribosomes
        size = max_coords - min_coords

        # Set the total size of the sample
        assert length_x > size[0]
        assert length_y > size[1]
        assert length_z > size[2]

        # Recentre coords on zero (needed for rotation)
        return coords - (max_coords + min_coords) / 2.0

    # Cast to numpy array
    coords = recentre(numpy.array(list(zip(x, y, z)), dtype=numpy.float64))

    # Generate some randomly oriented ribosome coordinates
    ribosomes = []
    for i in range(number_of_ribosomes):

        # Get a random rotation
        vector = random_uniform_rotation()

        # Create the scipy rotation object and apply the rotation
        rotation = scipy.spatial.transform.Rotation.from_rotvec(vector)
        ribosomes.append(recentre(rotation.apply(coords)))

    # Get the min and max coords or each ribosome and return the box size
    def ribosome_boxes(ribosomes):
        for transformed_coords in ribosomes:
            min_coords = numpy.min(transformed_coords, axis=0)
            max_coords = numpy.max(transformed_coords, axis=0)
            yield max_coords - min_coords

    # Put the ribosomes in the sample
    sample_element = []
    sample_x = []
    sample_y = []
    sample_z = []
    sample_occ = []
    sample_charge = []
    for i, translation in enumerate(
        distribute_boxes_uniformly(
            length_x, length_y, length_z, ribosome_boxes(ribosomes)
        )
    ):

        # Apply the translation
        ribosomes[i] += translation

        # Extend the arrays
        sample_element.extend(element)
        sample_x.extend(ribosomes[i][:, 0])
        sample_y.extend(ribosomes[i][:, 1])
        sample_z.extend(ribosomes[i][:, 2])
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

    # Check positions
    assert min(sample_x) > 0
    assert min(sample_y) > 0
    assert min(sample_z) > 0
    assert max(sample_x) < length_x
    assert max(sample_y) < length_y
    assert max(sample_z) < length_z

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


def create_custom_sample(filename=None):
    """
    Create the custom sample from file

    Args:
        filename: The sample filename

    Returns:
        object: The atom data

    """
    print(f"Reading sample information from {filename}")
    with open(filename, "rb") as infile:
        sample = pickle.load(infile)
    return sample


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
        "custom": create_custom_sample,
    }[name](**kwargs[name])
