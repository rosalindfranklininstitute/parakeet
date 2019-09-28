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
import copy
import itertools
import gemmi
import numpy
import pandas
import pickle
import os
import scipy.spatial.transform
import elfantasma.data
from math import acos, cos, pi, sin


class Sample(object):
    """
    A class to wrap the sample information

    """

    def __init__(self, atom_data=None, size=None):
        """
        Initialise the sample

        Args:
            atom_data (object): The sample atom_data
            size (tuple): The size of the sample box (units: A)

        """
        # Create and empty data frame
        if atom_data is None:
            atom_data = pandas.DataFrame()
            column_info = (
                ("model", "str"),
                ("chain", "str"),
                ("residue", "str"),
                ("atomic_number", "uint32"),
                ("x", "float64"),
                ("y", "float64"),
                ("z", "float64"),
                ("occ", "float64"),
                ("charge", "uint32"),
            )
            atom_data = pandas.DataFrame(
                dict(
                    (name, pandas.Series([], dtype=dtype))
                    for name, dtype in column_info
                )
            )

        # Set the atom data
        self.atom_data = atom_data

        # Add a couple of columns
        self.atom_data["sigma"] = pandas.Series(
            [0.085 for i in range(self.atom_data.shape[0])], dtype="float64"
        )
        self.atom_data["region"] = self.atom_data["model"]

        # Get the coordinates
        coords = self.atom_data[["x", "y", "z"]]

        # Get the min and max atom positions
        if coords.shape[0] == 0:
            self.min_coords = numpy.array([0, 0, 0], dtype="float64")
            self.max_coords = numpy.array([0, 0, 0], dtype="float64")
            self.sample_size = numpy.array([0, 0, 0], dtype="float64")
        else:
            self.min_coords = numpy.min(coords, axis=0)
            self.max_coords = numpy.max(coords, axis=0)
            self.sample_size = self.max_coords - self.min_coords

        # If box size is None then copy to sample size
        if size is None:
            self.resize(self.sample_size[:])
        else:
            self.resize(size)

        # Recentre the atoms in the box
        self.recentre()

    def resize(self, size=None):
        """
        Resize the sample box.

        This must be greater than the box around the sample

        Args:
            size (tuple): The size of the sample box (units: A)

        """
        if size is not None:
            assert numpy.all(numpy.greater_equal(size, self.sample_size))
        else:
            size = self.sample_size
        self.box_size = size

    def recentre(self, position=None):
        """
        Recentre the sample in the sample box

        Args:
            position (tuple): The position to centre on (otherwise the centre of the box)


        """
        if position is None:
            position = self.box_size / 2.0
        translation = numpy.array(position) - (self.max_coords + self.min_coords) / 2.0
        self.translate(translation)

    def translate(self, translation):
        """
        Translate the sample by some amount

        Args:
            translation (tuple): The translation

        """

        # Transform the sample
        self.atom_data[["x", "y", "z"]] += translation

        # Compute the new max and min
        self.min_coords += translation
        self.max_coords += translation

    def rotate(self, vector):
        """
        Perform a rotation

        Args:
            vector (tuple): The rotation vector

        """

        # Get the coordinates
        coords = self.atom_data[["x", "y", "z"]]

        # Perform the rotation
        rotation = scipy.spatial.transform.Rotation.from_rotvec(vector)
        coords = rotation.apply(coords)

        # Set the coordinates
        self.atom_data[["x", "y", "z"]] = coords

        # Get the min and max atom positions
        self.min_coords = numpy.min(coords, axis=0)
        self.max_coords = numpy.max(coords, axis=0)
        self.sample_size = self.max_coords - self.min_coords

    def extend(self, other):
        """
        Extend this sample with another

        Args:
            other (sample): Another sample object

        """
        # Get the maximum model index
        if self.atom_data.shape[0] == 0:
            region = 0
        else:
            region = self.atom_data["model"].max() + 1

        # Update the region number and set the model name as a string version of the region
        other.atom_data["model"] += region
        other.atom_data["region"] = other.atom_data["model"]

        # Set the atom data
        self.atom_data = self.atom_data.append(other.atom_data)

        # Update box sizes
        coords = self.atom_data[["x", "y", "z"]]
        self.min_coords = numpy.min(coords, axis=0)
        self.max_coords = numpy.max(coords, axis=0)
        self.sample_size = self.max_coords - self.min_coords
        self.box_size = self.sample_size

    def validate(self):
        """
        Just validate the box size

        """
        assert numpy.all(numpy.greater_equal(self.min_coords, (0, 0, 0)))
        assert numpy.all(numpy.less_equal(self.min_coords, self.box_size))

    def info(self):
        """
        Get some sample info

        Returns:
            str: Some sample info

        """
        lines = [
            "Sample information:",
            "    # models:      %d" % len(set(self.atom_data["model"])),
            "    # atoms:       %d" % self.atom_data.shape[0],
            "    Min x:         %.2f" % self.min_coords[0],
            "    Min y:         %.2f" % self.min_coords[1],
            "    Min z:         %.2f" % self.min_coords[2],
            "    Max x:         %.2f" % self.max_coords[0],
            "    Max y:         %.2f" % self.max_coords[1],
            "    Max z:         %.2f" % self.max_coords[2],
            "    Sample size x: %.2f" % self.sample_size[0],
            "    Sample size y: %.2f" % self.sample_size[1],
            "    Sample size z: %.2f" % self.sample_size[2],
            "    Box size x:    %.2f" % self.box_size[0],
            "    Box size y:    %.2f" % self.box_size[1],
            "    Box size z:    %.2f" % self.box_size[2],
        ]
        return "\n".join(lines)

    def as_gemmi_structure(self):
        """
        Get the gemmi structure

        Returns:
            object: The gemmi structure

        """

        # Sort the values ahead of time
        self.atom_data.sort_values(by=["model", "chain", "residue"])

        # Create a zip iterator over the arrays
        structure = zip(
            self.atom_data["model"],
            self.atom_data["chain"],
            self.atom_data["residue"],
            self.atom_data["atomic_number"],
            self.atom_data["x"],
            self.atom_data["y"],
            self.atom_data["z"],
            self.atom_data["occ"],
            self.atom_data["charge"],
        )

        # Create the structure
        gemmi_structure = gemmi.Structure()

        # Iterate over the models
        for model_name, model in itertools.groupby(structure, key=lambda x: x[0]):

            # Iterate over the chains in the model
            gemmi_model = gemmi.Model(str(model_name))
            for chain_name, chain in itertools.groupby(model, key=lambda x: x[1]):

                # Iterate over the residues in the chain
                gemmi_chain = gemmi.Chain(chain_name)
                for residue_name, residue in itertools.groupby(
                    chain, key=lambda x: x[2]
                ):

                    # Iterate over the atoms in the residue
                    gemmi_residue = gemmi.Residue()
                    gemmi_residue.name = residue_name
                    for _, _, _, atomic_number, x, y, z, occ, charge in residue:
                        gemmi_atom = gemmi.Atom()
                        gemmi_atom.element = gemmi.Element(atomic_number)
                        gemmi_atom.pos.x = x
                        gemmi_atom.pos.y = y
                        gemmi_atom.pos.z = z
                        gemmi_atom.occ = occ
                        gemmi_atom.charge = charge
                        gemmi_residue.add_atom(gemmi_atom, -1)

                    # Add the residue
                    gemmi_chain.add_residue(gemmi_residue, -1)

                # Add the chain
                gemmi_model.add_chain(gemmi_chain, -1)

            # Add the model
            gemmi_structure.add_model(gemmi_model, -1)

        # Return the structure
        return gemmi_structure

    def as_cif(self, filename):
        """
        Write the sample to a cif file

        Args:
            filename (str): The output filename

        """
        self.as_gemmi_structure().make_mmcif_document().write_file(filename)

    def as_pdb(self, filename):
        """
        Write the sample to a pdb file

        Args:
            filename (str): The output filename

        """
        self.as_gemmi_structure().write_pdb(filename)

    def as_pickle(self, filename):
        """
        Write the sample to a pickle file

        Args:
            filename (str): The output filename

        """
        with open(filename, "wb") as outfile:
            pickle.dump(self, outfile)

    def as_file(self, filename):
        """
        Write the sample to a file

        Args:
            filename (str): The output filename

        """
        extension = os.path.splitext(filename)[1].lower()
        if extension in [".cif"]:
            self.as_cif(filename)
        elif extension in [".pdb"]:
            self.as_pdb(filename)
        elif extension in [".p", ".pkl", ".pickle"]:
            self.as_pickle(filename)
        else:
            raise RuntimeError(f"File with unknown extension: {filename}")

    @classmethod
    def from_gemmi_structure(Class, structure):
        """
        Read the sample from a gemmi sucture

        Args:
            structure (object): The input structure

        Returns:
            object: The Sample object

        """

        # The column data
        column_info = (
            ("model", "uint32"),
            ("chain", "str"),
            ("residue", "str"),
            ("atomic_number", "uint32"),
            ("x", "float64"),
            ("y", "float64"),
            ("z", "float64"),
            ("occ", "float64"),
            ("charge", "uint32"),
        )

        # Iterate through the atoms
        def iterate_atoms(structure):
            for model_index, model in enumerate(structure):
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            yield (
                                model_index,
                                chain.name,
                                residue.name,
                                atom.element.atomic_number,
                                atom.pos.x,
                                atom.pos.y,
                                atom.pos.z,
                                atom.occ,
                                atom.charge,
                            )

        # Create a dictionary of column data
        def create_atom_data(structure, column_info):
            return dict(
                (name, pandas.Series(data, dtype=dtype))
                for data, (name, dtype) in zip(
                    zip(*iterate_atoms(structure)), column_info
                )
            )

        # Return the sample with the atom data as a pandas dataframe
        return Sample(pandas.DataFrame(create_atom_data(structure, column_info)))

    @classmethod
    def from_gemmi_file(Class, filename):
        """
        Read the sample from a file

        Args:
            filename (str): The input filename

        Returns:
            object: The Sample object

        """

        # Read the structure
        return Class.from_gemmi_structure(gemmi.read_structure(filename))

    @classmethod
    def from_pickle(Class, filename):
        """
        Read the sample from a file

        Args:
            filename (str): The input filename

        Returns:
            object: The Sample object

        """
        with open(filename, "rb") as infile:
            return pickle.load(infile)

    @classmethod
    def from_file(Class, filename):
        """
        Read the sample from a file

        Args:
            filename (str): The input filename

        Returns:
            object: The Sample object

        """
        extension = os.path.splitext(filename)[1].lower()
        if extension in [".cif", ".pdb"]:
            obj = Class.from_gemmi_file(filename)
        elif extension in [".p", ".pkl", ".pickle"]:
            obj = Class.from_pickle(filename)
        else:
            raise RuntimeError(f"File with unknown extension: {filename}")
        return obj


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

    # Create the sample
    sample = Sample.from_file(filename)

    # Set the size
    if length_x is None or length_y is None or length_z is None:
        size = sample.sample_size * 2.0
    else:
        size = (length_x, length_y, length_z)

    # Resize and recentre
    sample.resize(size)
    sample.recentre()
    print(sample.info())
    sample.validate()

    # Return the sample
    return sample


def distribute_boxes_uniformly(volume_size, boxes, max_tries=1000):
    """
    Find n random non overlapping positions for cubes within a volume

    Args:
        volume_size (float): The size of the volume
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
        upper = volume_size - box_size / 2
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

    # Create a single ribosome sample
    single_sample = Sample.from_file(filename)
    single_sample.recentre((0, 0, 0))

    # Set the sample size
    box_size = numpy.array([length_x, length_y, length_z])

    # Generate some randomly oriented ribosome coordinates
    ribosomes = []
    for i in range(number_of_ribosomes):

        # Get a random rotation
        vector = random_uniform_rotation()

        # Copy the ribosomes
        ribosome = copy.deepcopy(single_sample)
        ribosome.rotate(vector)
        ribosome.recentre((0, 0, 0))
        ribosomes.append(ribosome)

    # Get the ribosome sample size
    def ribosome_boxes(ribosomes):
        for ribosome in ribosomes:
            yield ribosome.sample_size

    # Put the ribosomes in the sample
    print("Placing ribosomes:")
    sample = Sample()
    for i, translation in enumerate(
        distribute_boxes_uniformly(box_size, ribosome_boxes(ribosomes))
    ):
        print(f"    Placing ribosome {i}")

        # Apply the translation
        ribosomes[i].translate(translation)

        # Extend the sample
        sample.extend(ribosomes[i])

    # Resize and print some info
    sample.resize(box_size)
    print(sample.info())
    sample.validate()

    # Return the sample
    return sample


def create_custom_sample(filename=None):
    """
    Create the custom sample from file

    Args:
        filename: The sample filename

    Returns:
        object: The atom data

    """
    print(f"Reading sample information from {filename}")
    return Sample.from_file(filename)


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
