#
# elfantasma.sample.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import itertools
import gemmi
import numpy
import pandas
import pickle
import os
import scipy.constants
import elfantasma.data
from math import acos, cos, pi, sin, sqrt, floor
from scipy.spatial.transform import Rotation


def translate(atom_data, translation):
    """
    Helper function to translate atom data

    Args:
        atom_data (object): The atom data
        translation (array): The translation

    """
    atom_data[["x", "y", "z"]] += translation
    return atom_data


def recentre(atom_data, position=None):
    """
    Helper function to reposition atom data

    Args:
        atom_data (object): The atom data
        translation (array): The translation

    """
    # Check the input
    if position is None:
        position = (0, 0, 0)

    # Get the coords
    coords = atom_data[["x", "y", "z"]]

    # Compute the translation
    translation = numpy.array(position) - (coords.max() + coords.min()) / 2.0

    # Do the translation
    return translate(atom_data, translation)


def extract_spec_atoms(atom_data):
    """
    Extract the spec atoms

    Args:
        atom_data (object): The atom data

    Returns:
        iterator: The iterator of specimen atoms

    """
    return zip(
        atom_data["atomic_number"],
        atom_data["x"],
        atom_data["y"],
        atom_data["z"],
        atom_data["sigma"],
        atom_data["occ"],
        atom_data["region"],
        atom_data["charge"],
    )


class Structure(object):
    """
    Hold a single set of atom data.

    Our simulated specimens tend to be lots of duplicate structures in random
    positions and orientations so it is more convenient just to save the atom
    coordinates once and then a list of where we want to put them in space.

    """

    def __init__(self, atom_data, positions=None, rotations=None):
        """
        Initialise the model

        Args:
            atom_data (object): The atom data
            positions (list): The list of X,Y,Z positions
            rotations (list): The list of orientations

        """

        # Set position
        if positions is None and rotations is None:
            positions = numpy.zeros(shape=(0, 3), dtype=numpy.float64)
            rotations = numpy.zeros(shape=(0, 3), dtype=numpy.float64)
        else:
            positions = numpy.array(positions, dtype=numpy.float64)
            rotations = numpy.array(rotations, dtype=numpy.float64)

        # Check input
        assert positions.shape[0] == rotations.shape[0]

        # Set the model stuff
        self.atom_data = recentre(atom_data)
        self.positions = positions
        self.rotations = rotations

    @property
    def num_models(self):
        """
        Returns:
            int: The number of models

        """
        assert self.positions.shape[0] == self.rotations.shape[0]
        return len(set(self.atom_data["model"])) * self.positions.shape[0]

    @property
    def num_atoms(self):
        """
        Returns:
            int: The number of atoms

        """
        assert self.positions.shape[0] == self.rotations.shape[0]
        return self.atom_data.shape[0] * self.positions.shape[0]

    @property
    def individual_bounding_boxes(self):
        """
        Compute the individual bounding boxes

        """
        # Check transformations are the same length
        assert self.positions.shape[0] == self.rotations.shape[0]

        # Compute min and max of all positions and rotations
        reference_coords = self.atom_data[["x", "y", "z"]]
        if reference_coords.shape[0] != 0:
            for position, rotation in zip(
                self.positions, Rotation.from_rotvec(self.rotations)
            ):
                coords = rotation.apply(reference_coords) + position
                yield (numpy.min(coords, axis=0), numpy.max(coords, axis=0))

    @property
    def individual_sample_sizes(self):
        """
        Compute the same size

        """
        for bbox in self.individual_bounding_boxes:
            yield bbox[1] - bbox[0]

    @property
    def bounding_box(self):
        """
        Compute the bounding box

        Returns:
            (min, max): The min and max coords

        """

        # Check transformations are the same length
        assert self.positions.shape[0] == self.rotations.shape[0]

        # Get the coordinates
        reference_coords = self.atom_data[["x", "y", "z"]]
        if reference_coords.shape[0] == 0 or self.positions.shape[0] == 0:
            min_coords = [[0, 0, 0]]
            max_coords = [[0, 0, 0]]
        else:
            min_coords, max_coords = zip(*self.individual_bounding_boxes)

        # Return the min and max
        return (
            numpy.min(numpy.array(min_coords), axis=0),
            numpy.max(numpy.array(max_coords), axis=0),
        )

    @property
    def spec_atoms(self):
        """
        Get the subset of data needed to the simulation

        Returns:
            list: A list of tuples

        """

        def spec_atoms_internal():

            # Get the reference coords
            reference_coords = self.atom_data[["x", "y", "z"]]

            # Iterate over the positions and yield the atom data
            for position, rotation in zip(
                self.positions, Rotation.from_rotvec(self.rotations)
            ):

                # Apply the rotation and translations
                coords = rotation.apply(reference_coords) + position

                # Yield the atom data
                yield zip(
                    self.atom_data["atomic_number"],
                    coords[:, 0],
                    coords[:, 1],
                    coords[:, 2],
                    self.atom_data["sigma"],
                    self.atom_data["occ"],
                    self.atom_data["region"],
                    self.atom_data["charge"],
                )

        # Chain all together
        return itertools.chain(*spec_atoms_internal())

    def select_atom_data_in_roi(self, roi):
        """
        Get the subset of atoms within the field of view

        Args:
            roi (array): The region of interest

        Returns:
            object: The atom data

        """

        # Split the roi
        (x0, y0), (x1, y1) = roi

        # Get the reference coords
        reference_coords = self.atom_data[["x", "y", "z"]]

        # Iterate over the positions and yield the atom data
        for position, rotation in zip(
            self.positions, Rotation.from_rotvec(self.rotations)
        ):

            # Apply the rotation and translations
            coords = rotation.apply(reference_coords) + position

            # Select the coords within the roi
            selection = (
                (coords[:, 0] >= x0)
                & (coords[:, 0] <= x1)
                & (coords[:, 1] >= y0)
                & (coords[:, 1] <= y1)
            )

            # Get the selection
            subset = self.atom_data.loc[selection].copy()
            subset[["x", "y", "z"]] = coords[selection]

            # Yield the atom data
            yield subset

    def append(self, position, rotation):
        """
        Append a model

        Args:
            position (array): The position
            rotation (array): The rotation

        """
        self.positions = numpy.append(self.positions, [position], axis=0)
        self.rotations = numpy.append(self.rotations, [rotation], axis=0)

    def translate(self, translation):
        """
        Translate the sample by some amount

        Args:
            translation (tuple): The translation

        """
        self.positions += translation

    def rotate(self, vector):
        """
        Perform a rotation

        Args:
            vector (tuple): The rotation vector

        """
        self.rotations = (
            Rotation.from_rotvec(vector) * Rotation.from_rotvec(self.rotations)
        ).as_rotvec()


class Sample(object):
    """
    A class to wrap the sample information

    """

    def __init__(self, structures=None, size=None, recentre=False):
        """
        Initialise the sample

        Args:
            structures (object): The structure data
            size (tuple): The size of the sample box (units: A)
            recentre (bool): Recentre the sample in the box

        """

        # Make sure we have a list
        if isinstance(structures, Structure):
            structures = [structures]

        # Set the structures
        self.structures = structures

        # Compute the min and max
        self.update()

        # Resize the container
        self.resize(size)

        # Recentre the atoms in the box
        if recentre:
            self.recentre()

    @property
    def num_models(self):
        """
        Returns:
            int: The number of models

        """
        return sum(s.num_models for s in self.structures)

    @property
    def num_atoms(self):
        """
        Returns:
            int: The number of atoms

        """
        return sum(s.num_atoms for s in self.structures)

    @property
    def bounding_box(self):
        """
        Compute the bounding box

        Returns:
            (min, max): The min and max coords

        """

        # Compute the min and max
        min_coords = []
        max_coords = []
        for structure in self.structures:
            bbox = structure.bounding_box
            min_coords.append(bbox[0])
            max_coords.append(bbox[1])

        # Return the min and max
        return (
            numpy.min(numpy.array(min_coords), axis=0),
            numpy.max(numpy.array(max_coords), axis=0),
        )

    @property
    def spec_atoms(self):
        """
        Get the subset of data needed to the simulation

        Returns:
            list: A list of tuples

        """
        return itertools.chain(*[s.spec_atoms for s in self.structures])

    def select_atom_data_in_roi(self, roi):
        """
        Get the subset of atoms within the field of view

        Args:
            roi (array): The region of interest

        Returns:
            object: The atom data

        """
        return pandas.concat(
            list(
                itertools.chain(
                    *[s.select_atom_data_in_roi(roi) for s in self.structures]
                )
            )
        )

    def update(self):
        """
        Update the statistics

        """

        # Compute the bounding box
        self.sample_bbox = self.bounding_box
        self.sample_size = self.sample_bbox[1] - self.sample_bbox[0]

    def resize(self, size=None, margin=1):
        """
        Resize the sample box.

        This must be greater than the box around the sample

        Args:
            size (tuple): The size of the sample box (units: A)
            margin (float): The margin when size is None (units: A)

        """
        if size is not None:
            assert numpy.all(numpy.greater_equal(size, self.sample_size))
        else:
            size = self.sample_size + margin * 2
        self.box_size = size

    def recentre(self, position=None):
        """
        Recentre the sample in the sample box

        Args:
            position (tuple): The position to centre on (otherwise the centre of the box)

        """
        # Check the input
        if position is None:
            position = self.box_size / 2.0

        # Compute the translation
        translation = (
            numpy.array(position) - (self.sample_bbox[1] + self.sample_bbox[0]) / 2.0
        )

        # Translate the structure
        self.translate(translation)

    def translate(self, translation):
        """
        Translate the sample by some amount

        Args:
            translation (tuple): The translation

        """
        # Apply translation to all structures
        for structure in self.structures:
            structure.translate(translation)

        # Compute the new bounding box
        self.sample_bbox += translation

    def rotate(self, vector):
        """
        Perform a rotation

        Args:
            vector (tuple): The rotation vector

        """
        # Apply rotation to all structures
        for structure in self.structures:
            structure.rotate(vector)

        # Compute the min and max coords
        self.update()

    def append(self, structure):
        """
        Extend this sample with another structure

        Args:
            other (sample): Another sample object

        """

        # Append the structure
        self.structures.append(structure)

        # Update stats and resize to sample
        self.update()
        self.resize()

    def validate(self):
        """
        Just validate the box size

        """
        self.update()
        assert numpy.all(numpy.greater_equal(self.sample_bbox[0], (0, 0, 0)))
        assert numpy.all(numpy.less_equal(self.sample_bbox[1], self.box_size))

    def info(self):
        """
        Get some sample info

        Returns:
            str: Some sample info

        """
        lines = [
            "Sample information:",
            "    # models:      %d" % self.num_models,
            "    # atoms:       %d" % self.num_atoms,
            "    Min x:         %.2f" % self.sample_bbox[0][0],
            "    Min y:         %.2f" % self.sample_bbox[0][1],
            "    Min z:         %.2f" % self.sample_bbox[0][2],
            "    Max x:         %.2f" % self.sample_bbox[1][0],
            "    Max y:         %.2f" % self.sample_bbox[1][1],
            "    Max z:         %.2f" % self.sample_bbox[1][2],
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

        # Sort the structure models
        for s in self.structures:
            s.atom_data.sort_values(by=["model", "chain", "residue"])

        # Get the atom data from the structure
        def atom_data(s):

            # Get the atom data from each model
            def atom_data_internal():

                # Get the reference coords
                reference_coords = s.atom_data[["x", "y", "z"]]

                # Iterate over the positions and yield the atom data
                for position, rotation in zip(
                    s.positions, Rotation.from_rotvec(s.rotations)
                ):

                    # Apply the rotation and translations
                    coords = rotation.apply(reference_coords) + position

                    # Yield the atom data
                    yield zip(
                        s.atom_data["model"],
                        s.atom_data["chain"],
                        s.atom_data["residue"],
                        s.atom_data["atomic_number"],
                        coords[:, 0],
                        coords[:, 1],
                        coords[:, 2],
                        s.atom_data["occ"],
                        s.atom_data["charge"],
                    )

            # Chain all together
            return itertools.chain(*atom_data_internal())

        # Get the iterator to the atom data
        structure = itertools.chain(*[atom_data(s) for s in self.structures])

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
            ("sigma", "float64"),
            ("region", "uint32"),
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
                                0.085,  # sigma: from MULTEM HRTEM example
                                0,  # region: non zero can lead to issues
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
        return Sample(
            Structure(
                pandas.DataFrame(create_atom_data(structure, column_info)),
                positions=[(0, 0, 0)],
                rotations=[(0, 0, 0)],
            )
        )

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
    def from_ligand_file(Class, filename):
        """
        Read the sample from a file

        Args:
            filename (str): The input filename

        Returns:
            object: The Sample object

        """
        # Create a single ribosome sample
        return Class.from_gemmi_structure(
            gemmi.make_structure_from_chemcomp_block(gemmi.cif.read(filename)[0])
        )

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

    # Create a single ribosome structure
    ribosomes = Structure(Sample.from_file(filename).structures[0].atom_data)

    # Set the sample size
    box_size = numpy.array([length_x, length_y, length_z])

    # Generate some randomly oriented ribosome coordinates
    for i in range(number_of_ribosomes):
        ribosomes.append((0, 0, 0), random_uniform_rotation())

    # Put the ribosomes in the sample
    print("Placing ribosomes:")
    for i, translation in enumerate(
        distribute_boxes_uniformly(box_size, ribosomes.individual_sample_sizes)
    ):
        ribosomes.positions[i] = translation

    # Resize and print some info
    sample = Sample(ribosomes, size=box_size)
    print(sample.info())
    sample.validate()

    # Return the sample
    return sample


def create_ribosomes_in_cylinder_sample(
    radius=1500, margin=500, length=10000, number_of_ribosomes=20
):
    """
    Create a sample with some Ribosomes in ice

    The cylinder will have it's circular cross section in the Y/Z plane and
    will be elongated in along the X axis. A margin will be put in the Y axis
    such that each image will fully contain the width of the cylinder.

    Args:
        radius (float): The radius of the cylinder (units: A)
        length (float): The length of the cylinder (units: A)
        margin (float): The margin around the cylinder (units: A)
        number_of_ribosomes (int): The number of ribosomes to place

    Returns:
        object: The atom data

    """
    print("Creating sample: ribosomes_in_cylinder")

    # Get the filename of the water.cif file
    filename = elfantasma.data.get_path("water.cif")

    # Create a single ribosome sample
    single_water = Sample.from_ligand_file(filename)

    # Compute the number of water molecules
    avogadros_number = scipy.constants.Avogadro
    molar_mass_of_water = 18.01528  # grams / mole
    density_of_water = 997  # kg / m^3
    volume_of_cylinder = pi * radius ** 2 * length  # A^3
    mass_of_cylinder = (density_of_water * 1000) * (
        volume_of_cylinder * 1e-10 ** 3
    )  # g
    number_of_waters = int(
        floor((mass_of_cylinder / molar_mass_of_water) * avogadros_number)
    )

    # Print some stuff of water
    print("Water information:")
    print("    Volume of cylinder: %g m^3" % (volume_of_cylinder * 1e-10 ** 3))
    print("    Density of water: %g kg/m^3" % density_of_water)
    print("    Mass of cylinder: %g kg" % (mass_of_cylinder / 1000))
    print("    Number of water molecules to place: %d" % number_of_waters)

    # Get the filename of the 4v5d.cif file
    filename = elfantasma.data.get_path("4v5d.cif")

    # Create a single ribosome structure
    ribosomes = Structure(Sample.from_file(filename).structures[0].atom_data)

    # Set the cuboid size and box size
    cuboid_size = numpy.array([length, sqrt(2) * radius, sqrt(2) * radius])
    box_size = numpy.array([length, 2 * (radius + margin), 2 * (radius + margin)])

    # Generate some randomly oriented ribosome coordinates
    print("Generating random orientations:")
    for i in range(number_of_ribosomes):
        ribosomes.append((0, 0, 0), random_uniform_rotation())

    # Put the ribosomes in the sample
    print("Placing ribosomes:")
    correction = margin + (1 - 1 / sqrt(2)) * radius
    for i, translation in enumerate(
        distribute_boxes_uniformly(cuboid_size, ribosomes.individual_sample_sizes)
    ):
        ribosomes.positions[i] = translation + numpy.array([0, correction, correction])

    # Create the sample
    sample = Sample(ribosomes, size=box_size)
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


def load(filename):
    """
    Open the sample from file

    Args:
        filename (str): The sample filename

    Returns:
        object: The test sample

    """
    return Sample.from_file(filename)


def new(name, **kwargs):
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
        "ribosomes_in_cylinder": create_ribosomes_in_cylinder_sample,
        "custom": create_custom_sample,
    }[name](**kwargs.get(name, {}))
