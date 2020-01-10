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
from collections import defaultdict
import h5py
import gemmi
import logging
import numpy
import pandas
import scipy.constants
from math import pi, sqrt, floor, ceil
from scipy.spatial.transform import Rotation
import elfantasma.data
import elfantasma.freeze

numpy.random.seed(0)

# Get the logger
logger = logging.getLogger(__name__)


def get_atom_sigma_sq(atom):
    """
    Get the sigma_sq from the atom

    Args:
        atom (object): A Gemmi atom object

    Returns:
        float: The positional sigma (sqrt(b_iso))

    """
    if atom.has_anisou():
        b_iso = atom.b_iso_from_aniso()
    else:
        b_iso = atom.b_iso
    return b_iso / (8 * pi ** 2)


def get_atom_sigma(atom):
    """
    Get the sigma from the atom

    Args:
        atom (object): A Gemmi atom object

    Returns:
        float: The positional sigma (sqrt(b_iso))

    """
    return sqrt(get_atom_sigma_sq(atom))


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


class AtomData(object):
    """
    A class to hold the atom data

    """

    # The column data
    column_data = {
        "atomic_number": "uint8",
        "x": "float32",
        "y": "float32",
        "z": "float32",
        "sigma": "float32",
        "occupancy": "float32",
        "charge": "uint8",
    }

    def __init__(self, data=None, **kwargs):
        """
        Initialise the class

        Either the data attribute or the individual arrays must be set.

        Args:
            data (object): The atom data table
            atomic_number (array): The array of atomic numbers
            x (array): The array of x positions
            y (array): The array of y positions
            z (array): The array of z positions
            sigma (array): The array of sigmas
            occupancy (array): The array of occupancies
            charge (array): The array of charges

        """
        if data is None:
            self.data = pandas.DataFrame(
                dict(
                    (name, pandas.Series(kwargs[name], dtype=dtype))
                    for name, dtype in AtomData.column_data.items()
                )
            )
        else:
            column_names = list(data.keys())
            for name in AtomData.column_data:
                assert name in column_names
            self.data = data

    def rotate(self, vector):
        """
        Rotate the atom data around the rotation vector

        Args:
            vector (array): The rotation vector

        """
        coords = self.data[["x", "y", "z"]]
        coords = Rotation.from_rotvec(vector).apply(coords).astype("float32")
        self.data["x"] = coords[:, 0]
        self.data["y"] = coords[:, 1]
        self.data["z"] = coords[:, 2]

    def translate(self, translation):
        """
        Translate the atom data

        Args:
            translation (array): The translation

        """
        self.data[["x", "y", "z"]] += numpy.array(translation)

    @classmethod
    def from_gemmi_structure(Class, structure):
        """
        Read the sample from a gemmi sucture

        Args:
            structure (object): The input structure

        Returns:
            object: The Sample object

        """

        # The column order
        column_info = ["atomic_number", "x", "y", "z", "sigma", "occupancy", "charge"]

        # Iterate through the atoms
        def iterate_atoms(structure):
            for model_index, model in enumerate(structure):
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            yield (
                                atom.element.atomic_number,
                                atom.pos.x,
                                atom.pos.y,
                                atom.pos.z,
                                get_atom_sigma(atom),
                                atom.occ,
                                atom.charge,
                            )

        # Create a dictionary of column data
        def create_atom_data(structure):
            return dict(
                (name, pandas.Series(data, dtype=AtomData.column_data[name]))
                for data, name in zip(zip(*iterate_atoms(structure)), column_info)
            )

        # Return the sample with the atom data as a pandas dataframe
        return AtomData(data=pandas.DataFrame(create_atom_data(structure)))

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


class SampleHDF5Adapter(object):
    """
    A class to handle the HDF5 group and dataset creation

    """

    class AtomsSubGroup(object):
        """
        A class to handle the atom sub groups

        """

        def __init__(self, handle):
            """
            Save the group handle

            Args:
                handle (object): The group handle

            """
            self.__handle = handle

        @property
        def atoms(self):
            """
            Return the atom data in the atom sub group

            """
            return pandas.DataFrame(
                dict(
                    (name, pandas.Series(self.__handle[name][:]))
                    for name in AtomData.column_data
                )
            )

        @atoms.setter
        def atoms(self, data):
            """
            Set the atom data in the sub group

            """

            # Create the datasets
            if len(self.__handle) == 0:
                for name, dtype in AtomData.column_data.items():
                    self.__handle.create_dataset(
                        name, (0,), maxshape=(None,), dtype=dtype, chunks=True
                    )

            # Get the size
            i0 = self.__handle["x"].shape[0]
            i1 = i0 + data["x"].shape[0]

            # Check size, resize and set data
            for name in AtomData.column_data.keys():
                assert i0 == self.__handle[name].shape[0]
                self.__handle[name].resize((i1,))
                self.__handle[name][i0:i1] = data[name]

        def __len__(self):
            """
            Returns the number of atoms

            """
            if len(self.__handle) == 0:
                return 0
            return self.__handle["x"].shape[0]

    class AtomsGroup(object):
        """
        A class to hold the atom group

        """

        def __init__(self, handle):
            """
            Save the group handle

            Args:
                handle (object): The group handle

            """
            self.__handle = handle

        def __len__(self):
            """
            Returns:
                int: The number of sub groups

            """
            return len(self.__handle)

        def __getitem__(self, item):
            """
            Get the subgroup with the given name

            Args:
                item (str): The subgroup name

            Returns:
                object: The subgroup

            """
            if item not in self.__handle:
                self.__handle.create_group(item)
            return SampleHDF5Adapter.AtomsSubGroup(self.__handle[item])

        def __iter__(self):
            """
            Iterate through the names of the subgroups

            """
            return self.keys()

        def keys(self):
            """
            Iterate through the names of the subgroups

            """
            return self.__handle.keys()

        def items(self):
            """
            Iterate through the names and the subgroups

            """
            for key, value in self.__handle.items():
                yield key, SampleHDF5Adapter.AtomsSubGroup(value)

        def values(self):
            """
            Iterate through the subgroups

            """
            for value in self.__handle.values():
                yield SampleHDF5Adapter.AtomsSubGroup(value)

        def zbins(self):
            """
            Iterate through the z bins

            """
            for key in self.keys():
                z = int(key.split("=")[1].split("A")[0])
                yield key, z

        def number_of_atoms(self):
            """
            Get the total number of atoms in all the subgroups

            Returns:
                int: The total number of atoms

            """
            count = 0
            for key, value in self.items():
                count += len(value)
            return count

    class MoleculeGroup(object):
        """
        A class to represent the molecule group

        """

        def __init__(self, handle):
            """
            Save the group handle

            Args:
                handle (object): The group handle

            """
            self.__handle = handle

        @property
        def atoms(self):
            """
            Get the atoms

            """

            # Get the atoms group
            group = self.__handle["atoms"]

            # Create a pandas data frame
            return pandas.DataFrame(
                dict(
                    (name, pandas.Series(group[name][:]))
                    for name in AtomData.column_data
                )
            )

        @property
        def positions(self):
            """
            Get the positions

            """
            return self.__handle["positions"]

        @property
        def orientations(self):
            """
            Get the orientations

            """
            return self.__handle["orientations"]

        @atoms.setter
        def atoms(self, data):
            """
            Set the atom data

            """

            # Create the atom data group
            group = self.__handle.create_group("atoms")

            # Set the columns
            for name, dtype in AtomData.column_data.items():
                group.create_dataset(name, data=data[name], dtype=dtype, chunks=True)

        @positions.setter
        def positions(self, positions):
            """
            Set the positions

            """
            self.__handle.create_dataset("positions", data=positions, dtype="float32")

        @orientations.setter
        def orientations(self, orientations):
            """
            Set the orientations

            """
            self.__handle.create_dataset(
                "orientations", data=orientations, dtype="float32"
            )

    class MoleculeListGroup(object):
        """
        A class to represent the molecule list group

        """

        def __init__(self, handle):
            """
            Save the group handle

            Args:
                handle (object): The group handle

            """
            self.__handle = handle

        def __len__(self):
            """
            Return the number of molecules

            """
            return len(self.__handle)

        def __getitem__(self, item):
            """
            Get the molecule by name

            """
            if item not in self.__handle:
                self.__handle.create_group(item)
            return SampleHDF5Adapter.MoleculeGroup(self.__handle[item])

        def __iter__(self):
            """
            Iterate through the molecule names

            """
            return self.keys()

        def keys(self):
            """
            Iterate through the molecule names

            """
            return self.__handle.keys()

        def items(self):
            """
            Iterate through the molecule names and groups

            """
            for key, value in self.__handle.items():
                yield key, SampleHDF5Adapter.MoleculeGroup(value)

        def values(self):
            """
            Iterate through the molecule groups

            """
            for value in self.__handle.values():
                yield SampleHDF5Adapter.MoleculeGroup(value)

    class SampleGroup(object):
        """
        A class to represent the sample group

        """

        def __init__(self, handle):
            """
            Save the group handle

            Args:
                handle (object): The group handle

            """
            self.__handle = handle

        @property
        def atoms(self):
            """
            Get the atom group

            """
            if "atoms" not in self.__handle:
                self.__handle.create_group("atoms")
            return SampleHDF5Adapter.AtomsGroup(self.__handle["atoms"])

        @property
        def molecules(self):
            """
            Get the molecule group

            """
            if "molecules" not in self.__handle:
                self.__handle.create_group("molecules")
            return SampleHDF5Adapter.MoleculeListGroup(self.__handle["molecules"])

        @property
        def bounding_box(self):
            """
            Get the bounding box

            """
            if "bounding_box" not in self.__handle:
                return numpy.zeros(shape=(2, 3), dtype="float32")
            return self.__handle["bounding_box"][:]

        @bounding_box.setter
        def bounding_box(self, bbox):
            """
            Set the bounding box

            """
            if "bounding_box" not in self.__handle:
                self.__handle.create_dataset("bounding_box", data=bbox, dtype="float32")
            else:
                self.__handle["bounding_box"][:] = bbox

        @property
        def containing_box(self):
            """
            Get the containing box

            """
            if "containing_box" not in self.__handle:
                return numpy.zeros(shape=(2, 3), dtype="float32")
            return self.__handle["containing_box"][:]

        @containing_box.setter
        def containing_box(self, box):
            """
            Set the containing box

            """
            if "containing_box" not in self.__handle:
                self.__handle.create_dataset(
                    "containing_box", data=box, dtype="float32"
                )
            else:
                self.__handle["containing_box"][:] = box

        @property
        def centre(self):
            """
            Get the centre

            """
            if "centre" not in self.__handle:
                return numpy.zeros(shape=(3,), dtype="float32")
            return self.__handle["centre"][:]

        @centre.setter
        def centre(self, centre):
            """
            Set the centre

            """
            if "centre" not in self.__handle:
                self.__handle.create_dataset("centre", data=centre, dtype="float32")
            else:
                self.__handle["centre"][:] = centre

        @property
        def shape(self):
            """
            Get the shape

            """
            if "shape" not in self.__handle:
                return {}

            # Get the shape
            shape = self.__handle["shape"][0]
            return {
                "type": shape,
                shape: dict(list(self.__handle["shape"].attrs.items())),
            }

        @shape.setter
        def shape(self, shape):
            """
            Set the shape

            """
            if "shape" not in self.__handle:
                self.__handle.create_dataset(
                    "shape", shape=(1,), dtype=h5py.string_dtype(encoding="utf-8")
                )

            # Set the shape
            self.__handle["shape"][:] = shape["type"]

            # Set the shape attributes
            for key, value in shape[shape["type"]].items():
                self.__handle["shape"].attrs[key] = value

    def __init__(self, filename=None, mode="r"):
        """

        Initialise the class with a filename

        Args:
            filename (str): The filename
            mode (str): The open mode

        """
        self.__handle = h5py.File(filename, mode=mode)

    @property
    def sample(self):
        """
        Get the sample group

        """
        if "sample" not in self.__handle:
            self.__handle.create_group("sample")
        return SampleHDF5Adapter.SampleGroup(self.__handle["sample"])

    def close(self):
        """
        Close the file

        """
        self.__handle.close()


class Sample(object):
    """
    A class to hold the sample data

    """

    def __init__(self, filename=None, mode="r"):
        """

        Initialise the class with a filename

        Args:
            filename (str): The filename
            mode (str): The open mode

        """

        # Open the HDF5 file
        self.__handle = SampleHDF5Adapter(filename, mode=mode)

        # The zstep between datasets
        self.zstep = 10  # A

    def close(self):
        """
        Close the HDF5 file

        """
        self.__handle.close()

    def atoms_dataset_name(self, z):
        return "Z=%06dA" % z

    def atoms_dataset_range(self, z0, z1):
        z0 = int(floor(z0 / float(self.zstep))) * self.zstep
        z1 = int(ceil(z1 / float(self.zstep))) * self.zstep
        for z in range(z0, z1, self.zstep):
            yield z, z + self.zstep

    @property
    def dimensions(self):
        """
        The dimensions of the bounding box

        Returns:
            array: The dimensions

        """
        x0, x1 = self.bounding_box
        return x1 - x0

    def add_atoms(self, atoms):
        """
        Add some atoms

        Args:
            atoms (object): The atom data to add to the sample

        """

        # Get the min/max coords
        x0 = atoms.data["x"].min()
        y0 = atoms.data["y"].min()
        z0 = atoms.data["z"].min()
        x1 = atoms.data["x"].max()
        y1 = atoms.data["y"].max()
        z1 = atoms.data["z"].max()

        # Check the old bounding box
        if self.number_of_atoms > 0:

            # Get the bounding box
            (bx0, by0, bz0), (bx1, by1, bz1) = self.bounding_box

            # Get the min/max coords
            x0 = min(x0, bx0)
            y0 = min(y0, by0)
            z0 = min(z0, bz0)
            x1 = max(x1, bx1)
            y1 = max(y1, by1)
            z1 = max(z1, bz1)

        # Set the bounding box
        self.__handle.sample.bounding_box = [(x0, y0, z0), (x1, y1, z1)]

        # Get the data bounds
        z_min = atoms.data["z"].min()
        z_max = atoms.data["z"].max()

        # Iterate over dataset ranges
        for z0, z1 in self.atoms_dataset_range(z_min, z_max):
            name = self.atoms_dataset_name(z0)
            self.__handle.sample.atoms[name].atoms = atoms.data[
                (atoms.data["z"] >= z0) & (atoms.data["z"] < z1)
            ]

    def del_atoms(self, z_min, z_max, deleter):
        """
        Delete atoms in the given range

        Args:
            z_min (float): The minimum z
            z_max (float): The maximum z
            deleter (func): Check atoms and return only those we want to keep

        """

        # Iterate over dataset ranges
        for z0, z1 in self.atoms_dataset_range(z_min, z_max):
            name = self.atoms_dataset_name(z0)
            atoms = self.__handle.sample.atoms[name].atoms
            self.__handle.sample.atoms[name].atoms = deleter(atoms)

    def add_molecule(self, atoms, positions, orientations, name=None):
        """
        Add a molecule.

        This adds the atoms of each copy of the molecule but also stores the
        molecule itself with the positions and orientations

        Args:
            atoms (object): The atom data
            positions (list): A list of positions
            orientations (list): A list of orientations
            name (str): The name of the molecule

        """

        # Convert to numpy arrays
        positions = numpy.array(positions, dtype="float32")
        orientations = numpy.array(orientations, dtype="float32")
        assert positions.shape == orientations.shape
        assert len(positions.shape) == 2
        assert positions.shape[1] == 3

        # Get the reference coords
        reference_coords = atoms.data[["x", "y", "z"]].to_numpy()

        # Loop through all the positions and rotations
        for position, rotation, orientation in zip(
            positions, Rotation.from_rotvec(orientations), orientations
        ):

            # Make a copy and apply the rotation and translation
            coords = (rotation.apply(reference_coords) + position).astype("float32")
            temp = atoms.data.copy()
            temp["x"] = coords[:, 0]
            temp["y"] = coords[:, 1]
            temp["z"] = coords[:, 2]

            # Add to the atoms
            self.add_atoms(AtomData(data=temp))

        # Add the molecule to it's own field
        if name is not None:
            self.__handle.sample.molecules[name].atoms = atoms.data
            self.__handle.sample.molecules[name].positions = positions
            self.__handle.sample.molecules[name].orientations = orientations

    @property
    def containing_box(self):
        """
        The box containing the sample

        """
        return self.__handle.sample.containing_box

    @containing_box.setter
    def containing_box(self, box):
        """
        The box containing the sample

        """
        self.__handle.sample.containing_box = box

    @property
    def centre(self):
        """
        The centre of the shape

        """
        return self.__handle.sample.centre

    @centre.setter
    def centre(self, centre):
        """
        The centre of the shape

        """
        self.__handle.sample.centre = centre

    @property
    def shape(self):
        """
        The ideal shape of the sample

        """
        return self.__handle.sample.shape

    @shape.setter
    def shape(self, shape):
        """
        The ideal shape of the sample

        """
        self.__handle.sample.shape = shape

    @property
    def bounding_box(self):
        """
        The bounding box

        Returns:
            tuple: lower, upper - the bounding box coordinates

        """
        return self.__handle.sample.bounding_box

    @property
    def molecules(self):
        """
        The list of molecule names

        Returns:
            list: The list of molecule names

        """
        return list(self.__handle.sample.molecules.keys())

    @property
    def number_of_molecular_models(self):
        """
        The number of molecular models

        Returns:
            int: The number of molecular models

        """
        return sum(
            molecule.positions.shape[0]
            for molecule in self.__handle.sample.molecules.values()
        )

    @property
    def number_of_molecules(self):
        """
        The number of molecules

        Returns:
            int: The number of molcules

        """
        return len(self.__handle.sample.molecules)

    def get_molecule(self, name):
        """
        Get the molcule data

        Args:
            name (str): The molecule name

        Returns:
            tuple (atoms, positions, orientations): The molecule data

        """
        return (
            AtomData(data=self.__handle.sample.molcules[name].atoms),
            self.__handle.sample.molecules[name].positions,
            self.__handle.sample.molecules[name].orientations,
        )

    def iter_molecules(self):
        """
        Iterate over the molecules

        Yields:
            tuple (name, data): The molecule data

        """
        for name in self.molecules:
            yield name, self.get_molecule(name)

    @property
    def number_of_atoms(self):
        """
        Returns:
            int: The number of atoms

        """
        return self.__handle.sample.atoms.number_of_atoms()

    def iter_atoms(self):
        """
        Iterate over the atoms

        Yields:
            object: The atom data

        """
        for name, z in self.__handle.sample.atoms.zbins():
            yield (z, z + self.zstep), AtomData(
                data=self.__handle.sample.atoms[name].atoms
            )

    def get_atoms_in_zrange(self, z0, z1, filter=None):
        """
        Get the subset of atoms within the field of view

        Args:
            z0 (float): The start of the z range
            z1 (float): The end of the z range
            filter (func): A function to filter the atoms

        Returns:
            object: The atom data

        """
        assert z0 < z1
        data = pandas.DataFrame()
        for name, z in self.__handle.sample.atoms.zbins():
            if z0 <= z + self.zstep and z <= z1:
                atoms = self.__handle.sample.atoms[name].atoms
                atoms = atoms[(atoms["z"] >= z0) & (atoms["z"] < z1)]
                if filter is not None:
                    atoms = filter(atoms)
                data = pandas.concat([data, atoms], ignore_index=True)
        return AtomData(data=data)

    def info(self):
        """
        Get some sample info

        Returns:
            str: Some sample info

        """
        lines = [
            "Sample information:",
            "    # Molecules:   %d" % self.number_of_molecules,
            "    # models:      %d" % self.number_of_molecular_models,
            "    # atoms:       %d" % self.number_of_atoms,
            "    Min x:         %.2f" % self.bounding_box[0][0],
            "    Min y:         %.2f" % self.bounding_box[0][1],
            "    Min z:         %.2f" % self.bounding_box[0][2],
            "    Max x:         %.2f" % self.bounding_box[1][0],
            "    Max y:         %.2f" % self.bounding_box[1][1],
            "    Max z:         %.2f" % self.bounding_box[1][2],
            "    Sample size x: %.2f" % self.dimensions[0],
            "    Sample size y: %.2f" % self.dimensions[1],
            "    Sample size z: %.2f" % self.dimensions[2],
            "    Min box x:     %.2f" % self.containing_box[0][0],
            "    Min box y:     %.2f" % self.containing_box[0][1],
            "    Min box z:     %.2f" % self.containing_box[0][2],
            "    Max box x:     %.2f" % self.containing_box[1][0],
            "    Max box y:     %.2f" % self.containing_box[1][1],
            "    Max box z:     %.2f" % self.containing_box[1][2],
            "    Centre x:      %.2f" % self.centre[0],
            "    Centre y:      %.2f" % self.centre[1],
            "    Centre z:      %.2f" % self.centre[2],
            "    Shape:         %s" % str(self.shape),
        ]
        return "\n".join(lines)


def random_uniform_rotation(size=1):
    """
    Return a uniform rotation vector sampled on a sphere

    Args:
        size (int): The number of vectors

    Returns:
        vector: The rotation vector

    """
    u1 = numpy.random.uniform(0, 1, size=size)
    u2 = numpy.random.uniform(0, 1, size=size)
    u3 = numpy.random.uniform(0, 1, size=size)
    theta = numpy.arccos(2 * u1 - 1)
    phi = 2 * pi * u2
    x = numpy.sin(theta) * numpy.cos(phi)
    y = numpy.sin(theta) * numpy.sin(phi)
    z = numpy.cos(theta)
    vector = numpy.array((x, y, z)).T
    vector *= 2 * pi * u3.reshape((u3.size, 1))
    return vector


def add_ice(sample, centre=None, shape=None, density=940.0):
    """
    Create a sample with just a load of water molecules

    Args:
        sample (object): The sample object
        shape (object): The shape description
        density (float): The water density

    Returns:
        object: The sample object

    """
    # Get the filename of the water.cif file
    filename = elfantasma.data.get_path("water.cif")

    # Create a single ribosome sample
    single_water = AtomData.from_ligand_file(filename)
    atoms = single_water.data[0:3]
    water_coords = atoms[["x", "y", "z"]].copy()
    water_coords -= water_coords.iloc[0].copy()

    # Set the volume in A^3
    if shape["type"] == "cube":
        length = shape["cube"]["length"]
        volume = length ** 3
        length_x = length
        length_y = length
        length_z = length
    elif shape["type"] == "cuboid":
        length_x = shape["cuboid"]["length_x"]
        length_y = shape["cuboid"]["length_y"]
        length_z = shape["cuboid"]["length_z"]
        volume = length_x * length_y * length_z
    elif shape["type"] == "cylinder":
        length = shape["cylinder"]["length"]
        radius = shape["cylinder"]["radius"]
        volume = pi * radius ** 2 * length
        length_x = 2 * radius
        length_y = 2 * radius
        length_z = length
    else:
        raise RuntimeError("Unknown shape")

    # Get the centre and offset
    centre_x, centre_y, centre_z = centre
    offset_x = centre_x - length_x / 2.0
    offset_y = centre_y - length_y / 2.0
    offset_z = centre_z - length_z / 2.0
    offset = numpy.array((offset_x, offset_y, offset_z), dtype="float32")

    # Determine the number of waters to place
    avogadros_number = scipy.constants.Avogadro
    molar_mass_of_water = 18.01528  # grams / mole
    density_of_water = density  # kg / m^3
    mass_of_water = (density_of_water * 1000) * (volume * 1e-10 ** 3)  # g
    number_of_waters = int(
        floor((mass_of_water / molar_mass_of_water) * avogadros_number)
    )

    # Van der Waals radius of water
    van_der_waals_radius = 2.7 / 2.0  # A

    # Compute the total volume in the spheres
    volume_of_spheres = (4.0 / 3.0) * pi * van_der_waals_radius ** 3 * number_of_waters

    # Create the grid
    grid = (
        int(floor(length_z / (2 * van_der_waals_radius))),
        int(floor(length_y / (2 * van_der_waals_radius))),
        int(floor(length_x / (2 * van_der_waals_radius))),
    )

    # Compute the node length and density
    node_length = max((length_z / grid[0], length_y / grid[1], length_x / grid[2]))
    sphere_density = number_of_waters / volume

    logger.info(
        f"Initialising Sphere Packer:\n"
        f"    Length X:           {length_x} A\n"
        f"    Length Y:           {length_y} A\n"
        f"    Length Z:           {length_z} A\n"
        f"    Volume:             {volume} A^3\n"
        f"    Density of water:   {density_of_water} Kg/m^3\n"
        f"    Mass of water:      {mass_of_water} g\n"
        f"    Number of waters:   {number_of_waters}\n"
        f"    Mean diameter:      {2*van_der_waals_radius} A\n"
        f"    Volume filled:      {100*volume_of_spheres/volume} %\n"
        f"    Packer grid:        ({grid[0]}, {grid[1]}, {grid[2]})\n"
        f"    Grid node length:   {node_length} A\n"
        f"    Density of spheres: {sphere_density} #/A^3\n"
    )

    # Create the sphere packer
    packer = elfantasma.freeze.SpherePacker(
        grid, node_length, sphere_density, van_der_waals_radius, max_iter=10
    )

    # Extract all the data
    logger.info("Generating water positions:")
    for z_index, z_slice in enumerate(packer):

        # Read the coordinates
        coords = []
        for node in z_slice:
            coords.extend(node)
        coords = numpy.array(coords, dtype="float32") + offset

        # Compute the rotation
        rotation = Rotation.from_rotvec(random_uniform_rotation(coords.shape[0]))

        # Rotate the Hydrogens around the Oxygen and translate
        O = rotation.apply(water_coords.iloc[0].copy()) + coords
        H1 = rotation.apply(water_coords.iloc[1].copy()) + coords
        H2 = rotation.apply(water_coords.iloc[2].copy()) + coords

        def create_atom_data(atomic_number, coords):

            # Create a new array
            def new_array(size, name, value):
                return (
                    numpy.zeros(shape=(size,), dtype=AtomData.column_data[name]) + value
                )

            # Create the new arrays
            atomic_number = new_array(coords.shape[0], "atomic_number", atomic_number)
            sigma = new_array(coords.shape[0], "sigma", 0.088)
            occupancy = new_array(coords.shape[0], "occupancy", 1.0)
            charge = new_array(coords.shape[0], "charge", 0)

            # Return the data frame
            return AtomData(
                data=pandas.DataFrame(
                    {
                        "atomic_number": atomic_number,
                        "x": coords[:, 0],
                        "y": coords[:, 1],
                        "z": coords[:, 2],
                        "sigma": sigma,
                        "occupancy": occupancy,
                        "charge": charge,
                    }
                )
            )

        # Add the sample atoms
        sample.add_atoms(create_atom_data(8, O))
        sample.add_atoms(create_atom_data(1, H1))
        sample.add_atoms(create_atom_data(1, H2))

        # Log some info
        logger.info(
            "    Z slice %d/%d: Num molecules: %d" % (z_index, len(packer), O.shape[0])
        )

    # Print some output
    logger.info(f"Sphere packer: Num unplaced:  {packer.num_unplaced_samples()}")

    # Return the sample
    return sample


def distribute_boxes_uniformly(volume_box, boxes, max_tries=1000):
    """
    Find n random non overlapping positions for cuboids within a volume

    Args:
        volume_box (array): The size of the volume
        boxes (array): The list of boxes
        max_tries (int): The maximum tries per cube

    Returns:
        list: A list of centre positions

    """

    # Cast to numpy array
    volume_lower, volume_upper = numpy.array(volume_box)
    boxes = numpy.array(boxes)

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
        lower = volume_lower + box_size / 2
        upper = volume_upper - box_size / 2
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


def random_uniform_rotation(size=1):
    """
    Return a uniform rotation vector sampled on a sphere

    Args:
        size (int): The number of vectors

    Returns:
        vector: The rotation vector

    """
    u1 = numpy.random.uniform(0, 1, size=size)
    u2 = numpy.random.uniform(0, 1, size=size)
    u3 = numpy.random.uniform(0, 1, size=size)
    theta = numpy.arccos(2 * u1 - 1)
    phi = 2 * pi * u2
    x = numpy.sin(theta) * numpy.cos(phi)
    y = numpy.sin(theta) * numpy.sin(phi)
    z = numpy.cos(theta)
    vector = numpy.array((x, y, z)).T
    vector *= 2 * pi * u3.reshape((u3.size, 1))
    return vector


def shape_bounding_box(centre, shape):
    """
    Compute the shape bounding box

    Args:
        centre (array): The centre of the shape
        shape (object): The shape

    Returns:
        array: bounding box

    """

    def cube_bounding_box(cube):
        length = cube["length"]
        return ((0, 0, 0), (length, length, length))

    def cuboid_bounding_box(cuboid):
        length_x = cuboid["length_x"]
        length_y = cuboid["length_y"]
        length_z = cuboid["length_z"]
        return ((0, 0, 0), (length_x, length_y, length_z))

    def cylinder_bounding_box(cylinder):
        length = cylinder["length"]
        radius = cylinder["radius"]
        return ((0, 0, 0), (2 * radius, 2 * radius, length))

    # The bounding box
    x0, x1 = numpy.array(
        {
            "cube": cube_bounding_box,
            "cuboid": cuboid_bounding_box,
            "cylinder": cylinder_bounding_box,
        }[shape["type"]](shape[shape["type"]])
    )

    # The offset
    offset = centre - (x1 + x0) / 2.0

    # Return the bounding box
    return (x0 + offset, x1 + offset)


def shape_enclosed_box(centre, shape):
    """
    Compute the shape enclosed box

    Args:
        centre (array): The centre of the shape
        shape (object): The shape

    Returns:
        array: enclosed box

    """

    def cube_enclosed_box(cube):
        length = cube["length"]
        return ((0, 0, 0), (length, length, length))

    def cuboid_enclosed_box(cuboid):
        length_x = cuboid["length_x"]
        length_y = cuboid["length_y"]
        length_z = cuboid["length_z"]
        return ((0, 0, 0), (length_x, length_y, length_z))

    def cylinder_enclosed_box(cylinder):
        length = cylinder["length"]
        radius = cylinder["radius"]
        return ((0, 0, 0), (length, sqrt(2) * radius, sqrt(2) * radius))

    # The enclosed box
    x0, x1 = numpy.array(
        {
            "cube": cube_enclosed_box,
            "cuboid": cuboid_enclosed_box,
            "cylinder": cylinder_enclosed_box,
        }[shape["type"]](shape[shape["type"]])
    )

    # The offset
    offset = centre - (x1 + x0) / 2.0

    # Return the bounding box
    return (x0 + offset, x1 + offset)


def is_shape_inside_box(box, centre, shape):
    """
    Check if the shape is inside the bounding box

    Args:
        box (array): The bounding box
        centre (array): The centre of the shape
        shape (object): The shape

    Returns:
        bool: Is the shape inside the box

    """

    # Get the shape bounding box
    (x0, y0, z0), (x1, y1, z1) = shape_bounding_box(centre, shape)

    # Check the bounding box is inside the containing box
    return (
        x0 >= 0
        and y0 >= 0
        and z0 >= 0
        and x1 <= box[0]
        and y1 <= box[1]
        and z1 <= box[2]
    )


def is_box_inside_shape(box, centre, shape):
    """
    Check if the box is inside the shape

    Args:
        box (array): The bounding box
        centre (array): The centre of the shape
        shape (object): The shape

    Returns:
        bool: Is the box is inside shape

    """

    def is_box_inside_cube(x0, x1, centre, cube):
        length = cube["length"]
        return (
            x0[0] >= centre[0] - length / 2.0
            and x0[1] >= centre[1] - length / 2.0
            and x0[2] >= centre[2] - length / 2.0
            and x1[0] <= centre[0] + length / 2.0
            and x1[1] <= centre[1] + length / 2.0
            and x1[2] <= centre[2] + length / 2.0
        )

    def is_box_inside_cuboid(x0, x1, centre, cuboid):
        length_x = cuboid["length_x"]
        length_y = cuboid["length_y"]
        length_z = cuboid["length_z"]
        return (
            x0[0] >= centre[0] - length_x / 2.0
            and x0[1] >= centre[1] - length_y / 2.0
            and x0[2] >= centre[2] - length_z / 2.0
            and x1[0] <= centre[0] + length_x / 2.0
            and x1[1] <= centre[1] + length_y / 2.0
            and x1[2] <= centre[2] + length_z / 2.0
        )

    def is_box_inside_cylinder(x0, x1, centre, cylinder):
        length = cylinder["length"]
        radius = cylinder["radius"]
        return (
            (x0[1] - centre[1]) ** 2 + (x0[2] - centre[2]) ** 2 <= radius ** 2
            and (x1[1] - centre[1]) ** 2 + (x0[2] - centre[2]) ** 2 <= radius ** 2
            and (x0[1] - centre[1]) ** 2 + (x1[2] - centre[2]) ** 2 <= radius ** 2
            and (x1[1] - centre[1]) ** 2 + (x1[2] - centre[2]) ** 2 <= radius ** 2
            and x0[0] >= centre - length / 2.0
            and x1[0] < centre + length / 2.0
        )

    return {
        "cube": is_box_inside_cube,
        "cuboid": is_box_inside_cuboid,
        "cylinder": is_box_inside_cylinder,
    }[shape["type"]](box[0], box[1], centre, shape[shape["type"]])


def dimensions_are_valid(box, centre, shape):
    """
    Check the dimensions are valid

    Args:
        box (array): The containing box
        centre (array): The centre of the sample in the box
        shape (object): The shape of the object

    Returns:
        bool: Are the dimensions valid

    """

    # Return if the shape is inside the bounding box
    return is_shape_inside_box(box, centre, shape)


def load(filename, mode="r"):
    """
    Load the sample

    Args:
        filename (str): The filename of the sample
        mode (str): The opening mode

    Returns:
        object: The test sample

    """
    return Sample(filename, mode=mode)


def new(filename, box=None, centre=None, shape=None, ice=None, **kwargs):
    """
    Create the sample

    Args:
        filename (str): The filename of the sample
        box (list): The containing box of the sample
        centre (list): The centering of the sample in the box
        shape (object): The shape of the sample
        ice (object): The ice description

    Returns:
        object: The test sample

    """

    # Check the dimensions are valid
    assert dimensions_are_valid(box, centre, shape)

    # Create the sample
    sample = Sample(filename, mode="w")

    # Set the sample box and shape
    sample.containing_box = ((0, 0, 0), box)
    sample.centre = centre
    sample.shape = shape

    # Add some ice
    if ice["generate"]:
        add_ice(sample, centre, shape, ice["density"])

    # Print some info
    logger.info(sample.info())


def add_single_molecule(sample, name):
    """
    Create a sample with a single molecule

    The molecule will be positioned at the centre

    Args:
        sample (object): The sample object
        name (str): The name of the molecule to add

    Returns:
        object: The sample

    """
    logger.info("Adding single %s molecule" % name)

    # Get the filename of the 4v5d.cif file
    filename = elfantasma.data.get_path("%s.cif" % name)

    # Get the atom data
    atoms = AtomData.from_gemmi_file(filename)
    atoms.data = recentre(atoms.data)

    # Get atom data bounds
    coords = atoms.data[["x", "y", "z"]]
    x0 = numpy.min(coords, axis=0) + sample.centre
    x1 = numpy.max(coords, axis=0) + sample.centre

    # Check the coords
    assert is_box_inside_shape((x0, x1), sample.centre, sample.shape)

    # Print some info
    logger.info(
        "\n".join(
            (
                "Name:   %s" % name,
                "Min x:  %.2f" % x0[0],
                "Min y:  %.2f" % x0[1],
                "Min z:  %.2f" % x0[2],
                "Max x:  %.2f" % x1[0],
                "Max y:  %.2f" % x1[1],
                "Max z:  %.2f" % x1[2],
                "Size x: %.2f" % (x1[0] - x0[0]),
                "Size y: %.2f" % (x1[1] - x0[1]),
                "Size z: %.2f" % (x1[2] - x0[2]),
                "Pos x:  %.2f" % (sample.centre[0]),
                "Pos y:  %.2f" % (sample.centre[1]),
                "Pos z:  %.2f" % (sample.centre[2]),
            )
        )
    )

    # Add the molecule
    sample.add_molecule(
        atoms, positions=[sample.centre], orientations=[(0, 0, 0)], name=name
    )

    # Print some information
    logger.info(sample.info())

    # Return the sample
    return sample


def add_multiple_molecules(sample, molecules):
    """
    Create a sample with multiple molecules

    The molecules will be positioned randomly in the sample

    Args:
        sample (object): The sample object
        molecules (object): The molecules

    Returns:
        object: The sample

    """

    # Setup some arrays
    atom_data = {}
    all_labels = []
    all_boxes = []
    all_positions = []
    all_orientations = []

    # Generate the orientations and boxes
    for name, number in molecules.items():

        # Skip if number is zero
        if number == 0:
            continue

        # Print some info
        logger.info("Adding %d %s molecules" % (number, name))

        # Get the filename of the 4v5d.cif file
        filename = elfantasma.data.get_path("%s.cif" % name)

        # Get the atom data
        atoms = AtomData.from_gemmi_file(filename)
        atoms.data = recentre(atoms.data)
        atom_data[name] = atoms

        # Generate some random orientations
        logger.info("    Generating random orientations")
        for rotation in Rotation.from_rotvec(random_uniform_rotation(number)):
            coords = rotation.apply(atoms.data[["x", "y", "z"]])
            all_orientations.append(rotation.as_rotvec())
            all_boxes.append(numpy.max(coords, axis=0) - numpy.min(coords, axis=0))
            all_labels.append(name)

    # Put the molecules in the sample
    logger.info("Placing molecules:")
    all_positions = distribute_boxes_uniformly(
        shape_enclosed_box(sample.centre, sample.shape), all_boxes
    )

    # Set the positions and orientations by molecule
    positions = defaultdict(list)
    orientations = defaultdict(list)
    for label, rotation, position, box in zip(
        all_labels, all_orientations, all_positions, all_boxes
    ):

        # Get atom data bounds
        x0 = box[0] + sample.centre
        x1 = box[1] + sample.centre

        # Check the coords
        assert is_box_inside_shape((x0, x1), sample.centre, sample.shape)

        # Construct arrays
        positions[label].append(position)
        orientations[label].append(rotation)

        # Print some info
        logger.info(
            "    placing %s at (%.2f, %.2f, %.2f) with orientation (%.2f, %.2f, %.2f)"
            % ((label,) + tuple(position) + tuple(rotation))
        )

    # Add the molecules
    for name in atom_data.keys():
        sample.add_molecule(
            atom_data[name],
            positions=positions[name],
            orientations=orientations[name],
            name=name,
        )


def add_molecules(filename, molecules=None, **kwargs):
    """
    Take a sample and add a load of molecules

    Args:
        filename (str): The filename of the sample
        molecules (dict): The molecules

    Returns:
        object: The test sample

    """

    # Total number of molecules
    total_number_of_molecules = sum(molecules.values())

    # Open the sample
    sample = Sample(filename, mode="r+")

    # Add the molecules
    if total_number_of_molecules == 0:
        raise RuntimeError("Need at least 1 molecule")
    elif total_number_of_molecules == 1:
        name = [key for key, value in molecules.items() if value > 0][0]
        add_single_molecule(sample, name)
    else:
        add_multiple_molecules(sample, molecules)

    # Show some info
    logger.info(sample.info())

    # Return the sample
    return sample
