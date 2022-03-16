#
# parakeet.sample.py
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
import numpy as np
import pandas
import scipy.constants
import time
from math import pi, sqrt, floor, ceil
from scipy.spatial.transform import Rotation
import parakeet.data
import parakeet.freeze

try:
    import multem
except ImportError:
    pass

# np.random.seed(0)

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
    b_iso = atom.b_iso
    return b_iso / (8 * pi**2)


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
    coords = atom_data[["x", "y", "z"]].to_numpy()
    coords += np.array(translation, dtype=coords.dtype)
    return atom_data.assign(x=coords[:, 0], y=coords[:, 1], z=coords[:, 2])


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
    coords = atom_data[["x", "y", "z"]].to_numpy()

    # Compute the translation
    translation = np.array(position, dtype=coords.dtype) - coords.mean(axis=0)

    # Do the translation
    return translate(atom_data, translation)


def number_of_water_molecules(volume, density=940.0):
    """
    Compute the number of water molecules

    Args:
        volume (float): The volume A^3
        density (float): The density Kg/m^3

    Returns:
        The number of water molcules

    """

    # Determine the number of waters to place
    avogadros_number = scipy.constants.Avogadro
    molar_mass_of_water = 18.01528  # grams / mole
    density_of_water = density  # kg / m^3
    mass_of_water = (density_of_water * 1000) * (volume * 1e-10**3)  # g
    return int(floor((mass_of_water / molar_mass_of_water) * avogadros_number))


def random_uniform_rotation(size=1):
    """
    Return a uniform rotation vector sampled on a sphere

    Args:
        size (int): The number of vectors

    Returns:
        vector: The rotation vector

    """
    u1 = np.random.uniform(0, 1, size=size)
    u2 = np.random.uniform(0, 1, size=size)
    u3 = np.random.uniform(0, 1, size=size)
    theta = np.arccos(2 * u1 - 1)
    phi = 2 * pi * u2
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    vector = np.array((x, y, z)).T
    vector *= 2 * pi * u3.reshape((u3.size, 1))
    return vector


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
    volume_lower, volume_upper = np.array(volume_box)
    boxes = np.array(boxes)

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
            q = np.random.uniform(lower, upper)
            if len(positions) == 0 or not overlapping(
                positions, box_sizes, q, box_size
            ):
                positions.append(q)
                box_sizes.append(box_size)
                break
            num_tries += 1
            if num_tries > max_tries:
                print("Number of particles placed = %d" % (len(box_sizes)))
                raise RuntimeError(f"Unable to place cube of size {box_size}")

    # Return the cube positions
    return positions


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
        radius = np.mean(cylinder["radius"])
        return ((0, 0, 0), (2 * radius, length, 2 * radius))

    # The bounding box
    x0, x1 = np.array(
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


def shape_bounding_cylinder(centre, shape):
    """
    Compute the shape cylinder

    Args:
        centre (array): The centre of the shape
        shape (object): The shape

    Returns:
        array: bounding cylinder

    """

    def cube_bounding_cylinder(cube):
        length = cube["length"]
        return (length, sqrt(length**2 + length**2))

    def cuboid_bounding_cylinder(cuboid):
        length_x = cuboid["length_x"]
        length_y = cuboid["length_y"]
        length_z = cuboid["length_z"]
        return (length_x, sqrt(length_y**2 + length_z**2))

    def cylinder_bounding_cylinder(cylinder):
        length = cylinder["length"]
        radius = cylinder["radius"]
        return (length, radius)

    # The bounding box
    length, radius = {
        "cube": cube_bounding_cylinder,
        "cuboid": cuboid_bounding_cylinder,
        "cylinder": cylinder_bounding_cylinder,
    }[shape["type"]](shape[shape["type"]])

    # Return the bounding box
    return (centre, length, radius)


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
        radius = np.mean(cylinder["radius"])
        return (
            (radius * (1 - 1 / sqrt(2)), 0, radius * (1 - 1 / sqrt(2))),
            (radius * (1 + 1 / sqrt(2)), length, radius * (1 + 1 / sqrt(2))),
        )

    # The enclosed box
    x0, x1 = np.array(
        {
            "cube": cube_enclosed_box,
            "cuboid": cuboid_enclosed_box,
            "cylinder": cylinder_enclosed_box,
        }[shape["type"]](shape[shape["type"]])
    )

    # The offset
    offset = centre - (x1 + x0) / 2.0

    # The margin
    margin = np.array(shape.get("margin", (0, 0, 0)))

    # Return the bounding box
    return (x0 + offset + margin, x1 + offset - margin)


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
        radius = np.mean(cylinder["radius"])
        return (
            (x0[0] - centre[0]) ** 2 + (x0[2] - centre[2]) ** 2 <= radius**2
            and (x1[0] - centre[0]) ** 2 + (x0[2] - centre[2]) ** 2 <= radius**2
            and (x0[0] - centre[0]) ** 2 + (x1[2] - centre[2]) ** 2 <= radius**2
            and (x1[0] - centre[0]) ** 2 + (x1[2] - centre[2]) ** 2 <= radius**2
            and x0[1] >= centre[1] - length / 2.0
            and x1[1] < centre[1] + length / 2.0
        )

    return {
        "cube": is_box_inside_cube,
        "cuboid": is_box_inside_cuboid,
        "cylinder": is_box_inside_cylinder,
    }[shape["type"]](box[0], box[1], centre, shape[shape["type"]])


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
            if len(data) > 0:
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
        if len(self.data) > 0:
            coords = self.data[["x", "y", "z"]].to_numpy()
            coords = Rotation.from_rotvec(vector).apply(coords).astype(coords.dtype)
            self.data["x"] = coords[:, 0]
            self.data["y"] = coords[:, 1]
            self.data["z"] = coords[:, 2]
        return self

    def translate(self, translation):
        """
        Translate the atom data

        Args:
            translation (array): The translation

        """
        if len(self.data) > 0:
            coords = self.data[["x", "y", "z"]].to_numpy()
            coords += np.array(translation, dtype=coords.dtype)
            self.data["x"] = coords[:, 0]
            self.data["y"] = coords[:, 1]
            self.data["z"] = coords[:, 2]
        return self

    def to_multem(self):
        """
        Convert to a multem atom list

        Returns:
            object: A multem atom list

        """
        if len(self.data) == 0:
            return multem.AtomList()
        return multem.AtomList(
            zip(
                self.data["atomic_number"].astype("uint8"),
                self.data["x"].astype("float32"),
                self.data["y"].astype("float32"),
                self.data["z"].astype("float32"),
                self.data["sigma"].astype("float32"),
                [float(1) for i in range(self.data.shape[0])],
                [int(0) for i in range(self.data.shape[0])],
                self.data["charge"].astype("uint8"),
            )
        )

    def rows(self):
        """
        Iterate through atoms rows

        Yield:
            tuple: atom data

        """
        return zip(
            self.data["atomic_number"],
            self.data["x"],
            self.data["y"],
            self.data["z"],
            self.data["sigma"],
            self.data["occupancy"],
            self.data["charge"],
        )

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
                            assert atom.element.atomic_number > 0
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

    @classmethod
    def from_text_file(Class, filename):
        """
        Read the sample from a text file

        Args:
            filename (str): The input filename

        Returns:
            object: The Sample object

        """

        # The column order
        column_info = ["atomic_number", "x", "y", "z", "sigma", "occupancy", "charge"]

        # Iterate through the atoms
        def iterate_atoms(infile):
            for line in infile.readlines():
                tokens = line.split()
                yield (
                    int(tokens[0]),
                    float(tokens[1]),
                    float(tokens[2]),
                    float(tokens[3]),
                    float(tokens[4]),
                    float(tokens[5]),
                    int(tokens[6]),
                )

        # Create a dictionary of column data
        def create_atom_data(infile):
            return dict(
                (name, pandas.Series(data, dtype=AtomData.column_data[name]))
                for data, name in zip(zip(*iterate_atoms(infile)), column_info)
            )

        # Return the atom data
        with open(filename, "r") as infile:
            return AtomData(data=pandas.DataFrame(create_atom_data(infile)))


class SampleHDF5Adapter(object):
    """
    A class to handle the HDF5 group and dataset creation

    """

    class AtomsSubGroup(object):
        """
        A class to handle the atom sub groups

        """

        def __init__(self, handle, name):
            """
            Save the group handle

            Args:
                handle (object): The group handle
                name (str): The dataset name

            """

            # Save the handle
            self.__handle = handle
            self.name = name

            # Create the datasets
            if name not in self.__handle:
                dtype = [(key, value) for key, value in AtomData.column_data.items()]
                self.__handle.create_dataset(
                    name, (0,), maxshape=(None,), dtype=dtype, chunks=True
                )

        @property
        def atoms(self):
            """
            Return the atom data in the atom sub group

            """
            return pandas.DataFrame.from_records(self.__handle[self.name][:])

        @atoms.setter
        def atoms(self, data):
            """
            Set the atom data in the sub group

            """

            # Check size, resize and set data
            self.__handle[self.name].resize((len(data),))
            self.__handle[self.name][:] = data.to_records()

        def extend(self, data):
            """
            Set the atom data in the sub group

            """
            # Get the size
            i0 = self.__handle[self.name].shape[0]
            i1 = i0 + data.shape[0]

            # Check size, resize and set data
            assert i0 == self.__handle[self.name].shape[0]
            self.__handle[self.name].resize((i1,))
            self.__handle[self.name][i0:i1] = data.to_records()

        def __len__(self):
            """
            Returns the number of atoms

            """
            if len(self.__handle) == 0:
                return 0
            return self.__handle[self.name].shape[0]

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
            return SampleHDF5Adapter.AtomsSubGroup(self.__handle, item)

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
            for key in self.__handle.keys():
                yield key, SampleHDF5Adapter.AtomsSubGroup(self.__handle, key)

        def values(self):
            """
            Iterate through the subgroups

            """
            for key in self.__handle.keys():
                yield SampleHDF5Adapter.AtomsSubGroup(self.__handle, key)

        def bins(self):
            """
            Iterate through the z bins

            """
            for key in self.keys():
                x, y, z = [int(k.split("=")[1]) for k in key.split(";")]
                yield key, np.array((x, y, z))

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
            return pandas.DataFrame.from_records(self.__handle["atoms"][:])

        @property
        def positions(self):
            """
            Get the positions

            """
            return self.__handle["positions"][:]

        @property
        def orientations(self):
            """
            Get the orientations

            """
            return self.__handle["orientations"][:]

        @atoms.setter
        def atoms(self, data):
            """
            Set the atom data

            """
            dtype = [(key, value) for key, value in AtomData.column_data.items()]
            self.__handle.create_dataset(
                "atoms", data=data.to_records(), dtype=dtype, chunks=True
            )

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
                return np.zeros(shape=(2, 3), dtype="float32")
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
                return np.zeros(shape=(2, 3), dtype="float32")
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
                return np.zeros(shape=(3,), dtype="float32")
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
            if isinstance(shape, bytes):
                shape = shape.decode("utf-8")
            return {
                "type": shape,
                "margin": self.__handle["shape"].attrs.get("margin", [0, 0, 0]),
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

            # Set the margin
            self.__handle["shape"].attrs["margin"] = shape.get("margin", (0, 0, 0))

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

        # The step between datasets, A 500^3 A^3 volume has around 4M water molecules
        # This seems to be a reasonable division size
        self.step = 100_000  # A

    def close(self):
        """
        Close the HDF5 file

        """
        self.__handle.close()

    def atoms_dataset_name(self, x):
        """
        Get the atom dataset name

        """
        return "X=%06d; Y=%06d; Z=%06d" % tuple(x)

    def atoms_dataset_range(self, x0, x1):
        """
        Get the atom dataset range

        Args:
            x0 (array): The minimum coord
            x1 (array): The maximum coord

        Yields:
            tuple: The coordinates of the sub ranges

        """
        x0 = (np.floor(np.array(x0) / float(self.step)) * self.step).astype("int32")
        x1 = (np.ceil(np.array(x1) / float(self.step)) * self.step).astype("int32")
        for z in range(x0[2], x1[2], self.step):
            for y in range(x0[1], x1[1], self.step):
                for x in range(x0[0], x1[0], self.step):
                    c = np.array((x, y, z))
                    yield c, c + self.step

    def atoms_dataset_range_3d(self, x0, x1):
        """
        Get the atom dataset range

        Args:
            x0 (array): The minimum coord
            x1 (array): The maximum coord

        Yields:
            tuple: The coordinates of the sub ranges

        """
        x0 = (np.floor(np.array(x0) / float(self.step)) * self.step).astype("int32")
        x1 = (np.ceil(np.array(x1) / float(self.step)) * self.step).astype("int32")
        shape = np.floor((x1 - x0) / self.step).astype("int32")
        X, Y, Z = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
        indices = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
        return indices, x0 + indices * self.step

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
        coords = atoms.data[["x", "y", "z"]].to_numpy()

        # Get the grid index and coordinates
        grid_index, x_min = self.atoms_dataset_range_3d(
            coords.min(axis=0), coords.max(axis=0)
        )

        # Compute the index of each coordinate
        index = np.floor((coords - x_min[0]) / self.step).astype("int32")
        size = np.max(grid_index, axis=0) + 1
        index = index[:, 2] + index[:, 1] * size[2] + index[:, 0] * size[2] * size[1]

        # Create an index list for each subgroup by splitting the sorted indices
        sorted_index = np.argsort(index)
        split_index, splits = np.unique(index[sorted_index], return_index=True)
        index_list = np.split(sorted_index, splits[1:])

        # Ensure the number is correct
        assert sum(map(len, index_list)) == atoms.data.shape[0]
        # assert np.max(split_index) < len(x_min)

        # Add the atoms to the subgroups
        for i in range(len(split_index)):
            try:
                name = self.atoms_dataset_name(x_min[split_index[i]])
                data = atoms.data.iloc[index_list[i]]
                self.__handle.sample.atoms[name].extend(data)
            except Exception:
                print("SKIPPING: ", i)

    def del_atoms(self, deleter):
        """
        Delete atoms in the given range

        Args:
            deleter (func): Check atoms and return only those we want to keep

        """

        # Iterate over dataset ranges
        for x_min, x_max in self.atoms_dataset_range(deleter.x0, deleter.x1):
            name = self.atoms_dataset_name(x_min)
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
        positions = np.array(positions, dtype="float32")
        orientations = np.array(orientations, dtype="float32")
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
    def shape_box(self):
        """
        Return the shape box

        """
        return shape_bounding_box(self.centre, self.shape)

    @property
    def shape_radius(self):
        """
        Return a radius

        """
        return shape_bounding_cylinder(self.centre, self.shape)[2]

    @property
    def bounding_box(self):
        """
        The bounding box

        Returns:
            tuple: lower, upper - the bounding box coordinates

        """
        return self.__handle.sample.bounding_box

    @bounding_box.setter
    def bounding_box(self, b):
        """
        The bounding box

        Returns:
            tuple: lower, upper - the bounding box coordinates

        """
        self.__handle.sample.bounding_box = b

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
            AtomData(data=self.__handle.sample.molecules[name].atoms),
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

    def iter_atom_groups(self):
        """
        Iterate over the atom groups

        Yields:
            object: The atom group

        """
        return self.__handle.sample.atoms.bins()

    def iter_atoms(self):
        """
        Iterate over the atoms

        Yields:
            object: The atom data

        """
        for name, x in self.__handle.sample.atoms.bins():
            yield (x, x + self.step), AtomData(
                data=self.__handle.sample.atoms[name].atoms
            )

    def get_atoms(self):
        """
        Get all the atoms

        Returns:
            object: The atom data

        """

        # Iterate over dataset ranges
        data = pandas.DataFrame()
        for (x0, x1), atoms in self.iter_atoms():
            data = pandas.concat([data, atoms.data], ignore_index=True)
        return AtomData(data=data)

    def get_atoms_in_group(self, name):
        """
        Get the atoms in the group

        """
        return AtomData(data=self.__handle.sample.atoms[name].atoms)

    def get_atoms_in_range(self, x0, x1, filter=None):
        """
        Get the subset of atoms within the field of view

        Args:
            x0 (float): The start of the z range
            x1 (float): The end of the z range
            filter (func): A function to filter the atoms

        Returns:
            object: The atom data

        """

        # Cast input
        x0 = np.array(x0)
        x1 = np.array(x1)

        # Check input
        assert (x0 < x1).all()

        # Iterate over dataset ranges
        data = pandas.DataFrame()
        for x_min, x_max in self.atoms_dataset_range(x0, x1):
            name = self.atoms_dataset_name(x_min)
            atoms = self.__handle.sample.atoms[name].atoms
            coords = atoms[["x", "y", "z"]]
            atoms = atoms[((coords >= x0) & (coords < x1)).all(axis=1)]
            if filter is not None:
                atoms = filter(atoms)
            data = pandas.concat([data, atoms], ignore_index=True)
        return AtomData(data=data)

    def get_atoms_in_fov(self, x0, x1):
        """
        Get the subset of atoms within the field of view

        Args:
            x0 (float): The start of the z range
            x1 (float): The end of the z range

        Returns:
            object: The atom data

        """

        # Cast input
        x0 = np.array(x0)
        x1 = np.array(x1)

        # Check input
        assert (x0 < x1).all()

        # Iterate over dataset ranges
        atoms = self.get_atoms().data
        if len(atoms) > 0:
            coords = atoms[["x", "y"]]
            atoms = atoms[((coords >= x0) & (coords < x1)).all(axis=1)]
        return AtomData(data=atoms)

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
        for name, molecule in self.iter_molecules():
            molecule_lines = [
                "    Molecule: %s" % name,
                "        Positions:\n%s"
                % "\n".join(["%s%s" % (" " * 12, str(p)) for p in molecule[1]]),
                "        Orientations:\n%s"
                % "\n".join(["%s%s" % (" " * 12, str(o)) for o in molecule[2]]),
            ]
            lines.extend(molecule_lines)

        return "\n".join(lines)


class AtomSliceExtractor(object):
    """
    A class to select atoms given a rotation and translation

    """

    class Slice(object):
        """
        A class to keep the slice result

        """

        def __init__(self, data, x_min, x_max):
            self.atoms = data
            self.x_min = x_min
            self.x_max = x_max

    def __init__(self, sample, translation, rotation, x0, x1, thickness=10):
        """
        Preprocess the slices

        The sample will be translated along the X axis and then rotated. All
        atoms within the range x0 -> x1 will then be extracted in a number of
        slices of given thickness

        Args:
            sample (object): The sample object
            translation (float): The translation along X
            rotation (float): The rotation around the X axis
            x0 (array): The (x, y) minimum coordinate
            x1 (array): The (x, y) maximum coordinate
            thickness (float): The slice thickness in A

        """

        # Check if the rectangles overlaps with each other
        def overlapping(a0, a1, b0, b1):
            if not (
                a0[0] > b1[0]
                or a1[0] < b0[0]
                or a0[1] > b1[1]
                or a1[1] < b0[1]
                or a0[2] > b1[2]
                or a1[2] < b0[2]
            ):
                return True
            return False

        # Set the items
        self.sample = sample
        self.centre = sample.centre
        self.translation = translation
        self.rotation = rotation * pi / 180.0
        self.x0 = x0
        self.x1 = x1
        self.thickness = thickness

        # Loop through the atom groups
        group_coords = []
        group_names = []
        for name, xmin in self.sample.iter_atom_groups():
            xmax = xmin + self.sample.step

            # The coordinates of the corners of the group
            y = np.array(
                [
                    (xmin[0], xmin[1], xmin[2]),
                    (xmin[0], xmin[1], xmax[2]),
                    (xmin[0], xmax[1], xmin[2]),
                    (xmin[0], xmax[1], xmax[2]),
                    (xmax[0], xmin[1], xmin[2]),
                    (xmax[0], xmin[1], xmax[2]),
                    (xmax[0], xmax[1], xmin[2]),
                    (xmax[0], xmax[1], xmax[2]),
                ]
            )

            # Rotate the group and translate the corners of the group and
            # append to the list of groupd
            group_names.append(name)
            group_coords.append(
                Rotation.from_rotvec((self.rotation, 0, 0)).apply(y - self.centre)
                + self.centre
                - (self.translation, 0, 0)
            )

        # Compute the min and max coordinates of all the rotated groups and
        # compute the number of slices we will have
        group_coords = np.array(group_coords)
        min_x = np.min(group_coords.reshape((-1, 3)), axis=0)
        max_x = np.max(group_coords.reshape((-1, 3)), axis=0)
        self.min_z = max(0, min_x[2])
        self.max_z = max_x[2]
        num_slices = ceil((self.max_z - self.min_z) / self.thickness)

        # Loop through the groups. Compute the minimum and maximum coordinates
        # of the group and create a new box. Loop through the slices and check
        # if the box overlaps with the slice box. If it does then add the name
        # of the group to the list of groups in that slice.
        self.__groups_in_slice = defaultdict(list)
        for name, group in zip(group_names, group_coords):
            min_x = np.min(group, axis=0)
            max_x = np.max(group, axis=0)
            for i in range(num_slices):
                z0 = self.min_z + i * thickness
                z1 = self.min_z + (i + 1) * thickness
                if overlapping(min_x, max_x, (x0[0], x0[1], z0), (x1[0], x1[1], z1)):
                    self.__groups_in_slice[i].append(name)

    def __getitem__(self, index):
        """
        Get the slice of atom data

        Args:
            index (int): The slice index

        Returns:
            object: The slice

        """

        # Get the min and max coordinate of the slice
        z0 = self.min_z + index * self.thickness
        z1 = self.min_z + (index + 1) * self.thickness
        x_min = (self.x0[0], self.x0[1], z0)
        x_max = (self.x1[0], self.x1[1], z1)

        # Rotate and translate the atoms and select only those atoms within the
        # slice
        def filter_atoms(atoms):
            coords = atoms[["x", "y", "z"]].to_numpy()
            coords = (
                Rotation.from_rotvec((0, self.rotation, 0)).apply(coords - self.centre)
                + self.centre
                - (0, self.translation, 0)
            ).astype("float32")
            atoms["x"] = coords[:, 0]
            atoms["y"] = coords[:, 1]
            atoms["z"] = coords[:, 2]
            atoms = atoms[((coords >= x_min) & (coords < x_max)).all(axis=1)]
            return atoms

        # Check the index
        assert index >= 0 and index < len(self.__groups_in_slice)

        # Loop through the ranges and extract the data
        data = []
        for name in self.__groups_in_slice[index]:
            atoms = filter_atoms(self.sample.get_atoms_in_group(name).data)
            if len(atoms) > 0:
                data.append(atoms)

        # Cat data
        if len(data) > 0:
            data = pandas.concat(data)
        else:
            data = pandas.DataFrame()

        # Return the slice
        return AtomSliceExtractor.Slice(AtomData(data=data), x_min, x_max)

    def __len__(self):
        """
        Returns:
            int: The number of slices

        """
        return len(self.__groups_in_slice)

    def __iter__(self):
        """
        Iterate through the slices

        """
        for i in range(len(self)):
            item = self[i]
            if len(item.atoms.data) > 0:
                yield self[i]


class AtomDeleter(object):
    """
    A class to delete atoms so we can place a molecule

    Create a grid with the molecule atom positions. Then test the atom
    positions in the sample to and only select those atoms which don't fall on
    molecule atom positions.

    """

    def __init__(self, atoms, position=(0, 0, 0), rotation=(0, 0, 0)):
        """
        Create the grid

        Args:
            atoms (object): the molcule atoms
            position (array): The molecule position
            rotation (array): The molecule rotation

        """

        # Compute the transformed coordinates
        coords = (
            Rotation.from_rotvec(rotation).apply(atoms.data[["x", "y", "z"]]) + position
        )

        # The min and max coordinates
        border = 1  # A
        self.x0 = np.floor(np.min(coords, axis=0)) - border
        self.x1 = np.ceil(np.max(coords, axis=0)) + border

        # The size of the box in A
        self.size = self.x1 - self.x0

        # The grid size
        self.grid_cell_size = 1.0  # 2 A
        min_distance = (
            2.0 / self.grid_cell_size
        )  # The distance at which to delete atoms

        # The dimensions of the grid
        grid_shape = np.ceil(self.size / self.grid_cell_size).astype("uint32")

        # Allocate the grid
        self.grid = np.ones(shape=grid_shape, dtype="bool")
        # self.grid = ~self.grid
        # X, Y, Z = np.mgrid[0:grid_shape[0],0:grid_shape[1],0:grid_shape[2]]
        # R = np.sqrt((X-grid_shape[0]/2)**2+(Y-grid_shape[1]/2)**2+(Z-grid_shape[2]/2)**2)
        # r = abs(self.x0[0] - self.x1[0])
        # print(R.max(), grid_shape[0]/2.0)
        # self.grid = R < (grid_shape[0]/2.0)
        # return

        # Get the indices and set the grid values. The fill in any holes using
        # a morphological fill. The distance transform finds the distance to
        # any zero holes so in this case we need to set the nodes with protein
        # to False then compute the distance and then set any nodes within a
        # given distance to True. So the mask is True where we want to delete
        # atoms.
        indices = np.floor((coords - self.x0) / self.grid_cell_size).astype("int32")
        index_x = indices[:, 0]
        index_y = indices[:, 1]
        index_z = indices[:, 2]
        assert (indices >= 0).all()
        self.grid[index_x, index_y, index_z] = False
        self.grid = ~self.grid
        # from matplotlib import pylab
        # pylab.imshow(self.grid[:,:,grid_shape[2]//2])
        # pylab.show()

        self.grid = (
            scipy.ndimage.morphology.distance_transform_edt(self.grid) < min_distance
        )
        return
        # self.grid = scipy.ndimage.morphology.binary_closing(self.grid, iterations=2)
        # self.grid = scipy.ndimage.morphology.binary_fill_holes(self.grid)
        # import mrcfile

        # outfile = mrcfile.new("temp.mrc", overwrite=True)
        # outfile.set_data(self.grid.astype("int8"))
        # logger.info("Filled grid with atom positions:")
        # logger.info("    x0: %g" % self.x0[0])
        # logger.info("    y0: %g" % self.x0[1])
        # logger.info("    z0: %g" % self.x0[2])
        # logger.info("    x1: %g" % self.x1[0])
        # logger.info("    y1: %g" % self.x1[1])
        # logger.info("    z1: %g" % self.x1[2])
        # logger.info("    num elements: %d" % self.grid.size)
        # logger.info("    num filled: %d" % np.count_nonzero(self.grid))

    def __call__(self, atoms):
        """
        Select atoms that don't fall on the molecule atom positions

        Args:
            atoms (object): The atoms to select from

        Returns:
            object: The selected atoms

        """

        # Get the atom coords
        coords = atoms[["x", "y", "z"]]

        # Compute the indices
        indices = (
            np.floor((coords - self.x0) / self.grid_cell_size)
            .astype("int32")
            .to_numpy()
        )
        # The coordinates inside the grid
        inside_grid = ((indices >= 0) & (indices < self.grid.shape)).all(axis=1)

        # The selection
        selection = np.zeros(shape=coords.shape[0], dtype="bool")
        selection[~inside_grid] = True

        # Compute the selection
        indices_inside_grid = indices[inside_grid]
        index_x = indices_inside_grid[:, 0]
        index_y = indices_inside_grid[:, 1]
        index_z = indices_inside_grid[:, 2]
        selection[inside_grid] = self.grid[index_x, index_y, index_z] == False

        # Print some info
        logger.info(
            "Deleted %d/%d atoms"
            % (len(selection) - np.count_nonzero(selection), coords.shape[0])
        )

        # Return the atoms
        return atoms[selection]


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


def add_ice(sample, centre=None, shape=None, density=940.0, pack=False):
    """
    Create a sample with just a load of water molecules

    Args:
        sample (object): The sample object
        shape (object): The shape description
        density (float): The water density

    Returns:
        object: The sample object

    """

    def shape_filter_coordinates(coords, centre, shape):
        def cube_filter_coordinates(coords, centre, cube):
            length = cube["length"]
            x0 = centre - length / 2.0
            x1 = centre + length / 2.0
            return coords[((coords >= x0) & (coords < x1)).all(axis=1)]

        def cuboid_filter_coordinates(coords, centre, cuboid):
            length_x = cuboid["length_x"]
            length_y = cuboid["length_y"]
            length_z = cuboid["length_z"]
            length = np.array((length_x, length_y, length_z))
            x0 = centre - length / 2.0
            x1 = centre + length / 2.0
            return coords[((coords >= x0) & (coords < x1)).all(axis=1)]

        def cylinder_filter_coordinates(coords, centre, cylinder):
            length = cylinder["length"]
            radius = cylinder["radius"]
            y0 = centre - length / 2.0
            y1 = centre + length / 2.0
            x = coords[:, 0]
            y = coords[:, 1]
            z = coords[:, 2]
            return coords[
                (y >= y0[0])
                & (y < y1[0])
                & ((z - centre[2]) ** 2 + (x - centre[1]) ** 2 <= radius**2)
            ]

        # Filter the coords
        return {
            "cube": cube_filter_coordinates,
            "cuboid": cuboid_filter_coordinates,
            "cylinder": cylinder_filter_coordinates,
        }[shape["type"]](coords, centre, shape[shape["type"]])

    # Cast input
    centre = np.array(centre)

    # Get the filename of the water.cif file
    filename = parakeet.data.get_path("water.cif")

    # Create a single ribosome sample
    single_water = AtomData.from_ligand_file(filename)
    atoms = single_water.data[0:3]
    water_coords = atoms[["x", "y", "z"]].copy()
    water_coords -= water_coords.iloc[0].copy()

    # Set the volume in A^3
    if shape["type"] == "cube":
        length = shape["cube"]["length"]
        volume = length**3
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
        volume = pi * radius**2 * length
        length_x = 2 * radius
        length_y = length
        length_z = 2 * radius
    else:
        raise RuntimeError("Unknown shape")

    # Get the centre and offset
    centre_x, centre_y, centre_z = centre
    offset_x = centre_x - length_x / 2.0
    offset_y = centre_y - length_y / 2.0
    offset_z = centre_z - length_z / 2.0
    offset = np.array((offset_x, offset_y, offset_z), dtype="float32")

    # Determine the number of waters to place
    avogadros_number = scipy.constants.Avogadro
    molar_mass_of_water = 18.01528  # grams / mole
    density_of_water = density  # kg / m^3
    mass_of_water = (density_of_water * 1000) * (volume * 1e-10**3)  # g
    number_of_waters = int(
        floor((mass_of_water / molar_mass_of_water) * avogadros_number)
    )

    # Uniform random or packed
    if not pack:

        # The water filename
        filename = parakeet.data.get_path("water.cif")

        # Get the water coords
        single_water = parakeet.sample.AtomData.from_ligand_file(filename)
        atoms = single_water.data[0:3]
        water_coords = atoms[["x", "y", "z"]].copy()

        # Min = 0
        water_coords -= water_coords.min()

        # Translation
        if shape["type"] != "cylinder":
            x = np.random.uniform(offset_x, offset_x + length_x, size=number_of_waters)
            y = np.random.uniform(offset_y, offset_y + length_y, size=number_of_waters)
            z = np.random.uniform(offset_z, offset_z + length_z, size=number_of_waters)
        else:
            r = radius * np.sqrt(np.random.uniform(0, 1, size=number_of_waters))
            t = np.random.uniform(0, 2 * pi, size=number_of_waters)
            x = centre_x + r * np.cos(t)
            y = np.random.uniform(offset_y, offset_y + length_z, size=number_of_waters)
            z = centre_z + r * np.sin(t)
        translation = np.array((x, y, z)).T

        # Random orientations
        rotation = Rotation.from_rotvec(random_uniform_rotation(number_of_waters))

        # Rotate the Hydrogens around the Oxygen and translate
        O = rotation.apply(water_coords.iloc[0].copy()) + translation
        H1 = rotation.apply(water_coords.iloc[1].copy()) + translation
        H2 = rotation.apply(water_coords.iloc[2].copy()) + translation

        def create_atom_data(atomic_number, coords):

            # Create a new array
            def new_array(size, name, value):
                return (
                    np.zeros(
                        shape=(size,), dtype=parakeet.sample.AtomData.column_data[name]
                    )
                    + value
                )

            # Create the new arrays
            atomic_number = new_array(coords.shape[0], "atomic_number", atomic_number)
            sigma = new_array(coords.shape[0], "sigma", 0.085)
            occupancy = new_array(coords.shape[0], "occupancy", 1.0)
            charge = new_array(coords.shape[0], "charge", 0)

            # Return the data frame
            return pandas.DataFrame(
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

        # Add the sample atoms
        data_buffer = []
        data_buffer.append(create_atom_data(8, O))
        data_buffer.append(create_atom_data(1, H1))
        data_buffer.append(create_atom_data(1, H2))

        sample.add_atoms(AtomData(data=pandas.concat(data_buffer, ignore_index=True)))

    else:

        # Van der Waals radius of water
        van_der_waals_radius = 2.7 / 2.0  # A

        # Compute the total volume in the spheres
        volume_of_spheres = (
            (4.0 / 3.0) * pi * van_der_waals_radius**3 * number_of_waters
        )

        # Create the grid. The sphere packer takes the grid is (z, y, x) but
        # because it does slices along Z and we want slices along X we flip the X
        # and Z grid spec here
        grid = (
            int(floor(length_x / (2 * van_der_waals_radius))),
            int(floor(length_y / (2 * van_der_waals_radius))),
            int(floor(length_z / (2 * van_der_waals_radius))),
        )

        # Compute the node length and density
        node_length = max((length_z / grid[2], length_y / grid[1], length_x / grid[0]))
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
        packer = parakeet.freeze.SpherePacker(
            grid, node_length, sphere_density, van_der_waals_radius, max_iter=10
        )

        # Extract all the data
        logger.info("Generating water positions:")
        start_time = time.time()
        max_buffer = 10_000_000
        data_buffer = []
        for x_index, x_slice in enumerate(packer):

            # Read the coordinates. The packer goes along the z axis so we need to
            # flip the coordinates since we want x slices
            coords = []
            for node in x_slice:
                coords.extend(node)
            coords = np.flip(np.array(coords, dtype="float32"), axis=1) + offset

            # Filter the coordinates by the shape to ensure no ice is outside the
            # shape. This is only really necessary for the cylinder shape
            coords = shape_filter_coordinates(coords, centre, shape)

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
                        np.zeros(shape=(size,), dtype=AtomData.column_data[name])
                        + value
                    )

                # Create the new arrays
                atomic_number = new_array(
                    coords.shape[0], "atomic_number", atomic_number
                )
                sigma = new_array(coords.shape[0], "sigma", 0.085)
                occupancy = new_array(coords.shape[0], "occupancy", 1.0)
                charge = new_array(coords.shape[0], "charge", 0)

                # Return the data frame
                return pandas.DataFrame(
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

            # Add the sample atoms
            data_buffer.append(create_atom_data(8, O))
            data_buffer.append(create_atom_data(1, H1))
            data_buffer.append(create_atom_data(1, H2))
            if sum(b.shape[0] for b in data_buffer) > max_buffer:
                logger.info(
                    "    Writing %d atoms" % sum(b.shape[0] for b in data_buffer)
                )
                sample.add_atoms(
                    AtomData(data=pandas.concat(data_buffer, ignore_index=True))
                )
                data_buffer = []

            # The estimates time left
            time_taken = time.time() - start_time
            estimated_time = (len(packer) - x_index) * time_taken / (x_index + 1)

            # Log some info
            logger.info(
                "    X slice %d/%d: Num molecules: %d (remaining %.d seconds)"
                % (x_index, len(packer), O.shape[0], estimated_time)
            )

        # Add anything remaining in the data buffer
        if len(data_buffer) > 0:
            logger.info("    Writing %d atoms" % sum(b.shape[0] for b in data_buffer))
            sample.add_atoms(
                AtomData(data=pandas.concat(data_buffer, ignore_index=True))
            )
            del data_buffer

        # Print some output
        logger.info(f"Sphere packer: Num unplaced:  {packer.num_unplaced_samples()}")

    # Return the sample
    return sample


def new(filename, box=None, centre=None, shape=None, ice=None, coords=None, **kwargs):
    """
    Create the sample

    Args:
        filename (str): The filename of the sample
        box (list): The containing box of the sample
        centre (list): The centering of the sample in the box
        shape (object): The shape of the sample
        ice (object): The ice description
        coords_filename (object): The coordinates

    Returns:
        object: The test sample

    """

    # Check the dimensions are valid
    assert is_shape_inside_box(box, centre, shape)

    # Create the sample
    sample = Sample(filename, mode="w")

    # Set the sample box and shape
    sample.containing_box = ((0, 0, 0), box)
    sample.centre = centre
    sample.shape = shape

    # Add some ice
    if ice is not None and ice["generate"]:
        add_ice(sample, centre, shape, ice["density"])

    # Add atoms from coordinates file
    if coords is not None and coords["filename"]:
        atoms = AtomData.from_gemmi_file(coords["filename"])
        if coords["recentre"]:
            atoms.data = recentre(atoms.data)
            position = sample.centre
        else:
            position = (0, 0, 0)

        # Set the orientation
        orientation = coords["orientation"]
        if orientation is None or len(orientation) != 3:
            orientation = random_uniform_rotation()[0]

        # Add the molecule
        sample.add_molecule(
            atoms, positions=[position], orientations=[orientation], name=None
        )

    # Print some info
    logger.info(sample.info())

    # Get the sample
    return sample


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
    filename = parakeet.data.get_pdb(name)

    # Get the atom data
    atoms = AtomData.from_gemmi_file(filename)
    atoms.data = recentre(atoms.data)

    # Get atom data bounds
    coords = atoms.data[["x", "y", "z"]]
    x0 = np.min(coords, axis=0) + sample.centre
    x1 = np.max(coords, axis=0) + sample.centre

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

    position = sample.centre

    # Delete the atoms where we want to place the molecules
    sample.del_atoms(AtomDeleter(atoms, position, (0, 0, 0)))

    # Add the molecule
    sample.add_molecule(
        atoms, positions=[position], orientations=[(0, 0, 0)], name=name
    )

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
    for name, value in molecules.items():

        # Get the type and instances
        mtype = value["type"]
        items = value["instances"]

        # Skip if number is zero
        if len(items) == 0:
            continue

        # Print some info
        logger.info("Adding %d %s molecules" % (len(items), name))

        # Get the filename of the PDB entry
        if mtype == "pdb":
            filename = parakeet.data.get_pdb(name)
        elif mtype == "local":
            filename = name

        # Get the atom data
        atoms = AtomData.from_gemmi_file(filename)
        atoms.data = recentre(atoms.data)
        atom_data[name] = atoms

        # Generate some random orientations
        logger.info("    Generating random orientations")
        for item in items:
            rotation = item.get("orientation", None)
            if rotation is None:
                rotation = random_uniform_rotation(1)[0]
            rotation = Rotation.from_rotvec(rotation)
            coords = rotation.apply(atoms.data[["x", "y", "z"]])
            all_orientations.append(rotation.as_rotvec())
            all_positions.append(item.get("position", None))
            all_boxes.append(np.max(coords, axis=0) - np.min(coords, axis=0))
            all_labels.append(name)

    # Put the molecules in the sample
    logger.info("Placing molecules:")
    if any(p is None or len(p) == 0 for p in all_positions):
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
        x0 = position - box / 2.0
        x1 = position + box / 2.0

        # Check the coords
        assert is_box_inside_shape((x0, x1), sample.centre, sample.shape)

        # Construct arrays
        positions[label].append(position)
        orientations[label].append(rotation)

        # Delete the atoms where we want to place the molecules
        sample.del_atoms(AtomDeleter(atom_data[label], position, rotation))

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

    # Convert to list of positions/orientations
    temp = {}
    print(molecules)
    for origin, items in molecules.items():
        if items is None:
            continue
        for item in items:

            # Get the key
            if origin == "local":
                key = item["filename"]
            elif origin == "pdb":
                key = item["id"]
            else:
                raise RuntimeError("Unknown origin")

            # Get the instances
            instances = item["instances"]

            # Set the instances
            temp[key] = {
                "type": origin,
                "instances": [{} for i in range(instances)]
                if isinstance(instances, int)
                else instances,
            }

    molecules = temp

    # The total number of molecules
    total_number_of_molecules = sum(map(len, molecules.values()))

    # Open the sample
    sample = Sample(filename, mode="r+")

    # Put the molecule in the centre if only one
    if total_number_of_molecules == 0:
        raise RuntimeError("Need at least 1 molecule")
    elif total_number_of_molecules == 1:
        key = [key for key, value in molecules.items() if len(value) > 0][0]
        if molecules[key]["instances"][0].get("position", None) is None:
            molecules[key]["instances"][0]["position"] = sample.centre
        if molecules[key]["instances"][0].get("orientation", None) is None:
            molecules[key]["instances"][0]["orientation"] = (0, 0, 0)

    # Add the molecules
    add_multiple_molecules(sample, molecules)

    # Show some info
    logger.info(sample.info())

    # Return the sample
    return sample


def mill(filename, box=None, centre=None, shape=None, **kwargs):
    """
    Mill the sample

    """

    def shape_filter_coordinates(coords, centre, shape):
        def cube_filter_coordinates(coords, centre, cube):
            length = cube["length"]
            x0 = centre - length / 2.0
            x1 = centre + length / 2.0
            return ((coords >= x0) & (coords < x1)).all(axis=1)

        def cuboid_filter_coordinates(coords, centre, cuboid):
            length_x = cuboid["length_x"]
            length_y = cuboid["length_y"]
            length_z = cuboid["length_z"]
            length = np.array((length_x, length_y, length_z))
            x0 = centre - length / 2.0
            x1 = centre + length / 2.0
            return ((coords >= x0) & (coords < x1)).all(axis=1)

        def cylinder_filter_coordinates(coords, centre, cylinder):
            length = cylinder["length"]
            radius = cylinder["radius"]
            y0 = centre - length / 2.0
            y1 = centre + length / 2.0
            x = coords["x"]
            y = coords["y"]
            z = coords["z"]
            return (
                (y >= y0[0])
                & (y < y1[0])
                & ((z - centre[2]) ** 2 + (x - centre[1]) ** 2 <= radius**2)
            )

        # Filter the coords
        return {
            "cube": cube_filter_coordinates,
            "cuboid": cuboid_filter_coordinates,
            "cylinder": cylinder_filter_coordinates,
        }[shape["type"]](coords, centre, shape[shape["type"]])

    class Deleter(object):
        def __init__(self, sample):
            self.sample = sample
            self.x0, self.x1 = self.sample.bounding_box

        def __call__(self, atoms):
            selection = shape_filter_coordinates(
                atoms[["x", "y", "z"]], self.sample.centre, self.sample.shape
            )
            return atoms[selection]

    # Open the sample
    sample = Sample(filename, mode="r+")

    # Set the sample box and shape
    sample.containing_box = ((0, 0, 0), box)
    sample.centre = centre
    sample.shape = shape

    # Delete the atoms
    sample.del_atoms(Deleter(sample))

    atoms = sample.get_atoms()
    coords = atoms.data[["x", "y", "z"]]
    x0 = coords.min()
    x1 = coords.max()
    sample.bounding_box = (x0, x1)

    # Print some info
    logger.info(sample.info())


def sputter(filename, element=None, thickness=20):
    """
    Add a sputter coating to the sample of the desired thickness

    This is very crude and adds atoms at random positions

    Params:
        atomic_number (int): The atomic number
        density (float): The density of atoms (g/cm3)
        thickness (float): The thickness (A)


    """

    # Open the sample
    sample = Sample(filename, mode="r+")

    # Get the sample shape
    shape = sample.shape
    centre = sample.centre

    # Set the volume in A^3
    if shape["type"] == "cube":
        length = shape["cube"]["length"]
        length_x = length
        length_y = length
        length_z = length
        sputter_length_x = length_x + thickness * 2
        sputter_length_y = length_y + thickness * 2
        sputter_length_z = length_z + thickness * 2
        shape_volume = length**3
        # sputter_volume = (
        #     sputter_length_x * sputter_length_y * sputter_length_z - shape_volume
        # )
    elif shape["type"] == "cuboid":
        length_x = shape["cuboid"]["length_x"]
        length_y = shape["cuboid"]["length_y"]
        length_z = shape["cuboid"]["length_z"]
        sputter_length_x = length_x + thickness * 2
        sputter_length_y = length_y + thickness * 2
        sputter_length_z = length_z + thickness * 2
        shape_volume = length_x * length_y * length_z
        # sputter_volume = (
        #     sputter_length_x * sputter_length_y * sputter_length_z - shape_volume
        # )
    elif shape["type"] == "cylinder":
        length = shape["cylinder"]["length"]
        radius = shape["cylinder"]["radius"]
        length_x = 2 * radius
        length_y = length
        length_z = 2 * radius
        sputter_length = length + thickness * 2
        sputter_radius = radius + thickness * 2
        shape_volume = pi * radius**2 * length
        # sputter_volume = pi * sputter_radius ** 2 * sputter_length - shape_volume
    else:
        raise RuntimeError("Unknown shape")

    # Determine the number of atoms to place
    if element is None:
        return
    elif element == "C":
        molar_mass = 12.0107  # grams / mole
        density = 2.3
        atomic_number = 6
    elif element == "Ir":
        molar_mass = 192.217  # grams / mole
        density = 22.56
        atomic_number = 77
    elif element == "Cr":
        molar_mass = 51.9961  # grams / mole
        density = 7.15
        atomic_number = 24
    elif element == "Pt":
        molar_mass = 195.084  # grams / mole
        density = 21.45
        atomic_number = 78
    else:
        raise RuntimeError("Unknown element %s" % element)
    avogadros_number = scipy.constants.Avogadro
    number_density = ((density * 1e-8**3) / molar_mass) * avogadros_number
    print("Placing %g %s atoms per A^3" % (number_density, element))

    # Get the centre and offset
    centre_x, centre_y, centre_z = centre
    offset_x = centre_x - length_x / 2.0
    offset_y = centre_y - length_y / 2.0
    offset_z = centre_z - length_z / 2.0
    # offset = np.array((offset_x, offset_y, offset_z), dtype="float32")

    # Translation
    if shape["type"] != "cylinder":
        # Generate positions in the thickness range
        volume = length_x * length_y * thickness
        number_of_atoms = int(number_density * volume)
        x = np.random.uniform(offset_x, offset_x + length_x, size=number_of_atoms)
        y = np.random.uniform(offset_y, offset_y + length_y, size=number_of_atoms)
        z = np.random.uniform(offset_z - thickness, offset_z, size=number_of_atoms)
        print("Placed %d atoms" % number_of_atoms)
        print("WARNING - ONLY APPLYING TO TOP OF SAMPLE")
    else:
        raise RuntimeError("Not implemented")
    position = np.array((x, y, z)).T

    def create_atom_data(atomic_number, coords):

        # Create a new array
        def new_array(size, name, value):
            return (
                np.zeros(
                    shape=(size,), dtype=parakeet.sample.AtomData.column_data[name]
                )
                + value
            )

        # Create the new arrays
        atomic_number = new_array(coords.shape[0], "atomic_number", atomic_number)
        sigma = new_array(coords.shape[0], "sigma", 0.085)
        occupancy = new_array(coords.shape[0], "occupancy", 1.0)
        charge = new_array(coords.shape[0], "charge", 0)

        # Return the data frame
        return pandas.DataFrame(
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

    # Add the sample atoms
    data_buffer = []
    data_buffer.append(create_atom_data(atomic_number, position))

    sample.add_atoms(AtomData(data=pandas.concat(data_buffer, ignore_index=True)))
