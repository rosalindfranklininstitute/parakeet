#
# parakeet.sample.new.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import logging
import numpy as np
import pandas
import scipy.constants
import time
import parakeet.config
import parakeet.data
import parakeet.freeze
from math import pi, floor
from scipy.spatial.transform import Rotation
from functools import singledispatch
from parakeet.sample import recentre
from parakeet.sample import Sample
from parakeet.sample import AtomData
from parakeet.sample import random_uniform_rotation


__all__ = ["new"]


# Get the logger
logger = logging.getLogger(__name__)


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
                    "group": -1,
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


@singledispatch
def new(config_file, sample_file: str) -> Sample:
    """
    Create an ice sample and save it

    Args:
        config_file: The input config filename
        sample_file: The sample filename

    """

    # Load the configuration
    config = parakeet.config.load(config_file)

    # Print some options
    parakeet.config.show(config)

    # Create the sample
    logger.info(f"Writing sample to {sample_file}")
    return _new_Config(config, sample_file)


@new.register(parakeet.config.Config)
def _new_Config(config: parakeet.config.Config, filename: str) -> Sample:
    """
    Create the sample

    Args:
        config: The sample configuration
        filename: The filename of the sample

    Returns:
        The sample object

    """
    return _new_Sample(config.sample, filename)


@new.register(parakeet.config.Sample)
def _new_Sample(config: parakeet.config.Sample, filename: str) -> Sample:
    """
    Create the sample

    Args:
        config: The sample configuration
        filename: The filename of the sample

    Returns:
        The sample object

    """
    box = config.box
    centre = config.centre
    shape = config.shape.model_dump()
    ice = config.ice
    coords = config.coords

    # Check the dimensions are valid
    assert parakeet.sample.is_shape_inside_box(box, centre, shape)

    # Create the sample
    sample = Sample(filename, mode="w")

    # Set the sample box and shape
    sample.containing_box = ((0, 0, 0), box)
    sample.centre = centre
    sample.shape = shape

    # Add some ice
    if ice is not None and ice.generate:
        add_ice(sample, centre, shape, ice.density)

    # Add atoms from coordinates file
    if coords is not None and coords.filename is not None:
        atoms = AtomData.from_gemmi_file(coords.filename)
        if coords.recentre:
            atoms.data = recentre(atoms.data)
            position = sample.centre
            orientation = (0, 0, 0)
        else:
            position = (0, 0, 0)
            orientation = (0, 0, 0)
        if coords.position:
            position = coords.position  # type: ignore
        if coords.orientation:
            orientation = coords.orientation  # type: ignore
        if coords.scale != 1.0:
            atoms.data["x"] = atoms.data["x"] * coords.scale
            atoms.data["y"] = atoms.data["y"] * coords.scale
            atoms.data["z"] = atoms.data["z"] * coords.scale

        # Add the molecule
        sample.add_molecule(
            atoms, positions=[position], orientations=[orientation], name=None
        )

    # Print some info
    logger.info(sample.info())

    # Get the sample
    return sample
