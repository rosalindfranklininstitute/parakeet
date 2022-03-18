#
# parakeet.freeze.__init__.py
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
import scipy.ndimage.morphology
from math import ceil, floor

from parakeet_ext import *  # noqa

# Get the logger
logger = logging.getLogger(__name__)


def freeze(atoms, x0, x1):
    """
    Given the input set of atoms add water molecules to random positions
    to simulate the sample in vitreous ice.

    Args:
        atoms (object): The list of atom positions
        x0 (array): The starting coordinate of the bounding box
        x1 (array): The ending coordinate of the bounding box

    Returns:
        object: The new list of atom positions with water molecules

    """

    # The grid size
    grid_size = 10.0  # 10 A

    # Get the min and max atom positions
    x = atoms["x"]
    y = atoms["y"]
    z = atoms["z"]
    min_x = np.min(x)
    max_x = np.max(x)
    min_y = np.min(y)
    max_y = np.max(y)
    min_z = np.min(z)
    max_z = np.max(z)

    # Ensure the atoms are within the box
    assert min_x >= x0[0]
    assert min_y >= x0[1]
    assert min_z >= x0[2]
    assert max_x <= x1[0]
    assert max_y <= x1[1]
    assert max_z <= x1[2]

    # Create a grid at 10A spacing to determine the shape of the input molecule
    shape = (
        int(ceil((x1[2] - x0[2]) / grid_size)),
        int(ceil((x1[1] - x0[1]) / grid_size)),
        int(ceil((x1[0] - x0[0]) / grid_size)),
    )
    logger.info("Allocating grid")
    logger.info("    shape: %s" % str(shape))
    grid = np.zeros(shape=shape, dtype="bool")

    # Fill in any spaces in the grid
    x_index = np.floor((x - x0[0]) / grid_size).astype("int32")
    y_index = np.floor((y - x0[1]) / grid_size).astype("int32")
    z_index = np.floor((z - x0[2]) / grid_size).astype("int32")
    grid[z_index, y_index, x_index] = True
    grid = scipy.ndimage.morphology.binary_fill_holes(grid)
    logger.info("Filled grid with atom positions:")
    logger.info("    x0: %g" % x0[0])
    logger.info("    y0: %g" % x0[1])
    logger.info("    z0: %g" % x0[2])
    logger.info("    x1: %g" % x1[0])
    logger.info("    y1: %g" % x1[1])
    logger.info("    z1: %g" % x1[2])
    logger.info("    num elements: %d" % grid.size)
    logger.info("    num filled: %d" % np.sum(grid))

    # Determine the remaining volume
    x_length = x1[0] - x0[0]
    y_length = x1[1] - x0[1]
    z_length = x1[2] - x0[2]
    total_volume = x_length * y_length * z_length  # A^3
    filled_volume = np.sum(grid) * 10**3
    remaining_volume = total_volume - filled_volume

    # Determine the number of waters to place
    avogadros_number = scipy.constants.Avogadro
    molar_mass_of_water = 18.01528  # grams / mole
    density_of_water = 997  # kg / m^3
    mass_of_water = (density_of_water * 1000) * (remaining_volume * 1e-10**3)  # g
    number_of_waters = int(
        floor((mass_of_water / molar_mass_of_water) * avogadros_number)
    )

    # Print some stuff of water
    logger.info("Water information:")
    logger.info("    Total volume %g m^3" % (total_volume * 1e-10**3))
    logger.info("    Sample volume %g m^3" % (filled_volume * 1e-10**3))
    logger.info("    Volume of water: %g m^3" % (remaining_volume * 1e-10**3))
    logger.info("    Density of water: %g kg/m^3" % density_of_water)
    logger.info("    Mass of water: %g kg" % (mass_of_water / 1000))
    logger.info("    Number of water molecules to place: %d" % number_of_waters)

    # Loop adding waters until complete
    logger.info("Placing waters:")
    number_to_place = number_of_waters
    water_coords = np.zeros(shape=(0, 3), dtype="float64")
    while number_to_place > 0:
        coords = np.random.uniform(x0, x1, size=(number_to_place, 3))
        x_index = np.floor((coords[:, 0] - x0[0]) / grid_size).astype("int32")
        y_index = np.floor((coords[:, 1] - x0[1]) / grid_size).astype("int32")
        z_index = np.floor((coords[:, 2] - x0[2]) / grid_size).astype("int32")
        selection = grid[z_index, y_index, x_index] == False
        coords = coords[selection, :]
        water_coords = np.concatenate((water_coords, coords), axis=0)
        number_to_place -= len(coords)
        logger.info("    placed %d waters" % len(coords))

    # Create the data frame for the atoms
    # FIXME Currently only adding oxygens since they are major contribution
    # to scattering of electrons; however, we should also add hydrogens
    logger.info("Creating water atom data")
    shape = number_of_waters
    water_atoms = pandas.DataFrame(
        {
            "model": np.zeros(shape=shape, dtype="uint32"),
            "chain": np.zeros(shape=shape, dtype="str"),
            "residue": np.zeros(shape=shape, dtype="str"),
            "atomic_number": np.full(shape=shape, fill_value=8, dtype="uint32"),
            "x": water_coords[:, 0],
            "y": water_coords[:, 1],
            "z": water_coords[:, 2],
            "occ": np.ones(shape=shape, dtype="float64"),
            "charge": np.zeros(shape=shape, dtype="uint32"),
            "sigma": np.full(shape=shape, fill_value=0.085, dtype="float64"),
            "region": np.zeros(shape=shape, dtype="uint32"),
        }
    )

    # Return the atom positions
    return pandas.concat([atoms, water_atoms])
