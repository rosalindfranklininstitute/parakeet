#
# parakeet.sample.new.sputter.py
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
import parakeet.config
import parakeet.data
import parakeet.freeze
from functools import singledispatch
from parakeet.sample import Sample
from parakeet.sample import AtomData
from parakeet.sample import Sample


__all__ = ["sputter"]


# Get the logger
logger = logging.getLogger(__name__)


@singledispatch
def sputter(config_file, sample_file: str) -> Sample:
    """
    Sputter the sample

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
    return _sputter_Config(config, sample_file)


@sputter.register(parakeet.config.Config)
def _sputter_Config(config: parakeet.config.Config, sample_file: str) -> Sample:
    """
    Take a sample and add a load of molecules

    Args:
        config: The sample configuration
        sample_file: The filename of the sample

    Returns:
        The sample object

    """
    # Open the sample
    sample = Sample(sample_file, mode="r+")

    # Do the work
    if config.sample.sputter:
        sample = _sputter_Sputter(config.sample.sputter, sample)
    else:
        logger.info("No sputter parameters found in config")
    return sample


@sputter.register(parakeet.config.Sputter)
def _sputter_Sputter(config: parakeet.config.Sputter, sample: Sample) -> Sample:
    """
    Add a sputter coating to the sample of the desired thickness

    This is very crude and adds atoms at random positions

    Params:
        config: The sputter configuration
        sample: The sample object


    """

    element = config.element
    thickness = config.thickness

    # Get the sample shape
    shape = sample.shape
    centre = sample.centre

    # Set the volume in A^3
    if shape["type"] == "cube":
        length = shape["cube"]["length"]
        length_x = length
        length_y = length
        length_z = length
        # sputter_length_x = length_x + thickness * 2
        # sputter_length_y = length_y + thickness * 2
        # sputter_length_z = length_z + thickness * 2
        # shape_volume = length**3
        # sputter_volume = (
        #     sputter_length_x * sputter_length_y * sputter_length_z - shape_volume
        # )
    elif shape["type"] == "cuboid":
        length_x = shape["cuboid"]["length_x"]
        length_y = shape["cuboid"]["length_y"]
        length_z = shape["cuboid"]["length_z"]
        # sputter_length_x = length_x + thickness * 2
        # sputter_length_y = length_y + thickness * 2
        # sputter_length_z = length_z + thickness * 2
        # shape_volume = length_x * length_y * length_z
        # sputter_volume = (
        #     sputter_length_x * sputter_length_y * sputter_length_z - shape_volume
        # )
    elif shape["type"] == "cylinder":
        length = shape["cylinder"]["length"]
        radius = shape["cylinder"]["radius"]
        length_x = 2 * radius
        length_y = length
        length_z = 2 * radius
        # sputter_length = length + thickness * 2
        # sputter_radius = radius + thickness * 2
        # shape_volume = pi * radius**2 * length
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

    return sample
