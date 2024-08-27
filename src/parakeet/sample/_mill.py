#
# parakeet.sample.mill.py
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
import parakeet.config
import parakeet.data
import parakeet.freeze
from functools import singledispatch
from parakeet.sample import Sample


__all__ = ["mill"]


# Get the logger
logger = logging.getLogger(__name__)


@singledispatch
def mill(config_file, sample_file: str) -> Sample:
    """
    Mill to the shape of the sample

    Args:
        config_file: The input config filename
        sample_file: The sample filename

    Returns:
        The sample object

    """
    # Load the configuration
    config = parakeet.config.load(config_file)

    # Print some options
    parakeet.config.show(config)

    # Create the sample
    logger.info(f"Writing sample to {sample_file}")
    return _mill_Config(config, sample_file)


@mill.register(parakeet.config.Config)
def _mill_Config(config: parakeet.config.Config, sample_file: str) -> Sample:
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
    return _mill_Sample(config.sample, sample)


@mill.register(parakeet.config.Sample)
def _mill_Sample(config: parakeet.config.Sample, sample: Sample) -> Sample:
    """
    Mill the sample

    Args:
        config: The input config
        sample: The sample model

    Returns:
        The sample object

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

    box = config.box
    centre = config.centre
    shape = config.shape.model_dump()

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

    return sample
