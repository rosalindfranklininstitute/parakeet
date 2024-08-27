#
# parakeet.sample.add_molecules.py
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
from collections import defaultdict
from functools import singledispatch
from scipy.spatial.transform import Rotation
from parakeet.sample import AtomData
from parakeet.sample import AtomDeleter
from parakeet.sample import Sample
from parakeet.sample import recentre
from parakeet.sample import random_uniform_rotation
from parakeet.sample import is_box_inside_shape
from parakeet.sample.distribute import distribute_particles_uniformly
from parakeet.sample.distribute import shape_volume_object


__all__ = ["add_molecules", "add_single_molecule"]


# Get the logger
logger = logging.getLogger(__name__)


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
    all_radii = []
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
            centre_of_mass = np.mean(coords, axis=0)
            radius = np.max(np.sqrt(np.sum((coords - centre_of_mass) ** 2, axis=1)))
            all_orientations.append(rotation.as_rotvec())
            all_positions.append(item.get("position", None))
            all_radii.append(radius)
            all_labels.append(name)

    # Put the molecules in the sample
    logger.info("Placing molecules:")
    if any(p is None or len(p) == 0 for p in all_positions):
        all_positions = distribute_particles_uniformly(
            shape_volume_object(sample.centre, sample.shape), np.array(all_radii)
        )

    # Set the positions and orientations by molecule
    positions = defaultdict(list)
    orientations = defaultdict(list)
    for label, rotation, position in zip(all_labels, all_orientations, all_positions):
        # # Get atom data bounds
        # x0 = position - box / 2.0
        # x1 = position + box / 2.0

        # # Check the coords
        # assert is_box_inside_shape((x0, x1), sample.centre, sample.shape)

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


@singledispatch
def add_molecules(config_file, sample_file: str) -> Sample:
    """
    Add molecules to the sample

    Args:
        config_file: The input config filename
        sample_file: The input sample filename

    """
    # Load the configuration
    config = parakeet.config.load(config_file)

    # Print some options
    parakeet.config.show(config)

    # Create the sample
    logger.info(f"Writing sample to {sample_file}")
    return _add_molecules_Config(config, sample_file)


@add_molecules.register(parakeet.config.Config)
def _add_molecules_Config(config: parakeet.config.Config, sample_file: str) -> Sample:
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
    return _add_molecules_Sample(config.sample, sample)


@add_molecules.register(parakeet.config.Sample)
def _add_molecules_Sample(config: parakeet.config.Sample, sample: Sample) -> Sample:
    """
    Take a sample and add a load of molecules

    Args:
        config: The sample configuration
        sample: The sample object

    Returns:
        The sample object

    """
    if config.molecules is not None:
        molecules = config.molecules.model_dump()
    else:
        molecules = {}

    # Convert to list of positions/orientations
    temp = {}
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
                "instances": (
                    [{} for i in range(instances)]
                    if isinstance(instances, int)
                    else instances
                ),
            }

    molecules = temp

    # The total number of molecules
    total_number_of_molecules = sum(
        map(lambda x: len(x["instances"]), molecules.values())
    )

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
