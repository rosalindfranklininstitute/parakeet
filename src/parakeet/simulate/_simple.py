#
# parakeet.simulate.simple.py
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
import parakeet.dqe
import parakeet.freeze
import parakeet.futures
import parakeet.inelastic
import parakeet.io
import parakeet.sample
from functools import singledispatch
from parakeet.microscope import Microscope
from parakeet.simulate.simulation import Simulation
from parakeet.simulate.engine import SimulationEngine


__all__ = ["simple"]


# Get the logger
logger = logging.getLogger(__name__)


class SimpleImageSimulator(object):
    """
    A class to do the actual simulation

    """

    def __init__(
        self,
        microscope=None,
        atoms=None,
        scan=None,
        simulation=None,
        device="gpu",
        gpu_id=None,
    ):
        self.microscope = microscope
        self.atoms = atoms
        self.scan = scan
        self.simulation = simulation
        self.device = device
        self.gpu_id = gpu_id

    def __call__(self, index):
        """
        Simulate a single frame

        Args:
            simulation (object): The simulation object
            index (int): The frame number

        Returns:
            tuple: (angle, image)

        """

        # Get the rotation angle
        angle = self.scan.angles[index]
        position = self.scan.positions[index]

        # The field of view
        nx = self.microscope.detector.nx
        ny = self.microscope.detector.ny
        pixel_size = self.microscope.detector.pixel_size
        margin = self.simulation["margin"]
        x_fov = nx * pixel_size
        y_fov = ny * pixel_size
        offset = margin * pixel_size

        # Get the specimen atoms
        logger.info(f"Simulating image {index+1}")

        # x0 = (-offset, -offset)
        # x1 = (x_fov + offset, y_fov + offset)

        # Create the multem system configuration
        simulate = SimulationEngine(
            self.device,
            self.gpu_id,
            self.microscope,
            self.simulation["slice_thickness"],
            self.simulation["margin"],
            "EWRS",
        )

        # Set the specimen size
        simulate.input.spec_lx = x_fov + offset * 2
        simulate.input.spec_ly = y_fov + offset * 2
        simulate.input.spec_lz = np.max(self.atoms.data["z"])

        # Set the atoms in the input after translating them for the offset
        simulate.input.spec_atoms = self.atoms.translate(
            (offset, offset, 0)
        ).to_multem()

        # Do the simulation
        image = simulate.image()

        # Print some info
        psi_tot = np.abs(image) ** 2
        logger.info("Ideal image min/max: %f/%f" % (np.min(psi_tot), np.max(psi_tot)))

        # Compute the image scaled with Poisson noise
        return (index, image, None)


def simulation_factory(
    microscope: Microscope,
    atoms: str,
    simulation: dict = None,
    multiprocessing: dict = None,
):
    """
    Create the simulation

    Args:
        microscope (object); The microscope object
        atoms (object): The atom data
        simulation (object): The simulation parameters
        multiprocessing (object): The multiprocessing parameters

    Returns:
        object: The simulation object

    """
    # Create the scan
    scan = parakeet.scan.new("still")

    # Get the margin
    margin = 0 if simulation is None else simulation.get("margin", 0)

    # Check multiprocessing settings
    if multiprocessing is None:
        multiprocessing = {"device": "gpu", "nproc": 1, "gpu_id": 0}
    else:
        assert multiprocessing["nproc"] in [None, 1]
        assert len(multiprocessing["gpu_id"]) == 1

    # Create the simulation
    return Simulation(
        image_size=(
            microscope.detector.nx + 2 * margin,
            microscope.detector.ny + 2 * margin,
        ),
        pixel_size=microscope.detector.pixel_size,
        scan=scan,
        simulate_image=SimpleImageSimulator(
            microscope=microscope,
            scan=scan,
            atoms=atoms,
            simulation=simulation,
            device=multiprocessing["device"],
            gpu_id=multiprocessing["gpu_id"][0],
        ),
    )


@singledispatch
def simple(config_file, atoms_file: str, output_file: str):
    """
    Simulate the image

    Args:
        config_file: The config filename
        atoms_file: The input atoms filename
        output_filename: The output filename

    """

    # Load the full configuration
    config = parakeet.config.load(config_file)

    # Print some options
    parakeet.config.show(config)

    # Do the work
    _simple_Config(config, atoms_file, output_file)


@simple.register(parakeet.config.Config)
def _simple_Config(config: parakeet.config.Config, atoms_file: str, output_file: str):
    """
    Simulate the image

    Args:
        config: The config object
        atoms_file: The input atoms filename
        output_filename: The output filename

    """
    # Create the microscope
    microscope = parakeet.microscope.new(config.microscope)

    # Create the exit wave data
    logger.info(f"Loading sample from {atoms_file}")
    atoms = parakeet.sample.AtomData.from_text_file(atoms_file)

    # Create the simulation
    simulation = simulation_factory(
        microscope=microscope,
        atoms=atoms,
        simulation=config.simulation.model_dump(),
        multiprocessing=config.multiprocessing.model_dump(),
    )

    # Create the writer
    logger.info(f"Opening file: {output_file}")
    writer = parakeet.io.new(
        output_file,
        shape=simulation.shape,
        pixel_size=simulation.pixel_size,
        dtype=np.complex64,
    )

    # Run the simulation
    simulation.run(writer)
