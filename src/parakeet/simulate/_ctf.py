#
# parakeet.simulate.ctf.py
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
import parakeet.sample
from parakeet.microscope import Microscope
from functools import singledispatch
from parakeet.simulate.simulation import Simulation
from parakeet.simulate.engine import SimulationEngine

# Get the logger
logger = logging.getLogger(__name__)


__all__ = ["ctf"]


class CTFSimulator(object):
    """
    A class to do the actual simulation

    The simulation is structured this way because the input data to the
    simulation is large enough that it makes an overhead to creating the
    individual processes.

    """

    def __init__(self, microscope=None, simulation=None):
        self.microscope = microscope
        self.simulation = simulation

    def __call__(self, index):
        """
        Simulate a single frame

        Args:
            simulation (object): The simulation object
            index (int): The frame number

        Returns:
            tuple: (angle, image)

        """

        # The field of view
        nx = self.microscope.detector.nx
        ny = self.microscope.detector.ny
        pixel_size = self.microscope.detector.pixel_size
        x_fov = nx * pixel_size
        y_fov = ny * pixel_size

        # Create the multem system configuration
        simulate = SimulationEngine(
            "cpu",
            0,
            self.microscope,
            self.simulation["slice_thickness"],
            self.simulation["margin"],
            "HRTEM",
        )
        simulate.input.nx = nx
        simulate.input.ny = ny

        # Set the specimen size
        simulate.input.spec_lx = x_fov
        simulate.input.spec_ly = y_fov
        simulate.input.spec_lz = x_fov  # self.sample.containing_box[1][2]

        # Run the simulation
        image = simulate.ctf()
        image = np.fft.fftshift(image)

        # Compute the image scaled with Poisson noise
        return (index, image, None)


def simulation_factory(microscope: Microscope, simulation: dict) -> Simulation:
    """
    Create the simulation

    Args:
        microscope (object); The microscope object
        simulation (object): The simulation parameters

    Returns:
        object: The simulation object

    """

    # Create the simulation
    return Simulation(
        image_size=(microscope.detector.nx, microscope.detector.ny),
        pixel_size=microscope.detector.pixel_size,
        simulate_image=CTFSimulator(microscope=microscope, simulation=simulation),
    )


@singledispatch
def ctf(config_file, output_file: str):
    """
    Simulate the ctf

    Args:
        config_file: The config filename
        output_file: The output ctf filename

    """

    # Load the full configuration
    config = parakeet.config.load(config_file)

    # Print some options
    parakeet.config.show(config)

    # Do the work
    _ctf_Config(config, output_file)


@ctf.register(parakeet.config.Config)
def _ctf_Config(config: parakeet.config.Config, output_file: str):
    """
    Simulate the ctf

    Args:
        config: The config object
        output_file: The output ctf filename

    """

    # Create the microscope
    microscope = parakeet.microscope.new(config.microscope)

    # Create the simulation
    simulation = simulation_factory(
        microscope=microscope, simulation=config.simulation.model_dump()
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
