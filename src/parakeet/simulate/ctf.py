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

# Get the logger
logger = logging.getLogger(__name__)


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
        system_conf = create_system_configuration("cpu")

        # Create the multem input multislice object
        input_multislice = create_input_multislice(
            self.microscope,
            self.simulation["slice_thickness"],
            self.simulation["margin"],
            "HRTEM",
        )
        input_multislice.nx = nx
        input_multislice.ny = ny

        # Set the specimen size
        input_multislice.spec_lx = x_fov
        input_multislice.spec_ly = y_fov
        input_multislice.spec_lz = x_fov  # self.sample.containing_box[1][2]

        # Run the simulation
        image = np.array(multem.compute_ctf(system_conf, input_multislice)).T
        image = np.fft.fftshift(image)

        # Compute the image scaled with Poisson noise
        return (index, 0, 0, image, None, None)


def ctf_internal(microscope: Microscope, simulation: dict):
    """
    Create the simulation

    Args:
        microscope (object); The microscope object
        exit_wave (object): The exit_wave object
        scan (object): The scan object
        device (str): The device to use
        simulation (object): The simulation parameters
        cluster (object): The cluster parameters

    Returns:
        object: The simulation object

    """

    # Create the simulation
    return parakeet.simulate.simulation.Simulation(
        image_size=(microscope.detector.nx, microscope.detector.ny),
        pixel_size=microscope.detector.pixel_size,
        simulate_image=CTFSimulator(microscope=microscope, simulation=simulation),
    )


@singledispatch
def ctf(config_file: str, output: str):
    """
    Simulate the ctf

    """

    # Load the full configuration
    config = parakeet.config.load(config_file)

    # Print some options
    parakeet.config.show(config)

    # Create the microscope
    microscope = parakeet.microscope.new(**config.microscope.dict())

    # Create the simulation
    simulation = ctf_internal(
        microscope=microscope, simulation=config.simulation.dict()
    )

    # Create the writer
    logger.info(f"Opening file: {output}")
    writer = parakeet.io.new(
        output,
        shape=simulation.shape,
        pixel_size=simulation.pixel_size,
        dtype=np.complex64,
    )

    # Run the simulation
    simulation.run(writer)


# Register function for single dispatch
ctf.register(ctf_internal)
