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
import warnings
from parakeet.microscope import Microscope
from functools import singledispatch
from parakeet.simulate.simulation import Simulation

Device = parakeet.config.Device


__all__ = ["simple"]


# Get the logger
logger = logging.getLogger(__name__)


# Try to input MULTEM
try:
    import multem
except ImportError:
    warnings.warn("Could not import MULTEM")


class SimpleImageSimulator(object):
    """
    A class to do the actual simulation

    """

    def __init__(
        self, microscope=None, atoms=None, scan=None, simulation=None, device="gpu"
    ):
        self.microscope = microscope
        self.atoms = atoms
        self.scan = scan
        self.simulation = simulation
        self.device = device

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

        # Set the rotation angle
        # input_multislice.spec_rot_theta = angle
        # input_multislice.spec_rot_u0 = simulation.scan.axis

        # x0 = (-offset, -offset)
        # x1 = (x_fov + offset, y_fov + offset)

        # Create the multem system configuration
        system_conf = parakeet.simulate.simulation.create_system_configuration(
            self.device
        )

        # Create the multem input multislice object
        input_multislice = parakeet.simulate.simulation.create_input_multislice(
            self.microscope,
            self.simulation["slice_thickness"],
            self.simulation["margin"],
            "EWRS",
        )

        # Set the specimen size
        input_multislice.spec_lx = x_fov + offset * 2
        input_multislice.spec_ly = y_fov + offset * 2
        input_multislice.spec_lz = np.max(self.atoms.data["z"])

        # Set the atoms in the input after translating them for the offset
        input_multislice.spec_atoms = self.atoms.translate(
            (offset, offset, 0)
        ).to_multem()

        # Run the simulation
        output_multislice = multem.simulate(system_conf, input_multislice)

        # Get the ideal image data
        # Multem outputs data in column major format. In C++ and Python we
        # generally deal with data in row major format so we must do a
        # transpose here.
        image = np.array(output_multislice.data[0].psi_coh).T

        # Print some info
        psi_tot = np.abs(image) ** 2
        logger.info("Ideal image min/max: %f/%f" % (np.min(psi_tot), np.max(psi_tot)))

        # Compute the image scaled with Poisson noise
        return (index, image, None)


def simulation_factory(
    microscope: Microscope,
    atoms: str,
    device: Device = Device.gpu,
    simulation: dict = None,
):
    """
    Create the simulation

    Args:
        microscope (object); The microscope object
        atoms (object): The atom data
        device (str): The device to use
        simulation (object): The simulation parameters
        cluster (object): The cluster parameters

    Returns:
        object: The simulation object

    """
    # Create the scan
    scan = parakeet.scan.new("still")

    # Get the margin
    margin = 0 if simulation is None else simulation.get("margin", 0)

    # Create the simulation
    return Simulation(
        image_size=(
            microscope.detector.nx + 2 * margin,
            microscope.detector.ny + 2 * margin,
        ),
        pixel_size=microscope.detector.pixel_size,
        scan=scan,
        cluster={"method": None},
        simulate_image=SimpleImageSimulator(
            microscope=microscope,
            scan=scan,
            atoms=atoms,
            simulation=simulation,
            device=device,
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


@simple.register
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
        device=config.device,
        simulation=config.simulation.dict(),
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
