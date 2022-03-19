#
# parakeet.simulate.image.py
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
from parakeet.microscope import Microscope
from parakeet.scan import Scan
from functools import singledispatch
from parakeet.simulate.simulation import Simulation
from parakeet.config import Device

# Get the logger
logger = logging.getLogger(__name__)


class ImageSimulator(object):
    """
    A class to do the actual simulation

    The simulation is structured this way because the input data to the
    simulation is large enough that it makes an overhead to creating the
    individual processes.

    """

    def __init__(
        self, microscope=None, optics=None, scan=None, simulation=None, device="gpu"
    ):
        self.microscope = microscope
        self.optics = optics
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

        # Check the angle and position
        assert abs(angle - self.optics.angle[index]) < 1e7
        assert (np.abs(position - self.optics.position[index]) < 1e7).all()

        # Get the specimen atoms
        logger.info(f"Simulating image {index+1}")

        # Compute the number of counts per pixel
        electrons_per_pixel = (
            self.microscope.beam.electrons_per_angstrom
            * self.microscope.detector.pixel_size**2
        )

        # Compute the electrons per pixel second
        electrons_per_second = electrons_per_pixel / self.scan.exposure_time
        energy = self.microscope.beam.energy

        # Get the image
        image = self.optics.data[index]

        # Get some other properties to propagate
        beam_drift = self.optics.drift[index]
        defocus = self.optics.defocus[index]

        # Apply the dqe in Fourier space
        if self.microscope.detector.dqe:
            logger.info("Applying DQE")
            dqe = parakeet.dqe.DQETable().dqe_fs(
                energy, electrons_per_second, image.shape
            )
            dqe = np.fft.fftshift(dqe)
            fft_image = np.fft.fft2(image)
            fft_image *= dqe
            image = np.real(np.fft.ifft2(fft_image))

        # Ensure all pixels are >= 0
        image = np.clip(image, 0, None)

        # Add Poisson noise
        # np.random.seed(index)
        image = np.random.poisson(image * electrons_per_pixel).astype("float64")

        # Print some info
        logger.info(
            "    Image min/max/mean: %g/%g/%.2g"
            % (np.min(image), np.max(image), np.mean(image))
        )

        # Compute the image scaled with Poisson noise
        return (index, angle, position, image.astype("float32"), beam_drift, defocus)


def image_internal(
    microscope: Microscope,
    optics: object,
    scan: Scan,
    device: Device = Device.gpu,
    simulation: dict = None,
    cluster: dict = None,
):
    """
    Create the simulation

    Args:
        microscope (object); The microscope object
        optics (object): The optics object
        scan (object): The scan object
        device (str): The device to use
        simulation (object): The simulation parameters
        cluster (object): The cluster parameters

    Returns:
        object: The simulation object

    """

    # Create the simulation
    return Simulation(
        image_size=(microscope.detector.nx, microscope.detector.ny),
        pixel_size=microscope.detector.pixel_size,
        scan=scan,
        cluster=cluster,
        simulate_image=ImageSimulator(
            microscope=microscope,
            optics=optics,
            scan=scan,
            simulation=simulation,
            device=device,
        ),
    )


@singledispatch
def image(config_file, optics_file: str, image_file: str):
    """
    Simulate the image with noise

    Args:
        config_file: The config filename
        optics_file: The optics image filename
        image_file: The output image filename

    """

    # Load the full configuration
    config = parakeet.config.load(config_file)

    # Print some options
    parakeet.config.show(config)

    # Create the microscope
    microscope = parakeet.microscope.new(**config.microscope.dict())

    # Create the exit wave data
    logger.info(f"Loading sample from {optics_file}")
    optics = parakeet.io.open(optics_file)

    # Create the scan
    scan = parakeet.scan.new(
        angles=optics.angle, positions=optics.position[:, 1], **config.scan.dict()
    )

    # Create the simulation
    simulation = image_internal(
        microscope=microscope,
        optics=optics,
        scan=scan,
        device=config.device,
        simulation=config.simulation.dict(),
        cluster=config.cluster.dict(),
    )

    # Create the writer
    logger.info(f"Opening file: {image_file}")
    writer = parakeet.io.new(
        image_file,
        shape=simulation.shape,
        pixel_size=simulation.pixel_size,
        dtype=np.float32,
    )

    # Run the simulation
    simulation.run(writer)


# Register function for single dispatch
image.register(image_internal)
