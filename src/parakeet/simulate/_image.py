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


__all__ = ["image"]


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
        self,
        microscope=None,
        optics=None,
        scan=None,
        simulation=None,
        device="gpu",
        gpu_id=None,
    ):
        self.microscope = microscope
        self.optics = optics
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
        exposure_time = self.scan.exposure_time[index]
        electrons_per_angstrom = self.scan.electrons_per_angstrom[index]
        if exposure_time <= 0:
            exposure_time = 1.0

        # Check the angle and position
        assert abs(angle - self.optics.header[index]["tilt_alpha"]) < 1e7

        # Get the specimen atoms
        logger.info(f"Simulating image {index+1}")

        # Compute the number of counts per pixel
        electrons_per_pixel = (
            electrons_per_angstrom * self.microscope.detector.pixel_size**2
        )

        # Compute the electrons per pixel second
        electrons_per_second = electrons_per_pixel / exposure_time
        energy = self.microscope.beam.energy

        # Get the image
        image = self.optics.data[index]

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

        # Get the image metadata
        metadata = np.asarray(self.optics.header[index])
        metadata["dose"] = electrons_per_angstrom
        metadata["dqe"] = self.microscope.detector.dqe
        metadata["gain"] = 1
        metadata["offset"] = 0

        # Compute the image scaled with Poisson noise
        return (index, image.astype("float32"), metadata)


def simulation_factory(
    microscope: Microscope,
    optics: object,
    scan: Scan,
    simulation: dict = None,
    multiprocessing: dict = None,
) -> Simulation:
    """
    Create the simulation

    Args:
        microscope (object); The microscope object
        optics (object): The optics object
        scan (object): The scan object
        simulation (object): The simulation parameters
        multiprocessing (object): The multiprocessing parameters

    Returns:
        object: The simulation object

    """
    # Check multiprocessing settings
    if multiprocessing is None:
        multiprocessing = {"device": "gpu", "nproc": 1, "gpu_id": 0}
    else:
        assert multiprocessing["nproc"] in [None, 1]
        assert len(multiprocessing["gpu_id"]) == 1

    # Create the simulation
    return Simulation(
        image_size=(microscope.detector.nx, microscope.detector.ny),
        pixel_size=microscope.detector.pixel_size,
        scan=scan,
        simulate_image=ImageSimulator(
            microscope=microscope,
            optics=optics,
            scan=scan,
            simulation=simulation,
            device=multiprocessing["device"],
            gpu_id=multiprocessing["gpu_id"][0],
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

    # Do the work
    _image_Config(config, optics_file, image_file)


@image.register(parakeet.config.Config)
def _image_Config(config: parakeet.config.Config, optics_file: str, image_file: str):
    """
    Simulate the image with noise

    Args:
        config: The config object
        optics_file: The optics image filename
        image_file: The output image filename

    """

    # Create the microscope
    microscope = parakeet.microscope.new(config.microscope)

    # Create the exit wave data
    logger.info(f"Loading sample from {optics_file}")
    optics = parakeet.io.open(optics_file)

    # Create the scan
    scan = optics.header.scan

    # Override the dose
    scan_new = parakeet.scan.new(
        electrons_per_angstrom=microscope.beam.electrons_per_angstrom,
        **config.scan.model_dump(),
    )
    scan.data["electrons_per_angstrom"] = scan_new.electrons_per_angstrom

    # Create the simulation
    simulation = simulation_factory(
        microscope=microscope,
        optics=optics,
        scan=scan,
        simulation=config.simulation.model_dump(),
        multiprocessing=config.multiprocessing.model_dump(),
    )

    # Create the writer
    logger.info(f"Opening file: {image_file}")
    writer = parakeet.io.new(
        image_file,
        shape=simulation.shape,
        pixel_size=simulation.pixel_size,
        dtype=np.float32,
    )

    # Propagate the particle positions
    writer.particle_positions = optics.particle_positions

    # Run the simulation
    simulation.run(writer)
