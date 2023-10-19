#
# parakeet.simulate.simulation.py
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
from typing import Tuple


# Get the logger
logger = logging.getLogger(__name__)


class Simulation(object):
    """
    An object to wrap the simulation

    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        pixel_size: float,
        scan=None,
        nproc=1,
        simulate_image=None,
    ):
        """
        Initialise the simulation

        Args:
            image_size: The image size
            scan (object): The scan object
            nproc: The number of processes
            simulate_image (func): The image simulation function

        """
        self.pixel_size = pixel_size
        self.image_size = image_size
        self.scan = scan
        self.nproc = nproc
        self.simulate_image = simulate_image

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Return
            The simulation data shape

        """
        nx = self.image_size[0]
        ny = self.image_size[1]
        nz = 1
        if self.scan is not None:
            nz = len(self.scan)
        return (nz, ny, nx)

    def angles(self) -> list:
        """
        Return:
            The simulation angles

        """
        if self.scan is None:
            return [(0, 0, 0)]
        return list(
            zip(self.scan.image_number, self.scan.fraction_number, self.scan.angles)
        )

    def run(self, writer=None):
        """
        Run the simulation

        Args:
            writer (object): Write each image to disk

        """

        # Check the shape of the writer
        if writer:
            assert writer.shape == self.shape

        # If we are executing in a single process just do a for loop
        if self.nproc is None or self.nproc <= 1:
            for i, (image_number, fraction_number, angle) in enumerate(self.angles()):
                logger.info(
                    f"    Running job: {i+1}/{self.shape[0]} for image {image_number} fraction {fraction_number} with tilt {angle} degrees"
                )
                _, image, metadata = self.simulate_image(i)
                if writer is not None:
                    writer.data[i, :, :] = image
                    if metadata is not None:
                        writer.header[i] = metadata
        else:
            # Set the maximum number of workers
            self.nproc = min(self.nproc, self.shape[0])
            logger.info("Initialising %d worker threads" % self.nproc)

            # Get the futures executor
            with parakeet.futures.factory(max_workers=self.nproc) as executor:
                # Submit all jobs
                futures = []
                for i, (image_number, fraction_number, angle) in enumerate(
                    self.angles()
                ):
                    logger.info(
                        f"    Running job: {i+1}/{self.shape[0]} for image {image_number} fraction {fraction_number} with tilt {angle} degrees"
                    )
                    futures.append(executor.submit(self.simulate_image, i))

                # Wait for results
                for j, future in enumerate(parakeet.futures.as_completed(futures)):
                    # Get the result
                    i, image, metadata = future.result()

                    # Set the output in the writer
                    if writer is not None:
                        writer.data[i, :, :] = image
                        if metadata is not None:
                            writer.header[i] = metadata

                    # Write some info
                    vmin = np.min(image)
                    vmax = np.max(image)
                    logger.info(
                        "    Processed job: %d (%d/%d); image min/max: %.2f/%.2f"
                        % (i + 1, j + 1, self.shape[0], vmin, vmax)
                    )
