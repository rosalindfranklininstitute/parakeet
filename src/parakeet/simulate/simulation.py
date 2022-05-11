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
from parakeet.scan import UniformAngularScan


# Get the logger
logger = logging.getLogger(__name__)


class Simulation(object):
    """
    An object to wrap the simulation

    """

    def __init__(
        self, image_size, pixel_size, scan=None, cluster=None, simulate_image=None
    ):
        """
        Initialise the simulation

        Args:
            image_size (tuple): The image size
            scan (object): The scan object
            cluster (object): The cluster spec
            simulate_image (func): The image simulation function

        """
        self.pixel_size = pixel_size
        self.image_size = image_size
        self.scan = scan
        self.cluster = cluster
        self.simulate_image = simulate_image

        # Single particle mode check
        if isinstance(scan, UniformAngularScan):
            self.scan.poses.write_star_file(self.scan.metadata_file)

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        Return
            tuple: The simulation data shape

        """
        nx = self.image_size[0]
        ny = self.image_size[1]
        nz = 1
        if self.scan is not None:
            nz = len(self.scan)
        return (nz, ny, nx)

    def angles(self) -> list[float]:
        """
        Return:
            The simulation angles

        """
        if self.scan is None:
            return [0]
        return self.scan.angles

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
        if self.cluster is None or self.cluster["method"] is None:
            for i, angle in enumerate(self.angles()):
                logger.info(
                    f"    Running job: {i+1}/{self.shape[0]} for {angle} degrees"
                )
                _, image, metadata = self.simulate_image(i)
                if writer is not None:
                    writer.data[i, :, :] = image
                    if metadata is not None:
                        writer.header[i] = metadata
        else:

            # Set the maximum number of workers
            self.cluster["max_workers"] = min(
                self.cluster["max_workers"], self.shape[0]
            )
            logger.info("Initialising %d worker threads" % self.cluster["max_workers"])

            # Get the futures executor
            with parakeet.futures.factory(**self.cluster) as executor:

                # Copy the data to each worker
                logger.info("Copying data to workers...")

                # Submit all jobs
                logger.info("Running simulation...")
                futures = []
                for i, angle in enumerate(self.scan.angles):
                    logger.info(
                        f"    Submitting job: {i+1}/{self.shape[0]} for {angle} degrees"
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
