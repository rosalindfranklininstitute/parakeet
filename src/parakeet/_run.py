#
# parakeet.simulate.exit_wave.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#

import logging
import parakeet.config
import parakeet.sample
import parakeet.simulate
from parakeet.config import Device
from parakeet.config import ClusterMethod


__all__ = ["exit_wave"]


# Get the logger
logger = logging.getLogger(__name__)


def run(
    config_file,
    sample_file: str,
    exit_wave_file: str,
    optics_file: str,
    image_file: str,
    device: Device = Device.gpu,
    cluster_method: ClusterMethod = None,
    cluster_max_workers: int = 1,
):
    """
    Simulate the TEM image from the sample

    Args:
        config_file: The config filename
        sample_file: The sample filename
        exit_wave_file: The exit wave filename
        optics_file: The optics filename
        image_file: The image filename
        device: The device to run on (CPU or GPU)
        cluster_method: The cluster method to use (default None)
        cluster_max_workers: The maximum number of cluster jobs

    """

    # Create the sample model
    parakeet.sample.new(config_file, sample_file)

    # Add the molecules
    parakeet.sample.add_molecules(config_file, sample_file)

    # Simulate the exit wave
    parakeet.simulate.exit_wave(
        config_file,
        sample_file,
        exit_wave_file,
        device=device,
        cluster_method=cluster_method,
        cluster_max_workers=cluster_max_workers,
    )

    # Simulate the optics
    parakeet.simulate.optics(
        config_file,
        exit_wave_file,
        optics_file,
        device=device,
        cluster_method=cluster_method,
        cluster_max_workers=cluster_max_workers,
    )

    # Simulate the image
    parakeet.simulate.image(config_file, optics_file, image_file)
