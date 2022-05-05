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
import parakeet.metadata
import parakeet.sample
import parakeet.simulate
from functools import singledispatch

Device = parakeet.config.Device
ClusterMethod = parakeet.config.ClusterMethod


__all__ = ["run"]


# Get the logger
logger = logging.getLogger(__name__)


@singledispatch
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

    # Load the configuration
    config = parakeet.config.load(config_file)

    # Set the command line args in a dict
    if device is not None:
        config.device = device
    if cluster_max_workers is not None:
        config.cluster.max_workers = cluster_max_workers
    if cluster_method is not None:
        config.cluster.method = cluster_method

    # Print some options
    parakeet.config.show(config)

    # Create the sample
    return _run_Config(config, sample_file, exit_wave_file, optics_file, image_file)


@run.register
def _run_Config(
    config: parakeet.config.Config,
    sample_file: str,
    exit_wave_file: str,
    optics_file: str,
    image_file: str,
):
    """
    Simulate the TEM image from the sample

    Args:
        config: The config object filename
        sample_file: The sample filename
        exit_wave_file: The exit wave filename
        optics_file: The optics filename
        image_file: The image filename

    """
    # Create the sample model
    sample = parakeet.sample.new(config, sample_file)

    # Add the molecules
    sample = parakeet.sample.add_molecules(config.sample, sample)  # type: ignore

    # Simulate the exit wave
    parakeet.simulate.exit_wave(config, sample, exit_wave_file)  # type: ignore

    # Simulate the optics
    parakeet.simulate.optics(
        config,
        exit_wave_file,
        optics_file,
    )

    # Simulate the image
    parakeet.simulate.image(config, optics_file, image_file)

    # Export the metadata
    # parakeet.metadata.export(config, sample)  # type: ignore
