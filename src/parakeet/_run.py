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
from parakeet.config import Device
from functools import singledispatch


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
    device: Device = None,
    nproc: int = None,
    gpu_id: list = None,
    steps: list = None,
):
    """
    Simulate the TEM image from the sample

    If steps is None then all steps are run, otherwise steps is
    a list which contains one or more of the following: all, sample, simulate,
    sample.new, sample.add_molecules, simulate.exit_wave, simulate.optics and
    simulate.image

    Args:
        config_file: The config filename
        sample_file: The sample filename
        exit_wave_file: The exit wave filename
        optics_file: The optics filename
        image_file: The image filename
        device: The device to run on (cpu or gpu)
        nproc: The number of processes
        gpu_id: The list of gpu ids
        steps: Choose the steps to run

    """

    # Load the configuration
    config = parakeet.config.load(config_file)

    # Set the command line args in a dict
    if device is not None:
        config.multiprocessing.device = device
    if nproc is not None:
        config.multiprocessing.nproc = nproc
    if gpu_id is not None:
        config.multiprocessing.gpu_id = gpu_id

    # Print some options
    parakeet.config.show(config)

    # Create the sample
    return _run_Config(
        config, sample_file, exit_wave_file, optics_file, image_file, steps
    )


@run.register(parakeet.config.Config)
def _run_Config(
    config: parakeet.config.Config,
    sample_file: str,
    exit_wave_file: str,
    optics_file: str,
    image_file: str,
    steps: list = None,
):
    """
    Simulate the TEM image from the sample

    Args:
        config: The config object filename
        sample_file: The sample filename
        exit_wave_file: The exit wave filename
        optics_file: The optics filename
        image_file: The image filename
        steps: The steps to run

    """

    # The sample steps
    sample_steps = ["sample.new", "sample.add_molecules"]

    # The simulate steps
    simulate_steps = ["simulate.exit_wave", "simulate.optics", "simulate.image"]

    # Setup the steps
    if steps is None:
        steps = ["all"]
    if "all" in steps:
        steps.extend(sample_steps)
        steps.extend(simulate_steps)
    elif "sample" in steps:
        steps.extend(sample_steps)
    elif "simulate" in steps:
        steps.extend(simulate_steps)
    steps = list(set(steps))

    # Create the sample model or open
    if "sample.new" in steps:
        sample = parakeet.sample.new(config, sample_file)
    else:
        sample = parakeet.sample.load(sample_file)

    # Add the molecules
    if "sample.add_molecules" in steps:
        sample = parakeet.sample.add_molecules(config.sample, sample)  # type: ignore

    # Simulate the exit wave
    if "simulate.exit_wave" in steps:
        parakeet.simulate.exit_wave(config, sample, exit_wave_file)  # type: ignore

    # Simulate the optics
    if "simulate.optics" in steps:
        parakeet.simulate.optics(
            config,
            exit_wave_file,
            optics_file,
        )

    # Simulate the image
    if "simulate.image" in steps:
        parakeet.simulate.image(config, optics_file, image_file)

    # Export the metadata
    # parakeet.metadata.export(config, sample)  # type: ignore
