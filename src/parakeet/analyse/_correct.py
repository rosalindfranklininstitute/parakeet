#
# parakeet.analyse.correct.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import os.path
import guanaco
import random
import parakeet.microscope
import parakeet.sample
from parakeet.config import Device
from functools import singledispatch


__all__ = ["correct"]


# Set the random seed
random.seed(0)


@singledispatch
def correct(
    config_file,
    image_file: str,
    corrected_file: str,
    num_defocus: int = 1,
    device: Device = None,
):
    """
    Correct the images using 3D CTF correction

    Args:
        config_file: The input config filename
        image_file: The input image filename
        corrected_file: The output CTF corrected filename
        num_defocus: The number of defoci
        device: The device to use (CPU or GPU)

    """

    # Load the full configuration
    config = parakeet.config.load(config_file)

    # Set the device
    if device is not None:
        config.multiprocessing.device = device

    # Print some options
    parakeet.config.show(config)

    # Do the reconstruction
    _correct_Config(
        config,
        image_file,
        corrected_file,
        num_defocus=num_defocus,
    )


@correct.register(parakeet.config.Config)
def _correct_Config(
    config: parakeet.config.Config,
    image_filename: str,
    corrected_filename: str,
    num_defocus: int = None,
):
    """
    Correct the images using 3D CTF correction

    """

    # Create the microscope
    microscope = parakeet.microscope.new(config.microscope)

    # Set the simulation params
    simulation = config.simulation

    # Ensure mrc file
    assert os.path.splitext(image_filename)[1] == ".mrc"

    # Get the parameters for the CTF correction
    nx = microscope.detector.nx
    pixel_size = microscope.detector.pixel_size
    energy = microscope.beam.energy
    defocus = -microscope.lens.c_10
    if num_defocus is None:
        num_defocus = int((nx * pixel_size) / 100)

    # Set the spherical aberration
    if simulation.inelastic_model == "cc_corrected":
        print("Setting spherical aberration to zero")
        spherical_aberration = 0.0
    else:
        spherical_aberration = microscope.lens.c_30

    astigmatism = microscope.lens.c_12
    astigmatism_angle = microscope.lens.phi_12
    phase_shift = 0

    # Do the reconstruction
    guanaco.correct_file(
        input_filename=image_filename,
        output_filename=corrected_filename,
        centre=None,
        energy=energy,
        defocus=defocus,
        num_defocus=num_defocus,
        spherical_aberration=spherical_aberration,
        astigmatism=astigmatism,
        astigmatism_angle=astigmatism_angle,
        phase_shift=phase_shift,
        device=config.multiprocessing.device,
    )
