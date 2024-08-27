#
# parakeet.analyse.reconstruct.py
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


__all__ = ["reconstruct"]


# Set the random seed
random.seed(0)


@singledispatch
def reconstruct(config_file, image_file: str, rec_file: str, device: Device = None):
    """
    Reconstruct the volume

    Args:
        config_file: The input config filename
        image_file: The input image filename
        rec_file: The output CTF corrected reconstruction filename
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
    _reconstruct_Config(
        config,
        image_file,
        rec_file,
    )


@reconstruct.register(parakeet.config.Config)
def _reconstruct_Config(
    config: parakeet.config.Config, image_filename: str, rec_filename: str
):
    """
    Reconstruct the volume and use 3D CTF correction beforehand if the input image is uncorrected

    """

    # Ensure mrc file
    assert os.path.splitext(image_filename)[1] == ".mrc"

    # Create the microscope
    microscope = parakeet.microscope.new(config.microscope)

    # Get the simulation parameters
    simulation = config.simulation.model_dump()

    # Set the corrected filename
    corrected_filename = os.path.join(os.path.dirname(rec_filename), "CORRECTED.dat")

    # Get the parameters for the CTF correction
    nx = microscope.detector.nx
    pixel_size = microscope.detector.pixel_size
    energy = microscope.beam.energy
    defocus = -microscope.lens.c_10
    num_defocus = int((nx * pixel_size) / 100)

    # Set the spherical aberration
    if simulation["inelastic_model"] == "cc_corrected":
        print("Setting spherical aberration to zero")
        spherical_aberration = 0.0
    else:
        spherical_aberration = microscope.lens.c_30

    astigmatism = microscope.lens.c_12
    astigmatism_angle = microscope.lens.phi_12
    phase_shift = 0

    # Do the reconstruction
    guanaco.reconstruct_file(
        input_filename=image_filename,
        output_filename=rec_filename,
        corrected_filename=corrected_filename,
        centre=None,
        energy=energy,
        defocus=defocus,
        num_defocus=num_defocus,
        spherical_aberration=spherical_aberration,
        astigmatism=astigmatism,
        astigmatism_angle=astigmatism_angle,
        phase_shift=phase_shift,
        angular_weights=True,
        device=config.multiprocessing.device,
    )
