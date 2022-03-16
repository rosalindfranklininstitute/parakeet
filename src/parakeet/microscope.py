#
# parakeet.microscope.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import parakeet.beam
import parakeet.detector
import parakeet.lens


class Microscope(object):
    """
    A class to encapsulate a microscope

    """

    def __init__(
        self, model=None, beam=None, lens=None, detector=None, phase_plate=False
    ):
        """
        Initialise the detector

        Args:
            model (str): The microscope model name
            beam (object): The beam object
            lens (object): The lens object
            detector (object): The detector object
            phase_plate (bool): The phase plate

        """
        self.model = model
        self.beam = beam
        self.lens = lens
        self.detector = detector
        self.phase_plate = phase_plate


def new(model=None, beam=None, lens=None, detector=None, phase_plate=False):
    """
    Make a new detector

    Args:
        model (str): The microscope model
        beam (dict): The beam parameters
        lens (dict): The objective lens parameters
        detector (dict): The detector parameters
        phase_plate (bool): The phase plate

    Returns:
        obj: The detector object

    """

    # Construct the basic models from the input
    beam = parakeet.beam.new(**beam)
    lens = parakeet.lens.new(**lens)
    detector = parakeet.detector.new(**detector)

    # Override the parameters for the different microscope models
    if model == "krios":
        beam.energy = 300
        beam.energy_spread = 2.66 * 1e-6
        beam.acceleration_voltage_spread = 0.8 * 1e-6
        lens.c_30 = 2.7
        lens.c_c = 2.7
        lens.current_spread = 0.33 * 1e-6
    elif model == "talos":
        beam.energy = 200
        beam.energy_spread = 2.66 * 1e-6
        beam.acceleration_voltage_spread = 0.8 * 1e-6
        lens.c_30 = 2.7
        lens.c_c = 2.7
        lens.current_spread = 0.33 * 1e-6
    elif model is not None:
        raise RuntimeError("Unknown microscope model")

    # Return the miroscope object
    return Microscope(
        model=model, beam=beam, lens=lens, detector=detector, phase_plate=phase_plate
    )
