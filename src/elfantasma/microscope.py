#
# elfantasma.microscope.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import elfantasma.beam
import elfantasma.detector
import elfantasma.lens


class Microscope(object):
    """
    A class to encapsulate a microscope

    """

    def __init__(self, model=None, beam=None, lens=None, detector=None):
        """
        Initialise the detector

        Args:
            model (str): The microscope model name
            beam (object): The beam object
            lens (object): The lens object
            detector (object): The detector object

        """
        self.model = model
        self.beam = beam
        self.lens = lens
        self.detector = detector


def new(model=None, beam=None, objective_lens=None, detector=None):
    """
    Make a new detector

    Args:
        model (str): The microscope model
        beam (dict): The beam parameters
        objective_lens (dict): The objective lens parameters
        detector (dict): The detector parameters

    Returns:
        obj: The detector object

    """

    # Construct the basic models from the input
    beam = elfantasma.beam.new(**beam)
    lens = elfantasma.lens.new(**objective_lens)
    detector = elfantasma.detector.new(**detector)

    # Override the parameters for the different microscope models
    if model == "krios":
        beam.energy = 300
        beam.energy_spread = 0.33 * 1e-6
        beam.acceleration_voltage_spread = 0.8 * 1e-6
        lens.c_30 = 2.7
        lens.c_c = 2.7
        lens.current_spread = 0.33 * 1e-6
    elif model == "talos":
        beam.energy = 200
        beam.energy_spread = 0.33 * 1e-6
        beam.acceleration_voltage_spread = 0.8 * 1e-6
        lens.c_30 = 2.7
        lens.c_c = 2.7
        lens.current_spread = 0.33 * 1e-6
    elif model is not None:
        raise RuntimeError("Unknown microscope model")

    # Return the miroscope object
    return Microscope(model=model, beam=beam, lens=lens, detector=detector)
