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


def new(self, model=None, beam=None, lens=None, detector=None):
    """
    Make a new detector

    Args:
        nx (int): The size of the detector in X
        ny (int): The size of the detector in Y
        pixel_size (float): The effective size of the pixels in A

    Returns:
        obj: The detector object

    """

    # Construct the basic models from the input
    beam = elfantasma.beam.new(**beam)
    lens = elfantasma.lens.new(**lens)
    detector = elfantasma.detector.new(**detector)

    # Override the parameters for the different microscope models
    if model == "krios":
        beam.E_0 = 300
        lens.c_30 = 2.7
    elif model == "talos":
        beam.E_0 = 200
        lens.c_30 = 2.7
    elif model is not None:
        raise RuntimeError("Unknown microscope model")

    # Return the miroscope object
    return Microscope(model=model, beam=beam, lens=lens, detector=detector)
