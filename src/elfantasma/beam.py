#
# elfantasma.beam.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#


class Beam(object):
    """
    A class to encapsulate a beam

    """

    def __init__(self, energy=300, flux=None):
        """
        Initialise the beam

        Args:
            energy (float): The beam energy (keV)
            flux (float): The flux (electrons / per second / per pixel(

        """
        self.energy = energy
        self.flux = flux


def new(energy=None, flux=None):
    """
    Create a beam

    Args:
        energy (float): The beam energy (keV)
        flux (float): The flux (electrons / per second / per pixel(

    Returns:
        object: The beam object

    """
    return Beam(energy=energy, flux=flux)
