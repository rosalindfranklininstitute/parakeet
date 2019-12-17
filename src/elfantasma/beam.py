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

    def __init__(
        self, energy=300, energy_spread=0, acceleration_voltage_spread=0, flux=None
    ):
        """
        Initialise the beam

        Args:
            energy (float): The beam energy (keV)
            energy_spread (float): dE / E where dE is the 1/e half width
            acceleration_voltage_spread (float): dV / V where dV is the 1 / e half width
            flux (float): The flux (electrons / per second / per pixel(

        """
        self.energy = energy
        self.energy_spread = energy_spread
        self.acceleration_voltage_spread = acceleration_voltage_spread
        self.flux = flux


def new(energy=None, energy_spread=None, acceleration_voltage_spread=None, flux=None):
    """
    Create a beam

    Args:
        energy (float): The beam energy (keV)
        energy_spread (float): dE / E where dE is the 1/e half width
        acceleration_voltage_spread (float): dV / V where dV is the 1 / e half width
        flux (float): The flux (electrons / per second / per pixel)

    Returns:
        object: The beam object

    """
    return Beam(
        energy=energy,
        energy_spread=energy_spread,
        acceleration_voltage_spread=acceleration_voltage_spread,
        flux=flux,
    )
