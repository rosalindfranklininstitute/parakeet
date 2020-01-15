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
        self,
        energy=300,
        energy_spread=0,
        acceleration_voltage_spread=0,
        electrons_per_angstrom=30,
    ):
        """
        Initialise the beam

        Args:
            energy (float): The beam energy (keV)
            energy_spread (float): dE / E where dE is the 1/e half width
            acceleration_voltage_spread (float): dV / V where dV is the 1 / e half width
            electrons_per_angstrom (float): The number of electrons per angstrom

        """
        self.energy = energy
        self.energy_spread = energy_spread
        self.acceleration_voltage_spread = acceleration_voltage_spread
        self.electrons_per_angstrom = electrons_per_angstrom


def new(
    energy=None,
    energy_spread=None,
    acceleration_voltage_spread=None,
    electrons_per_angstrom=None,
):
    """
    Create a beam

    Args:
        energy (float): The beam energy (keV)
        energy_spread (float): dE / E where dE is the 1/e half width
        acceleration_voltage_spread (float): dV / V where dV is the 1 / e half width
        electrons_per_angstrom (float): The number of electrons per angstrom

    Returns:
        object: The beam object

    """
    return Beam(
        energy=energy,
        energy_spread=energy_spread,
        acceleration_voltage_spread=acceleration_voltage_spread,
        electrons_per_angstrom=electrons_per_angstrom,
    )
