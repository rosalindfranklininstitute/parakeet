#
# parakeet.beam.py
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
        source_spread=0.1,
        electrons_per_angstrom=30,
        drift=None,
    ):
        """
        Initialise the beam

        Args:
            energy (float): The beam energy (keV)
            energy_spread (float): dE / E where dE is the 1/e half width
            acceleration_voltage_spread (float): dV / V where dV is the 1 / e half width
            source_spread (float): The source spread (mrad)
            electrons_per_angstrom (float): The number of electrons per angstrom
            drift (float): The beam drift sigma (A)

        """
        self.energy = energy
        self.energy_spread = energy_spread
        self.acceleration_voltage_spread = acceleration_voltage_spread
        self.source_spread = source_spread
        self.electrons_per_angstrom = electrons_per_angstrom
        self.drift = drift


def new(
    energy=None,
    energy_spread=None,
    acceleration_voltage_spread=None,
    source_spread=0.1,
    electrons_per_angstrom=None,
    drift=None,
):
    """
    Create a beam

    Args:
        energy (float): The beam energy (keV)
        energy_spread (float): dE / E where dE is the 1/e half width
        acceleration_voltage_spread (float): dV / V where dV is the 1 / e half width
        source_spread (float): The source spread (mrad)
        electrons_per_angstrom (float): The number of electrons per angstrom
        drift (float): The beam drift sigma (A)

    Returns:
        object: The beam object

    """
    return Beam(
        energy=energy,
        energy_spread=energy_spread,
        acceleration_voltage_spread=acceleration_voltage_spread,
        source_spread=source_spread,
        electrons_per_angstrom=electrons_per_angstrom,
        drift=drift,
    )
