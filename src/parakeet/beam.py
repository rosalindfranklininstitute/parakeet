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
import mrcfile
import numpy as np
import parakeet.config


class Beam(object):
    """
    A class to encapsulate a beam

    """

    def __init__(
        self,
        energy: float = 300,
        energy_spread: float = 0,
        acceleration_voltage_spread: float = 0,
        illumination_semiangle: float = 0.1,
        electrons_per_angstrom: float = 40,
        theta: float = 0,
        phi: float = 0,
        incident_wave: str = None,
    ):
        """
        Initialise the beam

        Args:
            energy: The beam energy (keV)
            energy_spread: dE / E where dE is the 1/e half width
            acceleration_voltage_spread: dV / V where dV is the 1 / e half width
            illumination_semiangle: The illumination semiangle spread (mrad)
            electrons_per_angstrom: The number of electrons per angstrom
            theta: The beam tilt theta
            phi: The beam tilt phi
            incident_wave: The wave function

        """
        self._energy = energy
        self._energy_spread = energy_spread
        self._acceleration_voltage_spread = acceleration_voltage_spread
        self._illumination_semiangle = illumination_semiangle
        self._electrons_per_angstrom = electrons_per_angstrom
        self._theta = theta
        self._phi = phi
        self._incident_wave_filename = incident_wave
        self._incident_wave: np.ndarray = None

    @property
    def energy(self) -> float:
        """
        The beam energy (keV)

        """
        return self._energy

    @energy.setter
    def energy(self, energy: float):
        self._energy = energy

    @property
    def energy_spread(self) -> float:
        """
        dE / E where dE is the 1/e half width

        """
        return self._energy_spread

    @energy_spread.setter
    def energy_spread(self, energy_spread: float):
        self._energy_spread = energy_spread

    @property
    def acceleration_voltage_spread(self) -> float:
        """
        dV / V where dV is the 1 / e half width

        """
        return self._acceleration_voltage_spread

    @acceleration_voltage_spread.setter
    def acceleration_voltage_spread(self, acceleration_voltage_spread: float):
        self._acceleration_voltage_spread = acceleration_voltage_spread

    @property
    def illumination_semiangle(self) -> float:
        """
        The illumination semiangle (mrad)

        """
        return self._illumination_semiangle

    @illumination_semiangle.setter
    def illumination_semiangle(self, illumination_semiangle: float):
        self._illumination_semiangle = illumination_semiangle

    @property
    def electrons_per_angstrom(self) -> float:
        """
        The number of electrons per angstrom

        """
        return self._electrons_per_angstrom

    @electrons_per_angstrom.setter
    def electrons_per_angstrom(self, electrons_per_angstrom: float):
        self._electrons_per_angstrom = electrons_per_angstrom

    @property
    def theta(self) -> float:
        """
        The beam tilt theta

        """
        return self._theta

    @theta.setter
    def theta(self, theta: float):
        self._theta = theta

    @property
    def phi(self) -> float:
        """
        The beam tilt phi

        """
        return self._phi

    @phi.setter
    def phi(self, phi: float):
        self._phi = phi

    @property
    def incident_wave(self) -> np.ndarray:
        if self._incident_wave is None and self._incident_wave_filename is not None:
            self._incident_wave = mrcfile.open(self._incident_wave_filename).data.copy()
            assert len(self._incident_wave.shape) == 2
        return self._incident_wave


def new(config: parakeet.config.Beam) -> Beam:
    """
    Create a beam

    Args:
        config: The beam model configuration

    Returns:
        The beam model

    """
    return Beam(
        energy=config.energy,
        energy_spread=config.energy_spread,
        acceleration_voltage_spread=config.acceleration_voltage_spread,
        illumination_semiangle=config.illumination_semiangle,
        electrons_per_angstrom=config.electrons_per_angstrom,
        theta=config.theta,
        phi=config.phi,
        incident_wave=config.incident_wave,
    )
