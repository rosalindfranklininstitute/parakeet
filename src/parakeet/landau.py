#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#

import numpy as np
import scipy.special
import scipy.integrate
import scipy.constants
from math import pi, log, sqrt


def electron_velocity(energy):
    """
    Calculate the velocity of an electron relative to the speed of light

    Args:
        energy (float): The electron energy (units: eV)

    Returns:
        float: v / c

    """

    # Get some constants
    c = scipy.constants.speed_of_light
    m0 = scipy.constants.electron_mass
    e = scipy.constants.elementary_charge
    eV = e * energy
    return sqrt(1 - (m0 * c**2 / (m0 * c**2 + eV)) ** 2)


def landau(l):
    """
    Calculate the landau universal function

    Params:
        l (float): The dimensionless variable

    Returns:
        float: The value of the landau distribution at l

    """
    u = np.arange(1e-30, 40, 0.01)
    v = np.exp(-pi * u / 2) * np.cos(l * u + u * np.log(u))
    psi = (1 / pi) * np.trapz(v, u)
    if psi < 1e-5:
        psi = 0
    return psi


def mpl_and_fwhm(energy, thickness):
    """
    Compute the MPL and FWHM

    Params:
        energy (float): The beam energy (keV)
        thickness (float): The thickness of the ice (A)

    Returns:
        tuple: (MPL, FWHM)

    """

    # The parameters
    Z = 7.42  # Atomic number
    A = 18.01  # g / mol
    rho = 0.94  # g/cm^3
    x = thickness * 1e-10  # m

    # Convert to SI
    rho *= 100**3 / 1000  # kg / m^3
    A /= 1000  # kg / mol

    # Some physical quantities
    Na = scipy.constants.Avogadro  # 1/mol
    c = scipy.constants.speed_of_light  # m/s
    m0 = scipy.constants.electron_mass  # Kg
    e = scipy.constants.elementary_charge  # C
    re = scipy.constants.value("classical electron radius")  # m
    # eps_0 = scipy.constants.value("vacuum electric permittivity")
    beta = electron_velocity(energy * 1000)
    I = e * 13.5 * Z  # Bethe's characteristic atomic energy (keV)
    gamma = 0.577215664901532860606512090  # Euler's constant

    # The peak and fwhm of the universal function
    lambda_M = -0.223
    lambda_FWHM = 4.018
    # lambda_2w = 8.960

    # Compute xi and eps and dE0
    xi = 2 * pi * Na * re**2 * m0 * c**2 * Z * rho * x / (beta**2 * A)
    eps = I**2 * (1 - beta**2) / (beta**2 * 2 * m0 * c**2)
    eps = eps / e
    xi = xi / e
    dE0 = xi * (np.log(xi / eps) + 1 - beta**2 - gamma)

    # Compute the MPL and FWHM energy loss
    dE_MP = lambda_M * xi + dE0
    dE_FWHM = lambda_FWHM * xi
    # dE_2w = lambda_2w * xi

    # Return the MPL and FWHM
    return dE_MP, dE_FWHM


def energy_loss_distribution(dE, energy=300, thickness=3000):
    """
    Compute the energy loss distribution

    Params:
        dE (array): The array of energy losses (eV)
        energy (float): The beam energy (keV)
        thickness (float): The thickness of the ice (A)

    Returns:
        array: The distribution

    """

    # The parameters
    Z = 7.42  # Atomic number
    A = 18.01  # g / mol
    rho = 0.94  # g/cm^3
    x = thickness * 1e-10

    # Convert to SI
    rho *= 100**3 / 1000  # kg / m^3
    A /= 1000  # kg / mol

    # Some physical quantities
    Na = scipy.constants.Avogadro  # 1/mol
    c = scipy.constants.speed_of_light  # m/s
    m0 = scipy.constants.electron_mass  # Kg
    e = scipy.constants.elementary_charge  # C
    re = scipy.constants.value("classical electron radius")  # m
    # eps_0 = scipy.constants.value("vacuum electric permittivity")
    beta = electron_velocity(energy * 1000)
    I = e * 13.5 * Z  # Bethe's characteristic atomic energy (keV)
    gamma = 0.577215664901532860606512090  # Euler's constant

    # Compute xi and eps and dE0
    xi = 2 * pi * Na * re**2 * m0 * c**2 * Z * rho * x / (beta**2 * A)
    eps = I**2 * (1 - beta**2) / (beta**2 * 2 * m0 * c**2)
    eps = eps / e
    xi = xi / e
    dE0 = xi * (log(xi / eps) + 1 - beta**2 - gamma)

    # Compute the points at which to compute psi
    lam = (dE - dE0) / xi
    phi = np.array([landau(xx) for xx in lam])
    phi = phi / np.sum(phi)

    # return the landau density
    return phi


class Landau(object):
    """
    A class to represent the Landau distribution

    """

    CACHE = None

    def __init__(self):
        """
        Initialise the class

        """

        # Generate the table of values for the universal function
        if Landau.CACHE is None:
            l0 = -10
            l1 = 200
            dl = 0.01
            lambda_ = np.arange(l0, l1, dl)
            phi = np.array([landau(xx) for xx in lambda_])
            Landau.CACHE = (l0, l1, dl, lambda_, phi)

        # Get the values from the cache
        self.l0, self.l1, self.dl, self.lambda_, self.phi = Landau.CACHE

    def __call__(self, dE, energy, thickness):
        """
        Compute the landau distribution for the given energy losses

        Params:
            dE (array): The array of energy losses
            energy (float): The beam energy (eV)
            thickness (float): The sample thickness (A)

        Returns:
            array: The energy loss distribution

        """
        return np.interp(
            self.dE_to_lambda(dE, energy, thickness), self.lambda_, self.phi
        )

    def dE_to_lambda(self, dE, energy, thickness):
        """
        Convert dE to lambda

        Params:
            dE (float): The energy loss (eV)
            energy (float): The beam energy (eV)
            thickness (float): The sample thickness (A)

        Returns:
            float: lambda

        """

        # The parameters
        Z = 7.42  # Atomic number
        A = 18.01  # g / mol
        rho = 0.94  # g/cm^3
        x = thickness * 1e-10  # m

        # Convert to SI
        rho *= 100**3 / 1000  # kg / m^3
        A /= 1000  # kg / mol

        # Some physical quantities
        Na = scipy.constants.Avogadro  # 1/mol
        c = scipy.constants.speed_of_light  # m/s
        m0 = scipy.constants.electron_mass  # Kg
        e = scipy.constants.elementary_charge  # C
        re = scipy.constants.value("classical electron radius")  # m
        # eps_0 = scipy.constants.value("vacuum electric permittivity")
        beta = electron_velocity(energy)
        I = e * 13.5 * Z  # Bethe's characteristic atomic energy (keV)
        gamma = 0.577215664901532860606512090  # Euler's constant

        # Compute xi and eps and dE0
        xi = 2 * pi * Na * re**2 * m0 * c**2 * Z * rho * x / (beta**2 * A)
        eps = I**2 * (1 - beta**2) / (beta**2 * 2 * m0 * c**2)
        eps = eps / e
        xi = xi / e
        dE0 = xi * (log(xi / eps) + 1 - beta**2 - gamma)

        # Convert dE to lambda
        l = (dE - dE0) / xi

        # Return lambda
        return l

    def lambda_to_dE(self, l, energy, thickness):
        """
        Convert lambda to dE

        Params:
            l (float): Lambda
            energy (float): The beam energy (eV)
            thickness (float): The sample thickness (A)

        Returns:
            float: The energy loss dE (eV)

        """

        # The parameters
        Z = 7.42  # Atomic number
        A = 18.01  # g / mol
        rho = 0.94  # g/cm^3
        x = thickness * 1e-10  # m

        # Convert to SI
        rho *= 100**3 / 1000  # kg / m^3
        A /= 1000  # kg / mol

        # Some physical quantities
        Na = scipy.constants.Avogadro  # 1/mol
        c = scipy.constants.speed_of_light  # m/s
        m0 = scipy.constants.electron_mass  # Kg
        e = scipy.constants.elementary_charge  # C
        re = scipy.constants.value("classical electron radius")  # m
        # eps_0 = scipy.constants.value("vacuum electric permittivity")
        beta = electron_velocity(energy)
        I = e * 13.5 * Z  # Bethe's characteristic atomic energy (keV)
        gamma = 0.577215664901532860606512090  # Euler's constant

        # Compute xi and eps and dE0
        xi = 2 * pi * Na * re**2 * m0 * c**2 * Z * rho * x / (beta**2 * A)
        eps = I**2 * (1 - beta**2) / (beta**2 * 2 * m0 * c**2)
        eps = eps / e
        xi = xi / e
        dE0 = xi * (log(xi / eps) + 1 - beta**2 - gamma)

        # Convert lambda to dE
        dE = l * xi + dE0

        # Return lambda
        return dE
