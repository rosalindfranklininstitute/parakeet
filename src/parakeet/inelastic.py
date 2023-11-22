#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import numpy as np
import parakeet.landau
import scipy.signal
from math import sqrt, pi, cos, exp, log, ceil, floor


def effective_thickness(shape, angle):
    """
    Compute the effective thickness

    """
    TINY = 1e-5
    if shape["type"] == "cube":
        D0 = shape["cube"]["length"]
        cos_angle = cos(pi * angle / 180.0)
        if abs(cos_angle) < TINY:
            cos_angle = TINY
        thickness = D0 / cos_angle
    elif shape["type"] == "cuboid":
        D0 = shape["cuboid"]["length_z"]
        cos_angle = cos(pi * angle / 180.0)
        if abs(cos_angle) < TINY:
            cos_angle = TINY
        thickness = D0 / cos_angle
    elif shape["type"] == "cylinder":
        thickness = shape["cylinder"]["radius"] * 2
    if isinstance(thickness, list):
        thickness = np.mean(thickness)
    return abs(thickness)


def zero_loss_fraction(shape, angle):
    """
    Compute the zero loss fraction

    """
    thickness = effective_thickness(shape, angle)
    mean_free_path = 3150  # A for Amorphous Ice at 300 keV
    electron_fraction = exp(-thickness / mean_free_path)
    return electron_fraction


def mp_loss_fraction(shape, angle):
    """
    Compute the inelastic fraction

    """
    thickness = effective_thickness(shape, angle)
    mean_free_path = 3150  # A for Amorphous Ice at 300 keV
    electron_fraction = exp(-thickness / mean_free_path)
    return 1.0 - electron_fraction


def fraction_of_electrons(shape, angle, model=None):
    """
    Compute the fraction of electrons

    """
    if model is None:
        fraction = 1.0
    elif model == "zero_loss":
        fraction = zero_loss_fraction(shape, angle)
    elif model == "mp_loss":
        fraction = mp_loss_fraction(shape, angle)
    elif model == "unfiltered":
        fraction = 1.0
    elif model == "cc_corrected":
        fraction = 1.0
    return fraction


def most_probable_loss(energy, shape, angle):
    """
    Compute the MPL peak and sigma

    Params:
        energy (float): Beam energy in keV

    Returns:
        tuple: (peak, sigma) of the energy loss distribution (eV)

    """
    thickness = effective_thickness(shape, angle)
    thickness = min(thickness, 100000)  # Maximum 10 um - to avoid issues at high tilt
    peak, fwhm = parakeet.landau.mpl_and_fwhm(energy, thickness)
    return peak, fwhm / (2 * sqrt(2 * log(2)))


class EnergyFilterOptimizer(object):
    """
    A simple class to find the optimal energy filter placement when trying to
    take into account the inelastic scattering component

    """

    def __init__(self, energy_spread=0.8, dE_min=-10, dE_max=200, dE_step=0.01):
        """
        Initialise the class

        Params:
            energy_spread (float): The energy spread (eV)
            dE_min (float): The minimum energy loss (eV)
            dE_max (float): The maximum energy loss (eV)
            dE_step (float): The energy loss step (eV)

        """

        # Save the energy spread
        self.energy_spread = energy_spread  # eV

        # Save an instance of the landau distribution class
        self.landau = parakeet.landau.Landau()

        # Set the energy losses to consider
        self.dE_min = dE_min
        self.dE_max = dE_max
        self.dE_step = dE_step

    def __call__(self, energy, thickness, filter_width=None):
        """
        Compute the optimum position for the energy filter

        Params:
            energy (float): The beam energy (eV)
            thickness (float): The sample thickness (A)
            filter_width (float): The energy filter width (eV)

        Returns:
            float: The optimal position to maximum number of electrons (eV)

        """

        # Check the input
        assert energy > 0
        assert thickness > 0

        # The energy loss distribution
        dE, distribution = self.energy_loss_distribution(energy, thickness)

        # Compute optimum given the position and filter width
        if filter_width is None:
            position = np.sum(dE * distribution) / np.sum(dE)
        else:
            size = len(distribution)
            kernel_size = filter_width / self.dE_step
            ks = kernel_size / 2
            kx = np.arange(-size // 2, size // 2 + 1)
            kernel = np.exp(-0.5 * (kx / ks) ** 80)
            kernel = kernel / np.sum(kernel)
            num = scipy.signal.fftconvolve(distribution, kernel, mode="same")
            position = dE[np.argmax(num)]

        # Return the filter position
        return position

    def elastic_fraction(self, energy, thickness):
        """
        Compute the elastic electron fraction

        Params:
            energy (float): The beam energy (eV)
            thickness (float): The sample thickness (A)

        Returns:
            float: The elastic electron fraction

        """
        # Compute the fractions for the zero loss and energy losses
        mean_free_path = 3150  # A for Amorphous Ice at 300 keV
        return exp(-thickness / mean_free_path)

    def energy_loss_distribution(self, energy, thickness):
        """
        Compute the energy loss distribution

        Params:
            energy (float): The beam energy (eV)
            thickness (float): The sample thickness (A)

        Returns:
            (array, array): dE (ev) and the energy loss distribution

        """

        # The energy losses to consider
        dE = np.arange(self.dE_min, self.dE_max, self.dE_step, dtype="float64")

        # The energy loss distribution
        energy_loss_distribution = self.landau(dE, energy, thickness)
        sum_ELD = np.sum(energy_loss_distribution)
        if sum_ELD > 0:
            energy_loss_distribution /= self.dE_step * sum_ELD

        # The zero loss distribution
        zero_loss_distribution = (1.0 / sqrt(pi * self.energy_spread**2)) * np.exp(
            -(dE**2) / self.energy_spread**2
        )

        # Compute the fractions for the zero loss and energy losses
        elastic_fraction = self.elastic_fraction(energy, thickness)

        # Return the distribution
        distribution = (
            elastic_fraction * zero_loss_distribution
            + (1 - elastic_fraction) * energy_loss_distribution
        )
        return dE, distribution

    def compute_elastic_component(self, energy, thickness, position, filter_width):
        """
        Compute the elastic fraction and energy spread

        Params:
            energy (float): The beam energy (eV)
            thickness (float): The sample thickness (A)
            position (float): The filter position (eV)
            filter_width (float): The energy filter width (eV)

        Returns:
            float, float: The electron fraction and energy spread (eV)

        """

        # The energy losses to consider
        dE = np.arange(self.dE_min, self.dE_max, self.dE_step)

        # The zero loss distribution
        P = (1.0 / sqrt(pi * self.energy_spread**2)) * np.exp(
            -(dE**2) / self.energy_spread**2
        )
        C = self.dE_step * np.cumsum(P)

        # Compute the fractions for the zero loss and energy losses
        fraction = self.elastic_fraction(energy, thickness)

        # Compute the number of electrons and the sigma
        if filter_width is not None:
            dE0 = position - filter_width / 2.0
            dE1 = position + filter_width / 2.0
            x0 = int(floor((dE0 - self.dE_min) / self.dE_step))
            x1 = int(ceil((dE1 - self.dE_min) / self.dE_step))
            x0 = max(x0, 0)
            x1 = min(x1, len(P) - 1)
            assert x1 > x0
            fraction *= C[x1] - C[x0]
            P = P[x0:x1]
            dE = dE[x0:x1]

        # Compute the spread
        if len(P) > 0 and np.sum(P) > 0:
            dE_mean = np.sum(P * dE) / np.sum(P)
            spread = sqrt(np.sum(P * (dE - dE_mean) ** 2) / np.sum(P)) * sqrt(2)
        else:
            spread = 0

        # Return the fraction and spread
        return fraction, spread

    def compute_inelastic_component(self, energy, thickness, position, filter_width):
        """
        Compute the inelastic fraction and energy spread

        Params:
            energy (float): The beam energy (eV)
            thickness (float): The sample thickness (A)
            position (float): The filter position (eV)
            filter_width (float): The energy filter width (eV)

        Returns:
            float, float: The electron fraction and energy spread (eV)

        """

        # The energy losses to consider
        dE = np.arange(self.dE_min, self.dE_max, self.dE_step)

        # The energy loss distribution
        P = self.landau(dE, energy, thickness)
        sum_P = np.sum(P)
        if sum_P > 0:
            P /= self.dE_step * sum_P
        C = np.cumsum(P) * self.dE_step

        # Compute the fractions for the zero loss and energy losses
        fraction = 1 - self.elastic_fraction(energy, thickness)

        # Compute the number of electrons and the sigma
        if filter_width is not None:
            dE0 = position - filter_width / 2.0
            dE1 = position + filter_width / 2.0
            x0 = int(floor((dE0 - self.dE_min) / self.dE_step))
            x1 = int(ceil((dE1 - self.dE_min) / self.dE_step))
            x0 = max(x0, 0)
            x1 = min(x1, len(P) - 1)
            assert x1 > x0
            fraction *= C[x1] - C[x0]
            P = P[x0:x1]
            dE = dE[x0:x1]

        # Compute the spread
        if len(P) > 0 and np.sum(P) > 0:
            peak, fwhm = parakeet.landau.mpl_and_fwhm(energy / 1000, thickness)
            sigma = fwhm / (2 * sqrt(2 * log(2)))

            # This is a hack to compute the variance because the heavy tail of
            # the landau distribution makes the variance undefined and this
            # causes problems for large filter sizes. There is probably a
            # better way to do this so maybe should look at fixing.
            P *= np.exp(-0.5 * (dE - peak) ** 2 / (2 * sigma) ** 2)

            P /= np.sum(P)
            dE_mean = np.sum(P * dE) / np.sum(P)
            spread = sqrt(np.sum(P * (dE - dE_mean) ** 2) / np.sum(P)) * sqrt(2)
        else:
            spread = 0

        # Return the fraction and spread
        return fraction, spread


def get_energy_bins(
    energy,
    thickness,
    energy_spread=0.8,
    filter_energy=None,
    filter_width=None,
    dE_max=200,
    dE_step=5,
):
    """
    Get some energy bins with weights
    """

    # Ensure min and max such that when we split the mean dE will be at
    # sensible locations (i.e. we will have one at zero)
    dE_step_sub = 0.01
    assert dE_step < 10
    dE_min = -dE_step * (0.5 + int(10 / dE_step))
    dE_max = dE_step * (0.5 + int(dE_max / dE_step))
    assert dE_min < 0
    assert dE_max > dE_min

    # Check the filter width and step size
    if filter_energy is not None and filter_width is not None:
        # Adjust the step size
        num_step = int(filter_width / dE_step)
        dE_step = filter_width / num_step

        # The min and max energies
        filter_min = filter_energy - filter_width / 2.0
        filter_max = filter_energy + filter_width / 2.0

        # Set the bins
        bins = []
        for i in range(num_step):
            E1 = filter_min + i * dE_step
            E2 = E1 + dE_step
            bins.append((E1, E2))

    else:
        # Number of steps
        num_step = int((dE_max - dE_min) / dE_step)

        bins = []
        for i in range(num_step):
            E1 = dE_min + i * dE_step
            E2 = E1 + dE_step
            bins.append((E1, E2))

    # Get the distribution of energy losses
    optimizer = EnergyFilterOptimizer(
        energy_spread,
        dE_min=dE_min + dE_step_sub / 2.0,
        dE_max=dE_max + dE_step_sub / 2.0,
        dE_step=dE_step_sub,
    )
    dE, distribution = optimizer.energy_loss_distribution(energy, thickness)
    distribution /= np.sum(distribution)

    # The maximum spread
    dE_spread_max = sqrt(dE_step**2 / 12) * sqrt(2)

    # Loop over the subdivisions and compute mean energy, spread and total
    # weight. For each bin we take the distribution and compute the weighted
    # mean energy loss and the weighted variance as the energy spread. The mean
    # will always be within the energy bin and the variance will always be > 0
    # and < the variance of the uniform distribution within the bin.
    TINY = 1e-5
    nbins = len(bins)
    bin_energy = np.zeros(nbins)
    bin_spread = np.zeros(nbins)
    bin_weight = np.zeros(nbins)
    for i in range(nbins):
        E1, E2 = bins[i]
        select = (dE >= E1) & (dE < E2)
        dE_sub = dE[select]
        P_sub = distribution[select]
        dE_mean = np.mean(dE_sub)
        P_tot = np.sum(P_sub)
        if P_tot > 1e-7:
            P_sub = P_sub / P_tot
            dE_mean = np.sum(P_sub * dE_sub)
            dE_spread = np.sum(P_sub * ((dE_sub - dE_mean) ** 2))
            dE_spread = sqrt(dE_spread) * sqrt(2)
        else:
            dE_spread = dE_spread_max
        fudge = np.sqrt((dE_step_sub**2 / 12) * len(P_sub))
        assert (E2 - E1) <= (dE_step + TINY)
        assert dE_mean >= E1
        assert dE_mean <= E2
        assert dE_spread >= 0
        # assert dE_spread <= (dE_spread_max + fudge)
        bin_energy[i] = energy - dE_mean
        bin_weight[i] = P_tot
        bin_spread[i] = dE_spread

    # Return the bins and weights
    return bin_energy, bin_spread, bin_weight
