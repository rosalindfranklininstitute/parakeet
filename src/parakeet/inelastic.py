#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import parakeet.landau
from math import sqrt, pi, cos, exp, log


def zero_loss_fraction(shape, angle):
    """
    Compute the zero loss fraction

    """
    TINY = 1e-10
    if shape["type"] == "cube":
        D0 = shape["cube"]["length"]
        thickness = D0 / (cos(pi * angle / 180.0) + TINY)
    elif shape["type"] == "cuboid":
        D0 = shape["cuboid"]["length_z"]
        thickness = D0 / (cos(pi * angle / 180.0) + TINY)
    elif shape["type"] == "cylinder":
        thickness = shape["cylinder"]["radius"] * 2
    mean_free_path = 3150  # A for Amorphous Ice at 300 keV
    electron_fraction = exp(-thickness / mean_free_path)
    return electron_fraction


def mp_loss_fraction(shape, angle):
    """
    Compute the inelastic fraction

    """
    TINY = 1e-10
    if shape["type"] == "cube":
        D0 = shape["cube"]["length"]
        thickness = D0 / (cos(pi * angle / 180.0) + TINY)
    elif shape["type"] == "cuboid":
        D0 = shape["cuboid"]["length_z"]
        thickness = D0 / (cos(pi * angle / 180.0) + TINY)
    elif shape["type"] == "cylinder":
        thickness = shape["cylinder"]["radius"] * 2
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

    """
    TINY = 1e-10
    if shape["type"] == "cube":
        D0 = shape["cube"]["length"]
        thickness = D0 / (cos(pi * angle / 180.0) + TINY)
    elif shape["type"] == "cuboid":
        D0 = shape["cuboid"]["length_z"]
        thickness = D0 / (cos(pi * angle / 180.0) + TINY)
    elif shape["type"] == "cylinder":
        thickness = shape["cylinder"]["radius"] * 2
    thickness = min(thickness, 100000)  # Maximum 10 um - to avoid issues at high tilt
    peak, fwhm = parakeet.landau.mpl_and_fwhm(energy, thickness)
    return peak, fwhm / (2 * sqrt(2 * log(2)))
