#
# parakeet.lens.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
import parakeet.config


class Lens(object):
    """
    A class to encapsulate a lens

    """

    def __init__(
        self,
        m: int = 0,
        c_10: float = 20,
        c_12: float = 0.0,
        phi_12: float = 0.0,
        c_21: float = 0.0,
        phi_21: float = 0.0,
        c_23: float = 0.0,
        phi_23: float = 0.0,
        c_30: float = 0.04,
        c_32: float = 0.0,
        phi_32: float = 0.0,
        c_34: float = 0.0,
        phi_34: float = 0.0,
        c_41: float = 0.0,
        phi_41: float = 0.0,
        c_43: float = 0.0,
        phi_43: float = 0.0,
        c_45: float = 0.0,
        phi_45: float = 0.0,
        c_50: float = 0.0,
        c_52: float = 0.0,
        phi_52: float = 0.0,
        c_54: float = 0.0,
        phi_54: float = 0.0,
        c_56: float = 0.0,
        phi_56: float = 0.0,
        inner_aper_ang: float = 0.0,
        outer_aper_ang: float = 0.0,
        c_c: float = 0.0,
        current_spread: float = 0.0,
    ):
        """
        Initialise the lens

        Args:
            m: The vortex momentum
            c_10: The defocus (A)
            c_12: The 2-fold astigmatism (A)
            phi_12: The azimuthal angle of 2-fold astigmatism (degrees)
            c_21: The axial coma (A)
            phi_21: The azimuthal angle of axial coma (degrees)
            c_23: The 3-fold astigmatism (A)
            phi_23: The azimuthal angle of 3-fold astigmatism (degrees)
            c_30: The 3rd order spherical aberration (mm)
            c_32: The axial star aberration (A)
            phi_32: The azimuthal angle of axial star aberration (degrees)
            c_34: The 4-fold astigmatism (A)
            phi_34: The azimuthal angle of 4-fold astigmatism (degrees)
            c_41: The 4th order axial coma (A)
            phi_41: The azimuthal angle of 4th order axial coma (degrees)
            c_43: The 3-lobe aberration (A)
            phi_43: The azimuthal angle of 3-lobe aberration (degrees)
            c_45: The 5-fold astigmatism (A)
            phi_45: The azimuthal angle of 5-fold astigmatism (degrees)
            c_50: The 5th order spherical aberration (mm)
            c_52: The 5th order axial star aberration (A)
            phi_52: The azimuthal angle of 5th order axial star aberration (degrees)
            c_54: The 5th order rosette aberration (A)
            phi_54: The azimuthal angle of 5th order rosette aberration (degrees)
            c_56: The 6-fold astigmatism (A)
            phi_56: The azimuthal angle of 6-fold astigmatism (degrees)
            inner_aper_ang: The inner aperture (mrad)
            outer_aper_ang: The outer aperture (mrad)
            c_c: The chromatic abberation (mm)
            current_spread: dI / I where dI is the 1/e half width

        """
        self.m = m
        self.c_10 = c_10
        self.c_12 = c_12
        self.phi_12 = phi_12
        self.c_21 = c_21
        self.phi_21 = phi_21
        self.c_23 = c_23
        self.phi_23 = phi_23
        self.c_30 = c_30
        self.c_32 = c_32
        self.phi_32 = phi_32
        self.c_34 = c_34
        self.phi_34 = phi_34
        self.c_41 = c_41
        self.phi_41 = phi_41
        self.c_43 = c_43
        self.phi_43 = phi_43
        self.c_45 = c_45
        self.phi_45 = phi_45
        self.c_50 = c_50
        self.c_52 = c_52
        self.phi_52 = phi_52
        self.c_54 = c_54
        self.phi_54 = phi_54
        self.c_56 = c_56
        self.phi_56 = phi_56
        self.inner_aper_ang = inner_aper_ang
        self.outer_aper_ang = outer_aper_ang
        self.c_c = c_c
        self.current_spread = current_spread


def new(config: parakeet.config.Lens) -> Lens:
    """
    Create a lens

    Args:
        config: The lens configuration

    Returns:
        The lens model object

    """
    return Lens(**config.model_dump())
