#
# parakeet.scan.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#
from abc import ABC, abstractmethod

import numpy as np
from math import pi
from scipy.stats import special_ortho_group
from scipy.spatial.transform import Rotation as R

from parakeet.pose import PoseSet


class Scan(ABC):
    """
    A base class defining the interface for image sampling in a simulation.

    """

    @property
    @abstractmethod
    def poses(self) -> PoseSet:
        pass

    @property
    @abstractmethod
    def exposure_times(self) -> np.ndarray:
        pass

    def __len__(self) -> int:
        """
        The number of images to sample

        """
        return len(self.poses)


class SingleAxisScan(Scan):
    """
    A scan of angles around a single rotation axis.

    """

    def __init__(self, axis=None, angles=None, positions=None, exposure_time=None):
        """
        Initialise the scan

        Args:
            axis (tuple): The rotation axis
            angles (list): The rotation angles (units: degrees)
            positions (list): The positions to shift (units: A)
            exposure_time (float): The exposure time (units: seconds)

        """
        if axis is None:
            self.axis = np.array((0, 1, 0))
        else:
            self.axis = axis
        if angles is None:
            self.angles = np.array([0])
        else:
            self.angles = np.array(angles)
        if positions is None:
            self.positions = numpy.zeros(shape=len(angles), dtype=numpy.float32)
        else:
            self.positions = np.array(positions)
        assert len(self.angles) == len(self.positions)
        self.exposure_time = exposure_time

    @property
    def poses(self) -> PoseSet:
        axis = np.array(self.axis)
        axis = axis / np.linalg.norm(axis)
        angles = np.array(self.angles) * pi / 180.0
        orientations = np.array([axis * a for a in angles])
        orientations = R.from_rotvec(orientations).as_matrix()
        return PoseSet(orientations, self.positions)

    @property
    def exposure_times(self):
        """
        Returns:
            np.ndarray: The exposure times array

        """
        return np.ones(len(self)) * self.exposure_time


class UniformAngularScan(Scan):
    """
    A uniform scan of orientations, no shifts.

    """

    def __init__(self, n: int):
        """
        Args:
            n (int): The number of uniform orientational samples

        """
        self.n = int(n)

    @property
    def exposure_time(self) -> float:
        1

    @property
    def angles(self) -> Rotation:
        # Draw n uniform samples from SO(3)
        return Rotation.from_matrix(special_ortho_group.rvs(dim=3, size=self.n))

    @property
    def positions(self):
        return numpy.zeros(self.n, dtype=numpy.float32)


def new(
    mode="still",
    axis=(0, 1, 0),
    angles=None,
    positions=None,
    start_angle=0,
    step_angle=0,
    start_pos=0,
    step_pos=0,
    num_images=1,
    exposure_time=1,
):
    """
    Create an scan

    If angles or positions is None they are generated form the other
    parameters.

    Args:
        mode (str): The type of scan (still, tilt_series, dose_symmetric, helical_scan)
        axis (array): The rotation axis
        angles (array): The rotation angles
        positions (array): The positions
        start_angle (float): The starting angle (deg)
        step_angle (float): The angle step (deg)
        start_pos (float): The starting position (A)
        step_pos (float): The step in position (A)
        num_images (int): The number of images
        exposure_time (float): The exposure time (seconds)

    Returns:
        object: The scan object

    """
    if mode == "single_particle":
        return UniformAngularScan(num_images)
    if angles is None:
        if mode == "still":
            angles = [start_angle]
        elif mode == "tilt_series":
            angles = start_angle + step_angle * numpy.arange(num_images)
        elif mode == "dose_symmetric":
            angles = start_angle + step_angle * numpy.arange(num_images)
            angles = numpy.array(sorted(angles, key=lambda x: abs(x)))
        elif mode == "helical_scan":
            angles = start_angle + step_angle * numpy.arange(num_images)
        else:
            raise RuntimeError(f"Scan mode not recognised: {mode}")
    if positions is None:
        if mode == "still":
            positions = [start_pos]
        elif mode == "tilt_series":
            positions = numpy.full(
                shape=len(angles), fill_value=start_pos, dtype=numpy.float32
            )
        elif mode == "dose_symmetric":
            positions = numpy.full(
                shape=len(angles), fill_value=start_pos, dtype=numpy.float32
            )
        elif mode == "helical_scan":
            positions = start_pos + step_pos * numpy.arange(num_images)
        else:
            raise RuntimeError(f"Scan mode not recognised: {mode}")
    return Scan(
        axis=axis, angles=angles, positions=positions, exposure_time=exposure_time
    )
