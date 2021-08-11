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
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.stats import special_ortho_group


@dataclass
class PoseSet:
    """A description of poses of the system being simulated.

    Parameters
    __________

    orientations : np.ndarray
        (n, 3, 3) array of rotation matrices describing the pose of the system.
    shifts : np.ndarray
        (n, 3) array of how to shift the sample volume (units: A)
    """
    orientations: np.ndarray
    shifts: np.ndarray

    @property
    def rotation_converter(self):
        return R.from_matrix(self.orientations)

    @property
    def euler_angles(self):
        """Euler angle representation of the orientations.

        The Euler angles are intrinsic, right handed rotations around ZYZ.
        This matches the convention used by XMIPP/RELION.
        """
        return self.rotation_converter.as_euler(seq='ZYZ', degrees=True)

    @property
    def axis_angle(self):
        """Axis-angle representation of the orientations.

        Magnitude of the rotation vector corresponds to the angle in radians.
        """
        return self.rotation_converter.as_rotvec()

    def __len__(self):
        n_orientations = self.orientations.shape[0]
        n_shifts = self.shifts.shape[0]
        if n_orientations != n_shifts:
            raise RuntimeError(
                'Number of orientations is different to the number of shifts.'
            )
        return n_shifts


class Scan(ABC):
    """A base class defining the interface for image sampling in a simulation.
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
        """The number of images to sample"""
        return len(self.poses)


class SingleAxisScan(object):
    """A scan of angles around a single rotation axis.
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
            self.axis = (0, 1, 0)
        else:
            self.axis = axis
        if angles is None:
            self.angles = [0]
        else:
            self.angles = angles
        if positions is None:
            self.positions = np.zeros(shape=len(angles), dtype=np.float32)
        else:
            self.positions = positions
        assert len(self.angles) == len(self.positions)
        self.exposure_time = exposure_time

    def __len__(self):
        """
        Returns:
            int: The number of images in the scan

        """
        assert len(self.angles) == len(self.positions)
        return len(self.angles)


class UniformAngularScan(Scan):
    """A uniform scan of orientations, no shifts.
    """

    def __init__(self, n: int):
        """
        Parameters
        __________
        n : int
            The number of uniform orientational samples
        """
        self.n = int(n)

    @property
    def exposure_times(self) -> np.ndarray:
        pass

    @property
    def poses(self) -> PoseSet:
        # Draw n uniform samples from SO(3)
        orientations = special_ortho_group.rvs(dim=3, size=self.n)
        shifts = np.zeros(shape=(self.n, 3))

        # Create and return PoseSet object
        return PoseSet(orientations, shifts)


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
    if angles is None:
        if mode == "still":
            angles = [start_angle]
        elif mode == "tilt_series":
            angles = start_angle + step_angle * np.arange(num_images)
        elif mode == "dose_symmetric":
            angles = start_angle + step_angle * np.arange(num_images)
            angles = np.array(sorted(angles, key=lambda x: abs(x)))
        elif mode == "helical_scan":
            angles = start_angle + step_angle * np.arange(num_images)
        else:
            raise RuntimeError(f"Scan mode not recognised: {mode}")
    if positions is None:
        if mode == "still":
            positions = [start_pos]
        elif mode == "tilt_series":
            positions = np.full(
                shape=len(angles), fill_value=start_pos, dtype=np.float32
            )
        elif mode == "dose_symmetric":
            positions = np.full(
                shape=len(angles), fill_value=start_pos, dtype=np.float32
            )
        elif mode == "helical_scan":
            positions = start_pos + step_pos * np.arange(num_images)
        else:
            raise RuntimeError(f"Scan mode not recognised: {mode}")
    return SingleAxisScan(
        axis=axis, angles=angles, positions=positions, exposure_time=exposure_time
    )
