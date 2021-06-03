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
import numpy


class Scan(object):
    """
    A class to encapsulate an scan

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
            self.positions = numpy.zeros(shape=len(angles), dtype=numpy.float32)
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


def new(
    mode="still",
    axis=(0, 1, 0),
    start_angle=0,
    step_angle=0,
    start_pos=0,
    step_pos=0,
    num_images=1,
    exposure_time=1,
):
    """
    Create an scan

    Args:
        axis (array): The rotation axis
        start_angle (float): The starting angle (deg)
        step_angle (float): The angle step (deg)
        start_pos (float): The starting position (A)
        step_pos (float): The step in position (A)
        num_images (int): The number of images
        exposure_time (float): The exposure time (seconds)

    Returns:
        object: The scan object

    """
    if mode == "still":
        angles = [start_angle]
        positions = [start_pos]
    elif mode == "tilt_series":
        angles = start_angle + step_angle * numpy.arange(num_images)
        positions = numpy.full(
            shape=len(angles), fill_value=start_pos, dtype=numpy.float32
        )
    elif mode == "helical_scan":
        angles = start_angle + step_angle * numpy.arange(num_images)
        positions = start_pos + step_pos * numpy.arange(num_images)
    else:
        raise RuntimeError(f"Scan mode not recognised: {mode}")
    return Scan(
        axis=axis, angles=angles, positions=positions, exposure_time=exposure_time
    )
